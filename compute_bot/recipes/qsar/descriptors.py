"""Molecular descriptors: RDKit 2D, Morgan bits, Mordred 2D and MOPAC PM6 (via the `mopac` binary).

Slow sources (Mordred, MOPAC) keep one JSON per molecule in a cache folder, so a restarted job skips molecules that
are already done. A molecule that fails gets NaN and an entry in `errors`; it never stops the run.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

MOPAC_KEYWORDS = "PM6 PRECISE GNORM=0.5 ENPART THREADS=1 T=1H"
MOPAC_VERSION = "pm6-v1"  # part of the cache key: change it when the keywords or the parsing change


def _quiet_rdkit():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def prepare_mols(smiles):
    """Parse SMILES and keep the largest fragment (salts and counter-ions out). Returns (mols, errors)."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    _quiet_rdkit()
    chooser = rdMolStandardize.LargestFragmentChooser()
    mols, errors = [], {}
    for i, s in enumerate(smiles):
        if s is None or (isinstance(s, float) and np.isnan(s)) or not str(s).strip():
            mols.append(None)
            errors[i] = "brak SMILES"
            continue
        m = Chem.MolFromSmiles(str(s).strip())
        if m is None:
            mols.append(None)
            errors[i] = f"nie umiem odczytać SMILES {s}"
            continue
        if len(Chem.GetMolFrags(m)) > 1:
            m = chooser.choose(m)
        mols.append(m)
    return mols, errors


def mol_key(mol):
    from rdkit import Chem
    return hashlib.sha1(Chem.MolToSmiles(mol).encode()).hexdigest()[:16]


def rdkit_block(mols):
    from rdkit.Chem import Descriptors

    _quiet_rdkit()
    names = [n for n, _ in Descriptors._descList]
    rows = []
    for m in mols:
        if m is None:
            rows.append(dict.fromkeys(names, np.nan))
        else:
            rows.append(Descriptors.CalcMolDescriptors(m, missingVal=np.nan))
    df = pd.DataFrame(rows, columns=names).astype(float)
    return df.replace([np.inf, -np.inf], np.nan)


def morgan_block(mols, radius=2, n_bits=1024):
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    data = np.full((len(mols), n_bits), np.nan)
    for i, m in enumerate(mols):
        if m is not None:
            data[i] = gen.GetFingerprintAsNumPy(m)
    return pd.DataFrame(data, columns=[f"Morgan_{j:04d}" for j in range(n_bits)])


def _write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, allow_nan=True))
    os.replace(tmp, path)


def mordred_block(mols, cache_dir, threads=1, progress=None, chunk=40):
    from mordred import Calculator
    from mordred import descriptors as md

    calc = Calculator(md, ignore_3D=True)
    names = [str(d) for d in calc.descriptors]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    keys = [mol_key(m) if m is not None else None for m in mols]
    todo = [i for i, k in enumerate(keys) if k and not (cache_dir / f"{k}.json").exists()]
    done = len([k for k in keys if k]) - len(todo)
    for start in range(0, len(todo), chunk):
        idx = todo[start:start + chunk]
        nproc = threads if len(idx) >= 20 and threads > 1 else 1
        res = calc.pandas([mols[i] for i in idx], nproc=nproc, quiet=True)
        res = res.fill_missing(np.nan) if hasattr(res, "fill_missing") else res
        res = res.apply(pd.to_numeric, errors="coerce").astype(float)
        for i, row in zip(idx, res.to_numpy(), strict=True):
            _write_json(cache_dir / f"{keys[i]}.json", [None if not np.isfinite(v) else float(v) for v in row])
        done += len(idx)
        if progress:
            progress(f"Mordred {done}/{len([k for k in keys if k])}")
    data = np.full((len(mols), len(names)), np.nan)
    for i, k in enumerate(keys):
        if k:
            vals = json.loads((cache_dir / f"{k}.json").read_text())
            data[i] = [np.nan if v is None else v for v in vals]
    return pd.DataFrame(data, columns=names)


# ---------------------------------------------------------------- MOPAC

_NUM = r"([-+]?\d+\.?\d*(?:[EeDd][-+]?\d+)?)"


def parse_mopac(text):
    """Descriptors from a MOPAC .out file. Raises ValueError when the job did not finish."""
    def grab(pattern):
        m = re.search(pattern, text)
        return float(m.group(1).replace("D", "E")) if m else np.nan

    hof = grab(r"FINAL HEAT OF FORMATION\s*=\s*" + _NUM)
    if np.isnan(hof):
        raise ValueError("MOPAC nie podał ciepła tworzenia (obliczenie się nie skończyło)")
    d = {"PM6_HOF": hof}
    m = re.search(r"HOMO LUMO ENERGIES \(EV\)\s*=\s*" + _NUM + r"\s+" + _NUM, text)
    if m:
        homo, lumo = float(m.group(1)), float(m.group(2))
    else:  # open shell: ALPHA SOMO LUMO (EV) = ...
        m = re.search(r"ALPHA SOMO LUMO \(EV\)\s*=\s*" + _NUM + r"\s+" + _NUM, text)
        homo, lumo = (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)
    d.update(PM6_HOMO=homo, PM6_LUMO=lumo, PM6_gap=lumo - homo)
    d["PM6_hardness"] = (lumo - homo) / 2
    d["PM6_electronegativity"] = -(homo + lumo) / 2
    d["PM6_electrophilicity"] = d["PM6_electronegativity"] ** 2 / (2 * d["PM6_hardness"]) if lumo > homo else np.nan
    d["PM6_IP"] = grab(r"IONIZATION POTENTIAL\s*=\s*" + _NUM)
    d["PM6_dipole"] = np.nan
    m = re.search(r"\n\s*SUM\s+" + r"\s+".join([_NUM] * 4), text)
    if m:
        d["PM6_dipole"] = float(m.group(4))
    d["PM6_Etot"] = grab(r"ETOT \(EONE \+ ETWO\)\s+" + _NUM)
    d["PM6_COSMO_area"] = grab(r"COSMO AREA\s*=\s*" + _NUM)
    d["PM6_COSMO_volume"] = grab(r"COSMO VOLUME\s*=\s*" + _NUM)
    charges, hydrogens = [], []
    block = re.search(r"NET ATOMIC CHARGES AND DIPOLE CONTRIBUTIONS(.*?)DIPOLE", text, re.S)
    if block:
        for line in block.group(1).splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    q = float(parts[2])
                except ValueError:
                    continue
                charges.append(q)
                if parts[1] == "H":
                    hydrogens.append(q)
    d["PM6_qmax"] = max(charges) if charges else np.nan
    d["PM6_qmin"] = min(charges) if charges else np.nan
    d["PM6_qHmax"] = max(hydrogens) if hydrogens else np.nan
    return d


MOPAC_COLUMNS = ["PM6_HOF", "PM6_HOMO", "PM6_LUMO", "PM6_gap", "PM6_hardness", "PM6_electronegativity",
                 "PM6_electrophilicity", "PM6_IP", "PM6_dipole", "PM6_Etot", "PM6_COSMO_area", "PM6_COSMO_volume",
                 "PM6_qmax", "PM6_qmin", "PM6_qHmax"]


def find_mopac():
    exe = shutil.which("mopac") or shutil.which("mopac", path=os.path.dirname(sys.executable))
    return exe


def embed_3d(mol, seed=42):
    """RDKit ETKDG v3 conformer, then MMFF94 (UFF when MMFF has no parameters). Returns the molecule with Hs."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mh, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mh, params) != 0:
            raise ValueError("RDKit nie wygenerował geometrii 3D")
    if AllChem.MMFFHasAllMoleculeParams(mh):
        AllChem.MMFFOptimizeMolecule(mh, maxIters=5000)
    else:
        AllChem.UFFOptimizeMolecule(mh, maxIters=5000)
    return mh


def mopac_input(mh, title="mol"):
    from rdkit import Chem

    charge = Chem.GetFormalCharge(mh)
    electrons = sum(a.GetAtomicNum() for a in mh.GetAtoms()) - charge
    kw = f"{MOPAC_KEYWORDS} CHARGE={charge}" + (" UHF DOUBLET" if electrons % 2 else "")
    conf = mh.GetConformer()
    lines = [kw, title, ""]
    for a in mh.GetAtoms():
        p = conf.GetAtomPosition(a.GetIdx())
        lines.append(f"{a.GetSymbol():2s} {p.x:12.6f} 1 {p.y:12.6f} 1 {p.z:12.6f} 1")
    return "\n".join(lines) + "\n"


def mopac_one(mol, key, cache_dir, exe, seed=42, timeout=3600):
    """PM6 optimisation of one molecule; result (dict or {'error': ...}) cached as <key>.json."""
    cache_dir = Path(cache_dir)
    out_json = cache_dir / f"{key}.json"
    if out_json.exists():
        return json.loads(out_json.read_text())
    work = cache_dir / f"tmp_{key}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        mh = embed_3d(mol, seed)
        (work / "mol.mop").write_text(mopac_input(mh))
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        subprocess.run([exe, "mol.mop"], cwd=work, env=env, capture_output=True, timeout=timeout, check=False)
        out = work / "mol.out"
        if not out.exists():
            raise ValueError("MOPAC nie zapisał wyniku")
        text = out.read_text(errors="replace")
        result = parse_mopac(text)
        shutil.copy(out, cache_dir / f"{key}.out")
    except (ValueError, subprocess.TimeoutExpired, OSError) as exc:
        result = {"error": str(exc)[:200]}
    _write_json(out_json, result)
    shutil.rmtree(work, ignore_errors=True)
    return result


def mopac_block(mols, cache_dir, threads=1, progress=None, seed=42):
    exe = find_mopac()
    if exe is None:
        raise FileNotFoundError("mopac")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    keys = {i: mol_key(m) + "_" + MOPAC_VERSION for i, m in enumerate(mols) if m is not None}
    results, errors = {}, {}
    total = len(keys)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, threads)) as pool:
        futures = {pool.submit(mopac_one, mols[i], k, cache_dir, exe, seed): i for i, k in keys.items()}
        for fut in as_completed(futures):
            i = futures[fut]
            res = fut.result()
            if "error" in res:
                errors[i] = "MOPAC: " + res["error"]
            else:
                results[i] = res
            done += 1
            if progress:
                progress(f"MOPAC PM6 {done}/{total}")
    data = pd.DataFrame([results.get(i, {}) for i in range(len(mols))], columns=MOPAC_COLUMNS).astype(float)
    return data, errors
