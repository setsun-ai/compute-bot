"""Receptor: protein chains only, waters/HETATM out, missing atoms and hydrogens at pH (pdbfixer), PDBQT.

PDBQT is written with Meeko 0.8 (Polymer + PDBQTWriterLegacy). Why Meeko rather than Open Babel: Meeko matches every
residue to a chemical template, so the protonation pdbfixer chose (HID/HIE/HIP, CYX, charged termini) becomes the
right AutoDock atom types and a residue that does not fit is reported instead of being silently mistyped; it is also
the toolkit that writes the ligands and reads the poses back. It is strict: a residue it cannot build (typically a
terminal residue with an odd OXT placement after pdbfixer) raises, so such residues are removed one at a time and
reported (flagged when they are near the box). Open Babel (`obabel -xr`) is the last resort: it always writes a file,
but perceives bonds and types heuristically.
"""
from __future__ import annotations

import io
import re
import shutil
import subprocess

import numpy as np

from compute_bot.recipes.dock.common import UserError

MAX_DROPPED = 25


def clean_structure(st, keep_waters: bool):
    """First model, first altloc, protein chains only; waters kept on request; everything else out.

    Returns (new gemmi Structure, info). info lists what was removed so the summary can say it.
    """
    import gemmi

    st = st.clone()
    while len(st) > 1:
        del st[1]
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    removed, waters, kept_chains, dropped_chains = {}, 0, [], []
    removed_atoms = []  # (resname, element, xyz) of removed single-atom groups: metal ions near the pocket matter
    model = st[0]
    for chain in model:
        has_protein = False
        for i in reversed(range(len(chain))):
            res = chain[i]
            info = gemmi.find_tabulated_residue(res.name)
            is_aa = info is not None and info.is_amino_acid()
            polymer = res.entity_type == gemmi.EntityType.Polymer or (
                res.entity_type == gemmi.EntityType.Unknown and res.het_flag == "A")
            if polymer and is_aa:
                has_protein = True
                continue
            if res.is_water():
                if keep_waters:
                    continue
                waters += 1
            else:
                removed[res.name] = removed.get(res.name, 0) + 1
                if len(res) == 1:
                    removed_atoms.append((res.name, res[0].element.name, res[0].pos.tolist()))
            del chain[i]
        (kept_chains if has_protein else dropped_chains).append(chain.name)
    # a chain without amino acids (DNA, a peptide-free ligand chain) is removed with all its residues
    for i in reversed(range(len(model))):
        chain = model[i]
        if chain.name in dropped_chains:
            for res in chain:
                removed[res.name] = removed.get(res.name, 0) + 1
            del model[i]
    st.remove_empty_chains()
    if not any(len(ch) for ch in st[0]):
        raise UserError("W strukturze nie ma łańcucha białkowego – nie ma do czego dokować.")
    return st, {"chains": sorted(set(kept_chains)), "removed_hetero": removed, "removed_waters": waters,
                "dropped_chains": [c for c in dropped_chains if c not in kept_chains], "removed_atoms": removed_atoms}


def add_hydrogens(pdb_text: str, ph: float) -> str:
    """pdbfixer: non-standard residues to standard, missing heavy atoms (no loop modelling), hydrogens at pH."""
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    fixer = PDBFixer(pdbfile=io.StringIO(pdb_text))
    fixer.findMissingResidues()
    fixer.missingResidues = {}  # do not invent loops: a modelled loop in the pocket would bias docking
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(float(ph))
    out = io.StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, out, keepIds=True)
    return out.getvalue()


def _residue_key(line: str) -> str:
    return f"{line[21].strip()}:{line[22:27].strip()}"


def drop_residues(pdb_text: str, keys) -> str:
    keys = set(keys)
    return "".join(ln for ln in pdb_text.splitlines(keepends=True)
                   if not (ln.startswith(("ATOM", "HETATM")) and _residue_key(ln) in keys))


def meeko_pdbqt(pdb_text: str):
    """(pdbqt, pdb of what went in, removed residue keys). Raises RuntimeError when Meeko keeps failing."""
    from meeko import PDBQTWriterLegacy, Polymer

    dropped = []
    while True:
        try:
            poly = Polymer.from_pdb_string(pdb_text, allow_bad_res=True, default_altloc="A")
            break
        except RuntimeError as exc:
            m = re.search(r"key ([A-Za-z0-9]*:-?\d+[A-Za-z]?)", str(exc))
            if not m or len(dropped) >= MAX_DROPPED or m.group(1) in dropped:
                raise
            dropped.append(m.group(1))
            pdb_text = drop_residues(pdb_text, [m.group(1)])
    ignored = list(poly.get_ignored_monomers())
    if ignored:  # allow_bad_res left these out of the PDBQT: drop them from the PDB too, so files agree
        dropped.extend(ignored)
        pdb_text = drop_residues(pdb_text, ignored)
    rigid, _flex = PDBQTWriterLegacy.write_from_polymer(poly)
    return rigid, pdb_text, dropped


def obabel_pdbqt(pdb_text: str) -> str:
    exe = shutil.which("obabel")
    if not exe:
        raise UserError("Nie udało się przygotować receptora (Meeko zawiódł, brak Open Babel).")
    proc = subprocess.run([exe, "-ipdb", "-opdbqt", "-xr", "-xh"], input=pdb_text, capture_output=True, text=True,
                          timeout=600)
    if proc.returncode != 0 or "ATOM" not in proc.stdout:
        raise UserError("Nie udało się przygotować receptora (Meeko i Open Babel zawiodły). Sprawdź plik PDB.")
    return "".join(ln + "\n" for ln in proc.stdout.splitlines() if ln.startswith(("ATOM", "HETATM", "TER")))


def prepare(pdb_text: str, ph: float):
    """Hydrogens + PDBQT. Returns (pdbqt, receptor pdb with H, method, dropped residue keys)."""
    with_h = add_hydrogens(pdb_text, ph)
    try:
        pdbqt, with_h, dropped = meeko_pdbqt(with_h)
        return pdbqt, with_h, "Meeko", dropped
    except (RuntimeError, ValueError, KeyError, ImportError) as exc:
        return obabel_pdbqt(with_h), with_h, f"Open Babel (Meeko: {str(exc)[:120]})", []


def pdb_atoms(pdb_text: str):
    """[(chain, resname, resnum, atom name, element, (x, y, z))] for ATOM/HETATM lines."""
    out = []
    for ln in pdb_text.splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        name = ln[12:16].strip()
        element = ln[76:78].strip() or re.sub(r"[^A-Za-z]", "", name)[:1]
        out.append((ln[21].strip(), ln[17:20].strip(), ln[22:27].strip(), name, element.capitalize(),
                    (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))))
    return out


def residues_near(pdb_text: str, keys, center, size, margin=4.0):
    """Which of the residue keys have an atom inside the box (+margin) in the original structure."""
    keys = set(keys)
    c, h = np.asarray(center), np.asarray(size) / 2 + margin
    near = set()
    for chain, _resname, resnum, _name, _el, xyz in pdb_atoms(pdb_text):
        key = f"{chain}:{resnum}"
        if key in keys and np.all(np.abs(np.asarray(xyz) - c) <= h):
            near.add(key)
    return sorted(near)
