"""Input structure: find or download it, check it (models, chains, gaps, missing atoms, HETATM, waters, altlocs)
and write a clean protein-only PDB for pdb2gmx (repairing missing heavy atoms with pdbfixer when needed)."""
from __future__ import annotations

import re
import urllib.request
from collections import Counter
from pathlib import Path

from .gmxtools import RecipeError

STRUCTURE_EXT = (".pdb", ".ent", ".cif", ".mmcif")

# heavy atoms of the 20 standard residues (OXT not counted: termini are rebuilt by pdb2gmx)
HEAVY = {
    "ALA": "N CA C O CB", "ARG": "N CA C O CB CG CD NE CZ NH1 NH2", "ASN": "N CA C O CB CG OD1 ND2",
    "ASP": "N CA C O CB CG OD1 OD2", "CYS": "N CA C O CB SG", "GLN": "N CA C O CB CG CD OE1 NE2",
    "GLU": "N CA C O CB CG CD OE1 OE2", "GLY": "N CA C O", "HIS": "N CA C O CB CG ND1 CD2 CE1 NE2",
    "ILE": "N CA C O CB CG1 CG2 CD1", "LEU": "N CA C O CB CG CD1 CD2", "LYS": "N CA C O CB CG CD CE NZ",
    "MET": "N CA C O CB CG SD CE", "PHE": "N CA C O CB CG CD1 CD2 CE1 CE2 CZ", "PRO": "N CA C O CB CG CD",
    "SER": "N CA C O CB OG", "THR": "N CA C O CB OG1 CG2", "TRP": "N CA C O CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
    "TYR": "N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH", "VAL": "N CA C O CB CG1 CG2",
}
HEAVY = {k: v.split() for k, v in HEAVY.items()}


def plural(n: int, one: str, few: str, many: str) -> str:
    if n == 1:
        return f"{n} {one}"
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return f"{n} {few}"
    return f"{n} {many}"


# --- input ----------------------------------------------------------------------------------------------------------

def find_input(work: Path, code: str) -> Path:
    """The structure to use: params.pdb (downloaded once into the work dir) or the single structure file sent."""
    code = code.strip().upper()
    if code:
        if not re.fullmatch(r"[0-9][A-Z0-9]{3}", code):
            raise RecipeError(f"pdb={code}? Kod PDB to 4 znaki, np. 1L2Y")
        for ext in (".pdb", ".cif"):
            for name in (code + ext, code.lower() + ext):
                if (work / name).exists():
                    return work / name
        return download(code, work)
    found = sorted(p for p in work.iterdir() if p.is_file() and p.suffix.lower() in STRUCTURE_EXT)
    if not found:
        raise RecipeError("brak struktury: wyślij plik .pdb/.cif albo podaj kod, np. pdb=1L2Y")
    if len(found) > 1:
        raise RecipeError("kilka plików ze strukturą (" + ", ".join(p.name for p in found) +
                          "): zostaw jeden albo podaj pdb=KOD")
    return found[0]


def download(code: str, work: Path) -> Path:
    errors = []
    for ext in (".pdb", ".cif"):  # large entries exist only as mmCIF
        url = f"https://files.rcsb.org/download/{code}{ext}"
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
        except OSError as exc:
            errors.append(f"{url}: {exc}")
            continue
        if data:
            tmp = work / f".{code}{ext}.part"
            tmp.write_bytes(data)
            tmp.rename(work / f"{code}{ext}")
            print(f"Pobrano {code}{ext} z RCSB", flush=True)
            return work / f"{code}{ext}"
    raise RecipeError(f"nie udało się pobrać {code} z RCSB ({'; '.join(errors)[:300]}). Wyślij plik .pdb")


# --- check ----------------------------------------------------------------------------------------------------------

def _is_amino_acid(gemmi, name: str) -> bool:
    info = gemmi.find_tabulated_residue(name)
    return info is not None and info.is_amino_acid()


def _scan_model(gemmi, st, model) -> dict:
    chains, hetero, nonstandard, missing = [], Counter(), Counter(), {}
    waters = altloc = 0
    hydrogens = False
    for ch in model:
        residues = []
        for res in ch:
            if res.is_water():
                waters += 1
                continue
            if not _is_amino_acid(gemmi, res.name):
                hetero[res.name] += 1
                continue
            residues.append(res)
            names = set()
            for at in res:
                altloc += at.has_altloc()
                hydrogens |= at.is_hydrogen()
                names.add(at.name)
            if res.name in HEAVY:
                lost = [a for a in HEAVY[res.name] if a not in names]
                if lost:
                    missing[f"{ch.name}:{res.seqid.num}{res.seqid.icode.strip()} {res.name}"] = lost
            else:
                nonstandard[res.name] += 1
        if not residues:
            continue
        breaks = []
        for a, b in zip(residues, residues[1:], strict=False):
            c, n = a.find_atom("C", "*"), b.find_atom("N", "*")
            if c is not None and n is not None and c.pos.dist(n.pos) > 2.0:
                breaks.append(f"{a.seqid.num}–{b.seqid.num}")
        seq = "".join((gemmi.find_tabulated_residue(r.name).one_letter_code or "x").upper() for r in residues)
        seqres = None
        polymer = ch.get_polymer()
        if len(polymer):
            entity = st.get_entity_of(polymer)
            if entity is not None and entity.full_sequence:
                seqres = len(entity.full_sequence)
        chains.append({"name": ch.name, "residues": len(residues), "sequence": seq, "seqres": seqres,
                       "first": residues[0].seqid.num, "last": residues[-1].seqid.num, "breaks": breaks})
    return {"chains": chains, "waters": waters, "hetero": dict(hetero), "nonstandard": dict(nonstandard),
            "altloc_atoms": altloc, "hydrogens": hydrogens, "missing_atoms": missing}


def inspect_structure(path: Path, model: int = 1) -> dict:
    """What is in the file (for the report); the per-chain details are for `model` (1-based)."""
    import gemmi

    st = gemmi.read_structure(str(path))
    st.setup_entities()
    if len(st) == 0:
        raise RecipeError(f"{Path(path).name}: brak atomów w pliku")
    meta = {k: st.info[k] for k in st.info}
    info = {"file": Path(path).name, "title": meta.get("_struct.title", ""),
            "method": meta.get("_exptl.method", ""), "resolution": st.resolution or None, "models": len(st)}
    info.update(_scan_model(gemmi, st, st[min(max(model, 1), len(st)) - 1]))
    return info


def describe_structure(info: dict, model: int) -> str:
    """Owner-facing report of the structure check (Polish)."""
    lines = [f"Plik: {info['file']}" + (f" – {info['title']}" if info.get("title") else "")]
    method = info.get("method") or "?"
    res = f", rozdzielczość {info['resolution']:.2f} Å" if info.get("resolution") else ""
    lines.append(f"Metoda: {method}{res}; {plural(info['models'], 'model', 'modele', 'modeli')}"
                 + (f", wybrany model {model}" if info["models"] > 1 else ""))
    for c in info["chains"]:
        extra = f", SEQRES {c['seqres']}" if c.get("seqres") else ""
        lines.append(f"Łańcuch {c['name']}: {c['residues']} reszt ({c['first']}–{c['last']}{extra}) {c['sequence']}")
        if c.get("seqres") and c["seqres"] > c["residues"]:
            lines.append(f"  brakuje {c['seqres'] - c['residues']} reszt względem SEQRES")
        if c["breaks"]:
            lines.append(f"  przerwy w łańcuchu między resztami: {', '.join(c['breaks'])}")
    lines.append(f"Woda krystaliczna: {info['waters']}")
    het = ", ".join(f"{k} ({v})" for k, v in sorted(info["hetero"].items())) or "brak"
    lines.append(f"HETATM (ligandy, jony): {het}")
    if info["nonstandard"]:
        lines.append("Niestandardowe aminokwasy: " + ", ".join(f"{k} ({v})" for k, v in info["nonstandard"].items()))
    lines.append(f"Atomy z altloc: {info['altloc_atoms']}; wodory w pliku: {'tak' if info['hydrogens'] else 'nie'}")
    if info["missing_atoms"]:
        lines.append("Brakujące atomy ciężkie: " + "; ".join(f"{k} {' '.join(v)}"
                                                            for k, v in info["missing_atoms"].items()))
    else:
        lines.append("Brakujące atomy ciężkie: brak")
    return "\n".join(lines)


# --- clean protein --------------------------------------------------------------------------------------------------

def prepare_structure(path: Path, out: Path, model: int, chains: str) -> list[str]:
    """Write `out`: one model, chosen chains, protein only, first altloc, no hydrogens; repaired with pdbfixer when
    heavy atoms or internal residues are missing or residues are non-standard. Returns notes for the owner."""
    import gemmi

    st = gemmi.read_structure(str(path))
    st.setup_entities()
    notes = []
    if model > len(st):
        raise RecipeError(f"model={model}, a plik ma {plural(len(st), 'model', 'modele', 'modeli')}")
    for i in reversed(range(len(st))):
        if i != model - 1:
            del st[i]
    m = st[0]
    names = list(dict.fromkeys(ch.name for ch in m))
    if chains.strip():
        want = [c.strip() for c in chains.split(",") if c.strip()]
        absent = [c for c in want if c not in names]
        if absent:
            raise RecipeError(f"nie ma łańcucha {', '.join(absent)}; w modelu {model} są: {', '.join(names)}")
        for i in reversed(range(len(m))):
            if m[i].name not in want:
                del m[i]
        notes.append(f"Zostawiono łańcuchy: {', '.join(want)} (z {', '.join(names)})")
    waters, hetero = 0, Counter()
    for ch in m:
        for i in reversed(range(len(ch))):
            res = ch[i]
            if res.is_water():
                waters += 1
                del ch[i]
            elif not _is_amino_acid(gemmi, res.name):
                hetero[res.name] += 1
                del ch[i]
    if waters:
        notes.append(f"Usunięto wodę krystaliczną ({waters}); białko dostaje nowe pudełko wody")
    if hetero:
        notes.append("Usunięto HETATM (ligandy/jony, pole siłowe ich nie zna): " +
                     ", ".join(f"{k} ({v})" for k, v in sorted(hetero.items())))
    scan = _scan_model(gemmi, st, m)
    if scan["altloc_atoms"]:
        notes.append(f"Altloc: zostawiono pierwszy wariant ({scan['altloc_atoms']} atomów z altloc)")
    if scan["hydrogens"]:
        notes.append("Wodory z pliku pominięte – pdb2gmx dodaje je od nowa (-ignh)")
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    st.remove_empty_chains()
    if not len(st[0]):
        raise RecipeError("po usunięciu wody i ligandów nie zostało białko")
    gaps = [f"{c['name']}:{b}" for c in scan["chains"] for b in c["breaks"]]
    has_seqres = all(c.get("seqres") for c in scan["chains"])
    if gaps and not has_seqres:
        raise RecipeError(f"przerwy w łańcuchu ({', '.join(gaps)}), a plik nie ma SEQRES – uzupełnij brakujące "
                          "reszty (np. Modeller) albo wyślij pełny plik z RCSB")
    if scan["missing_atoms"] or scan["nonstandard"] or gaps:
        tmp = out.with_name(out.stem + "_przed_pdbfixer.pdb")
        st.write_pdb(str(tmp))
        notes += _pdbfixer(tmp, out)
    else:
        st.write_minimal_pdb(str(out))
    return notes


def _pdbfixer(src: Path, out: Path) -> list[str]:
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(src))
    notes = []
    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    inner, ends = [], []
    for key in list(fixer.missingResidues):
        chain = chains[key[0]]
        names = fixer.missingResidues[key]
        where = f"{chain.id}: {'-'.join(names[:6])}{'…' if len(names) > 6 else ''}"
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            ends.append(where)
            del fixer.missingResidues[key]  # loose termini are not modelled
        else:
            inner.append(where)
    fixer.findNonstandardResidues()
    swapped = [f"{r.name}{r.id}→{new}" for r, new in fixer.nonstandardResidues]
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    atoms = [f"{r.chain.id}:{r.id} {r.name} {' '.join(a.name for a in lost)}" for r, lost in fixer.missingAtoms.items()]
    fixer.addMissingAtoms()
    with open(out, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    if swapped:
        notes.append("pdbfixer: zamieniono niestandardowe reszty " + ", ".join(swapped))
    if atoms:
        notes.append("pdbfixer: dodano brakujące atomy ciężkie: " + "; ".join(atoms))
    if inner:
        notes.append("pdbfixer: wstawiono brakujące reszty wewnątrz łańcucha (" + "; ".join(inner) +
                     ") – sprawdź tę pętlę")
    if ends:
        notes.append("Brakujące reszty na końcach łańcucha nie zostały dodane (" + "; ".join(ends) + ")")
    return notes
