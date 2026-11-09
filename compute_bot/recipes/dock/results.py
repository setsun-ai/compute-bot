"""Results: ranking, contacts, complexes for viewing, plots and the Polish summary."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

RANKING_COLUMNS = ["rank", "name", "smiles", "affinity_kcal_mol", "ligand_efficiency", "heavy_atoms", "mw", "logp",
                   "hbd", "hba", "rot_bonds", "lipinski_violations", "reference", "docked_smiles", "pose_sdf",
                   "pose_pdbqt", "notes"]
CONTACT_CUTOFF = 4.0  # Å, any heavy atom pair
HBOND_CUTOFF = 3.5  # Å, N/O ... N/O: a geometric candidate only, no angles


def rank(records):
    """Docked ligands, best (most negative) affinity first; ties: higher ligand efficiency, then name."""
    done = [r for r in records if r.get("affinity") is not None]
    return sorted(done, key=lambda r: (r["affinity"], -(r.get("ligand_efficiency") or 0.0), r["name"]))


def format_eta(seconds) -> str:
    if seconds is None:
        return "?"
    minutes = max(1, math.ceil(seconds / 60))
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60} min"


def contacts(lig_atoms, rec_atoms, cutoff=CONTACT_CUTOFF, hb_cutoff=HBOND_CUTOFF):
    """Residues with a heavy atom within cutoff of the ligand, and N/O pairs within hb_cutoff (H-bond candidates).

    lig_atoms: [(name, element, xyz)]; rec_atoms: [(chain, resname, resnum, atom name, element, xyz)].
    Residues come in receptor order; H-bond candidates by distance.
    """
    lig = [a for a in lig_atoms if a[1].upper() not in ("H", "D")]
    rec = [a for a in rec_atoms if a[4].upper() not in ("H", "D")]
    if not lig or not rec:
        return [], []
    lx = np.array([a[2] for a in lig], dtype=float)
    rx = np.array([a[5] for a in rec], dtype=float)
    d = np.linalg.norm(rx[:, None, :] - lx[None, :, :], axis=2)  # (n_rec, n_lig)
    residues, seen = [], set()
    for i in np.where(d.min(axis=1) <= cutoff)[0]:
        chain, resname, resnum = rec[i][:3]
        key = f"{chain}:{resname}{resnum}"
        if key not in seen:
            seen.add(key)
            residues.append(key)
    hbonds = []
    polar = {"N", "O"}
    for i, j in zip(*np.where(d <= hb_cutoff), strict=True):
        if rec[i][4].upper() in polar and lig[j][1].upper() in polar:
            chain, resname, resnum, atom = rec[i][:4]
            hbonds.append({"ligand_atom": lig[j][0], "residue": f"{chain}:{resname}{resnum}", "atom": atom,
                           "distance": round(float(d[i, j]), 2)})
    hbonds.sort(key=lambda h: h["distance"])
    return residues, hbonds


def mol_atoms(mol, conf_id=0):
    """[(name, element, xyz)] of an RDKit pose, names like C1, N2 by atom index."""
    pos = mol.GetConformer(conf_id).GetPositions()
    return [(f"{a.GetSymbol()}{a.GetIdx() + 1}", a.GetSymbol(), tuple(pos[a.GetIdx()])) for a in mol.GetAtoms()]


def complex_pdb(receptor_pdb: str, mol, conf_id=0, resname="LIG", chain="L") -> str:
    """Receptor + one pose as HETATM (with CONECT for the ligand) in one PDB, for PyMOL/Chimera."""
    lines = [ln for ln in receptor_pdb.splitlines() if ln.startswith(("ATOM", "HETATM", "TER"))]
    serial = sum(1 for ln in lines if ln.startswith(("ATOM", "HETATM", "TER")))
    pos = mol.GetConformer(conf_id).GetPositions()
    first = serial + 1
    for a in mol.GetAtoms():
        serial += 1
        el = a.GetSymbol()
        name = f"{el}{a.GetIdx() + 1}"[:4]
        name = name if len(name) == 4 or len(el) == 2 else f" {name}"
        x, y, z = pos[a.GetIdx()]
        lines.append(f"HETATM{serial % 100000:>5} {name:<4} {resname:>3} {chain}   1    {x:8.3f}{y:8.3f}{z:8.3f}"
                     f"  1.00  0.00          {el.upper():>2}")
    for a in mol.GetAtoms():
        nbrs = [first + n.GetIdx() for n in a.GetNeighbors()]
        if nbrs:
            tail = "".join(f"{n % 100000:>5}" for n in nbrs[:4])
            lines.append(f"CONECT{(first + a.GetIdx()) % 100000:>5}{tail}")
    return "\n".join(lines) + "\nEND\n"


def write_ranking(path: Path, ranked) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RANKING_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(ranked, start=1):
            row = dict(r, rank=i, affinity_kcal_mol=r["affinity"], reference="tak" if r.get("reference") else "",
                       notes="; ".join(r.get("notes", [])), **r.get("props", {}))
            w.writerow(row)


def write_failed(path: Path, failed) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "smiles", "error"])
        for f in failed:
            w.writerow([f["name"], f.get("smiles", ""), f.get("error", "")])


def plots(out_dir: Path, ranked):
    """Affinity histogram and affinity vs MW (reference ligand marked). Returns PNG paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not ranked:
        return []
    paths = []
    aff = [r["affinity"] for r in ranked]
    ref = [r for r in ranked if r.get("reference")]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(aff, bins=min(30, max(5, len(aff) // 3)), color="#4c72b0", edgecolor="white")
    for r in ref:
        ax.axvline(r["affinity"], color="#c44e52", ls="--", label=f"ligand natywny ({r['affinity']:.1f})")
    ax.set_xlabel("najlepsza energia wiązania [kcal/mol]")
    ax.set_ylabel("liczba ligandów")
    ax.set_title(f"Rozkład wyników dokowania (n = {len(aff)})")
    if ref:
        ax.legend()
    fig.tight_layout()
    p = out_dir / "histogram_energii.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(6, 4))
    mw = [r.get("props", {}).get("mw", np.nan) for r in ranked]
    ax.scatter(mw, aff, s=18, color="#4c72b0", alpha=0.8, label="ligandy")
    for r in ref:
        ax.scatter([r["props"]["mw"]], [r["affinity"]], s=60, color="#c44e52", marker="*", label="ligand natywny")
    for r in ranked[:5]:
        ax.annotate(r["name"][:18], (r.get("props", {}).get("mw", np.nan), r["affinity"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("masa cząsteczkowa [g/mol]")
    ax.set_ylabel("najlepsza energia [kcal/mol]")
    ax.set_title("Energia a wielkość cząsteczki")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "energia_vs_masa.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


def fit(text: str, limit: int = 2500) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 25].rsplit("\n", 1)[0] + "\n… (pełne wyniki w wyniki/)\n"
