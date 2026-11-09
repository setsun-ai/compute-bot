"""Docking one ligand at a time with per-ligand records, so a restarted job skips what is already done."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

MAX_ATTEMPTS = 2  # a ligand that killed the process this many times is recorded as failed, not tried forever


def signature(smiles: str, settings: dict) -> str:
    """What a result depends on: the input SMILES and the docking settings (box, scoring, effort, seed, pH...)."""
    blob = json.dumps({"smiles": smiles, **settings}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def record_path(out_dir: Path, name: str) -> Path:
    return Path(out_dir) / f"{name}.json"


def write_record(out_dir: Path, record: dict) -> None:
    path = record_path(out_dir, record["name"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=1, ensure_ascii=False))
    os.replace(tmp, path)  # atomic: a reboot leaves either the old record or the new one


def read_record(out_dir: Path, name: str):
    path = record_path(out_dir, name)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def is_current(record, sig: str) -> bool:
    return bool(record) and record.get("signature") == sig and record.get("status") in ("done", "failed")


def pending(ligands, out_dir: Path, settings: dict):
    """Ligands without a finished record for the same SMILES and settings (failed ones count as finished)."""
    return [lig for lig in ligands
            if not is_current(read_record(out_dir, lig["name"]), signature(lig["smiles"], settings))]


def mark_running(out_dir: Path, name: str) -> int:
    """Count attempts at a ligand; the marker survives a crash. Returns the attempt number (1 = first)."""
    path = Path(out_dir) / f"{name}.running"
    try:
        n = int(path.read_text().strip() or 0)
    except (OSError, ValueError):
        n = 0
    path.write_text(str(n + 1))
    return n + 1


def clear_running(out_dir: Path, name: str) -> None:
    (Path(out_dir) / f"{name}.running").unlink(missing_ok=True)


def make_vina(receptor_pdbqt: Path, center, size, scoring: str, threads: int, seed: int):
    """A Vina engine with maps for all atom types: computed once, reused for every ligand."""
    from vina import Vina

    v = Vina(sf_name=scoring, cpu=int(threads), seed=int(seed), verbosity=0)
    v.set_receptor(str(receptor_pdbqt))
    v.compute_vina_maps(center=list(center), box_size=list(size))
    return v


def dock(v, ligand_pdbqt: str, exhaustiveness: int, n_poses: int):
    """(poses PDBQT text, [[affinity, rmsd_lb, rmsd_ub], ...]) for one ligand."""
    v.set_ligand_from_string(ligand_pdbqt)
    v.dock(exhaustiveness=int(exhaustiveness), n_poses=int(n_poses))
    energies = v.energies(n_poses=int(n_poses))
    return v.poses(n_poses=int(n_poses)), [[round(float(x), 3) for x in row[:3]] for row in energies]
