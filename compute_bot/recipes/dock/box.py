"""The docking box: from center+size, a co-crystallised ligand, pocket residues, or the only drug-like ligand."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from compute_bot.recipes.dock.common import UserError

MIN_BOX = 10.0  # Å; smaller boxes leave Vina no room to rotate even a fragment
LARGE_VOLUME = 27000.0  # Å^3; above this Vina's docs advise raising exhaustiveness

# Crystallisation additives, buffers, detergents and glycans: HETATM groups that are almost never "the ligand"
ADDITIVES = frozenset("""
SO4 PO4 PI GOL EDO PEG PGE PG4 1PE P6G 2PE PE4 PE5 ACT ACY ACE DMS MPD MRD MES EPE TRS CIT FMT IMD BME DTT TLA
MLI SUC TAR NO3 SCN AZI IOD BR CL NA K MG CA ZN MN FE CO NI CU CD HG UNX UNL UNK BOG LMT LDA OLC OLA DMU HTG
C8E SDS NAG NDG MAN BMA FUC GAL GLC SIA BGC XYP FLC IPA EOH MOH BU3 PGO CXS HEZ NHE 15P P33 12P CPS
""".split())


@dataclass
class Ligand:
    resname: str
    chain: str
    resnum: str  # number + insertion code, as in the file
    names: list
    elements: list
    coords: np.ndarray  # heavy atoms only, (n, 3)

    @property
    def label(self):
        return f"{self.resname} {self.chain}:{self.resnum}"

    def pdb_block(self) -> str:
        lines = []
        for i, (name, el, xyz) in enumerate(zip(self.names, self.elements, self.coords, strict=True), start=1):
            atom = name if len(name) == 4 or len(el) == 2 else f" {name}"
            lines.append(f"HETATM{i:>5} {atom:<4} {self.resname:>3} {self.chain[:1] or 'A'}"
                         f"{str(self.resnum)[:4]:>4}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00"
                         f"          {el.upper():>2}")
        return "\n".join(lines) + "\nEND\n"


@dataclass
class Box:
    center: list
    size: list
    source: str  # center | ligand_ref | residues | auto
    note: str = ""
    ligand: Ligand | None = None  # the reference ligand to redock, when there is one
    warnings: list = field(default_factory=list)

    @property
    def volume(self):
        return float(np.prod(self.size))


def box_from_coords(coords, padding: float, min_size: float = MIN_BOX):
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    center = (lo + hi) / 2
    size = np.maximum(hi - lo + 2 * padding, min_size)
    return [round(float(c), 3) for c in center], [round(float(s), 3) for s in size]


def _residue_to_ligand(chain, res) -> Ligand:
    atoms = [a for a in res if a.element.name not in ("H", "D")]
    return Ligand(res.name, chain.name, f"{res.seqid.num}{res.seqid.icode.strip()}",
                  [a.name for a in atoms], [a.element.name for a in atoms],
                  np.array([a.pos.tolist() for a in atoms], dtype=float))


def _is_polymer(res) -> bool:
    import gemmi

    return res.entity_type == gemmi.EntityType.Polymer


def hetero_groups(st):
    """(chain, residue) of every non-polymer, non-water group in the first model."""
    out = []
    for chain in st[0]:
        for res in chain:
            if res.is_water() or _is_polymer(res):
                continue
            out.append((chain, res))
    return out


def druglike_ligands(st):
    """Ligand candidates: non-polymer groups with >= 6 heavy atoms that aren't additives/glycans."""
    out = []
    for chain, res in hetero_groups(st):
        heavy = sum(1 for a in res if a.element.name not in ("H", "D"))
        if res.name.upper() in ADDITIVES or heavy < 6:
            continue
        out.append(_residue_to_ligand(chain, res))
    return out


def find_ligand(st, spec: str) -> Ligand:
    """The co-crystallised ligand by residue name, optionally with chain: 'BTN' or 'A:BTN'."""
    spec = spec.strip().upper()
    chain_name, _, resname = spec.rpartition(":")
    found = [(c, r) for c, r in hetero_groups(st)
             if r.name.upper() == resname and (not chain_name or c.name.upper() == chain_name)]
    if not found:
        names = sorted({r.name for _, r in hetero_groups(st)})
        have = ", ".join(names) if names else "brak ligandów w pliku"
        raise UserError(f"ligand_ref={spec}: nie ma takiego ligandu w strukturze (HETATM: {have}).")
    lig = _residue_to_ligand(*found[0])
    if len(lig.coords) == 0:
        raise UserError(f"ligand_ref={spec}: ligand nie ma atomów ciężkich.")
    return lig


def parse_residues(text: str):
    """'A:45, A:88, B:12A, 7' -> [('A', 45, ''), ('A', 88, ''), ('B', 12, 'A'), (None, 7, '')]."""
    out = []
    for part in re.split(r"[,;\s]+", text.strip()):
        if not part:
            continue
        m = re.fullmatch(r"(?:([A-Za-z0-9]{1,4})[:_])?(-?\d+)([A-Za-z]?)", part)
        if not m:
            raise UserError(f"residues: nie rozumiem '{part}'. Podaj np. residues=A:45,A:88.")
        out.append((m.group(1), int(m.group(2)), m.group(3).upper()))
    if not out:
        raise UserError("residues: pusta lista reszt.")
    return out


def box_from_residues(st, specs, padding: float, min_size: float = MIN_BOX):
    coords, labels = [], []
    for chain_name, num, icode in specs:
        hits = [(c, r) for c in st[0] for r in c
                if (chain_name is None or c.name == chain_name) and r.seqid.num == num
                and r.seqid.icode.strip().upper() == icode and not r.is_water()]
        label = f"{chain_name + ':' if chain_name else ''}{num}{icode}"
        if not hits:
            raise UserError(f"residues: nie ma reszty {label} w strukturze.")
        chains = sorted({c.name for c, _ in hits})
        if len(chains) > 1:
            raise UserError(f"residues: reszta {label} jest w łańcuchach {', '.join(chains)}; "
                            f"podaj łańcuch, np. {chains[0]}:{num}{icode}.")
        c, r = hits[0]
        coords.extend(a.pos.tolist() for a in r)
        labels.append(f"{c.name}:{r.seqid.num}{r.seqid.icode.strip()} {r.name}")
    center, size = box_from_coords(coords, padding, min_size)
    return center, size, ", ".join(labels)


def choose_box(st, params) -> Box:
    """Box from params in order center > ligand_ref > residues > the only drug-like ligand; else a Polish error."""
    padding = float(params.get("padding", 6.0))
    ref_spec = (params.get("ligand_ref") or "").strip()
    residues = (params.get("residues") or "").strip()
    ligand = find_ligand(st, ref_spec) if ref_spec else None
    warnings = []
    if params.get("center"):
        center = [float(x) for x in params["center"]]
        size = [float(x) for x in params["size"]]
        box = Box(center, size, "center", f"środek {fmt_vec(center)}, rozmiar {fmt_vec(size)} (z parametrów)", ligand)
        if ligand is not None and not inside(box, ligand.coords.mean(axis=0)):
            warnings.append(f"ligand {ligand.label} leży poza pudełkiem – redocking nie ma sensu.")
    elif ligand is not None:
        center, size = box_from_coords(ligand.coords, padding)
        box = Box(center, size, "ligand_ref", f"z ligandu {ligand.label} + margines {padding:g} Å", ligand)
        if residues:
            warnings.append("podano i ligand_ref, i residues – pudełko z ligandu, residues pominięte.")
    elif residues:
        center, size, note = box_from_residues(st, parse_residues(residues), padding)
        box = Box(center, size, "residues", f"z reszt: {note} + margines {padding:g} Å")
    else:
        candidates = druglike_ligands(st)
        names = sorted({c.resname for c in candidates})
        if len(names) != 1:
            found = f"W strukturze są: {', '.join(names)}. " if names else "W strukturze nie ma ligandu. "
            raise UserError("Nie wiem, gdzie dokować. " + found +
                            "Podaj ligand_ref=NAZWA (ligand krystaliczny), residues=A:45,A:88 (reszty kieszeni) "
                            "albo center=x,y,z (i size=x,y,z).")
        ligand = candidates[0]
        center, size = box_from_coords(ligand.coords, padding)
        copies = len(candidates)
        note = f"z jedynego ligandu w strukturze: {ligand.label} + margines {padding:g} Å"
        if copies > 1:
            note += f" (kopii: {copies}, wzięto pierwszą)"
        box = Box(center, size, "auto", note, ligand)
    if box.volume > LARGE_VOLUME:
        warnings.append(f"duże pudełko ({box.volume:.0f} Å³) – rozważ exhaustiveness=16 lub więcej.")
    box.warnings = warnings
    return box


def inside(box: Box, xyz, margin: float = 0.0) -> bool:
    c, s = np.asarray(box.center), np.asarray(box.size) / 2 + margin
    return bool(np.all(np.abs(np.asarray(xyz) - c) <= s))


def fmt_vec(v):
    return "(" + ", ".join(f"{x:.1f}" for x in v) + ")"


def vina_box_text(center, size, comment: str = "") -> str:
    """A Vina config fragment (usable as `vina --config box.txt`)."""
    lines = [f"# {line}" for line in comment.splitlines() if line.strip()]
    for axis, c in zip("xyz", center, strict=True):
        lines.append(f"center_{axis} = {c:.3f}")
    for axis, s in zip("xyz", size, strict=True):
        lines.append(f"size_{axis} = {s:.3f}")
    return "\n".join(lines) + "\n"


def box_corners_pdb(center, size) -> str:
    """The 8 box corners as pseudo-atoms with the 12 edges as CONECT: load next to the receptor in PyMOL."""
    c, h = np.asarray(center, dtype=float), np.asarray(size, dtype=float) / 2
    corners = [c + h * np.array([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    lines = []
    for i, p in enumerate(corners, start=1):
        lines.append(f"HETATM{i:>5}  C   BOX X   1    {p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00           C")
    for i in range(8):
        neighbours = [j for j in range(8) if bin(i ^ j).count("1") == 1]
        lines.append("CONECT" + f"{i + 1:>5}" + "".join(f"{j + 1:>5}" for j in neighbours))
    return "\n".join(lines) + "\nEND\n"
