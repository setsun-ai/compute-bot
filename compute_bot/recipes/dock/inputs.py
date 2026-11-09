"""Input boundary: finding the owner's files, reading ligand tables (csv/smi/sdf/mol2), the PDB code, downloads."""
from __future__ import annotations

import csv
import io
import re
import time
import urllib.request
import zipfile
from pathlib import Path

from compute_bot.recipes.dock.common import UserError

RECEPTOR_EXT = (".pdb", ".ent", ".cif", ".mmcif", ".pdb.gz", ".cif.gz", ".ent.gz")
LIGAND_EXT = {".csv": "csv", ".tsv": "csv", ".smi": "smi", ".smiles": "smi", ".sdf": "sdf",
              ".sd": "sdf", ".mol": "sdf", ".mol2": "mol2"}
SKIP_DIRS = {"wyniki", "__MACOSX", "_rozpakowane_tmp"}
SKIP_FILES = {"params.json", "summary.txt", "highlights.txt", "spec.json", "status.json"}

SMILES_HEADERS = ("smiles", "smi", "canonical_smiles", "isomeric_smiles", "smiles_string", "structure_smiles")
NAME_HEADERS = ("name", "id", "nazwa", "compound", "compound_id", "compound_name", "mol_id", "molecule",
                "molecule_name", "ligand", "ligand_id", "title", "identifier", "chembl_id", "zinc_id",
                "molecule_chembl_id", "cid")

RCSB_ENTRY = "https://files.rcsb.org/download/{code}.{ext}"
RCSB_IDEAL_SDF = "https://files.rcsb.org/ligands/download/{code}_ideal.sdf"


# ---------------------------------------------------------------- text helpers

def decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1250"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _quiet_rdkit():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def looks_like_smiles(text: str) -> bool:
    """True when RDKit parses the text as a molecule (plain numbers and words like 'name' do not count)."""
    from rdkit import Chem

    text = (text or "").strip()
    if not text or " " in text or re.fullmatch(r"[-+0-9.,eE]+", text):
        return False
    _quiet_rdkit()
    mol = Chem.MolFromSmiles(text)
    return mol is not None and mol.GetNumAtoms() > 0


def unique_names(names, reserved=()):
    """File-safe, unique ligand names in input order; empty names become lig_0001 etc."""
    used = set(reserved)
    out = []
    for i, name in enumerate(names):
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "")).strip("._-")[:60] or f"lig_{i + 1:04d}"
        candidate, k = base, 2
        while candidate in used:
            candidate, k = f"{base}_{k}", k + 1
        used.add(candidate)
        out.append(candidate)
    return out


def check_pdb_code(code: str) -> str:
    code = str(code).strip().upper()
    if re.fullmatch(r"[0-9][A-Z0-9]{3}", code) or re.fullmatch(r"PDB_[0-9A-Z]{8}", code):
        return code
    raise UserError(f"receptor={code}? Kod PDB ma 4 znaki i zaczyna się cyfrą, np. receptor=1STP.")


# ---------------------------------------------------------------- ligand tables

def _row(name, smiles, source, error=""):
    return {"name": (name or "").strip(), "smiles": (smiles or "").strip(), "source": source, "error": error}


def parse_csv_text(text: str, source: str = "csv"):
    """Rows from a delimited table: SMILES column by header (or by content), optional name/id column."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise UserError(f"{source}: plik jest pusty.")
    sample = "\n".join(lines[:20])
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = "\t" if "\t" in lines[0] else ";" if ";" in lines[0] else ","
    table = [r for r in csv.reader(io.StringIO("\n".join(lines)), delimiter=delimiter)]
    header = [h.strip() for h in table[0]]
    lowered = [h.lower().strip().replace(" ", "_") for h in header]
    body = table[1:]

    def column(i):
        return [r[i].strip() if i < len(r) else "" for r in body]

    smiles_col = next((lowered.index(h) for h in SMILES_HEADERS if h in lowered), None)
    if smiles_col is None:
        smiles_col = next((i for i, h in enumerate(lowered) if "smiles" in h), None)
    if smiles_col is None:
        # no telling header: a headerless file, or a column whose values parse as SMILES
        if any(looks_like_smiles(h) for h in header) and not any(h in NAME_HEADERS for h in lowered):
            body = table
            header = [f"kolumna{i + 1}" for i in range(len(header))]
            lowered = [h.lower() for h in header]
        best, best_score = None, 0.0
        for i in range(len(header)):
            values = [v for v in column(i)[:50] if v]
            if values:
                score = sum(looks_like_smiles(v) for v in values) / len(values)
                if score > best_score:
                    best, best_score = i, score
        if best is None or best_score < 0.6:
            raise UserError(f"{source}: nie widzę kolumny ze SMILES (kolumny: {', '.join(header)}). "
                            "Nazwij ją 'smiles'.")
        smiles_col = best
    name_col = next((lowered.index(h) for h in NAME_HEADERS if h in lowered and lowered.index(h) != smiles_col), None)
    if name_col is None:
        name_col = next((i for i, h in enumerate(lowered) if i != smiles_col and
                         ("name" in h or "nazwa" in h or h.endswith("_id") or h.startswith("id_"))), None)
    rows = []
    for k, r in enumerate(body):
        smi = r[smiles_col].strip() if smiles_col < len(r) else ""
        name = r[name_col].strip() if name_col is not None and name_col < len(r) else ""
        if not smi and not name:
            continue
        rows.append(_row(name or f"{Path(source).stem}_{k + 1}", smi, source, "" if smi else "pusty SMILES"))
    if not rows:
        raise UserError(f"{source}: brak wierszy z ligandami.")
    return rows


def parse_smi_text(text: str, source: str = "smi"):
    """'SMILES [name]' per line, optional header line, '#' comments; a delimited table goes to parse_csv_text."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise UserError(f"{source}: plik jest pusty.")
    first = lines[0].split()[0]
    if ("," in lines[0] or ";" in lines[0]) and not looks_like_smiles(first):
        return parse_csv_text(text, source)
    if first.lower() in SMILES_HEADERS or (not looks_like_smiles(first) and first.lower() in ("smiles", "smile")):
        lines = lines[1:]
    rows = []
    for k, ln in enumerate(lines):
        parts = ln.split(None, 1)
        name = parts[1].strip() if len(parts) > 1 else f"{Path(source).stem}_{k + 1}"
        rows.append(_row(name, parts[0], source))
    if not rows:
        raise UserError(f"{source}: brak ligandów.")
    return rows


def _name_from_props(mol, title: str, fallback: str):
    if title and title.strip():
        return title.strip()
    if mol is not None:
        props = {p.lower(): p for p in mol.GetPropNames()}
        for key in NAME_HEADERS:
            if key in props and str(mol.GetProp(props[key])).strip():
                return str(mol.GetProp(props[key])).strip()
    return fallback


def _mol_to_smiles(mol):
    from rdkit import Chem

    if mol.GetNumConformers() and mol.GetConformer().Is3D():
        Chem.AssignStereochemistryFrom3D(mol)
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def parse_sdf_text(text: str, source: str = "sdf"):
    """One row per SDF record; a record RDKit can't read stays as a row with an error (not fatal)."""
    from rdkit import Chem

    _quiet_rdkit()
    blocks = [b for b in re.split(r"^\$\$\$\$[ \t]*\r?$\n?", text, flags=re.M) if b.strip()]
    rows = []
    for k, block in enumerate(blocks):
        title = block.splitlines()[0] if block.splitlines() else ""
        supplier = Chem.SDMolSupplier()
        supplier.SetData(block + "\n$$$$\n", sanitize=True, removeHs=False)
        mol = next(iter(supplier), None)
        fallback = f"{Path(source).stem}_{k + 1}"
        if mol is None:
            rows.append(_row(title.strip() or fallback, "", source, "RDKit nie odczytał rekordu SDF"))
            continue
        rows.append(_row(_name_from_props(mol, title, fallback), _mol_to_smiles(mol), source))
    if not rows:
        raise UserError(f"{source}: brak cząsteczek w pliku SDF.")
    return rows


def parse_mol2_text(text: str, source: str = "mol2"):
    from rdkit import Chem

    _quiet_rdkit()
    parts = re.split(r"(?=@<TRIPOS>MOLECULE)", text)
    blocks = [p for p in parts if p.startswith("@<TRIPOS>MOLECULE")]
    rows = []
    for k, block in enumerate(blocks):
        lines = block.splitlines()
        title = lines[1].strip() if len(lines) > 1 else ""
        name = title or f"{Path(source).stem}_{k + 1}"
        mol = Chem.MolFromMol2Block(block, sanitize=True, removeHs=False)
        smi = _mol_to_smiles(mol) if mol is not None else _obabel_convert(block, "mol2")
        rows.append(_row(name, smi or "", source, "" if smi else "nie udało się odczytać rekordu MOL2"))
    if not rows:
        raise UserError(f"{source}: brak cząsteczek w pliku MOL2.")
    return rows


def _obabel_convert(block: str, fmt: str) -> str:
    try:
        from openbabel import pybel
    except ImportError:
        return ""
    try:
        return pybel.readstring(fmt, block).write("can").split()[0]
    except Exception:  # noqa: BLE001 - Open Babel raises plain IOError/ValueError on bad input
        return ""


def read_ligand_file(path: Path):
    kind = LIGAND_EXT[path.suffix.lower()]
    text = decode(path.read_bytes())
    name = path.name
    return {"csv": parse_csv_text, "smi": parse_smi_text, "sdf": parse_sdf_text, "mol2": parse_mol2_text}[kind](
        text, name)


# ---------------------------------------------------------------- discovering files

def _is_receptor(path: Path) -> bool:
    return path.name.lower().endswith(RECEPTOR_EXT)


def find_inputs(work: Path):
    """(receptor files, ligand files) among the owner's files; results and hidden files are skipped."""
    receptors, ligand_files = [], []
    for path in sorted(work.rglob("*")):
        rel = path.relative_to(work)
        if not path.is_file() or any(p in SKIP_DIRS or p.startswith(".") for p in rel.parts):
            continue
        if path.name in SKIP_FILES:
            continue
        if _is_receptor(path):
            receptors.append(path)
        elif path.suffix.lower() in LIGAND_EXT:
            ligand_files.append(path)
    return receptors, ligand_files


def unpack_zips(work: Path) -> list[Path]:
    """Unpack zips the runner left packed (normally it unpacks them itself). Paths outside work are refused."""
    done = []
    for z in sorted(work.glob("*.zip")):
        target = work / f"{z.stem}_zip"
        if target.exists():
            continue
        with zipfile.ZipFile(z) as zf:
            for member in zf.namelist():
                if member.startswith("/") or ".." in Path(member).parts:
                    raise UserError(f"{z.name}: podejrzana ścieżka w zipie ({member}).")
            zf.extractall(target)
        done.append(target)
    return done


def pick_receptor(paths: list[Path]) -> Path:
    if len(paths) == 1:
        return paths[0]
    preferred = [p for p in paths if re.search(r"rec|prot|bialk|białk|target", p.name, re.I)]
    if len(preferred) == 1:
        return preferred[0]
    raise UserError("Kilka plików struktury: " + ", ".join(p.name for p in paths) +
                    ". Zostaw jeden albo nazwij receptor np. receptor.pdb.")


# ---------------------------------------------------------------- downloads

def fetch(url: str, timeout: float = 60, tries: int = 3) -> bytes:
    last = None
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - network errors come in many types; retried, then re-raised
            last = exc
            time.sleep(2 * (attempt + 1))
    raise last


def download_entry(code: str, dest_dir: Path) -> Path:
    """mmCIF of a PDB entry from RCSB (PDB format as a fallback), cached in dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("cif", "pdb"):
        path = dest_dir / f"{code}.{ext}"
        if path.exists() and path.stat().st_size > 0:
            return path
    errors = []
    for ext in ("cif", "pdb"):
        try:
            data = fetch(RCSB_ENTRY.format(code=code, ext=ext))
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            continue
        path = dest_dir / f"{code}.{ext}"
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        return path
    raise UserError(f"Nie udało się pobrać {code} z RCSB ({errors[-1] if errors else 'brak sieci?'}). "
                    "Sprawdź kod albo wyślij plik .pdb/.cif.")


def download_ideal_sdf(resname: str, dest_dir: Path) -> str | None:
    """Ideal coordinates + bond orders of a chemical component from the PDB CCD; None when offline."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{resname}_ideal.sdf"
    if path.exists() and path.stat().st_size > 0:
        return path.read_text()
    try:
        data = fetch(RCSB_IDEAL_SDF.format(code=resname), timeout=30, tries=2)
    except Exception:  # noqa: BLE001 - optional step, the caller falls back to Open Babel bond perception
        return None
    path.write_bytes(data)
    return decode(data)
