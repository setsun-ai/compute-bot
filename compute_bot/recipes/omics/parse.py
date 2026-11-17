"""Input shapes -> Dataset: DIA-NN pg/gg_matrix and long report, MS-DIAL alignment export, Agilent Profinder / MFE
tables, MZmine 3 feature lists, generic feature x sample tables (with a sample sheet, feature annotation and internal
standards in separate files or sheets of one workbook), samples-in-rows tables.

Everything is matched by content, not by file name alone: the sample sheet is the table with a column that matches
the matrix column names, the feature table matches the matrix row IDs, the internal standards table has the same
sample columns. Whatever cannot be found in metadata is inferred from the column names and recorded in
`Dataset.inferred`, so the summary can say what was guessed.
"""
from __future__ import annotations

import csv
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


class InputError(Exception):
    """A problem with the owner's files or parameters, worded in Polish for the ERROR line."""


NA_STRINGS = ["", "NA", "N/A", "NaN", "nan", "null", "NULL", "-", "#N/A", "na", "None", "n/a"]
TABLE_EXT = {".csv", ".tsv", ".txt", ".tab", ".xlsx", ".xls", ".xlsm", ".parquet"}
SKIP_FILES = {"params.json", "summary.txt", "highlights.txt"}
SKIP_DIRS = {"wyniki", "__MACOSX", ".omics_cache"}
UNZIP_DIR = "_rozpakowane"
RAW_EXT = re.compile(r"\.(wiff2?|raw|d|mzml|mzxml|mgf|dia|abf|mzdata|cdf|scan|baf|tdf|mzdat)$", re.I)

DIANN_META = {"Protein.Group", "Protein.Ids", "Protein.Names", "Genes", "First.Protein.Description",
              "N.Sequences", "N.Proteotypic.Sequences", "Proteotypic", "Protein.Q.Value", "Global.PG.Q.Value"}
DIANN_QUANT = ["PG.MaxLFQ", "Genes.MaxLFQ", "PG.Normalised", "PG.Quantity", "Genes.MaxLFQ.Unique", "Genes.Quantity"]
MSDIAL_LABELS = {"class": "Class", "file type": "File type", "injection order": "Injection order",
                 "batch id": "Batch ID", "analytical order": "Injection order"}

GROUP_ALIASES = ["group", "grupa", "class", "condition", "treatment", "phenotype", "diagnosis", "label", "status"]
PROFINDER_TAG = re.compile(r"^\[([^\]]+)\]\s*(.+?)\s*$")
PROFINDER_QUANT = ["area", "height", "vol", "volume", "abund", "abundance", "quant"]
MZMINE_LEGACY = re.compile(r"^(.+?)\s+Peak (area|height)$", re.I)
MZMINE_MODULAR = re.compile(r"^datafile:(.+):(area|height)$", re.I)
MZMINE_NAME_COLS = ("row identity (main id)", "row identity (all ids)", "compound_name", "lipid_annotations",
                    "annotation", "row comment")
# deuterium / 13C labels in a name ("(d7)", "-d9", " d7") but not the sphingoid base "d18:1"
DEUTERIUM_RE = re.compile(r"(?<![A-Za-z0-9])d(\d{1,2})(?![\d:;])|[(\-]D\d{1,2}\b|(?<![A-Za-z0-9])13C\d*\b"
                          r"|(?<![A-Za-z0-9])15N\d*\b|\[2H\d*\]|\[13C\d*\]")
ISTD_WORD_RE = re.compile(r"(?i)(?<![A-Za-z])(istd|internal[ _]?standard|int[ _]?std)(?![A-Za-z])"
                          r"|(?<![A-Za-z])IS(?![A-Za-z])")
NOT_A_TABLE = re.compile(r"(?i)readme|dictionary|s[lł]ownik|legend")
# "DA 12_1B 1207": subject, group (time point), biological sample + extraction letter, run number
DESIGN_NAME = re.compile(r"^(?P<s>[A-Za-z][A-Za-z0-9]*?)[\s_\-]+(?P<g>\d+)[\s_\-]+(?P<b>\d+)(?P<e>[A-Za-z])"
                         r"(?:[\s_\-]+(?P<run>\d+))?$")
TYPE_ALIASES = ["sample_type", "sampletype", "sample type", "file type", "file_type", "type", "typ"]
ORDER_ALIASES = ["run_order", "injection order", "injection_order", "run order", "runorder", "order",
                 "acquisition_order", "analytical order", "kolejnosc"]
BATCH_ALIASES = ["batch", "batch id", "batch_id", "analytical_batch", "partia", "plate", "measurement_day", "day"]
SUBJECT_ALIASES = ["subject_id", "subject", "patient_id", "patient", "individual", "donor", "donor_id", "osoba",
                   "dawczyni"]
ANNOT_NUMERIC = {"mz", "m/z", "mass", "mass (avg)", "rt (avg)", "row m/z", "row retention time", "row id",
                  "rt", "rt_min", "retention_time", "rt(min)", "average rt(min)", "average mz",
                 "n.sequences", "n.proteotypic.sequences", "charge", "isotope_index_candidate", "alignment id",
                 "annotation_level_simulated", "unique_peptides_simulated", "protein_q_value_simulated",
                 "taxonomy_id", "reference rt", "reference m/z", "fill %", "total score", "s/n average",
                 "rt similarity", "dot product", "reverse dot product", "fragment presence %", "msi_level"}
META_HINTS = ("inject", "sample", "subject", "meta", "design", "sheet", "probk", "pheno", "clinic", "osob", "pacjent",
              "dictionary", "slownik", "feature", "annot", "cech", "standards_metadata")
MATRIX_HINTS = ("intens", "matrix", "abund", "quant", "area", "height", "pg_", "gg_", "lfq", "peak", "macierz")
IS_HINTS = ("internal_standard", "internal-standard", "istd", "internal standards", "wzorce")
FEATURE_COLS = {
    "gene": ["gene_symbol", "gene", "genes", "gene names", "gene name", "gene_name", "gene names (primary)"],
    "accession": ["uniprot_accession", "accession", "uniprot", "protein_id", "protein.ids", "protein.group",
                  "protein ids", "entry", "majority protein ids"],
    "name": ["metabolite_name", "metabolite name", "compound", "compound_name", "name", "protein_name",
             "protein names", "protein.names", "first.protein.description"],
    "formula": ["molecular_formula_candidate", "formula", "molecular_formula"],
    "mz": ["mz", "m/z", "average mz", "precursor_mz", "row m/z", "mass (avg)", "mass"],
    "rt": ["rt_min", "rt", "average rt(min)", "retention_time", "rt(min)", "row retention time", "rt (avg)"],
    "cluster": ["analytical_cluster_id", "cluster_id", "pcgroup", "feature_group"],
    "adduct": ["adduct_candidate", "adduct", "adduct type", "adduct_type"],
}
ACCESSION_RE = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(-\d+)?$")
CONTROL_RE = re.compile(r"control|ctrl|kontrol|healthy|zdrow|^wt$|wild|baseline|placebo|sham|vehicle|untreat|^ref",
                        re.I)


@dataclass
class Dataset:
    name: str                   # file the matrix came from
    layer: str                  # proteomics | metabolomics
    source: str                 # what the matrix looked like (DIA-NN gg_matrix, MS-DIAL, table, ...)
    X: pd.DataFrame             # features x samples, linear intensities, NaN = missing
    features: pd.DataFrame      # index = X.index; standard columns name/gene/accession/... plus the original ones
    samples: pd.DataFrame       # index = X.columns; _type _raw_type _group _batch _order _subject _file + sheet
    istd: pd.DataFrame | None = None
    notes: list = field(default_factory=list)
    inferred: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)    # things about the input worth a sentence in the summary


# ---------------------------------------------------------------- names

def sample_stem(name):
    """'D:\\SCIEX OS Data\\x\\P1_3.wiff' -> 'P1_3'."""
    s = re.split(r"[\\/]", str(name).strip())[-1]
    prev = None
    while prev != s:
        prev, s = s, RAW_EXT.sub("", s)
    return s


def _key(name):
    """Matching key of a sample name: 'AB 1_2B 101' and 'AB_1_2B_101.mzML' are the same injection."""
    return re.sub(r"[^a-z0-9]+", "_", sample_stem(name).lower()).strip("_")


def is_internal_standard(name, formula=None):
    """Isotope-labelled or explicitly marked standards: '15:0-18:1(d7) PC', 'SM d9', 'x ISTD', formula with D."""
    if formula is not None and not (isinstance(formula, float) and np.isnan(formula)):
        if re.search(r"(?<![A-Za-z])D\d*(?![a-z])", re.sub(r"\s+", "", str(formula))) and \
                re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", re.sub(r"\s+", "", str(formula))):
            return True
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return False
    text = str(name)
    if re.search(r"(?i)w?vitamin", text):
        return False
    return bool(DEUTERIUM_RE.search(text) or ISTD_WORD_RE.search(text))


def canonical_type(value):
    """Sample type in the recipe's words: sample, qc, blank, replicate, conditioning, other (None if empty)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    v = re.sub(r"[\s\-]+", "_", str(value).strip().lower())
    if v in ("", "nan", "na", "none"):
        return None
    if "condition" in v or v in ("equilibration", "system_suitability"):
        return "conditioning"
    if "blank" in v or v in ("blk", "solvent", "empty", "ślepa", "slepa"):
        return "blank"
    if "replicate" in v or v in ("tech_rep", "techrep", "reinjection", "replikat"):
        return "replicate"
    if v.startswith("qc") or "pool" in v or v.endswith("_qc") or v in ("pqc", "sqc"):
        return "qc"
    if v in ("standard", "std", "calibration", "cal", "calibrant"):
        return "other"
    return "sample"


def _type_from_name(stem):
    low = stem.lower()
    if re.search(r"blank|blk|solvent|empty", low):
        return "blank"
    if re.search(r"(^|[^a-z])(qc|pool|pooled)([^a-z]|$)", low) or low.startswith("qc"):
        return "qc"
    return "sample"


def _group_from_name(stem):
    m = re.match(r"^(.*[A-Za-z].*?)[_\-\s.]+\d+$", stem)
    if m:
        return m.group(1)
    m = re.match(r"^(.*?[A-Za-z])\d+$", stem)
    return m.group(1) if m else None


def infer_from_names(names):
    """Sample type and group guessed from column names (P1_3 -> sample of group P1, QC1 -> qc, Blank_2 -> blank).

    Names like 'DA 12_1B 1207' (subject, time point, sample + extraction letter, run number) also give `_subject`
    and the time point as the group."""
    rows = []
    for n in names:
        stem = sample_stem(n)
        t = _type_from_name(stem)
        rows.append({"_type": t, "_group": _group_from_name(stem) if t == "sample" else None, "_subject": None})
    out = pd.DataFrame(rows, index=list(names))
    st = list(out.index[out._type == "sample"])
    hits = {n: DESIGN_NAME.match(" ".join(sample_stem(n).split())) for n in st}
    if len(st) >= 4 and np.mean([m is not None for m in hits.values()]) >= 0.8:
        for n, m in hits.items():
            out.loc[n, "_group"] = m["g"] if m else None
            out.loc[n, "_subject"] = m["s"] if m else None
        out.attrs["design"] = True
        return out
    groups = out.loc[out._type == "sample", "_group"]
    if groups.notna().sum() and groups.nunique() == groups.notna().sum():
        out["_group"] = None        # every sample its own "group": no grouping in the names
    return out


# ---------------------------------------------------------------- files

def collect_files(workdir):
    """Table files in the work dir; zips (if the runner did not unpack them) go to _rozpakowane/<name>/."""
    workdir = Path(workdir)
    for z in sorted(workdir.glob("*.zip")):
        target = workdir / UNZIP_DIR / z.stem
        if target.exists():
            continue
        try:
            with zipfile.ZipFile(z) as zf:
                tmp = target.with_name(target.name + ".tmp")
                for m in zf.infolist():
                    p = Path(m.filename)
                    if m.is_dir() or p.is_absolute() or ".." in p.parts or "__MACOSX" in p.parts:
                        continue
                    dest = tmp / p
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(zf.read(m))
                tmp.mkdir(parents=True, exist_ok=True)
                tmp.rename(target)
        except zipfile.BadZipFile:
            raise InputError(f"Nie mogę rozpakować {z.name} – uszkodzony zip.") from None
    files = []
    for p in sorted(workdir.rglob("*")):
        rel = p.relative_to(workdir)
        if not p.is_file() or any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts):
            continue
        if p.name in SKIP_FILES or p.suffix.lower() not in TABLE_EXT:
            continue
        files.append(p)
    return files


def _sniff_sep(lines):
    head = "\n".join(lines[:20])
    counts = {sep: head.count(sep) for sep in ("\t", ",", ";")}
    return max(counts, key=counts.get) if max(counts.values()) else ","


def _find_header_row(rows):
    for i, r in enumerate(rows[:40]):
        if any(str(c).strip() == "Alignment ID" for c in r):
            return i
    return 0


def _filled(cell):
    return not (cell is None or (isinstance(cell, float) and np.isnan(cell)) or str(cell).strip() == "")


def _title_rows(rows):
    """Number of rows above the header that are a title or empty (one filled cell at most, table below wider)."""
    width = [sum(_filled(c) for c in r) for r in rows[:15]]
    if not width or max(width) < 2:
        return 0
    need = max(2, 0.5 * max(width))
    first = next(i for i, w in enumerate(width) if w >= need)
    return first if all(w <= 1 for w in width[:first]) else 0


def _coerce_numeric(df):
    for c in df.columns:
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            nonnull = df[c].notna() & (df[c].astype(str).str.strip() != "")
            if nonnull.sum() and conv[nonnull].notna().mean() >= 0.95:
                df[c] = conv
    return df


def _sheet_table(raw):
    rows = raw.values.tolist()
    h = _find_header_row(rows)
    block = [["" if pd.isna(c) else str(c) for c in r] for r in rows[:h]] if h else None
    if not h:
        h = _title_rows(rows)
    df = raw.iloc[h + 1:].reset_index(drop=True)
    df.columns = _unique_names(["" if pd.isna(c) else str(c).strip() for c in raw.iloc[h]])
    df = df.replace({s: np.nan for s in NA_STRINGS if s})
    df = df.dropna(how="all").reset_index(drop=True)       # e.g. the empty second header row of Profinder
    return _coerce_numeric(df), block


def read_tables(path):
    """Every table of a file: [(sheet name or None, DataFrame, MS-DIAL header block or None)]; a workbook gives one
    per non-empty sheet (a title row above the header is skipped)."""
    if path.suffix.lower() not in (".xlsx", ".xls", ".xlsm"):
        df, block = read_table(path)
        return [(None, df, block)]
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
    except Exception as e:  # noqa: BLE001 - any reader failure means an unreadable workbook for the owner
        raise InputError(f"Nie mogę otworzyć {path.name}: {e}") from None
    out = []
    for name, raw in sheets.items():
        if raw.shape[0] < 2 or raw.shape[1] < 1:
            continue
        df, block = _sheet_table(raw)
        out.append((str(name), df, block))
    if len(out) == 1:
        out = [(None, out[0][1], out[0][2])]
    return out


def read_table(path):
    """DataFrame plus the MS-DIAL header block above the column names (list of rows) or None."""
    suf = path.suffix.lower()
    if suf == ".parquet":
        try:
            return pd.read_parquet(path), None
        except ImportError:
            raise InputError(f"{path.name}: do czytania .parquet brakuje pakietu pyarrow – wyślij report.tsv "
                             "(DIA-NN zapisuje go obok) albo pg_matrix.tsv.") from None
    if suf in (".xlsx", ".xls", ".xlsm"):
        tabs = read_tables(path)
        return (tabs[0][1], tabs[0][2]) if tabs else (pd.DataFrame(), None)
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        lines = [f.readline() for _ in range(40)]
    lines = [ln for ln in lines if ln]
    if not lines:
        return pd.DataFrame(), None
    sep = _sniff_sep(lines)
    rows = list(csv.reader(lines, delimiter=sep))
    h = _find_header_row(rows)
    block = rows[:h] if h else None
    if not h:
        h = _title_rows(rows)
    try:
        df = pd.read_csv(path, sep=sep, skiprows=h, na_values=NA_STRINGS, keep_default_na=True, low_memory=False,
                         encoding="utf-8-sig", decimal="," if sep == ";" else ".")
    except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError):
        return pd.DataFrame(), None
    return _coerce_numeric(df.dropna(how="all").reset_index(drop=True)), block


def _unique_names(names):
    seen, out = {}, []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}.{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


# ---------------------------------------------------------------- recognising tables

def _numeric_cols(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and str(c).strip().lower()
            not in ANNOT_NUMERIC and not str(c).lower().startswith("unnamed")]


def _looks_like_matrix(name, df):
    low = name.lower()
    num = _numeric_cols(df)
    if len(df) < 2 or len(num) < 3:
        return False
    if any(h in low for h in IS_HINTS):
        return True
    if any(h in low for h in MATRIX_HINTS):
        return True
    if any(h in low for h in META_HINTS):
        return False
    if len(num) < 0.6 * len(df.columns):
        return False
    vals = df[num]
    binary = sum(set(vals[c].dropna().unique()) <= {0, 1} for c in num)
    integer = sum(bool(np.all(np.mod(vals[c].dropna(), 1) == 0)) for c in num)
    return binary < 0.2 * len(num) and integer < 0.5 * len(num)


def _first_id_col(df):
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[0]


def _matrix_generic(df):
    idc = _first_id_col(df)
    num = [c for c in _numeric_cols(df) if c != idc]
    X = df[num].astype(float)
    X.index = df[idc].astype(str).where(df[idc].notna(), None)
    feats = df[[c for c in df.columns if c not in num]].copy()
    feats.index = X.index
    return X, feats, idc


def _matrix_diann(df, notes):
    idc = "Protein.Group" if "Protein.Group" in df.columns else "Genes"
    src = "DIA-NN pg_matrix" if idc == "Protein.Group" else "DIA-NN gg_matrix"
    samples = [c for c in df.columns if c not in DIANN_META and pd.api.types.is_numeric_dtype(df[c])]
    X = df[samples].astype(float)
    X.index = df[idc].astype(str).where(df[idc].notna() & (df[idc].astype(str) != ""), None)
    feats = df[[c for c in df.columns if c in DIANN_META]].copy()
    feats.index = X.index
    return X, feats, src


def _matrix_diann_long(df, notes):
    quant = next(c for c in DIANN_QUANT if c in df.columns)
    idc = "Protein.Group" if "Protein.Group" in df.columns else "Genes"
    n0 = len(df)
    for qcol in ("Q.Value", "PG.Q.Value", "Lib.PG.Q.Value"):
        if qcol in df.columns:
            df = df[df[qcol] <= 0.01]
    if len(df) < n0:
        notes.append(f"raport DIA-NN: zostawiono {len(df)}/{n0} wierszy z Q.Value/PG.Q.Value ≤ 0.01")
    run = "Run" if "Run" in df.columns else "File.Name"
    X = df.pivot_table(index=idc, columns=run, values=quant, aggfunc="max")
    X.columns = [str(c) for c in X.columns]
    X.index = X.index.astype(str)
    keep = [c for c in ("Protein.Group", "Protein.Ids", "Protein.Names", "Genes", "First.Protein.Description")
            if c in df.columns and c != idc]
    feats = df.groupby(idc)[keep].first() if keep else pd.DataFrame(index=X.index)
    feats.index = feats.index.astype(str)
    feats = feats.reindex(X.index)
    notes.append(f"raport DIA-NN w formacie długim: ilości z kolumny {quant}")
    return X.astype(float), feats


def _msdial_block(block, columns):
    """Per sample column: Class / File type / Injection order / Batch ID from the rows above the header."""
    info = {}
    for row in block:
        for j, cell in enumerate(row):
            lab = MSDIAL_LABELS.get(str(cell).strip().lower())
            if lab:
                info[lab] = {columns[k]: row[k] for k in range(j + 1, min(len(row), len(columns)))}
                break
    return info


def _matrix_msdial(df, block, notes):
    columns = list(df.columns)
    info = _msdial_block(block or [], columns)
    ftype = info.get("File type", {})
    if ftype:
        samples = [c for c, v in ftype.items() if str(v).strip() and str(v).strip().lower()
                   not in ("average", "stdev", "std", "sd", "nan")]
    else:
        start = columns.index("MS/MS spectrum") + 1 if "MS/MS spectrum" in columns else None
        samples = columns[start:] if start else _numeric_cols(df)
    samples = [c for c in samples if pd.api.types.is_numeric_dtype(df[c])]
    X = df[samples].astype(float)
    X.index = df["Alignment ID"].astype(str)
    feats = df[[c for c in columns if c not in samples and c in columns[:columns.index(samples[0])]]].copy() \
        if samples else df.copy()
    feats.index = X.index
    sheet = pd.DataFrame({k: pd.Series(v) for k, v in info.items()}).reindex(samples)
    if len(sheet.columns):
        sheet.insert(0, "_file", samples)
        notes.append("MS-DIAL: klasy, typy plików, kolejność i partie wzięte z wierszy nad tabelą")
    else:
        sheet = None
    return X, feats, sheet


def _profinder_blocks(columns):
    """{tag: {sample: column}} for columns named like '[Area] sample' (Agilent Profinder / MassHunter MFE)."""
    blocks = {}
    for c in columns:
        m = PROFINDER_TAG.match(str(c))
        if m:
            blocks.setdefault(m.group(1).strip(), {})[" ".join(m.group(2).split())] = c
    return blocks


def _profinder_quant(blocks):
    for want in PROFINDER_QUANT:
        for tag, cols in blocks.items():
            if tag.lower() == want and len(cols) >= 2:
                return tag
    return None


def _matrix_profinder(df, notes):
    """Per-sample blocks [Mass] / [RT] / [Area|Height|Vol] / [Score]...: the intensity block is the matrix, the
    summary columns (Name, Formula, Mass (avg), RT (avg)) describe the features."""
    blocks = _profinder_blocks(df.columns)
    tag = _profinder_quant(blocks)
    cols = blocks[tag]
    X = df[list(cols.values())].apply(pd.to_numeric, errors="coerce").astype(float)
    X.columns = list(cols)
    tagged = {c for b in blocks.values() for c in b.values()}
    feats = df[[c for c in df.columns if c not in tagged and not df[c].isna().all()]].copy()
    low = {str(c).strip().lower(): c for c in feats.columns}
    mass = pd.to_numeric(feats[low["mass (avg)"]], errors="coerce") if "mass (avg)" in low else None
    rt = pd.to_numeric(feats[low["rt (avg)"]], errors="coerce") if "rt (avg)" in low else None
    ids = pd.Series([None] * len(df), index=df.index, dtype=object)
    if mass is not None:
        ids = mass.map("{:.4f}".format) + ("@" + rt.map("{:.2f}".format) if rt is not None else "")
        ids = ids.where(mass.notna(), None)
        notes.append("Profinder podaje masę neutralną (Mass) – w kolumnie mz i na wykresach to masa, nie m/z jonu")
    if "name" in low:
        name = feats[low["name"]]
        ids = name.astype(str).str.strip().where(name.notna(), ids)
    else:
        notes.append("cechy bez nazw – identyfikator = masa@RT")
    X.index = feats.index = ids.values
    others = ", ".join(f"[{t}]" for t in blocks if t != tag)
    notes.append(f"Profinder: macierz z bloku [{tag}] ({len(cols)} próbek); pominięte bloki: {others or 'brak'}")
    return X, feats, f"Agilent Profinder ({tag})"


def _mzmine_columns(columns):
    """{sample file: column} of the MZmine quantity (area preferred), or None when this is not an MZmine export."""
    low = {str(c).strip().lower() for c in columns}
    for pattern, need in ((MZMINE_LEGACY, {"row id"}), (MZMINE_MODULAR, {"id"})):
        if not need <= low:
            continue
        found = {}
        for c in columns:
            m = pattern.match(str(c).strip())
            if m:
                found.setdefault(m.group(2).lower(), {})[m.group(1).strip()] = c
        for kind in ("area", "height"):
            if len(found.get(kind, {})) >= 2:
                return kind, found[kind]
    return None


def _matrix_mzmine(df, notes):
    kind, cols = _mzmine_columns(df.columns)
    X = df[list(cols.values())].apply(pd.to_numeric, errors="coerce").astype(float)
    X.columns = list(cols)
    low = {str(c).strip().lower(): c for c in df.columns}
    idc = low.get("row id", low.get("id"))
    quant = re.compile(r"(?i)^datafile:| Peak [\w/ ]+$")
    feats = df[[c for c in df.columns if not quant.search(str(c)) and not df[c].isna().all()]].copy()
    name = pd.Series([None] * len(df), index=df.index, dtype=object)
    for c in df.columns:
        lc = str(c).strip().lower()
        if any(lc == k or lc.endswith(":" + k) for k in MZMINE_NAME_COLS) and not pd.api.types.is_numeric_dtype(df[c]):
            name = name.where(name.notna(), df[c])
    feats["name"] = name
    ids = pd.to_numeric(df[idc], errors="coerce")
    X.index = feats.index = [str(int(v)) if pd.notna(v) else None for v in ids]
    notes.append(f"MZmine: macierz z kolumn Peak {kind} ({len(cols)} plików); 0 = niewykryte")
    return X, feats, f"MZmine ({kind})"


def _maybe_transposed(name, df):
    """Samples in rows (MetaboAnalyst style: Sample, Label/Group, feature columns)."""
    if any(h in name.lower() for h in META_HINTS):
        return None
    cols = {str(c).strip().lower(): c for c in df.columns}
    gcol = next((cols[a] for a in GROUP_ALIASES if a in cols), None)
    if gcol is None or pd.api.types.is_numeric_dtype(df[gcol]):
        return None
    num = [c for c in _numeric_cols(df) if c != gcol]
    if len(num) < 4 or len(num) <= len(df) or df[gcol].nunique() > max(10, len(df) // 2):
        return None
    idc = _first_id_col(df)
    if idc == gcol:
        ids = [f"S{i + 1}" for i in range(len(df))]
    else:
        ids = df[idc].astype(str).tolist()
    X = df[num].astype(float).T
    X.columns = ids
    X.index = [str(c) for c in X.index]
    sheet = pd.DataFrame({"_file": ids, "group": df[gcol].values}, index=ids)
    return X, pd.DataFrame(index=X.index), sheet


# ---------------------------------------------------------------- matching tables

def _match_fraction(keys, values):
    vals = {_key(v) for v in values if pd.notna(v)}
    return np.mean([k in vals for k in keys]) if keys else 0.0


def _find_sheet(X, tables, group_col=""):
    """The table with a column matching the matrix column names (best match; a tie goes to the table that also has
    the group column, then to the first one)."""
    keys = [_key(c) for c in X.columns]
    want = str(group_col or "").strip().lower()
    best, best_rank = None, None
    for t in tables:
        has_group = want in {str(c).strip().lower() for c in t["df"].columns}
        for c in t["df"].columns:
            col = t["df"][c]
            if pd.api.types.is_float_dtype(col):
                continue
            frac = _match_fraction(keys, col.astype(str))
            rank = (round(frac, 6), has_group)
            if frac >= 0.5 and (best is None or rank > best_rank):
                best, best_rank = (frac, t, c), rank
    return best


def _string_like(col):
    s = col.dropna().astype(str)
    return len(s) >= 2 and s.str.contains(r"[A-Za-z]").mean() > 0.5


def _shared_key(sheet, df):
    """Join key by shared column names: one column whose values are (nearly) unique in `df`, otherwise two columns
    that are unique together (Donor_ID + Timepoint). Returns (sheet columns, df columns) or None."""
    left = {str(c).strip().lower(): c for c in sheet.columns if not str(c).startswith("_")}
    right = {str(c).strip().lower(): c for c in df.columns}
    shared = [k for k in left if k in right and k]
    if not shared:
        return None

    def rows(frame, cols):
        every = zip(*[frame[c].tolist() for c in cols], strict=True)
        return [tuple(str(v).strip() for v in r) for r in every if not any(pd.isna(v) for v in r)]

    def match(keys):
        lt, rt = rows(sheet, [left[k] for k in keys]), rows(df, [right[k] for k in keys])
        if not lt or not rt:
            return 0.0, 1.0
        seen = set(rt)
        return float(np.mean([r in seen for r in lt])), 1 - len(seen) / len(rt)

    best = None
    for k in shared:
        frac, dup = match([k])
        if frac >= 0.5 and dup <= 0.1 and (best is None or frac > best[0]):
            best = (frac, [k])
    if best is None:
        for i, a in enumerate(shared):
            for b in shared[i + 1:]:
                frac, dup = match([a, b])
                if frac >= 0.5 and dup == 0 and (best is None or frac > best[0]):
                    best = (frac, [a, b])
    if best is None:
        return None
    return [left[k] for k in best[1]], [right[k] for k in best[1]]


def _merge_secondary(sheet, tables, notes):
    for t in tables:
        df = t["df"]
        key = _shared_key(sheet, df)
        if key is not None:
            lcols, rcols = key
            def joined(frame, cols):
                rows = zip(*[frame[c].tolist() for c in cols], strict=True)
                return [None if any(pd.isna(v) for v in r) else " + ".join(str(v).strip() for v in r) for r in rows]

            rk = joined(df, rcols)
            ok = [k is not None for k in rk]
            right = df[ok].copy()
            right.index = pd.Index([k for k in rk if k is not None], dtype=object)
            dup = right.index.duplicated()
            if dup.any():
                which = ", ".join(sorted(set(right.index[dup]))[:3])
                notes.append(f"{t['name']}: powtórzony klucz {which} – wzięto pierwszy wiersz")
                right = right[~dup]
            add = [c for c in right.columns if c not in sheet.columns and not right[c].isna().all()]
            mapped = right.reindex(pd.Index(joined(sheet, lcols), dtype=object))[add]
            mapped.index = sheet.index
            sheet = pd.concat([sheet, mapped], axis=1)
            t["used"] = t["secondary"] = True
            notes.append(f"dołączono {t['name']} do arkusza próbek po kolumnie {' + '.join(map(str, lcols))} "
                         f"({', '.join(map(str, add[:6]))}{'…' if len(add) > 6 else ''})")
            continue
        best = None
        for cu in df.columns:
            if not _string_like(df[cu]) or not df[cu].dropna().astype(str).is_unique:
                continue
            vals = set(df[cu].dropna().astype(str))
            for cs in sheet.columns:
                if str(cs).startswith("_") or not _string_like(sheet[cs]):
                    continue
                sv = sheet[cs].dropna().astype(str)
                frac = sv.isin(vals).mean()
                if frac >= 0.5 and (best is None or frac > best[0]):
                    best = (frac, cu, cs)
        if best is None:
            continue
        _, cu, cs = best
        right = df.dropna(subset=[cu]).copy()
        right[cu] = right[cu].astype(str)
        right = right.set_index(cu)
        add = [c for c in right.columns if c not in sheet.columns]
        mapped = right.reindex(sheet[cs].astype(str).where(sheet[cs].notna(), None))[add]
        mapped.index = sheet.index
        sheet = pd.concat([sheet, mapped], axis=1)
        t["used"] = t["secondary"] = True     # may serve the next layer's sheet too (subjects.csv)
        notes.append(f"dołączono {t['name']} do arkusza próbek po kolumnie {cs} ({', '.join(map(str, add[:6]))}"
                     f"{'…' if len(add) > 6 else ''})")
    return sheet


# ---------------------------------------------------------------- design read from sample names

def _name_tokens(name):
    return [t for t in re.split(r"[^A-Za-z0-9]+", sample_stem(name)) if t]


def _name_rule(names, values, min_rows=6, min_support=1.5, min_consistency=0.9):
    """Which part of the sample names tells a sheet column (Donor_ID = 1st token, Timepoint = 'M' + 2nd token)?

    Learned from the samples that have a value: per kind of name (QC/blank/sample, number of tokens) the token
    position whose tokens map to the column values consistently. Returns a function name -> value or None."""
    rules = {}
    known = [(n, str(v)) for n, v in zip(names, values, strict=True) if pd.notna(v)]
    kinds = {}
    for n, v in known:
        tok = _name_tokens(n)
        kinds.setdefault((_type_from_name(sample_stem(n)), len(tok)), []).append((tok, v))
    for kind, rows in kinds.items():
        if len(rows) < min_rows or len({v for _, v in rows}) < 2:
            continue
        best = None
        for i in range(kind[1]):
            by_tok = {}
            for tok, v in rows:
                by_tok.setdefault(tok[i].lower(), []).append(v)
            if len(rows) / len(by_tok) < min_support:
                continue
            major = {k: max(set(vs), key=vs.count) for k, vs in by_tok.items()}
            cons = np.mean([major[tok[i].lower()] == v for tok, v in rows])
            if cons >= min_consistency and (best is None or cons > best[0]):
                best = (cons, i, major)
        if best is None:
            continue
        _, i, major = best
        frames = set()
        for k, v in major.items():
            at = v.lower().find(k)
            frames.add((v[:at], v[at + len(k):]) if at >= 0 else None)
        frame = frames.pop() if len(frames) == 1 else None      # one prefix/suffix around the token for all values
        rules[kind] = (i, major, frame)

    def rule(name):
        tok = _name_tokens(name)
        r = rules.get((_type_from_name(sample_stem(name)), len(tok)))
        if r is None:
            return None
        i, major, frame = r
        if tok[i].lower() in major:
            return major[tok[i].lower()]
        return frame[0] + tok[i] + frame[1] if frame is not None else None

    return rule if rules else None


def _near_matches(columns, free_keys):
    """Matrix columns without a sheet row vs sheet rows without a column: the same first token and the same run
    number at the end ('AB 1_2A 107' and 'AB_1_1A_107.mzML') is the same file described differently."""
    out, taken = {}, set()
    for c in columns:
        tc = _key(c).split("_")
        if len(tc) < 3 or not re.fullmatch(r"\d{2,}", tc[-1]):
            continue
        cand = [k for k in free_keys if k not in taken and (tk := k.split("_")) and len(tk) == len(tc)
                and tk[0] == tc[0] and tk[-1] == tc[-1]]
        if len(cand) == 1:
            out[c] = cand[0]
            taken.add(cand[0])
    return out


def _fill_from_names(sheet, columns, match, notes, warnings, inferred, done):
    """Design columns (group, subject/pair, batch) for samples the sheet does not describe, read from their names
    by rules learned on the described ones; disagreements between a name and its sheet row are reported."""
    exact = [n for n in sheet.index if match.get(n) == "exact"]
    loose = [n for n in sheet.index if match.get(n) != "exact"]
    filled = set(inferred.get("from_names_samples", []))
    for col in columns:
        if col is None or col not in sheet.columns or col in done:
            continue
        rule = _name_rule(exact, sheet.loc[exact, col])
        if rule is None:
            continue
        done.add(col)
        differ = [n for n in exact if pd.notna(sheet.loc[n, col]) and rule(n) is not None
                  and rule(n) != str(sheet.loc[n, col])]
        if differ:
            warnings.append(f"arkusz i nazwa pliku mówią co innego o {col}: "
                            + "; ".join(f"{n}: arkusz {sheet.loc[n, col]}, nazwa {rule(n)}" for n in differ[:3])
                            + ("…" if len(differ) > 3 else "") + " (użyto arkusza)")
        got = []
        if sheet[col].dtype != object:
            sheet[col] = sheet[col].astype(object)
        for n in loose + [n for n in exact if pd.isna(sheet.loc[n, col])]:
            v = rule(n)
            if v is not None and (match.get(n) != "exact" or pd.isna(sheet.loc[n, col])):
                if pd.isna(sheet.loc[n, col]) or str(sheet.loc[n, col]) != v:
                    sheet.loc[n, col] = v
                    got.append(n)
        if got:
            filled.update(got)
            notes.append(f"{col} z nazw plików dla {len(got)} próbek bez (zgodnego) wiersza w arkuszu: "
                         + ", ".join(map(str, got[:8])) + ("…" if len(got) > 8 else ""))
    inferred["from_names_samples"] = sorted(filled)
    inferred["from_names"] = len(filled)
    return sheet


def _find_features(X, tables):
    ids = set(X.index.astype(str))
    best = None
    for t in tables:
        for c in t["df"].columns:
            vals = t["df"][c].dropna().astype(str)
            if not len(vals):
                continue
            frac = len(ids & set(vals)) / max(1, len(ids))
            if frac >= 0.8 and (best is None or frac > best[0]):
                best = (frac, t, c)
    return best


def _pick(cols_lower, aliases):
    return next((cols_lower[a] for a in aliases if a in cols_lower), None)


def standardize_features(feats, X):
    """Standard columns name/gene/accession/formula/mz/rt/cluster/adduct next to the original ones."""
    f = feats.reindex(X.index).copy()
    low = {str(c).strip().lower(): c for c in f.columns}
    for std, aliases in FEATURE_COLS.items():
        if std in f.columns and std not in low.values():
            continue
        src = _pick(low, aliases)
        if src is None:
            f[std] = np.nan
            continue
        vals = f[src]
        if std in ("gene", "accession"):
            vals = vals.astype(str).str.split(";").str[0].str.strip().where(vals.notna())
            vals = vals.replace({"": np.nan, "nan": np.nan})
        if std == "accession":
            vals = vals.where(vals.astype(str).str.match(ACCESSION_RE.pattern))
        f[std] = vals.values
    if f["accession"].isna().all():
        acc = pd.Series(X.index, index=X.index).astype(str).str.split(";").str[0]
        f["accession"] = acc.where(acc.str.match(ACCESSION_RE.pattern))
    name = f["name"].astype(object).where(f["name"].notna())
    unknown = name.astype(str).str.lower().isin(["unknown", "nan", "none", "", "w/o ms2"]) | name.isna()
    name = name.where(~unknown, f["gene"])
    f["name"] = name
    return f


def _layer_of(source, fname, feats, X):
    if source.startswith("DIA-NN"):
        return "proteomics"
    if source == "MS-DIAL" or source.startswith(("Agilent Profinder", "MZmine")):
        return "metabolomics"
    low = fname.lower()
    if "prot" in low:
        return "proteomics"
    if "metab" in low or "lipid" in low:
        return "metabolomics"
    if feats["accession"].notna().mean() > 0.5 or feats["gene"].notna().mean() > 0.5:
        return "proteomics"
    if feats["mz"].notna().mean() > 0.5 or feats["formula"].notna().mean() > 0.3:
        return "metabolomics"
    ids = pd.Series(X.index.astype(str)).str.split(";").str[0]
    if ids.str.match(ACCESSION_RE.pattern).mean() > 0.5:
        return "proteomics"
    return "metabolomics"


# ---------------------------------------------------------------- samples

def _resolve(low, wanted, aliases):
    """Sheet column for a parameter value (case-insensitive) or the first known alias; None when absent."""
    key = str(wanted or "").strip().lower()
    if key and key in low:
        return low[key]
    return _pick(low, aliases)


def build_samples(X, sheet, group_col, notes, inferred, pair_col=""):
    """Canonical sample columns from the sheet (when there is one) or the column names."""
    names = list(X.columns)
    guess = infer_from_names(names)
    out = pd.DataFrame(index=names)
    if sheet is not None:
        sheet = sheet.copy()
        sheet.index = names
        out = pd.concat([out, sheet.drop(columns=[c for c in sheet.columns if c == "_file"])], axis=1)
    low = {str(c).strip().lower(): c for c in out.columns if not str(c).startswith("_")}

    tcol = _pick(low, TYPE_ALIASES)
    if tcol is not None:
        out["_raw_type"] = out[tcol].astype(object)
        t = out[tcol].map(canonical_type)
        out["_type"] = t.where(t.notna(), guess["_type"])
        inferred["type"] = False
    else:
        out["_raw_type"] = guess["_type"]
        out["_type"] = guess["_type"]
        inferred["type"] = True

    gcol = low.get(group_col.strip().lower()) if group_col else None
    if gcol is None and group_col and group_col.strip().lower() != "group" and sheet is not None:
        raise InputError(f"W arkuszu próbek nie ma kolumny group={group_col}. Kolumny: "
                         f"{', '.join(str(c) for c in sheet.columns if not str(c).startswith('_'))[:400]}")
    if gcol is None:
        gcol = _pick(low, GROUP_ALIASES)
    if gcol is not None:
        g = out[gcol].astype(object).where(out[gcol].notna())
        # a "class" of QC/blank in the sheet also tells the type (MS-DIAL Class row, MetaboAnalyst labels)
        gt = g.map(lambda v: canonical_type(v) if isinstance(v, str) and canonical_type(v) in ("qc", "blank")
                   and re.search(r"qc|pool|blank", v, re.I) else None)
        if tcol is None:
            out.loc[gt.notna(), "_type"] = gt[gt.notna()]
            out.loc[gt.notna(), "_raw_type"] = gt[gt.notna()]
        out["_group"] = g.astype(str).where(g.notna())
        inferred["group"] = False
        inferred["group_col"] = str(gcol)
    else:
        out["_group"] = guess["_group"]
        inferred["group"] = True
        inferred["group_col"] = None
        if guess.attrs.get("design"):
            notes.append("z nazw plików (wzór: osoba grupa_próbka+litera numer): osoba, grupa (punkt czasowy) i "
                         "replikaty ekstrakcji")
    out.loc[~out._type.isin(["sample", "replicate"]), "_group"] = None

    ocol = _pick(low, ORDER_ALIASES)
    order = pd.to_numeric(out[ocol], errors="coerce") if ocol is not None else None
    if order is not None and order.notna().mean() > 0.8:
        out["_order"] = order.astype(float)
        inferred["order"] = False
        gaps = out["_order"].isna()
        if gaps.any():      # injections the sequence does not list: after the last known one, in column order
            out.loc[gaps, "_order"] = out["_order"].max() + np.arange(1, int(gaps.sum()) + 1)
            notes.append(f"{int(gaps.sum())} iniekcji bez kolejności w arkuszu – dopisane na końcu: "
                         + ", ".join(map(str, out.index[gaps][:8])))
    else:
        out["_order"] = np.arange(1, len(names) + 1, dtype=float)
        inferred["order"] = True
    bcol = _pick(low, BATCH_ALIASES)
    if bcol is not None and out[bcol].notna().any():
        batch = out[bcol].astype(object).where(out[bcol].notna())
        for other in (low[a] for a in BATCH_ALIASES if a in low and low[a] != bcol):
            both = out[[bcol, other]].dropna().astype(str).drop_duplicates()
            if batch.isna().any() and len(both) and both[other].is_unique:      # day7 always goes with Batch_7
                lookup = dict(zip(both[other], both[bcol], strict=True))
                fill = out[other].map(lambda v, m=lookup: m.get(str(v)))
                n_fill = int((batch.isna() & fill.notna()).sum())
                if n_fill:
                    batch = batch.where(batch.notna(), fill)
                    notes.append(f"partia uzupełniona z kolumny {other} dla {n_fill} iniekcji")
        out["_batch"] = batch.map(lambda v: "?" if pd.isna(v) else str(v))
        if (out["_batch"] == "?").any():
            notes.append(f"{int((out['_batch'] == '?').sum())} iniekcji bez partii (partia „?” – bez korekty partii)")
    else:
        out["_batch"] = "1"
    scol = _pick(low, SUBJECT_ALIASES)
    if scol is not None:
        out["_subject"] = out[scol].astype(object).where(out[scol].notna())
    else:
        out["_subject"] = guess["_subject"] if guess.attrs.get("design") else None
    want = str(pair_col or "").strip()
    if not want:
        out["_pair"] = None
    elif want.lower() in low:
        pc = low[want.lower()]
        out["_pair"] = out[pc].astype(object).where(out[pc].notna())
        inferred["pair_col"] = str(pc)
    elif sheet is None and guess.attrs.get("design"):
        out["_pair"] = guess["_subject"]
        inferred["pair_col"] = "osoba z nazw plików"
        notes.append(f"pair={want}: brak arkusza próbek – parą jest osoba odczytana z nazw plików")
    else:
        raise InputError(f"Nie ma kolumny pair={want}" + (" w arkuszu próbek. Kolumny: " + ", ".join(
            str(c) for c in sheet.columns if not str(c).startswith("_"))[:400] if sheet is not None
            else " – bez arkusza próbek nie wiem, które próbki tworzą parę (dodaj arkusz z tą kolumną)."))
    out.loc[~out._type.isin(["sample", "replicate"]), "_pair"] = None
    return out


# ---------------------------------------------------------------- whole dataset

def _clean_matrix(X, notes):
    idx = pd.Series(X.index, dtype=object)
    empty = idx.isna() | (idx.astype(str).str.strip() == "")
    if empty.any():
        idx[empty] = [f"cecha_{i + 1}" for i in np.flatnonzero(empty)]
        notes.append(f"{int(empty.sum())} cech bez identyfikatora – nazwane cecha_N")
    dup = idx.duplicated(keep=False)
    if dup.any():
        n_dup = int(idx.duplicated().sum())
        counts = {}
        new = []
        for v in idx:
            counts[v] = counts.get(v, 0) + 1
            new.append(v if counts[v] == 1 else f"{v}_{counts[v]}")
        idx = pd.Series(new)
        notes.append(f"{n_dup} powtórzonych identyfikatorów cech – dopisano _2, _3…")
    X = X.copy()
    X.index = idx.astype(str).values
    vals = X.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    looks_log = bool(finite.size) and np.percentile(finite, 99) < 40 and finite.max() < 60 \
        and finite.min() > -20 and (np.median(finite) > 5 or (finite < 0).any())
    if looks_log:
        X = 2.0 ** X
        notes.append("wartości wyglądają na log2 – przeliczone na skalę liniową (2^x)")
    else:
        nonpos = int((vals <= 0).sum())
        if nonpos:
            X = X.where(X > 0)
            notes.append(f"{nonpos} wartości ≤ 0 potraktowano jako brak (niewykryte)")
    return X


def _rename_samples(X, sheet):
    stems = [sample_stem(c) for c in X.columns]
    files = list(X.columns)
    if len(set(stems)) == len(stems):
        X = X.copy()
        X.columns = stems
        if sheet is not None:
            sheet = sheet.copy()
            sheet.index = stems
    return X, sheet, files


def load_datasets(workdir, group_col="group", layer="auto", pair_col=""):
    files = collect_files(workdir)
    if not files:
        raise InputError("Nie widzę żadnej tabeli (csv/tsv/txt/xlsx/parquet ani zipa z nimi).")
    tables = []
    for p in files:
        for sheet_name, df, block in read_tables(p):
            if df.shape[0] == 0 or df.shape[1] < 2:
                continue
            tables.append({"path": p, "name": p.name if sheet_name is None else f"{p.name}[{sheet_name}]",
                           "df": df, "block": block, "used": False})
    if not tables:
        raise InputError("Pliki nie wyglądają na tabele: " + ", ".join(p.name for p in files[:5]))

    found = []
    for t in tables:
        df, name, notes = t["df"], t["name"], []
        cols = set(df.columns)
        sheet = None
        if "Run" in cols and cols & {"Protein.Group", "Genes"} and cols & set(DIANN_QUANT):
            X, feats = _matrix_diann_long(df, notes)
            src = "DIA-NN raport (format długi)"
        elif "Protein.Group" in cols or {"Genes", "N.Sequences"} <= cols:
            X, feats, src = _matrix_diann(df, notes)
        elif _profinder_quant(_profinder_blocks(df.columns)) is not None:
            X, feats, src = _matrix_profinder(df, notes)
        elif _mzmine_columns(df.columns) is not None:
            X, feats, src = _matrix_mzmine(df, notes)
        elif t["block"] is not None or "Alignment ID" in cols:
            X, feats, sheet = _matrix_msdial(df, t["block"], notes)
            src = "MS-DIAL"
        elif (tr := _maybe_transposed(name, df)) is not None:
            X, feats, sheet = tr
            src = "tabela próbki × cechy (odwrócona)"
        elif _looks_like_matrix(name, df):
            X, feats, _ = _matrix_generic(df)
            src = "tabela cechy × próbki"
        else:
            continue
        if X.shape[1] < 2 or X.shape[0] < 2:
            continue
        found.append({"t": t, "X": X, "feats": feats, "sheet": sheet, "src": src, "notes": notes})
    if not found:
        raise InputError("Nie znalazłem macierzy intensywności (cechy × próbki). Wyślij DIA-NN pg_matrix/gg_matrix, "
                         "raport DIA-NN, eksport MS-DIAL (Alignment) albo tabelę: 1. kolumna = ID cechy, dalej próbki.")

    # internal standards: a small matrix with the same sample columns as a bigger one
    mains, is_tables = [], {}
    for f in sorted(found, key=lambda f: -f["X"].size):
        host = next((m for m in mains if f["X"].shape[0] < max(100, 0.1 * m["X"].shape[0])
                     and _match_fraction([_key(c) for c in m["X"].columns], f["X"].columns) >= 0.8), None)
        if host is not None and (any(h in f["t"]["name"].lower() for h in IS_HINTS)
                                 or f["X"].shape[0] <= 30):
            is_tables[id(host)] = f
            f["t"]["used"] = True
        else:
            mains.append(f)
    for m in mains:
        m["t"]["used"] = True
    others = [t for t in tables if not t["used"]]

    datasets = []
    for m in mains:
        notes = list(m["notes"])
        inferred = {}
        X = _clean_matrix(m["X"], notes)
        if len(m["feats"]) == len(X):       # row for row the same features: follow the cleaned (unique) IDs
            m["feats"] = m["feats"].copy()
            m["feats"].index = X.index
        sheet = m["sheet"]
        sheet_from = "plik" if sheet is not None else None
        best = _find_sheet(X, [t for t in others if t is not m["t"]], group_col)
        match = {}
        if best is not None:
            frac, t, c = best
            df = t["df"].copy()
            df["_k"] = df[c].map(_key)
            df = df[df[c].notna()].drop_duplicates("_k").set_index("_k")
            keys = [_key(x) for x in X.columns]
            ext = df.reindex(keys).drop(columns=[], errors="ignore")
            ext.index = X.columns
            match = {x: "exact" for x, k in zip(X.columns, keys, strict=True) if k in df.index}
            near = _near_matches([x for x in X.columns if x not in match], [k for k in df.index if k not in keys])
            for x, k in near.items():
                ext.loc[x] = df.loc[k]
                match[x] = "near"
            if near:
                warns = m.setdefault("warnings", [])
                warns.append(f"{len(near)} plików dopasowano do arkusza tylko po numerze na końcu nazwy (opis w "
                             "arkuszu jest inny, grupę/osobę wzięto z nazwy pliku): "
                             + "; ".join(f"{x} ↔ {df.loc[k, c]}" for x, k in list(near.items())[:4])
                             + ("…" if len(near) > 4 else ""))
            unused_rows = [str(df.loc[k, c]) for k in df.index if k not in keys and k not in near.values()]
            if unused_rows:
                notes.append(f"{len(unused_rows)} wierszy arkusza bez kolumny w macierzy: "
                             + ", ".join(unused_rows[:6]) + ("…" if len(unused_rows) > 6 else ""))
            if sheet is not None:
                sheet.index = X.columns
                ext = pd.concat([ext, sheet[[k for k in sheet.columns if k not in ext.columns]]], axis=1)
            sheet = ext
            t["used"] = True
            t["sheet_of"] = m["t"]["name"]
            sheet_from = t["name"]
            lost = [x for x in X.columns if x not in match]
            notes.append(f"arkusz próbek: {t['name']} (kolumna {c}, dopasowano {len(match)}/{X.shape[1]} próbek)")
            if lost:
                notes.append(f"{len(lost)} próbek bez wiersza w arkuszu: " + ", ".join(map(str, lost[:8]))
                             + ("…" if len(lost) > 8 else ""))
        m["sheet_final"], m["sheet_from"], m["Xc"], m["notes2"], m["inferred"] = sheet, sheet_from, X, notes, inferred
        m["match"] = match

    remaining = [t for t in tables if not t["used"]]
    for m in mains:
        X, notes, inferred = m["Xc"], m["notes2"], m["inferred"]
        feats = m["feats"]
        fb = _find_features(X, [t for t in remaining if "sheet_of" not in t])
        if fb is not None:
            frac, t, c = fb
            df = t["df"].drop_duplicates(c).copy()
            df.index = df[c].astype(str)
            feats = pd.concat([feats.reindex(X.index), df.reindex(X.index).drop(
                columns=[k for k in df.columns if k in feats.columns])], axis=1)
            t["used"] = True
            notes.append(f"opis cech: {t['name']} ({frac:.0%} cech)")
        sheet = m["sheet_final"]
        warnings = m.setdefault("warnings", [])
        if sheet is not None:
            sheet.index = X.columns
            match, done = m["match"], set()

            def design_cols(sh):
                low = {str(c).strip().lower(): c for c in sh.columns if not str(c).startswith("_")}
                return [_resolve(low, group_col, GROUP_ALIASES), _resolve(low, pair_col, SUBJECT_ALIASES),
                        _pick(low, BATCH_ALIASES)]

            partial = len(match) < X.shape[1] or "near" in match.values()
            if partial or sheet[[c for c in design_cols(sheet) if c is not None]].isna().any().any():
                sheet = _fill_from_names(sheet, design_cols(sheet), match, notes, warnings, inferred, done)
            sec = [t for t in remaining if (not t["used"] or t.get("secondary")) and "sheet_of" not in t]
            sheet = _merge_secondary(sheet, sec, notes)
            if partial:
                sheet = _fill_from_names(sheet, design_cols(sheet), match, notes, warnings, inferred, done)
        X2, sheet, files = _rename_samples(X, sheet)
        if sheet is not None:
            sheet.index = X2.columns
        samples = build_samples(X2, sheet, group_col, notes, inferred, pair_col=pair_col)
        samples["_file"] = files
        feats = standardize_features(feats, X2)
        istd = None
        if id(m) in is_tables:
            f = is_tables[id(m)]
            std = f["X"].copy()
            stems = [sample_stem(c) for c in std.columns]
            if len(set(stems)) == len(stems):
                std.columns = stems
            istd = std.reindex(columns=X2.columns)
            istd = istd.where(istd > 0)
            notes.append(f"wzorce wewnętrzne: {f['t']['name']} ({istd.shape[0]} wzorców)")
        lay = _layer_of(m["src"], m["t"]["name"], feats, X2)
        if istd is None and lay == "metabolomics":
            formula = feats["formula"] if "formula" in feats else pd.Series(None, index=feats.index)
            std = pd.Series([is_internal_standard(n, f) for n, f in zip(feats["name"], formula, strict=True)],
                            index=X2.index)
            if std.any() and not std.all():
                istd = X2[std].where(X2[std] > 0)
                X2, feats = X2[~std], feats[~std]
                notes.append(f"wzorce wewnętrzne: {int(std.sum())} cech rozpoznanych po nazwie/wzorze (znakowane "
                             "izotopowo) – wyjęte z macierzy przed filtrami, służą do kontroli iniekcji i dryfu: "
                             + ", ".join(map(str, istd.index[:8])) + ("…" if std.sum() > 8 else ""))
        datasets.append(Dataset(name=m["t"]["name"], layer=lay, source=m["src"], X=X2, features=feats,
                                samples=samples, istd=istd, notes=notes, inferred=inferred, warnings=warnings))

    unused = [t["name"] for t in tables if not t["used"] and not NOT_A_TABLE.search(t["name"])]
    if unused:
        for d in datasets:
            d.notes.append("nieużyte pliki: " + ", ".join(unused[:6]))
    if layer != "auto":
        chosen = [d for d in datasets if d.layer == layer]
        if not chosen and len(datasets) == 1:
            datasets[0].layer = layer
            datasets[0].notes.append(f"rodzaj danych ustawiony parametrem layer={layer}")
            chosen = datasets
        if not chosen:
            raise InputError(f"layer={layer}, ale macierze wyglądają na: "
                             + ", ".join(f"{d.name} ({d.layer})" for d in datasets))
        datasets = chosen
    # one matrix per layer: the biggest wins, the rest is mentioned
    by_layer = {}
    for d in sorted(datasets, key=lambda d: -d.X.size):
        if d.layer in by_layer:
            by_layer[d.layer].notes.append(f"pominięto drugą macierz tej warstwy: {d.name}")
        else:
            by_layer[d.layer] = d
    return [by_layer[k] for k in sorted(by_layer, key=lambda k: k != "proteomics")]


def choose_ref(groups, ref):
    """Reference group: the `ref` parameter, otherwise a control-looking name, otherwise the first one."""
    levels = sorted({str(g) for g in groups if g is not None and not (isinstance(g, float) and np.isnan(g))})
    if ref:
        hit = [g for g in levels if g.lower() == ref.strip().lower()]
        if not hit:
            raise InputError(f"ref={ref}: nie ma takiej grupy. Grupy: {', '.join(levels)}")
        return hit[0], False
    ctrl = [g for g in levels if CONTROL_RE.search(g)]
    return (ctrl[0] if ctrl else levels[0]), True
