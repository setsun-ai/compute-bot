"""Input shapes -> Dataset: DIA-NN pg/gg_matrix and long report, MS-DIAL alignment export, generic feature x sample
tables (with a sample sheet, feature annotation and internal standards in separate files), samples-in-rows tables.

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
TYPE_ALIASES = ["sample_type", "sampletype", "sample type", "file type", "file_type", "type", "typ"]
ORDER_ALIASES = ["run_order", "injection order", "injection_order", "run order", "runorder", "order",
                 "acquisition_order", "analytical order", "kolejnosc"]
BATCH_ALIASES = ["batch", "batch id", "batch_id", "partia", "plate"]
SUBJECT_ALIASES = ["subject_id", "subject", "patient_id", "patient", "individual", "donor", "osoba"]
ANNOT_NUMERIC = {"mz", "m/z", "mass", "rt", "rt_min", "retention_time", "rt(min)", "average rt(min)", "average mz",
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
    "mz": ["mz", "m/z", "average mz", "precursor_mz"],
    "rt": ["rt_min", "rt", "average rt(min)", "retention_time", "rt(min)"],
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


# ---------------------------------------------------------------- names

def sample_stem(name):
    """'D:\\SCIEX OS Data\\x\\P1_3.wiff' -> 'P1_3'."""
    s = re.split(r"[\\/]", str(name).strip())[-1]
    prev = None
    while prev != s:
        prev, s = s, RAW_EXT.sub("", s)
    return s


def _key(name):
    return sample_stem(name).lower()


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
    """Sample type and group guessed from column names (P1_3 -> sample of group P1, QC1 -> qc, Blank_2 -> blank)."""
    rows = []
    for n in names:
        stem = sample_stem(n)
        t = _type_from_name(stem)
        rows.append({"_type": t, "_group": _group_from_name(stem) if t == "sample" else None})
    out = pd.DataFrame(rows, index=list(names))
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


def _coerce_numeric(df):
    for c in df.columns:
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            nonnull = df[c].notna() & (df[c].astype(str).str.strip() != "")
            if nonnull.sum() and conv[nonnull].notna().mean() >= 0.95:
                df[c] = conv
    return df


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
        try:
            raw = pd.read_excel(path, sheet_name=0, header=None, dtype=object)
        except Exception as e:  # noqa: BLE001 - any reader failure means an unreadable workbook for the owner
            raise InputError(f"Nie mogę otworzyć {path.name}: {e}") from None
        rows = raw.values.tolist()
        h = _find_header_row(rows)
        block = [["" if pd.isna(c) else str(c) for c in r] for r in rows[:h]] if h else None
        df = raw.iloc[h + 1:].reset_index(drop=True)
        df.columns = _unique_names(["" if pd.isna(c) else str(c) for c in raw.iloc[h]])
        df = df.replace({s: np.nan for s in NA_STRINGS if s})
        return _coerce_numeric(df), block
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        lines = [f.readline() for _ in range(40)]
    lines = [ln for ln in lines if ln]
    if not lines:
        return pd.DataFrame(), None
    sep = _sniff_sep(lines)
    rows = list(csv.reader(lines, delimiter=sep))
    h = _find_header_row(rows)
    block = rows[:h] if h else None
    try:
        df = pd.read_csv(path, sep=sep, skiprows=h, na_values=NA_STRINGS, keep_default_na=True, low_memory=False,
                         encoding="utf-8-sig", decimal="," if sep == ";" else ".")
    except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError):
        return pd.DataFrame(), None
    return _coerce_numeric(df), block


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


def _find_sheet(X, tables):
    keys = [_key(c) for c in X.columns]
    best = None
    for t in tables:
        for c in t["df"].columns:
            col = t["df"][c]
            if pd.api.types.is_float_dtype(col):
                continue
            frac = _match_fraction(keys, col.astype(str))
            if frac >= 0.5 and (best is None or frac > best[0]):
                best = (frac, t, c)
    return best


def _string_like(col):
    s = col.dropna().astype(str)
    return len(s) >= 2 and s.str.contains(r"[A-Za-z]").mean() > 0.5


def _merge_secondary(sheet, tables, notes):
    for t in tables:
        df = t["df"]
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
        notes.append(f"dołączono {t['path'].name} do arkusza próbek po kolumnie {cs} ({', '.join(map(str, add[:6]))}"
                     f"{'…' if len(add) > 6 else ''})")
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
    if source == "MS-DIAL":
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

def build_samples(X, sheet, group_col, notes, inferred):
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
    out.loc[~out._type.isin(["sample", "replicate"]), "_group"] = None

    ocol = _pick(low, ORDER_ALIASES)
    order = pd.to_numeric(out[ocol], errors="coerce") if ocol is not None else None
    if order is not None and order.notna().mean() > 0.8:
        out["_order"] = order.astype(float)
        inferred["order"] = False
    else:
        out["_order"] = np.arange(1, len(names) + 1, dtype=float)
        inferred["order"] = True
    bcol = _pick(low, BATCH_ALIASES)
    if bcol is not None and out[bcol].notna().any():
        out["_batch"] = out[bcol].astype(str).where(out[bcol].notna(), "?")
    else:
        out["_batch"] = "1"
    scol = _pick(low, SUBJECT_ALIASES)
    out["_subject"] = out[scol].astype(object).where(out[scol].notna()) if scol is not None else None
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


def load_datasets(workdir, group_col="group", layer="auto"):
    files = collect_files(workdir)
    if not files:
        raise InputError("Nie widzę żadnej tabeli (csv/tsv/txt/xlsx/parquet ani zipa z nimi).")
    tables = []
    for p in files:
        df, block = read_table(p)
        if df.shape[0] == 0 or df.shape[1] < 2:
            continue
        tables.append({"path": p, "df": df, "block": block, "used": False})
    if not tables:
        raise InputError("Pliki nie wyglądają na tabele: " + ", ".join(p.name for p in files[:5]))

    found = []
    for t in tables:
        df, name, notes = t["df"], t["path"].name, []
        cols = set(df.columns)
        sheet = None
        if "Run" in cols and cols & {"Protein.Group", "Genes"} and cols & set(DIANN_QUANT):
            X, feats = _matrix_diann_long(df, notes)
            src = "DIA-NN raport (format długi)"
        elif "Protein.Group" in cols or {"Genes", "N.Sequences"} <= cols:
            X, feats, src = _matrix_diann(df, notes)
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
        if host is not None and (any(h in f["t"]["path"].name.lower() for h in IS_HINTS)
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
        sheet = m["sheet"]
        sheet_from = "plik" if sheet is not None else None
        best = _find_sheet(X, [t for t in others if t["path"] != m["t"]["path"]])
        if best is not None:
            frac, t, c = best
            df = t["df"].copy()
            df["_k"] = df[c].map(_key)
            df = df.drop_duplicates("_k").set_index("_k")
            ext = df.reindex([_key(x) for x in X.columns]).drop(columns=[], errors="ignore")
            ext.index = X.columns
            if sheet is not None:
                sheet.index = X.columns
                ext = pd.concat([ext, sheet[[k for k in sheet.columns if k not in ext.columns]]], axis=1)
            sheet = ext
            t["used"] = True
            t["sheet_of"] = m["t"]["path"].name
            sheet_from = t["path"].name
            missing = int(df.reindex([_key(x) for x in X.columns]).isna().all(axis=1).sum())
            notes.append(f"arkusz próbek: {t['path'].name} (kolumna {c}, dopasowano {frac:.0%} próbek)")
            if missing:
                notes.append(f"{missing} próbek bez wiersza w arkuszu")
        m["sheet_final"], m["sheet_from"], m["Xc"], m["notes2"], m["inferred"] = sheet, sheet_from, X, notes, inferred

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
            notes.append(f"opis cech: {t['path'].name} ({frac:.0%} cech)")
        sheet = m["sheet_final"]
        if sheet is not None:
            sec = [t for t in remaining if (not t["used"] or t.get("secondary")) and "sheet_of" not in t]
            sheet = _merge_secondary(sheet, sec, notes)
        X2, sheet, files = _rename_samples(X, sheet)
        if sheet is not None:
            sheet.index = X2.columns
        samples = build_samples(X2, sheet, group_col, notes, inferred)
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
            notes.append(f"wzorce wewnętrzne: {f['t']['path'].name} ({istd.shape[0]} wzorców)")
        lay = _layer_of(m["src"], m["t"]["path"].name, feats, X2)
        datasets.append(Dataset(name=m["t"]["path"].name, layer=lay, source=m["src"], X=X2, features=feats,
                                samples=samples, istd=istd, notes=notes, inferred=inferred))

    unused = [t["path"].name for t in tables if not t["used"]]
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
