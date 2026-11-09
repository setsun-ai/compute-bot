"""Reading the owner's table once, at the boundary: header rows, junk cells, column roles."""
from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


class InputError(Exception):
    """Something in the owner's file or parameters that they have to fix, in Polish."""


_EXCEL_ERROR = re.compile(r"^#(VALUE!|DIV/0!|N/A|NAME\?|REF!|NUM!|NULL!|SPILL!|CALC!|GETTING_DATA)$", re.I)
_JUNK = {"", "na", "n/a", "nan", "none", "null", "-", "--", "?", "brak", "b.d.", "bd", "#n/a"}


def normkey(text):
    """Lowercase ASCII key of a column name: 'Zbiór\xa0' -> 'zbior', 'log D' -> 'log d'."""
    text = str(text).replace("ł", "l").replace("Ł", "L")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", text).strip()


def _tokens(name):
    return re.sub(r"[^a-z0-9]+", " ", normkey(name)).split()


def _empty(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def clean_cell(v):
    """Strings stripped of \xa0 and spaces; Excel errors and empty markers become None."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, str):
        s = re.sub(r"\s+", " ", unicodedata.normalize("NFC", v).replace("\xa0", " ")).strip()
        if s.lower() in _JUNK or _EXCEL_ERROR.match(s):
            return None
        return s
    return v


def to_number(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int | float | np.integer | np.floating):
        return float(v) if np.isfinite(v) else None
    s = str(v).replace("−", "-").replace(" ", "")
    if re.fullmatch(r"[-+]?\d+,\d+([eE][-+]?\d+)?", s):
        s = s.replace(",", ".")
    try:
        x = float(s)
    except ValueError:
        return None
    return x if np.isfinite(x) else None


def _read_text(path):
    data = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1250", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", "replace")


def _read_delimited(path):
    text = _read_text(path)
    lines = [ln for ln in text.splitlines() if ln.strip()][:50]
    if Path(path).suffix.lower() == ".tsv":
        sep = "\t"
    else:
        counts = {s: [ln.count(s) for ln in lines] for s in ("\t", ";", ",")}
        # a separator that appears the same (non-zero) number of times on every line; ';' before ',' (decimal commas)
        consistent = [s for s, c in counts.items() if c and min(c) > 0 and len(set(c)) == 1]
        if consistent:
            sep = consistent[0]
        else:
            try:
                sep = csv.Sniffer().sniff("\n".join(lines), delimiters="\t;,").delimiter
            except csv.Error:
                sep = max(counts, key=lambda s: sum(counts[s]))
    rows = list(csv.reader(io.StringIO(text), delimiter=sep))
    width = max((len(r) for r in rows), default=0)
    return pd.DataFrame([r + [None] * (width - len(r)) for r in rows], dtype=object)


def _parses_as_smiles(values):
    try:
        from rdkit import Chem, RDLogger
    except ImportError:
        return np.zeros(len(values), bool)
    RDLogger.DisableLog("rdApp.*")
    try:
        return np.array([Chem.MolFromSmiles(str(v)) is not None for v in values])
    finally:
        RDLogger.EnableLog("rdApp.*")


def _read_smi(path):
    rows = []
    for line in _read_text(path).splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            rows.append(line.split())
    if not rows:
        raise InputError("Plik .smi jest pusty.")
    width = max(len(r) for r in rows)
    rows = [r + [None] * (width - len(r)) for r in rows]
    if not _parses_as_smiles([rows[0][0]])[0]:
        header, rows = rows[0], rows[1:]
    else:
        header = ["SMILES", "name"] + [f"kolumna_{i + 1}" for i in range(2, width)]
    header = [h if h else f"kolumna_{i + 1}" for i, h in enumerate(header)]
    return pd.DataFrame([header] + rows, dtype=object)


def read_raw(path):
    """The first non-empty sheet / the whole delimited file as a grid of objects, no header interpreted yet."""
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        try:
            sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object)
        except ImportError as exc:
            raise InputError("Nie umiem otworzyć starego .xls – zapisz plik jako .xlsx albo .csv.") from exc
        for df in sheets.values():
            if df.notna().sum().sum() >= 4:
                return df
        raise InputError("Arkusz jest pusty.")
    if ext == ".smi":
        return _read_smi(path)
    return _read_delimited(path)


def _is_number(v):
    return to_number(v) is not None


def _second_header(cells):
    """True when row 2 continues the header: labels where row 1 is empty, no numbers, data below."""
    if len(cells) < 2:
        return False
    r0, r1 = cells.iloc[0].tolist(), cells.iloc[1].tolist()
    filled = [j for j, v in enumerate(r1) if not _empty(v)]
    if not filled or any(_is_number(r1[j]) for j in filled):
        return False
    if not any(_empty(r0[j]) for j in filled):
        return False
    if len(cells) == 2:
        return True
    r2 = cells.iloc[2].tolist()
    return np.mean([_is_number(r2[j]) for j in filled]) >= 0.5


def read_table(path):
    """Parsed table: one merged header, clean names, numeric columns as float, others as text or None.

    `df.attrs['unparsed']` maps a numeric column to row positions whose cell was not empty but not a number.
    """
    cells = read_raw(path).map(clean_cell)
    cells = cells.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    cells.columns = range(cells.shape[1])
    if cells.empty:
        raise InputError("Tabela jest pusta.")
    # a lone title in the first row ("Dane do projektu") above the real header
    if len(cells) > 2 and cells.shape[1] >= 3 and cells.iloc[0].notna().sum() == 1 and cells.iloc[1].notna().sum() >= 3:
        cells = cells.iloc[1:].reset_index(drop=True)
    two = _second_header(cells)
    h0 = cells.iloc[0].tolist()
    h1 = cells.iloc[1].tolist() if two else [None] * len(h0)
    names, seen = [], {}
    for j, (a, b) in enumerate(zip(h0, h1, strict=True)):
        nm = str(b if not _empty(b) else a if not _empty(a) else f"kolumna_{j + 1}")
        nm = re.sub(r"\s+", " ", nm.replace("\xa0", " ")).strip()
        if nm in seen:
            seen[nm] += 1
            nm = f"{nm}_{seen[nm]}"
        else:
            seen[nm] = 1
        names.append(nm)
    data = cells.iloc[2 if two else 1:].reset_index(drop=True)
    data.columns = names
    out, unparsed = {}, {}
    for c in names:
        vals = [None if _empty(v) else v for v in data[c].tolist()]
        filled = [v for v in vals if v is not None]
        nums = [to_number(v) for v in vals]
        n_num = sum(x is not None for x in nums)
        if filled and n_num / len(filled) >= 0.7:
            out[c] = pd.Series([np.nan if x is None else x for x in nums], dtype=float)
            bad = [i for i, (v, x) in enumerate(zip(vals, nums, strict=True)) if v is not None and x is None]
            if bad:
                unparsed[c] = bad
        else:
            out[c] = pd.Series([None if v is None else str(v) for v in vals], dtype=object)
    df = pd.DataFrame(out)
    df.attrs["unparsed"] = unparsed
    return df


# ---------------------------------------------------------------- column roles

_TARGET_STRONG = {"activity", "aktywnosc", "pic50", "pec50", "pki", "pkd", "ic50", "ec50", "target", "odpowiedz",
                  "response", "endpoint", "klasa", "class", "label", "yexp", "yobs"}
_TARGET_WEAK = {"y", "exp", "obs"}  # only in short names: PaDEL has ETA_Shape_Y, MOMI-Y ...
_NAME_TOKENS = {"name", "nazwa", "compound", "compounds", "zwiazek", "zwiazki", "molecule", "czasteczka",
                "substancja", "chemical", "id"}
_SPLIT_KEYS = {"zbior", "set", "split", "podzial", "subset", "dataset", "zestaw"}
_ID_TOKENS = {"lp", "no", "nr", "id", "index", "idx"}
_FORMULA = re.compile(r"^([A-Z][a-z]?\d*)+$")
MAX_CLASSES = 10


@dataclass
class Columns:
    target: str | None
    name: str | None
    smiles: str | None
    split: str | None
    given: list = field(default_factory=list)
    task: str = "regression"
    notes: list = field(default_factory=list)


def _find(df, wanted, what):
    key = normkey(wanted)
    for c in df.columns:
        if normkey(c) == key:
            return c
    listed = ", ".join(map(str, list(df.columns)[:15])) + (" …" if df.shape[1] > 15 else "")
    raise InputError(f"Nie ma kolumny {wanted} ({what}). Kolumny w pliku: {listed}")


def _is_id_like(name, s):
    """Row numbers: a column called Lp/No/ID with unique integers, or 1..n / 0..n-1 in row order.

    The name alone is not enough: PaDEL has nR and nO (ring and oxygen counts)."""
    v = s.dropna().to_numpy()
    if len(v) < 3 or not np.all(v == np.round(v)):
        return False
    if normkey(name).strip(" .#") in _ID_TOKENS and len(np.unique(v)) == len(v):
        return True
    d = np.diff(v)
    return bool(len(v) >= 5 and np.all(d == 1) and v[0] in (0, 1))


def detect_columns(df, target="", smiles="", name="", split_col=""):
    numeric = [c for c in df.columns if df[c].dtype == float]
    text = [c for c in df.columns if c not in numeric]
    cols = Columns(None, None, None, None)

    if split_col:
        cols.split = _find(df, split_col, "split_col")
    else:
        cols.split = next((c for c in df.columns if normkey(c).replace(" ", "") in _SPLIT_KEYS), None)

    if smiles:
        cols.smiles = _find(df, smiles, "smiles")
    else:
        by_name = [c for c in text if "smiles" in normkey(c) or normkey(c) == "smi"]
        cols.smiles = by_name[0] if by_name else None
        if cols.smiles is None:
            for c in text:
                if c == cols.split:
                    continue
                vals = [v for v in df[c] if not _empty(v)]
                if len(vals) < 2 or all(_FORMULA.match(v) for v in vals):
                    continue
                if _parses_as_smiles(vals).mean() >= 0.8:
                    cols.smiles = c
                    break

    taken = {cols.split, cols.smiles}
    if target:
        cols.target = _find(df, target, "target")
    else:
        free = [c for c in df.columns if c not in taken]
        strong = [c for c in free if "logk" in normkey(c).replace(" ", "") or set(_tokens(c)) & _TARGET_STRONG]
        weak = [c for c in free if len(_tokens(c)) <= 2 and set(_tokens(c)) & _TARGET_WEAK]
        cand = strong or weak
        if len(cand) > 1:
            raise InputError(f"Nie wiem, która kolumna to odpowiedź: {', '.join(cand)}. Podaj target=<kolumna>.")
        if cand:
            cols.target = cand[0]
        else:
            nums = [c for c in numeric if c not in taken and not _is_id_like(c, df[c])]
            if len(nums) == 1:
                cols.target = nums[0]
                cols.notes.append(f"Odpowiedź: jedyna kolumna liczbowa {nums[0]}.")
            elif nums:
                raise InputError("Nie wiem, która kolumna to odpowiedź (y). Podaj target=<kolumna>, np. "
                                 f"target={nums[-1]}.")
    taken.add(cols.target)

    if name:
        cols.name = _find(df, name, "name")
    else:
        free = [c for c in text if c not in taken]
        named = [c for c in free if set(_tokens(c)) & _NAME_TOKENS]
        cols.name = (named or free or [None])[0]
    taken.add(cols.name)

    cols.given = [c for c in numeric if c not in taken and not _is_id_like(c, df[c])]

    if cols.target is not None:
        y = df[cols.target]
        if y.dtype == float:
            if y.dropna().nunique() <= 2 and y.notna().sum() > 2:
                cols.task = "classification"
        else:
            n_cls = y.dropna().nunique()
            if n_cls < 2 or n_cls > MAX_CLASSES:
                raise InputError(f"Kolumna {cols.target} ma {n_cls} różnych wartości tekstowych – to nie wygląda ani "
                                 "na liczby, ani na klasy. Popraw ją albo podaj target=.")
            cols.task = "classification"
    return cols
