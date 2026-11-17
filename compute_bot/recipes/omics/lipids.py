"""Lipid names -> class, chains, double bonds; class sums, saturation profile, class enrichment (offline), and the
numbers behind the lipidomics plots (Kendrick mass defect, Van Krevelen ratios, Venn regions). Pure functions.

Names are read in the shorthand people actually type: `PC 34:1`, `PC(16:0/18:1)`, `15:0-18:1 PC`, `SMd34:1`,
`Cer d18:1/24:0`, `Lyso-PC18:1`, `PEO-38:5`, `TG48:4 ISO1`. What does not look like a lipid gives None.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats as ss

from .parse import DEUTERIUM_RE
from .stats import bh

CH2 = 14.01565006
# alias (lower case) -> class; lyso forms get their own class (LPC, LPE, ...)
CLASS_ALIASES = {
    "pc": "PC", "pe": "PE", "ps": "PS", "pi": "PI", "pg": "PG", "pa": "PA", "cl": "CL", "bmp": "BMP",
    "lpc": "LPC", "lpe": "LPE", "lps": "LPS", "lpi": "LPI", "lpg": "LPG", "lpa": "LPA",
    "sm": "SM", "cer": "Cer", "ceramide": "Cer", "hexcer": "HexCer", "glccer": "HexCer", "galcer": "HexCer",
    "hex2cer": "Hex2Cer", "laccer": "Hex2Cer", "hex3cer": "Hex3Cer", "cerp": "CerP", "c1p": "CerP", "spb": "SPB",
    "s1p": "S1P", "shexcer": "SHexCer", "gm3": "GM3",
    "tg": "TG", "tag": "TG", "dg": "DG", "dag": "DG", "mg": "MG", "mag": "MG",
    "ce": "CE", "che": "CE", "fa": "FA", "ffa": "FA", "car": "CAR", "acca": "CAR", "acar": "CAR",
    "mgdg": "MGDG", "dgdg": "DGDG", "sqdg": "SQDG", "we": "WE", "nae": "NAE", "fahfa": "FAHFA",
    "phosphatidylcholine": "PC", "phosphatidylethanolamine": "PE", "sphingomyelin": "SM", "triacylglycerol": "TG",
    "diacylglycerol": "DG", "monoacylglycerol": "MG",
}
LYSO = {"PC": "LPC", "PE": "LPE", "PS": "LPS", "PI": "LPI", "PG": "LPG", "PA": "LPA"}
ONE_CHAIN = {"LPC", "LPE", "LPS", "LPI", "LPG", "LPA", "FA", "CE", "MG", "CAR", "SPB", "S1P", "NAE"}
SPHINGO = {"SM", "Cer", "HexCer", "Hex2Cer", "Hex3Cer", "CerP", "SHexCer", "GM3", "SPB", "S1P"}
CHAIN_RE = re.compile(r"(?<![\d.:])(?:(?P<pre>[dtm])|(?P<eth>[OP])-)?(?P<c>\d{1,2}):(?P<db>\d{1,2})"
                      r"(?P<ox>;O\d?|;\d|\(OH\))?(?![\d:.])")
ADDUCT_RE = re.compile(r"\s*\[[^\]]*\]\s*\d*[+-]?\s*$")
ISOMER_RE = re.compile(r"(?i)(\s+(izo|iso|isomer)\s*\d*|\s*_\d+)\s*$")
ELEMENTS = {"C", "H", "D", "N", "O", "P", "S", "Na", "K", "Cl", "F", "Br", "I", "Se", "Si", "Li", "B", "Fe", "Mg",
            "Ca", "Zn", "Cu"}
FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
SAT_ORDER = ["SFA", "MUFA", "PUFA"]


def saturation(db):
    """SFA / MUFA / PUFA by the number of double bonds in the whole molecule (0 / 1 / >= 2)."""
    return "SFA" if db == 0 else "MUFA" if db == 1 else "PUFA"


def _parse_one(text):
    s = ADDUCT_RE.sub("", text.strip())
    prev = None
    while prev != s:
        prev, s = s, ISOMER_RE.sub("", s)
    labelled = bool(DEUTERIUM_RE.search(s))
    s = DEUTERIUM_RE.sub(" ", s)
    lyso = bool(re.search(r"(?i)lyso", s))
    s = re.sub(r"(?i)lyso[\s\-]*", "", s)
    found = list(CHAIN_RE.finditer(s))
    if not found:
        return None
    rest = CHAIN_RE.sub(" ", s)
    cls_text = re.sub(r"[\s()\[\]/_\-,;]+", "", rest).lower()
    cls = CLASS_ALIASES.get(cls_text)
    if cls is None:
        return None
    lo, hi = s.find("("), s.rfind(")")
    if lo >= 0 and hi > lo:
        inside = [m for m in found if lo < m.start() < hi]
        if inside and len(inside) < len(found):      # "18:1 SM (d18:1/18:1)": the brackets hold the full species
            found = inside
    if lyso and cls in LYSO:
        cls = LYSO[cls]
    chains = [(int(m["c"]), int(m["db"])) for m in found]
    hydrox = next((m["pre"] for m in found if m["pre"]), None)
    ox = next((m["ox"] for m in found if m["ox"]), None)
    if hydrox is None and ox and cls in SPHINGO:
        n = re.sub(r"\D", "", ox)
        hydrox = {"": "m", "1": "m", "2": "d", "3": "t"}.get(n)
    ether = next((m["eth"] for m in found if m["eth"]), None)
    carbons, db = sum(c for c, _ in chains), sum(d for _, d in chains)
    if carbons == 0:
        return None
    resolved = len(chains) > 1 or cls in ONE_CHAIN
    species = f"{cls} {(ether + '-') if ether else ''}{hydrox or ''}{carbons}:{db}"
    return {"cls": cls, "carbons": carbons, "db": db, "chains": chains if resolved else None, "hydrox": hydrox,
            "ether": ether, "lyso": cls.startswith("L") and cls in LYSO.values(), "labelled": labelled,
            "species": species}


def parse_lipid(name):
    """Lipid shorthand -> dict(cls, carbons, db, chains, hydrox, ether, lyso, labelled, species) or None.

    `chains` is a list of (carbons, double bonds) when the name gives them (or the class has one chain), None for a
    sum composition. Of two alternatives (`SMd33:1/SMt32:2`) the first one counts."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    text = str(name).strip()
    if not text or ":" not in text:
        return None
    if "/" in text:
        parts = [_parse_one(p) for p in text.split("/")]
        if all(p is not None for p in parts):
            return parts[0]
    return _parse_one(text)


def annotate(names):
    """One row per feature: lipid_class, carbons, db, chains, hydrox, ether, lyso, saturation, species (NaN = not a
    lipid name)."""
    rows = []
    for n in names:
        p = parse_lipid(n)
        if p is None:
            rows.append({})
            continue
        rows.append({"lipid_class": p["cls"], "carbons": p["carbons"], "db": p["db"],
                     "chains": "/".join(f"{c}:{d}" for c, d in p["chains"]) if p["chains"] else None,
                     "hydrox": p["hydrox"], "ether": p["ether"], "lyso": p["lyso"], "labelled": p["labelled"],
                     "saturation": saturation(p["db"]), "species": p["species"]})
    cols = ["lipid_class", "carbons", "db", "chains", "hydrox", "ether", "lyso", "labelled", "saturation", "species"]
    return pd.DataFrame(rows, index=getattr(names, "index", None), columns=cols)


def lipid_mode(setting, names, ann, min_fraction=0.3, min_lipids=3):
    """lipids=tak/nie/auto: auto turns the lipid analysis on when at least ~30 % of the named features parse."""
    if setting == "nie":
        return False
    n = int(ann["lipid_class"].notna().sum())
    if setting == "tak":
        return n > 0
    named = int(pd.Series(names).notna().sum())
    return n >= min_lipids and named > 0 and n / named >= min_fraction


# ---------------------------------------------------------------- classes

def class_sums(lin, labels):
    """Sum of linear intensities per label (class, saturation, ...) and its share of the labelled signal, per
    sample. Features without a label are left out."""
    lab = labels.reindex(lin.index)
    ok = lab.notna()
    sums = lin[ok].groupby(lab[ok]).sum(min_count=1)
    return sums, sums / sums.sum()


def class_enrichment(ann, significant, min_size=1):
    """Fisher exact test (one-sided) per lipid class: is the class over-represented among the significant features?
    Background = all lipid features that were tested. BH across classes."""
    cls = ann["lipid_class"].dropna()
    sig = cls.index.isin(list(significant))
    rows = []
    for k, idx in cls.groupby(cls).groups.items():
        inside = cls.index.isin(idx)
        if inside.sum() < min_size:
            continue
        a, b = int((sig & inside).sum()), int((sig & ~inside).sum())
        c, d = int((~sig & inside).sum()), int((~sig & ~inside).sum())
        odds, p = ss.fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append({"klasa": k, "w_klasie": a + c, "istotne": a,
                     "oczekiwane": round((a + b) * (a + c) / max(1, len(cls)), 2), "odds_ratio": odds, "p": p})
    out = pd.DataFrame(rows, columns=["klasa", "w_klasie", "istotne", "oczekiwane", "odds_ratio", "p"])
    out["q"] = bh(out["p"]) if len(out) else []
    return out.sort_values(["p", "klasa"]).reset_index(drop=True)


def distributions(ann, up=(), down=()):
    """Counts of total chain length and of double bonds: all lipids, the significantly higher and lower ones."""
    a = ann[ann["lipid_class"].notna()]
    rows = []
    for col, label in (("carbons", "dlugosc_lancuchow"), ("db", "wiazania_podwojne")):
        for v in sorted(a[col].dropna().unique()):
            ids = a.index[a[col] == v]
            rows.append({"cecha": label, "wartosc": int(v), "wszystkie": len(ids),
                         "w_gore": int(ids.isin(list(up)).sum()), "w_dol": int(ids.isin(list(down)).sum())})
    return pd.DataFrame(rows, columns=["cecha", "wartosc", "wszystkie", "w_gore", "w_dol"])


# ---------------------------------------------------------------- masses and formulas

def kendrick(mass):
    """Kendrick mass on the CH2 base: (nominal Kendrick mass, Kendrick mass defect = nominal - exact). Members of
    a CH2 homologous series share the defect and differ by 14 in nominal mass."""
    km = np.asarray(mass, float) * 14.0 / CH2
    nominal = np.round(km)
    return nominal, nominal - km


def parse_formula(text):
    """'C41 H73 D7 N O8 P' or 'C6H12O6' -> {'C': 41, ...}; None when the text is not a molecular formula."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    s = re.sub(r"\s+", "", str(text))
    if not s or not re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", s):
        return None
    out = {}
    for el, n in FORMULA_RE.findall(s):
        if el not in ELEMENTS:
            return None
        out[el] = out.get(el, 0) + (int(n) if n else 1)
    if "C" not in out or not ({"H", "D"} & set(out)):
        return None
    return out


def van_krevelen(formulas):
    """H/C and O/C per feature with a readable formula (deuterium counts as hydrogen)."""
    rows = {}
    for i, f in formulas.items():
        p = parse_formula(f)
        if p:
            rows[i] = {"H_C": (p.get("H", 0) + p.get("D", 0)) / p["C"], "O_C": p.get("O", 0) / p["C"],
                       "N_C": p.get("N", 0) / p["C"]}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["H_C", "O_C", "N_C"])


def presence(X, groups, min_fraction=0.5):
    """Feature x group: detected (not NaN) in at least `min_fraction` of the group's injections."""
    g = groups.dropna().astype(str)
    return pd.DataFrame({k: X[list(idx)].notna().mean(axis=1) >= min_fraction
                         for k, idx in g.groupby(g).groups.items()})


def venn_counts(present):
    """Regions of a Venn diagram: tuple of the groups a feature is present in -> number of features."""
    cols = list(present.columns)
    out = {}
    for row in present.to_numpy(bool):
        key = tuple(c for c, v in zip(cols, row, strict=True) if v)
        if key:
            out[key] = out.get(key, 0) + 1
    return out
