"""Tests for the lipidomics mode of the omics recipe: Profinder / MZmine / multi-sheet metadata shapes (synthetic,
built here), paired tests and replicate averaging, lipid name parsing, class statistics and the lipid plots' numbers,
plus an end-to-end run the way the runner starts it."""
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")
pytest.importorskip("statsmodels")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")
pytest.importorskip("openpyxl")

from compute_bot.recipes import RECIPES  # noqa: E402
from compute_bot.recipes.omics import lipids, parse, prep, stats  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- paired design and replicates

def _paired_data(n_subj=10, n_feat=100, spiked=20, effect=0.6, seed=0):
    rng = np.random.default_rng(seed)
    subj = [f"S{i:02d}" for i in range(n_subj)]
    cols, grp, pair = [], {}, {}
    data = {}
    base = rng.normal(20, 1, n_feat)
    person = {s: rng.normal(0, 2.0, n_feat) for s in subj}        # large between-subject spread
    eff = np.zeros(n_feat)
    eff[:spiked] = effect
    for g in ("T0", "T1"):
        for s in subj:
            c = f"{s}_{g}"
            cols.append(c)
            grp[c], pair[c] = g, s
            data[c] = base + person[s] + (eff if g == "T1" else 0) + rng.normal(0, 0.1, n_feat)
    L = pd.DataFrame(data, index=[f"F{i:03d}" for i in range(n_feat)])[cols]
    return L, pd.Series(grp), pd.Series(pair), list(L.index[:spiked])


def test_paired_tests_find_what_unpaired_misses():
    L, groups, pairs, spiked = _paired_data()
    unpaired = stats.feature_tests(L, groups, ref="T0", test="welch")
    assert (unpaired.loc[spiked, "q"] < 0.05).sum() <= 1
    for test in ("paired_t", "wilcoxon"):
        res = stats.feature_tests(L, groups, ref="T0", test=test, pairs=pairs)
        assert res.attrs["test"] == test and res.attrs["n_pairs"] == 10 and res.attrs["dropped_pairs"] == 0
        assert (res.loc[spiked, "q"] < 0.05).mean() >= 0.9, test
        assert (res.drop(index=spiked)["q"] < 0.05).sum() <= 3
        assert np.allclose(res.loc[spiked, "log2FC"], 0.6, atol=0.15)
    auto = stats.feature_tests(L, groups, ref="T0", test="auto", pairs=pairs)
    assert auto.attrs["test"] == "paired_t"


def test_paired_test_drops_incomplete_pairs_and_matches_scipy():
    from scipy import stats as ss
    L, groups, pairs, _ = _paired_data(n_subj=8)
    gone = "S03_T1"
    L2, g2, p2 = L.drop(columns=gone), groups.drop(gone), pairs.drop(gone)
    res = stats.feature_tests(L2, g2, ref="T0", test="paired_t", pairs=p2)
    assert res.attrs["n_pairs"] == 7 and res.attrs["dropped_pairs"] == 1
    keep = [s for s in sorted(set(pairs)) if s != "S03"]
    a = L.loc["F000", [f"{s}_T0" for s in keep]].to_numpy()
    b = L.loc["F000", [f"{s}_T1" for s in keep]].to_numpy()
    assert res.loc["F000", "p"] == pytest.approx(ss.ttest_rel(b, a).pvalue)
    w = stats.feature_tests(L2, g2, ref="T0", test="wilcoxon", pairs=p2)
    assert w.loc["F000", "p"] == pytest.approx(ss.wilcoxon(b - a).pvalue)


def test_replicate_averaging_halves_n_and_removes_pseudoreplication():
    rng = np.random.default_rng(1)
    n_feat, n_subj = 2000, 6
    cols, grp, subj = [], {}, {}
    data = {}
    for g in ("A", "B"):
        for i in range(n_subj):
            bio = rng.normal(0, 1.0, n_feat)                       # no group effect at all
            for rep in ("1", "2"):
                c = f"{g}{i}_r{rep}"
                cols.append(c)
                grp[c], subj[c] = g, f"{g}{i}"
                data[c] = 20 + bio + rng.normal(0, 0.05, n_feat)
    L = pd.DataFrame(data)[cols]
    samples = pd.DataFrame({"_group": pd.Series(grp), "_subject": pd.Series(subj), "_type": "sample"})
    units = prep.replicate_units(samples, cols)
    assert units.nunique() == 12 and units["A0_r1"] == units["A0_r2"] != units["A1_r1"]
    Lu, gu, info = stats.collapse_replicates(L, units, pd.Series(grp))
    assert Lu.shape == (n_feat, 12) and gu.value_counts().to_dict() == {"A": 6, "B": 6}
    assert info["n_injections"] == 24 and info["n_units"] == 12
    first = units[units == units["A0_r1"]].index
    assert np.allclose(Lu[units["A0_r1"]], L[first].mean(axis=1))
    naive = stats.feature_tests(L, pd.Series(grp), ref="A", test="ttest")
    honest = stats.feature_tests(Lu, gu, ref="A", test="ttest")
    assert (naive["p"] < 0.05).mean() > 0.12          # replicates counted as subjects: inflated false positives
    assert (honest["p"] < 0.05).mean() < 0.08


def test_whole_sample_outlier_is_not_a_bad_injection():
    names = [f"S{i}_{g}_{r}" for i in range(6) for g in "AB" for r in "12"]
    samples = pd.DataFrame({"_type": "sample", "_group": [n.split("_")[1] for n in names],
                            "_subject": [n.split("_")[0] for n in names]}, index=names)
    flags = pd.DataFrame({"rejected": False, "reason": ""}, index=names)
    flags.loc[["S1_B_1", "S1_B_2"], ["rejected", "reason"]] = [True, "braki 19% (z=5.2)"]     # both replicates
    flags.loc["S2_A_1", ["rejected", "reason"]] = [True, "braki 30% (z=9.0)"]                 # one of two
    flags.loc[["S3_A_1", "S3_A_2"], ["rejected", "reason"]] = [True, "IS z=-4.0"]             # standards say so
    spared = prep.spare_whole_samples(samples, flags)
    assert spared == ["S1 B"]
    assert not flags.loc[["S1_B_1", "S1_B_2"], "rejected"].any()
    assert flags.loc[["S2_A_1", "S3_A_1", "S3_A_2"], "rejected"].all()


def test_replicate_units_from_names_without_subjects():
    names = ["K_1A", "K_1B", "K_2A", "K_2B", "L_1A", "L_1B", "QC_1"]
    samples = pd.DataFrame({"_group": ["K"] * 4 + ["L"] * 2 + [None], "_subject": None,
                            "_type": ["sample"] * 6 + ["qc"]}, index=names)
    units = prep.replicate_units(samples, names[:6])
    assert units["K_1A"] == units["K_1B"] != units["K_2A"] and units.nunique() == 3
    plain = pd.DataFrame({"_group": ["P1"] * 3, "_subject": None, "_type": "sample"}, index=["P1_1", "P1_2", "P1_3"])
    assert prep.replicate_units(plain, list(plain.index)).nunique() == 3     # _1/_2 alone are samples


# ---------------------------------------------------------------- lipid names

LIPID_NAMES = [
    # name, class, carbons, double bonds, chains, hydroxylation, ether, lyso
    ("PC 34:1", "PC", 34, 1, None, None, None, False),
    ("PC(16:0/18:1)", "PC", 34, 1, [(16, 0), (18, 1)], None, None, False),
    ("PC 16:0_18:1", "PC", 34, 1, [(16, 0), (18, 1)], None, None, False),
    ("TG 52:2", "TG", 52, 2, None, None, None, False),
    ("TG(16:0/18:1/18:1)", "TG", 52, 2, [(16, 0), (18, 1), (18, 1)], None, None, False),
    ("TAG 54:3", "TG", 54, 3, None, None, None, False),
    ("SM d18:1/18:1", "SM", 36, 2, [(18, 1), (18, 1)], "d", None, False),
    ("SMd34:1", "SM", 34, 1, None, "d", None, False),
    ("SMt42:1", "SM", 42, 1, None, "t", None, False),
    ("SM 34:1;O2", "SM", 34, 1, None, "d", None, False),
    ("Cer d18:1/24:0", "Cer", 42, 1, [(18, 1), (24, 0)], "d", None, False),
    ("Cer(t18:0/24:0)", "Cer", 42, 0, [(18, 0), (24, 0)], "t", None, False),
    ("HexCer d18:1/16:0", "HexCer", 34, 1, [(18, 1), (16, 0)], "d", None, False),
    ("LPC 18:0", "LPC", 18, 0, [(18, 0)], None, None, True),
    ("Lyso-PC18:1", "LPC", 18, 1, [(18, 1)], None, None, True),
    ("LysoPC16:0 izo 1", "LPC", 16, 0, [(16, 0)], None, None, True),
    ("LYsoPE18:2", "LPE", 18, 2, [(18, 2)], None, None, True),
    ("PE O-38:5", "PE", 38, 5, None, None, "O", False),
    ("PEO-38:5", "PE", 38, 5, None, None, "O", False),
    ("PC P-36:4", "PC", 36, 4, None, None, "P", False),
    ("PC(O-16:0/20:4)", "PC", 36, 4, [(16, 0), (20, 4)], None, "O", False),
    ("TGO-50:1", "TG", 50, 1, None, None, "O", False),
    ("15:0-18:1 PC", "PC", 33, 1, [(15, 0), (18, 1)], None, None, False),
    ("DG 36:2", "DG", 36, 2, None, None, None, False),
    ("DG32:0", "DG", 32, 0, None, None, None, False),
    ("CE 18:2", "CE", 18, 2, [(18, 2)], None, None, False),
    ("FA 20:4", "FA", 20, 4, [(20, 4)], None, None, False),
    ("MG 16:0", "MG", 16, 0, [(16, 0)], None, None, False),
    ("TG48:4  ISO1", "TG", 48, 4, None, None, None, False),
    ("TG54:7_2", "TG", 54, 7, None, None, None, False),
    ("SMd33:1/SMt32:2", "SM", 33, 1, None, "d", None, False),
    ("PI 38:4 [M+NH4]+", "PI", 38, 4, None, None, None, False),
    ("PA46:3", "PA", 46, 3, None, None, None, False),
]
NOT_LIPIDS = ["Glucose", "Caffeine", "L-Alanine", "Unknown", "337.3360@3.75", "ATP", "Vitamin D3", "", None,
              "Peptide 12:34 fragment", "C18H36O2"]


@pytest.mark.parametrize("row", LIPID_NAMES, ids=[r[0] for r in LIPID_NAMES])
def test_lipid_name_parser(row):
    name, cls, c, db, chains, hydrox, ether, lyso = row
    got = lipids.parse_lipid(name)
    assert got is not None, name
    assert (got["cls"], got["carbons"], got["db"]) == (cls, c, db)
    assert got["chains"] == chains
    assert got["hydrox"] == hydrox and got["ether"] == ether and got["lyso"] == lyso


def test_lipid_parser_rejects_non_lipids_and_marks_labelled():
    for n in NOT_LIPIDS:
        assert lipids.parse_lipid(n) is None, n
    got = lipids.parse_lipid("15:0-18:1(d7) PC")
    assert got["cls"] == "PC" and got["carbons"] == 33 and got["labelled"]
    assert lipids.parse_lipid("18:1(d7) Lyso PC")["cls"] == "LPC"
    assert lipids.saturation(0) == "SFA" and lipids.saturation(1) == "MUFA" and lipids.saturation(4) == "PUFA"


def test_internal_standards_recognised_by_name_and_formula():
    yes = ["15:0-18:1(d7) PC", "18:1 SM (d18:1/18:1)-d9", "18:1(d7) Lyso PC", "Ceramide-d7", "DG d7", "SM d9",
           "d18:1-18:1(d9) SM", "PC Lyso d7 izo 1", "Caffeine-13C3 ISTD", "IS-PC"]
    no = ["SMd34:1", "SM d18:1/18:1", "Cer d18:1/24:0", "PC 34:1", "TG48:4  ISO1", "Vitamin D3", "DG32:0"]
    for n in yes:
        assert parse.is_internal_standard(n), n
    for n in no:
        assert not parse.is_internal_standard(n), n
    assert parse.is_internal_standard("lipid X", "C41 H73 D7 N O8 P")
    assert not parse.is_internal_standard("lipid X", "C42 H82 N O8 P")


def test_auto_lipid_mode_needs_a_third_of_named_features():
    names = pd.Series(["PC 34:1", "TG 52:2", "SM d18:1/16:0", "Glucose", "Citrate", None, None, "LPC 18:0"])
    ann = lipids.annotate(names)
    assert ann["lipid_class"].notna().sum() == 4
    assert lipids.lipid_mode("auto", names, ann) and not lipids.lipid_mode("nie", names, ann)
    few = pd.Series(["PC 34:1"] + [f"Metabolite {i}" for i in range(9)])
    assert not lipids.lipid_mode("auto", few, lipids.annotate(few))
    assert lipids.lipid_mode("tak", few, lipids.annotate(few))


# ---------------------------------------------------------------- KMD, Van Krevelen, Venn, classes

def test_kendrick_mass_defect_constant_in_ch2_series():
    ch2 = 14.01565006
    series = 733.5622 - 3 * ch2 + ch2 * np.arange(8)              # PC 30:0 ... PC 37:0
    nominal, kmd = lipids.kendrick(series)
    assert np.ptp(kmd) < 1e-6
    assert list(np.diff(nominal)) == [14] * 7
    other = lipids.kendrick(np.array([733.5622 - 2.01565]))[1]     # one more double bond: another KMD line
    assert abs(other[0] - kmd[0]) > 0.01


def test_van_krevelen_ratios_from_formulas():
    assert lipids.parse_formula("C6H12O6") == {"C": 6, "H": 12, "O": 6}
    f = lipids.parse_formula("C41 H73 D7 N O8 P")
    assert f == {"C": 41, "H": 73, "D": 7, "N": 1, "O": 8, "P": 1}
    vk = lipids.van_krevelen(pd.Series({"a": "C6H12O6", "b": "C41 H73 D7 N O8 P", "c": None, "d": "not a formula"}))
    assert vk.loc["a", "H_C"] == pytest.approx(2.0) and vk.loc["a", "O_C"] == pytest.approx(1.0)
    assert vk.loc["b", "H_C"] == pytest.approx(80 / 41) and vk.loc["b", "O_C"] == pytest.approx(8 / 41)
    assert set(vk.index) == {"a", "b"}
    assert lipids.parse_formula("PC 34:1") is None


def test_venn_counts():
    present = pd.DataFrame({"M1": [1, 1, 0, 1, 0], "M12": [1, 0, 1, 1, 0]}, dtype=bool)
    assert lipids.venn_counts(present) == {("M1",): 1, ("M12",): 1, ("M1", "M12"): 2}
    three = pd.DataFrame({"A": [1, 1, 1, 0], "B": [1, 1, 0, 0], "C": [1, 0, 0, 1]}, dtype=bool)
    v = lipids.venn_counts(three)
    assert v[("A", "B", "C")] == 1 and v[("A", "B")] == 1 and v[("A",)] == 1 and v[("C",)] == 1
    assert v.get(("B", "C"), 0) == 0


def _lipid_set():
    names = ([f"TG {c}:{d}" for c in range(40, 60) for d in range(5)]
             + [f"PC {c}:{d}" for c in range(30, 40) for d in range(5)]
             + [f"PE {c}:{d}" for c in range(30, 40) for d in range(5)])
    return pd.Series(names, index=[f"L{i:03d}" for i in range(len(names))])


def test_class_enrichment_finds_enriched_class():
    names = _lipid_set()
    ann = lipids.annotate(names)
    pcs = list(ann.index[ann.lipid_class == "PC"])
    tgs = list(ann.index[ann.lipid_class == "TG"])
    enr = lipids.class_enrichment(ann, pcs[:30] + tgs[:3]).set_index("klasa")
    assert enr.loc["PC", "q"] < 0.01 and enr.loc["PC", "istotne"] == 30 and enr.loc["PC", "w_klasie"] == 50
    assert enr.loc["TG", "q"] > 0.5 and enr.loc["PE", "q"] > 0.5
    assert enr["q"].idxmin() == "PC"


def test_class_sums_shares_and_saturation_profile():
    names = pd.Series({"a": "PC 34:1", "b": "PC 36:0", "c": "TG 52:2", "d": "Glucose", "e": "TG 48:0"})
    ann = lipids.annotate(names)
    lin = pd.DataFrame({"s1": [10.0, 30.0, 50.0, 999.0, 10.0], "s2": [20.0, 20.0, 40.0, 5.0, 20.0]}, index=names.index)
    sums, shares = lipids.class_sums(lin, ann["lipid_class"])
    assert sums.loc["PC", "s1"] == 40 and sums.loc["TG", "s2"] == 60       # the non-lipid is not in any class
    assert np.allclose(shares.sum(), 1.0) and shares.loc["PC", "s1"] == pytest.approx(0.4)
    _, sat = lipids.class_sums(lin, ann["saturation"])
    assert sat.loc["SFA", "s1"] == pytest.approx(0.4) and sat.loc["MUFA", "s1"] == pytest.approx(0.1)
    assert sat.loc["PUFA", "s1"] == pytest.approx(0.5)
    dist = lipids.distributions(ann, up=["a", "c"], down=["e"])
    row = dist[(dist.cecha == "wiazania_podwojne") & (dist.wartosc == 0)].iloc[0]
    assert row["wszystkie"] == 2 and row["w_dol"] == 1 and row["w_gore"] == 0


# ---------------------------------------------------------------- input shapes (synthetic, Profinder / MZmine)

DONORS = ["DA", "DB", "DC", "DD", "DE", "DF", "DG", "DH"]
IS_NAMES = ["15:0-18:1(d7) PC", "18:1 SM (d18:1/18:1)-d9", "TG d7"]
FORMULA = {"PC": "C42 H82 N O8 P", "TG": "C55 H102 O6", "SM": "C39 H79 N2 O6 P", "PE": "C39 H76 N O8 P",
           "DG": "C39 H72 O5", "LysoPC": "C26 H52 N O7 P"}


def _design(extra_donor=True):
    """Injections of a paired longitudinal study: donors x (M1, M12) x extraction replicates A/B, QCs, one blank."""
    rows, run = [], 100
    for d in DONORS + (["DX"] if extra_donor else []):
        for tp in (("12",) if d == "DX" else ("1", "12")):
            for ext in "AB":
                run += 1
                rows.append({"col": f"{d} {tp}_1{ext} {run}", "file": f"{d}_{tp}_1{ext}_{run}.mzML", "type": "Sample",
                             "donor": d, "tp": f"M{tp}", "ext": ext})
    rng = np.random.default_rng(5)
    rows = [rows[i] for i in rng.permutation(len(rows))]
    half = len(rows) // 2
    seq = []
    for day, part in (("day1", rows[:half]), ("day2", rows[half:])):
        q = 0
        for k, r in enumerate(part):
            if k % 6 == 0:
                q += 1
                seq.append({"col": f"QC_{day} {q}", "file": f"QC_{day}_{q}.mzML", "type": "QC", "donor": "POOL",
                            "tp": "QC", "ext": None, "day": day})
            seq.append(dict(r, day=day))
        seq.append({"col": f"QC_{day} 9", "file": f"QC_{day}_9.mzML", "type": "QC", "donor": "POOL", "tp": "QC",
                    "ext": None, "day": day})
    seq.append({"col": "Blank extraction _day2", "file": "Blank_extraction_day2.mzML", "type": "Blank", "donor": None,
                "tp": "Blank", "ext": None, "day": "day2"})
    d = pd.DataFrame(seq)
    d["order"] = np.arange(1, len(d) + 1)
    return d


def _lipid_names():
    names = ([f"PC{c}:{d}" for c in (32, 34, 36, 38) for d in (0, 1, 2, 4)]
             + [f"TG{c}:{d}" for c in (46, 48, 50, 52, 54, 56) for d in (0, 1, 2, 3)]
             + [f"SMd{c}:{d}" for c in (34, 36, 40, 42) for d in (1, 2)]
             + [f"PE{c}:{d}" for c in (34, 36, 38) for d in (1, 2)]
             + [f"DG{c}:{d}" for c in (32, 34, 36) for d in (0, 1)]
             + ["Lyso-PC16:0", "Lyso-PC18:1", "LysoPC18:0 izo 1", "PEO-38:5"])
    return names


def _intensities(design, names, effect=1.0, seed=0):
    """Linear areas, lipids x injections: PCs are `effect` log2 higher at M12; donors differ a lot from each other."""
    rng = np.random.default_rng(seed)
    n = len(names)
    base = rng.normal(22, 1.2, n)
    person = {d: rng.normal(0, 1.2, n) for d in design.donor.dropna().unique()}
    eff = np.array([effect if x.startswith("PC") else 0.0 for x in names])
    cols = {}
    for r in design.itertuples():
        if r.type == "Blank":
            v = np.full(n, np.nan)
            v[:2] = base[:2] - 6
        elif r.type == "QC":
            v = base + eff / 2 + rng.normal(0, 0.05, n)
        else:
            v = base + person[r.donor] + (eff if r.tp == "M12" else 0) + rng.normal(0, 0.08, n)
        cols[r.col] = 2.0 ** v
    return pd.DataFrame(cols, index=names)


def _profinder(design, X, targeted=True, block="Area"):
    """A Profinder-style table: summary columns, then [Mass] / [RT] / [<block>] / [Score] columns per sample."""
    n = len(X)
    rng = np.random.default_rng(2)
    mass = np.round(np.linspace(450, 900, n), 4)
    rt = np.round(np.linspace(2, 30, n), 2)
    df = pd.DataFrame()
    if targeted:
        df["Name"] = X.index
        df["Formula"] = [next((f for k, f in FORMULA.items() if x.replace("-", "").startswith(k)),
                              "C41 H73 D7 N O8 P") for x in X.index]
    df["Mass (avg)"] = mass
    df["RSD (Mass, ppm)"] = 2.1
    df["RT (avg)"] = rt
    df["Height (avg)"] = np.round(X.mean(axis=1).to_numpy() / 10)
    df["Found"] = X.notna().sum(axis=1).to_numpy()
    df["Missed"] = X.isna().sum(axis=1).to_numpy()
    df["%RSD (Tgt)"] = 9.5
    blocks = {}
    for c in X.columns:
        blocks[f"[Mass] {c}"] = np.round(mass + rng.normal(0, 0.001, n), 4)
    for c in X.columns:
        blocks[f"[RT] {c}"] = np.round(rt + rng.normal(0, 0.02, n), 3)
    for c in X.columns:
        blocks[f"[{block}] {c}"] = np.round(X[c].to_numpy())
    for c in X.columns:
        blocks[f"[Score (Tgt)] {c}"] = 99.0
    df = pd.concat([df, pd.DataFrame(blocks)], axis=1)
    df["Column1"] = np.nan
    return df


def _write_targeted_xlsx(path, df):
    """Header, an empty second row, then the data (as Profinder exports it), plus an empty second sheet."""
    with pd.ExcelWriter(path) as xl:
        pd.DataFrame([list(df.columns), [None] * df.shape[1]] + df.values.tolist()).to_excel(
            xl, sheet_name="targeted table", header=False, index=False)
        pd.DataFrame().to_excel(xl, sheet_name="Arkusz1", header=False, index=False)


def _write_metadata_xlsx(path, design, drop_files=()):
    """README + Donor / Timepoint / Sample / Sequence sheets, each with a title row above the header."""
    d = design[~design.col.isin(drop_files)]
    st = d[d.type == "Sample"]
    sample = pd.DataFrame({"Sample_ID": [f.replace(".mzML", "") for f in d.file], "File_name": d.file,
                           "Sample_type": d.type, "Donor_ID": d.donor, "Timepoint": d.tp, "ExtractionRep_ID": d.ext,
                           "Analytical_batch": ["Batch_" + x[-1] for x in d.day], "Measurement_day": d.day})
    seq = pd.DataFrame({"Injection_order": d.order, "File_name": d.file, "Sample_type": d.type,
                        "Measurement_day": d.day})
    donors = sorted(st.donor.unique())
    donor = pd.DataFrame({"Donor_ID": donors, "Maternal_age": [25 + i for i in range(len(donors))],
                          "BMI_category": ["normal" if i % 2 == 0 else "overweight" for i in range(len(donors))]})
    tp = st[["donor", "tp"]].drop_duplicates().rename(columns={"donor": "Donor_ID", "tp": "Timepoint"})
    tp["Diet_pattern"] = ["omnivore" if i % 3 else "vegetarian" for i in range(len(tp))]
    readme = pd.DataFrame([["Zestaw danych – test", None], [None, None], ["Design", "paired"], ["QC", "pooled"]])

    def titled(df, title):
        return pd.DataFrame([[title] + [None] * (df.shape[1] - 1), list(df.columns)] + df.values.tolist())

    with pd.ExcelWriter(path) as xl:
        readme.to_excel(xl, sheet_name="README", header=False, index=False)
        titled(donor, "Donor_metadata — cechy stałe").to_excel(xl, sheet_name="Donor_metadata", header=False,
                                                              index=False)
        titled(tp, "Timepoint_metadata — cechy zmienne").to_excel(xl, sheet_name="Timepoint_metadata", header=False,
                                                                 index=False)
        titled(sample, "Sample_metadata — 1 wiersz = 1 plik").to_excel(xl, sheet_name="Sample_metadata",
                                                                      header=False, index=False)
        titled(seq, "Sequence, kolejność").to_excel(xl, sheet_name="Sequence", header=False, index=False)


def _targeted_dataset(tmp_path, drop_files=(), xlsx=True):
    design = _design()
    names = _lipid_names()
    X = _intensities(design, names + IS_NAMES)
    X.loc[IS_NAMES] = 2.0 ** np.random.default_rng(3).normal(21, 0.05, (len(IS_NAMES), X.shape[1]))
    df = _profinder(design, X)
    if xlsx:
        _write_targeted_xlsx(tmp_path / "targeted data.xlsx", df)
    else:
        df.to_csv(tmp_path / "targeted data.csv", index=False)
    _write_metadata_xlsx(tmp_path / "metadata_M1_M12.xlsx", design, drop_files=drop_files)
    return design, names


def test_parse_profinder_targeted_xlsx_with_multisheet_metadata(tmp_path):
    design, names = _targeted_dataset(tmp_path)
    [ds] = parse.load_datasets(tmp_path, group_col="Timepoint", layer="auto", pair_col="Donor_ID")
    assert ds.layer == "metabolomics" and "Profinder" in ds.source and "Area" in ds.source
    assert ds.X.shape == (len(names), len(design))
    assert ds.istd is not None and set(ds.istd.index) == set(IS_NAMES)       # deuterated standards set aside
    assert not set(IS_NAMES) & set(ds.X.index)
    assert ds.features.loc["PC34:1", "formula"] == "C42 H82 N O8 P"
    assert ds.features["mz"].notna().all() and ds.features["rt"].notna().all()
    s = ds.samples
    assert s._type.value_counts().to_dict() == design.type.str.lower().value_counts().to_dict()
    col = design.col.iloc[5]
    one = design.iloc[5]
    assert s.loc[col, "_order"] == one.order and s.loc[col, "_batch"] == "Batch_" + one.day[-1]
    st = design[design.type == "Sample"].set_index("col")
    assert (s.loc[st.index, "_group"] == st.tp).all() and (s.loc[st.index, "_pair"] == st.donor).all()
    assert "Maternal_age" in s.columns and "Diet_pattern" in s.columns        # donor and donor+timepoint sheets
    assert s.loc[st.index, "Diet_pattern"].notna().all()
    joined = " ".join(ds.notes)
    assert "Sequence" in joined and "Donor_metadata" in joined and "Timepoint_metadata" in joined
    assert not ds.inferred["group"] and not ds.inferred["order"]


def test_metadata_gaps_filled_from_sample_names(tmp_path):
    design = _design()
    st = design[design.type == "Sample"]
    missing = [st.col.iloc[0], st.col.iloc[7]]
    _targeted_dataset(tmp_path, drop_files=missing, xlsx=False)
    [ds] = parse.load_datasets(tmp_path, group_col="Timepoint", layer="auto", pair_col="Donor_ID")
    s = ds.samples
    for c in missing:
        r = design.set_index("col").loc[c]
        assert s.loc[c, "_group"] == r.tp and s.loc[c, "_pair"] == r.donor and s.loc[c, "_type"] == "sample"
    assert ds.inferred["from_names"] == 2
    assert any("z nazw" in n for n in ds.notes + ds.warnings)


def test_profinder_duplicate_names_get_unique_ids(tmp_path):
    design = _design(extra_donor=False)
    names = ["TG42:3", "TG42:3", "PC34:1", "TG48:4  ISO1", "TG48:4  ISO2", "SMd34:1"]
    _profinder(design, _intensities(design, names)).to_csv(tmp_path / "targeted.csv", index=False)
    [ds] = parse.load_datasets(tmp_path, group_col="group", layer="auto")
    assert list(ds.X.index[:2]) == ["TG42:3", "TG42:3_2"]
    assert list(ds.features["name"][:2]) == ["TG42:3", "TG42:3"] and any("powtórzonych" in n for n in ds.notes)


def test_parse_profinder_untargeted_csv(tmp_path):
    design = _design(extra_donor=False)
    X = _intensities(design, [f"x{i}" for i in range(40)])
    df = _profinder(design, X, targeted=False, block="Vol")
    df.to_csv(tmp_path / "nontargeted table.csv", index=False)
    [ds] = parse.load_datasets(tmp_path, group_col="group", layer="auto")
    assert "Profinder" in ds.source and "Vol" in ds.source
    assert ds.X.shape == (40, len(design))
    assert list(ds.X.columns)[:2] == [" ".join(c.split()) for c in design.col[:2]]
    first = ds.X.index[0]
    assert first == f"{df['Mass (avg)'][0]:.4f}@{df['RT (avg)'][0]:.2f}"
    assert ds.features.loc[first, "mz"] == pytest.approx(df["Mass (avg)"][0])
    assert ds.features.loc[first, "rt"] == pytest.approx(df["RT (avg)"][0])
    assert ds.X.iloc[3, 4] == pytest.approx(round(X.iloc[3, 4]))             # the intensity block, not [Mass]
    s = ds.samples
    assert (s._type == "qc").sum() == (design.type == "QC").sum() and (s._type == "blank").sum() == 1
    # no sheet: donor, timepoint and extraction replicate come from names like "DA 12_1B 117"
    st = design[design.type == "Sample"].set_index("col")
    assert set(s.loc[s._type == "sample", "_group"]) == {"1", "12"}
    assert (s.loc[[" ".join(c.split()) for c in st.index], "_subject"].to_numpy() == st.donor.to_numpy()).all()
    assert ds.inferred["group"]


def _mzmine_tables():
    files = ["ctrl_1.mzML", "ctrl_2.mzML", "case_1.mzML", "case_2.mzML", "QC_1.mzML"]
    rng = np.random.default_rng(0)
    n = 6
    area = rng.uniform(1e5, 1e6, (n, len(files))).round()
    area[2, 1] = 0
    legacy = pd.DataFrame({"row ID": range(1, n + 1), "row m/z": np.linspace(500, 800, n).round(4),
                           "row retention time": np.linspace(1, 9, n).round(3),
                           "row identity (main ID)": ["PC 34:1", None, "TG 52:2", None, None, "SM d18:1/16:0"]})
    for j, f in enumerate(files):
        legacy[f"{f} Peak m/z"] = legacy["row m/z"]
        legacy[f"{f} Peak height"] = (area[:, j] / 7).round()
        legacy[f"{f} Peak area"] = area[:, j]
    legacy["Unnamed: 99"] = np.nan
    new = pd.DataFrame({"id": range(1, n + 1), "rt": legacy["row retention time"], "mz": legacy["row m/z"],
                        "height": 1.0, "area": 2.0, "charge": 1,
                        "compound_db_identity:compound_name": legacy["row identity (main ID)"]})
    for j, f in enumerate(files):
        new[f"datafile:{f}:mz"] = new["mz"]
        new[f"datafile:{f}:rt"] = new["rt"]
        new[f"datafile:{f}:height"] = (area[:, j] / 7).round()
        new[f"datafile:{f}:area"] = area[:, j]
    return legacy, new, area, files


@pytest.mark.parametrize("style", ["legacy", "modular"])
def test_parse_mzmine_feature_list(tmp_path, style):
    legacy, new, area, files = _mzmine_tables()
    (legacy if style == "legacy" else new).to_csv(tmp_path / "features_quant.csv", index=False)
    [ds] = parse.load_datasets(tmp_path, group_col="group", layer="auto")
    assert "MZmine" in ds.source and ds.layer == "metabolomics"
    assert list(ds.X.columns) == ["ctrl_1", "ctrl_2", "case_1", "case_2", "QC_1"]
    assert ds.X.shape == (6, 5) and ds.X.loc["1", "case_1"] == area[0, 2]
    assert np.isnan(ds.X.loc["3", "ctrl_2"])                                  # zero area = not detected
    assert ds.features.loc["2", "mz"] == pytest.approx(legacy["row m/z"][1])
    assert ds.features.loc["2", "rt"] == pytest.approx(legacy["row retention time"][1])
    assert ds.features.loc["1", "name"] == "PC 34:1" and pd.isna(ds.features.loc["2", "name"])
    assert ds.samples.loc["QC_1", "_type"] == "qc" and ds.samples.loc["case_2", "_group"] == "case"


# ---------------------------------------------------------------- end to end, like the runner

def run_recipe(work, **given):
    params = RECIPES["omics"].validate({k: str(v) for k, v in given.items()})
    (work / "params.json").write_text(json.dumps(params))
    env = dict(os.environ, PYTHONPATH=str(REPO), COMPUTE_THREADS="2", COMPUTE_BOT_OFFLINE="1", MPLBACKEND="Agg")
    return subprocess.run([sys.executable, "-m", "compute_bot.recipes.omics"], cwd=work, env=env,
                          capture_output=True, text=True, timeout=600)


def test_end_to_end_paired_lipidomics_zip(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    design, names = _targeted_dataset(src)
    with zipfile.ZipFile(tmp_path / "lipidy.zip", "w") as z:
        for p in src.iterdir():
            z.write(p, p.name)
    for p in src.iterdir():
        p.unlink()
    src.rmdir()
    # a quarter of the lipids (all PCs) goes up: median normalisation would push the rest down, the spiked
    # standards do not move
    r = run_recipe(tmp_path, group="Timepoint", pair="Donor_ID", blank_ratio=10, normalize="is")
    assert r.returncode == 0, r.stderr[-3000:]
    summary = (tmp_path / "summary.txt").read_text()
    assert len(summary) <= 2500
    assert "paired_t" in summary and "8 par" in summary and "Donor_ID" in summary
    assert "M12" in summary and "M1" in summary
    assert "Lipid" in summary and "PC" in summary
    w = tmp_path / "wyniki"
    for f in ["statystyka.csv", "kroki.txt", "lipidy_adnotacja.csv", "lipidy_klasy.csv", "lipidy_nasycenie.csv",
              "lipidy_wzbogacenie_klas.csv", "lipidy_istotne_rozklad.csv", "wzorce_is.csv", "venn.csv",
              "lipidy_klasy.png", "rt_mz.png", "kmd.png", "van_krevelen.png", "venn.png", "is_dryf.png",
              "volcano.png"]:
        assert (w / f).exists(), f
    res = pd.read_csv(w / "statystyka.csv", index_col=0)
    pcs = [n for n in names if n.startswith("PC")]
    assert not set(IS_NAMES) & set(res.index)
    assert (res.loc[[p for p in pcs if p in res.index], "q"] < 0.05).mean() >= 0.8
    assert (res.drop(index=[p for p in pcs if p in res.index])["q"] < 0.05).sum() <= 3
    assert (res.loc[[p for p in pcs if p in res.index], "log2FC"] > 0.5).all()
    assert res["n_M1"].max() == 8 and res["n_M12"].max() == 8             # donors, not injections
    enr = pd.read_csv(w / "lipidy_wzbogacenie_klas.csv")
    assert enr.sort_values("q").iloc[0]["klasa"] == "PC"
    cls = pd.read_csv(w / "lipidy_klasy.csv").set_index("klasa")
    assert cls.loc["PC", "log2FC"] > 0.5 and cls.loc["PC", "q"] < 0.05 and cls.loc["TG", "q"] > 0.05
    assert cls.loc["PC", "udzial_M12"] > cls.loc["PC", "udzial_M1"]
    steps = (w / "kroki.txt").read_text()
    assert "wzorce wewnętrzne" in steps and "replikat" in steps and "Timepoint_metadata" in steps
    assert "bez pary" in steps and "DX" in steps                             # the donor with one timepoint only
    hl = (tmp_path / "highlights.txt").read_text().split()
    assert 1 <= len(hl) <= 4 and all((tmp_path / p).exists() for p in hl)
    assert any("lipidy_klasy" in p for p in hl)
    first = (w / "statystyka.csv").read_text()
    r2 = run_recipe(tmp_path, group="Timepoint", pair="Donor_ID", blank_ratio=10, normalize="is")
    assert r2.returncode == 0 and (w / "statystyka.csv").read_text() == first


def test_end_to_end_replicates_kept_warns_about_pseudoreplication(tmp_path):
    _targeted_dataset(tmp_path, xlsx=False)
    r = run_recipe(tmp_path, group="Timepoint", replicates="keep", lipids="nie")
    assert r.returncode == 0, r.stderr[-3000:]
    summary = (tmp_path / "summary.txt").read_text()
    assert "pseudoreplikac" in summary.lower()
    assert not (tmp_path / "wyniki" / "lipidy_klasy.csv").exists()
    res = pd.read_csv(tmp_path / "wyniki" / "statystyka.csv", index_col=0)
    assert res["n_M1"].max() == 16


def test_end_to_end_unknown_pair_column(tmp_path):
    _targeted_dataset(tmp_path, xlsx=False)
    r = run_recipe(tmp_path, group="Timepoint", pair="Nie_ma")
    assert r.returncode != 0
    last = r.stderr.strip().splitlines()[-1]
    assert last.startswith("ERROR:") and "Donor_ID" in last
