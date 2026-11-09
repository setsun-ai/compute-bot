"""Tests for the omics recipe: pure steps on synthetic data with a known truth, parsers for every input shape
(built here, never real course data), mocked network, and an end-to-end run the way the runner starts it."""
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

from compute_bot.recipes import RECIPES  # noqa: E402
from compute_bot.recipes.omics import annotate, parse, prep, stats  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- synthetic data with a known truth

def synth(n_feat=200, n_per_group=12, groups=("CONTROL", "CASE"), n_batches=2, qc_every=3, n_blanks=2,
          spiked=20, fc=1.5, blank_only=10, drift=0.0, batch_shift=0.0, mcar=0.03, seed=0):
    """Linear intensities (features x injections), injection table and the truth.

    Injections per batch: blanks, QC, then samples with a pooled QC after every `qc_every` samples and at the end.
    Features 0..spiked-1 are higher by `fc` (log2) in the last group; the last `blank_only` features are background:
    as high in blanks as in samples.
    """
    rng = np.random.default_rng(seed)
    n_s = n_per_group * len(groups)
    subj = [f"S{i + 1:03d}" for i in range(n_s)]
    grp = [groups[i % len(groups)] for i in range(n_s)]
    order = rng.permutation(n_s)
    per_batch = np.array_split(order, n_batches)
    rows = []
    for b, idx in enumerate(per_batch):
        seq = [("blank", None)] * n_blanks + [("qc", None)]
        for k, i in enumerate(idx):
            seq.append(("sample", i))
            if (k + 1) % qc_every == 0:
                seq.append(("qc", None))
        if seq[-1][0] != "qc":
            seq.append(("qc", None))
        for w, (t, i) in enumerate(seq):
            rows.append({"batch": f"B{b + 1}", "run_order_within_batch": w + 1, "type": t,
                         "subject_id": subj[i] if i is not None else None,
                         "group": grp[i] if i is not None else None})
    inj = pd.DataFrame(rows)
    inj["run_order"] = np.arange(1, len(inj) + 1)
    inj["injection_id"] = [f"INJ_{k:03d}" for k in inj.run_order]
    inj = inj.set_index("injection_id")

    mu = rng.normal(20, 2, n_feat)
    feats = [f"F{i:04d}" for i in range(n_feat)]
    effect = np.zeros(n_feat)
    effect[:spiked] = fc
    bio = {s: rng.normal(0, 0.3, n_feat) for s in subj}
    L = np.full((n_feat, len(inj)), np.nan)
    amp = rng.uniform(0.5, 1.5, n_feat) * drift
    shift = rng.normal(batch_shift, 0.1 * abs(batch_shift) + 1e-9, n_feat)
    n_in_batch = inj.groupby("batch").size()
    for j, (_, r) in enumerate(inj.iterrows()):
        b = int(r.batch[1:]) - 1
        tech = rng.normal(0, 0.08, n_feat)
        pos = r.run_order_within_batch / n_in_batch[r.batch]
        technical = amp * np.sin(np.pi * pos) * 1.0 - amp * 0.5 + b * shift + tech
        if r.type == "sample":
            v = mu + bio[r.subject_id] + (effect if r.group == groups[-1] else 0) + technical
        elif r.type == "qc":
            v = mu + effect / len(groups) + technical
        else:
            v = np.full(n_feat, np.nan)
        if blank_only:
            if r.type == "blank":
                v[-blank_only:] = mu[-blank_only:] + rng.normal(0, 0.1, blank_only)
            else:
                v[-blank_only:] = mu[-blank_only:] - 0.5 + rng.normal(0, 0.1, blank_only)
        if r.type != "blank":
            v[rng.random(n_feat) < mcar] = np.nan
        L[:, j] = v
    X = pd.DataFrame(2.0 ** L, index=feats, columns=inj.index)
    samples = pd.DataFrame({
        "_type": inj["type"], "_group": inj["group"], "_batch": inj["batch"],
        "_order": inj["run_order"].astype(float), "_subject": inj["subject_id"]}, index=inj.index)
    truth = {"spiked": feats[:spiked], "blank_only": feats[n_feat - blank_only:] if blank_only else []}
    return X, samples, inj, truth


def cv(lin):
    return (lin.std(axis=1) / lin.mean(axis=1)).median()


# ---------------------------------------------------------------- pure steps

def test_robust_z_and_bad_injection_flagged():
    X, samples, _, _ = synth(blank_only=0)
    bad = samples.index[samples._type == "sample"][5]
    X[bad] = X[bad] * 0.2
    X.loc[X.sample(frac=0.4, random_state=1).index, bad] = np.nan
    flags = prep.flag_bad_injections(X, samples)
    assert flags.loc[bad, "rejected"]
    assert flags["rejected"].sum() == 1
    z = prep.robust_z(pd.Series([1.0, 1.1, 0.9, 1.0, 10.0]))
    assert z.iloc[-1] > 10 and abs(z.iloc[0]) < 1


def test_bad_injection_via_internal_standards():
    X, samples, _, _ = synth(blank_only=0)
    rng = np.random.default_rng(3)
    istd = pd.DataFrame(2.0 ** rng.normal(18, 0.05, (4, X.shape[1])), index=[f"IS{i}" for i in range(4)],
                        columns=X.columns)
    bad = samples.index[samples._type == "sample"][3]
    istd[bad] = istd[bad] / 4
    flags = prep.flag_bad_injections(X, samples, istd=istd)
    assert flags.loc[bad, "rejected"] and "IS" in flags.loc[bad, "reason"]
    assert flags["rejected"].sum() == 1


def test_blank_filter_removes_background_only():
    X, samples, _, truth = synth()
    keep, info = prep.blank_filter(X, samples, ratio=3.0)
    removed = set(X.index[~keep])
    assert removed == set(truth["blank_only"])
    assert prep.blank_filter(X, samples, ratio=0)[0].all()


def test_na_filter_keeps_feature_present_in_one_group():
    X, samples, _, _ = synth(blank_only=0, mcar=0)
    st = samples.index[samples._type == "sample"]
    ctrl = st[samples.loc[st, "_group"] == "CONTROL"]
    case = st[samples.loc[st, "_group"] == "CASE"]
    X.loc["F0100", ctrl] = np.nan            # only in CASE -> kept (present in one group)
    X.loc["F0101", st[: int(len(st) * 0.8)]] = np.nan   # mostly missing everywhere -> removed
    X.loc["F0101", case] = np.nan
    keep, frac = prep.na_filter(X, samples, na_max=0.5)
    assert keep["F0100"] and not keep["F0101"]
    assert keep.sum() == len(X) - 1


def test_halfmin_and_min_imputation():
    X = pd.DataFrame({"a": [10.0, np.nan, 4.0], "b": [np.nan, 8.0, 2.0], "c": [6.0, 2.0, np.nan]},
                     index=["f1", "f2", "f3"])
    h = prep.impute(X, "halfmin")
    assert h.loc["f1", "b"] == 3.0 and h.loc["f2", "a"] == 1.0 and h.loc["f3", "c"] == 1.0
    m = prep.impute(X, "min")
    assert m.loc["f1", "b"] == 6.0
    assert prep.impute(X, "none").isna().sum().sum() == 3
    k = prep.impute(X, "knn")
    assert k.notna().all().all()


def test_pqn_recovers_dilution_factors():
    rng = np.random.default_rng(1)
    base = 2.0 ** rng.normal(18, 2, 500)
    dil = np.array([1.0, 0.5, 2.0, 1.5, 0.8, 1.2, 0.7, 1.0])
    X = pd.DataFrame(np.outer(base, dil) * 2.0 ** rng.normal(0, 0.05, (500, 8)),
                     columns=[f"s{i}" for i in range(8)])
    X.iloc[:20, 2] *= 8   # a few genuinely changed features must not move the factor
    samples = pd.DataFrame({"_type": "sample", "_batch": "1"}, index=X.columns)
    L, factors = prep.normalize(np.log2(X), "pqn", samples)
    rel = factors / np.median(factors) / (dil / np.median(dil))
    assert np.allclose(rel, 1, atol=0.03)
    assert np.log2(X).sub(L, axis=0).std().std() < 1 and L.median().std() < 0.05


def test_median_and_sum_normalization_equalize_samples():
    X, samples, _, _ = synth(blank_only=0)
    cols = samples.index[samples._type != "blank"]
    L = np.log2(prep.impute(X[cols], "halfmin"))
    L.iloc[:, 3] += 1.0
    N, _ = prep.normalize(L, "median", samples.loc[cols])
    assert N.median().std() < 0.01
    # the total is dominated by the strongest features: the +1 shift goes, the medians match only roughly
    N, f = prep.normalize(L, "sum", samples.loc[cols])
    assert abs(np.log2(f.iloc[3]) - np.log2(f.drop(f.index[3])).median() - 1) < 0.3
    assert abs((N.iloc[:, 3] - N.drop(columns=N.columns[3]).median(axis=1)).median()) < 0.3


def test_is_normalization_uses_standards():
    X, samples, _, _ = synth(blank_only=0)
    cols = samples.index[samples._type != "blank"]
    L = np.log2(prep.impute(X[cols], "halfmin"))
    load = pd.Series(np.random.default_rng(2).normal(0, 0.5, len(cols)), index=cols)
    L = L.add(load, axis=1)
    istd = pd.DataFrame(2.0 ** (18 + np.tile(load.values, (3, 1))), index=["a", "b", "c"], columns=cols)
    N, f = prep.normalize(L, "is", samples.loc[cols], istd=istd)
    assert np.corrcoef(np.log2(f), load)[0, 1] > 0.99
    assert N.median().std() < L.median().std()


def test_loess_smoother_reproduces_lines():
    x = np.array([1, 5, 9, 13, 17, 21.0])
    S = prep.loess_matrix(x, np.arange(1, 22.0))
    assert np.allclose(S.sum(axis=1), 1)
    assert np.allclose(S @ (3 * x + 2), 3 * np.arange(1, 22.0) + 2)


def test_drift_removed_by_qc_loess_and_cv_drops():
    X, samples, _, _ = synth(drift=1.0, blank_only=0, qc_every=2)
    cols = samples.index[samples._type != "blank"]
    s = samples.loc[cols]
    mask = X[cols].notna()
    L = np.log2(prep.impute(X[cols], "halfmin"))
    qc = s.index[s._type == "qc"]
    before = prep.qc_cv(L, s, mask).median()
    C, info = prep.drift_correct(L, s, mask, mode="auto")
    after = prep.qc_cv(C, s, mask).median()
    assert info["corrected_batches"] == 2
    assert before > 0.2 and after < before / 2
    st = s.index[s._type == "sample"]
    # the drift is a hump within each batch: the technical part of the sample spread goes down
    assert C[st].std(axis=1).median() < 0.9 * L[st].std(axis=1).median()
    assert C[qc].notna().all().all()
    # auto needs >= 5 QCs per batch: with few QCs nothing happens
    X2, s2, _, _ = synth(drift=1.0, blank_only=0, qc_every=12)
    c2 = s2.index[s2._type != "blank"]
    L2 = np.log2(prep.impute(X2[c2], "halfmin"))
    C2, info2 = prep.drift_correct(L2, s2.loc[c2], X2[c2].notna(), mode="auto")
    assert info2["corrected_batches"] == 0 and np.allclose(C2.values, L2.values)


def test_batch_shift_removed():
    X, samples, _, _ = synth(batch_shift=1.0, blank_only=0)
    cols = samples.index[samples._type != "blank"]
    s = samples.loc[cols]
    L = np.log2(prep.impute(X[cols], "halfmin"))
    b1, b2 = s.index[s._batch == "B1"], s.index[s._batch == "B2"]
    assert (L[b2].median(axis=1) - L[b1].median(axis=1)).median() > 0.8
    C, info = prep.batch_correct(L, s, X[cols].notna(), mode="auto")
    assert info["method"] == "qc"
    assert abs((C[b2].median(axis=1) - C[b1].median(axis=1)).median()) < 0.1
    same, info0 = prep.batch_correct(L, s, X[cols].notna(), mode="none")
    assert info0["method"] == "none"


def test_d_ratio_and_cv_on_measured_values_only():
    X, samples, _, _ = synth(blank_only=0)
    cols = samples.index[samples._type != "blank"]
    s = samples.loc[cols]
    L = np.log2(X[cols])
    d = prep.d_ratio(L, s, L.notna())
    assert d.median() < 0.5      # technical sd 0.08 vs biological 0.3
    c = prep.qc_cv(L, s, L.notna())
    assert 0.02 < c.median() < 0.15


def test_welch_finds_spiked_features_at_fdr():
    X, samples, _, truth = synth(n_per_group=10, blank_only=0, mcar=0)
    st = samples.index[samples._type == "sample"]
    L = np.log2(X[st])
    res = stats.feature_tests(L, samples.loc[st, "_group"], ref="CONTROL", test="welch")
    sig = set(res.index[res.q < 0.05])
    assert set(truth["spiked"]) <= sig
    assert len(sig - set(truth["spiked"])) <= 3
    assert res.loc[truth["spiked"], "log2FC"].between(1.0, 2.0).all()
    from scipy import stats as ss
    a = L.loc["F0000", samples.loc[st, "_group"] == "CASE"]
    b = L.loc["F0000", samples.loc[st, "_group"] == "CONTROL"]
    assert np.isclose(res.loc["F0000", "p"], ss.ttest_ind(a, b, equal_var=False).pvalue)


def test_auto_test_choice_and_nonparametric():
    X, samples, _, truth = synth(n_per_group=8, blank_only=0, mcar=0, groups=("CONTROL", "A", "B"))
    st = samples.index[samples._type == "sample"]
    L = np.log2(X[st])
    res = stats.feature_tests(L, samples.loc[st, "_group"], ref="CONTROL", test="auto")
    assert res.attrs["test"] == "anova"
    assert {"log2FC_A", "log2FC_B"} <= set(res.columns)
    assert (res.loc[truth["spiked"], "q"] < 0.05).mean() > 0.9
    kw = stats.feature_tests(L, samples.loc[st, "_group"], ref="CONTROL", test="kruskal")
    assert (kw.loc[truth["spiked"], "q"] < 0.05).mean() > 0.5


def test_bh_matches_scipy():
    from scipy.stats import false_discovery_control
    p = np.random.default_rng(0).random(300) ** 3
    assert np.allclose(stats.bh(p), false_discovery_control(p))
    q = stats.bh(np.array([0.01, np.nan, 0.04]))
    assert np.isnan(q[1]) and np.allclose(q[[0, 2]], [0.02, 0.04])


def test_lm_with_covariates_matches_statsmodels():
    import statsmodels.api as sm
    X, samples, _, truth = synth(n_per_group=10, blank_only=0, mcar=0, n_feat=60)
    st = samples.index[samples._type == "sample"]
    rng = np.random.default_rng(5)
    cov = pd.DataFrame({"age": rng.normal(60, 10, len(st)), "sex": rng.choice(["F", "M"], len(st))}, index=st)
    L = np.log2(X[st])
    L = L + np.outer(np.r_[np.ones(10), np.zeros(50)], (cov.age - 60) / 10)  # age confounds 10 features
    res = stats.feature_tests(L, samples.loc[st, "_group"], ref="CONTROL", test="lm", covariates=cov)
    design = pd.DataFrame({"const": 1.0, "case": (samples.loc[st, "_group"] == "CASE").astype(float),
                           "age": cov.age, "sexM": (cov.sex == "M").astype(float)}, index=st)
    for f in ["F0000", "F0011", "F0040"]:
        fit = sm.OLS(L.loc[f, st].values, design.values).fit()
        assert np.isclose(res.loc[f, "p"], fit.pvalues[1], rtol=1e-6)
        assert np.isclose(res.loc[f, "log2FC"], fit.params[1], rtol=1e-6)
    # missing values -> per-feature statsmodels path gives the same answer on the remaining samples
    L.iloc[0, 0] = np.nan
    res2 = stats.feature_tests(L, samples.loc[st, "_group"], ref="CONTROL", test="lm", covariates=cov)
    ok = L.loc["F0000"].notna().values
    fit = sm.OLS(L.loc["F0000", st].values[ok], design.values[ok]).fit()
    assert np.isclose(res2.loc["F0000", "p"], fit.pvalues[1], rtol=1e-6)


def test_plsda_vip_cv_and_permutation():
    X, samples, _, truth = synth(n_per_group=12, blank_only=0, mcar=0, fc=2.0)
    st = samples.index[samples._type == "sample"]
    L = np.log2(X[st])
    out = stats.plsda(L, samples.loc[st, "_group"], n_perm=30, threads=1, seed=0)
    assert out["accuracy"] > 0.9 and out["q2"] > 0.3 and out["perm_p"] < 0.1
    top = out["vip"].sort_values(ascending=False).index[:20]
    assert len(set(top) & set(truth["spiked"])) >= 15
    again = stats.plsda(L, samples.loc[st, "_group"], n_perm=30, threads=1, seed=0)
    assert again["perm_p"] == out["perm_p"] and np.allclose(again["vip"], out["vip"])


def test_pca_scores_shape():
    X, samples, _, _ = synth(blank_only=0, mcar=0)
    st = samples.index[samples._type == "sample"]
    p = stats.pca(np.log2(X[st]))
    assert p["scores"].shape[0] == len(st) and p["explained"][0] > p["explained"][1]


# ---------------------------------------------------------------- parsers

WIN = "D:\\SCIEX OS Data\\Lab\\Data\\"


def _gg_matrix_text():
    names = ["P1_3", "P1_2", "P1_1", "QC1", "QC0", "Blank_1", "P2_3", "P2_2", "P2_1", "QC2", "P3_1", "P3_2", "P3_3"]
    cols = [WIN + n + ".wiff" for n in names]
    rng = np.random.default_rng(0)
    lines = ["\t".join(["Genes", "N.Sequences", "N.Proteotypic.Sequences"] + cols)]
    for g in ["ADI1", "ADK", "ALB", "APOA1", "TP53"]:
        vals = ["" if (rng.random() < 0.2 or "Blank" in c) else f"{rng.uniform(10, 300):.4f}" for c in cols]
        lines.append("\t".join([g, "3", "3"] + vals))
    return "\n".join(lines) + "\n", names


def test_sample_stem_and_type_inference():
    assert parse.sample_stem(WIN + "P1_3.wiff") == "P1_3"
    assert parse.sample_stem("/data/run/QC_04.raw") == "QC_04"
    assert parse.sample_stem("x/HF_01.mzML") == "HF_01"
    inf = parse.infer_from_names(["P1_3", "P1_1", "QC1", "Blank_2", "pool_3", "CTRL3", "Ctrl4", "P2_1"])
    assert list(inf._type) == ["sample", "sample", "qc", "blank", "qc", "sample", "sample", "sample"]
    assert inf.loc["P1_3", "_group"] == "P1" and inf.loc["CTRL3", "_group"] == "CTRL"
    assert inf.loc["Ctrl4", "_group"] == "Ctrl" and pd.isna(inf.loc["QC1", "_group"])


def test_parse_gg_matrix_windows_paths(tmp_path):
    text, names = _gg_matrix_text()
    (tmp_path / "report.gg_matrix.tsv").write_text(text)
    [ds] = parse.load_datasets(tmp_path, group_col="group", layer="auto")
    assert ds.layer == "proteomics"
    assert list(ds.X.columns) == names
    assert ds.X.shape == (5, 13) and "ADI1" in ds.X.index
    assert ds.samples.loc["QC1", "_type"] == "qc" and ds.samples.loc["Blank_1", "_type"] == "blank"
    assert ds.samples.loc["P2_3", "_group"] == "P2"
    assert ds.inferred["group"] and ds.inferred["order"]
    assert ds.features.loc["ADI1", "gene"] == "ADI1"


def test_parse_pg_matrix_with_sample_sheet_override(tmp_path):
    cols = ["/data/A_1.raw", "/data/A_2.raw", "/data/A_3.raw", "/data/B_1.raw", "/data/B_2.raw", "/data/B_3.raw"]
    df = pd.DataFrame({"Protein.Group": ["P04637", "O75110;O75111", "Q9BV57"],
                       "Protein.Ids": ["P04637", "O75110;O75111", "Q9BV57"],
                       "Protein.Names": ["P53_HUMAN", "AT9A_HUMAN", "MTND_HUMAN"],
                       "Genes": ["TP53", "ATP9A", "ADI1"],
                       "First.Protein.Description": ["p53", "ATPase", "ARD"]})
    for i, c in enumerate(cols):
        df[c] = [100.0 + i, np.nan, 50.0 * (i + 1)]
    df.to_csv(tmp_path / "report.pg_matrix.tsv", sep="\t", index=False)
    sheet = pd.DataFrame({"file": ["A_1.raw", "A_2.raw", "A_3.raw", "B_1.raw", "B_2.raw", "B_3.raw"],
                          "condition": ["ctrl", "ctrl", "treated", "treated", "treated", "ctrl"],
                          "order": [6, 5, 4, 3, 2, 1]})
    sheet.to_excel(tmp_path / "probki.xlsx", index=False)
    [ds] = parse.load_datasets(tmp_path, group_col="condition", layer="auto")
    assert ds.layer == "proteomics"
    assert ds.samples.loc["A_3", "_group"] == "treated"   # sheet overrides name prefix "A"
    assert not ds.inferred["group"]
    assert ds.samples.loc["B_3", "_order"] == 1 and not ds.inferred["order"]
    assert ds.features.loc["O75110;O75111", "accession"] == "O75110"
    assert ds.features.loc["P04637", "gene"] == "TP53"


def test_parse_diann_long_report(tmp_path):
    rows = []
    for run in ["S1", "S2", "S3"]:
        for pg, gene, q in [("P04637", "TP53", 0.001), ("Q9BV57", "ADI1", 0.001), ("P99999", "BAD", 0.2)]:
            for prec in range(2):
                rows.append({"File.Name": f"C:\\raw\\{run}.raw", "Run": run, "Protein.Group": pg, "Genes": gene,
                             "Protein.Names": gene + "_HUMAN", "Precursor.Id": f"PEP{prec}",
                             "Q.Value": q, "PG.Q.Value": q, "PG.MaxLFQ": 1000.0 * (1 + len(run) + prec * 0)})
    pd.DataFrame(rows).to_csv(tmp_path / "report.tsv", sep="\t", index=False)
    [ds] = parse.load_datasets(tmp_path, group_col="group", layer="auto")
    assert ds.X.shape == (2, 3) and "P99999" not in ds.X.index
    assert any("Q.Value" in n for n in ds.notes)


def test_parse_msdial_alignment_with_header_rows(tmp_path):
    meta_cols = ["Alignment ID", "Average Rt(min)", "Average Mz", "Metabolite name", "Adduct type", "Formula"]
    samples = ["Blank_01", "QC_01", "CTRL_01", "CASE_01", "QC_02", "CTRL_02", "CASE_02"]
    header = [
        [""] * (len(meta_cols) - 1) + ["Class"] + ["Blank", "QC", "CTRL", "CASE", "QC", "CTRL", "CASE", "CTRL"],
        [""] * (len(meta_cols) - 1) + ["File type"] + ["Blank", "QC", "Sample", "Sample", "QC", "Sample", "Sample",
                                                       "Average"],
        [""] * (len(meta_cols) - 1) + ["Injection order"] + ["1", "2", "3", "4", "5", "6", "7", ""],
        [""] * (len(meta_cols) - 1) + ["Batch ID"] + ["1", "1", "1", "1", "1", "1", "1", ""],
    ]
    table = [meta_cols + samples + ["CTRL"]]
    for i, name in enumerate(["Glucose", "Unknown", "Citrate"]):
        table.append([str(i), f"{1 + i:.2f}", f"{180 + i:.4f}", name, "[M+H]+", "C6H12O6"]
                     + [str(0 if s.startswith("Blank") else 1000 * (i + 1) + k) for k, s in enumerate(samples)]
                     + ["999"])
    text = "\n".join("\t".join(r) for r in header + table) + "\n"
    (tmp_path / "Height_0_2024.txt").write_text(text)
    [ds] = parse.load_datasets(tmp_path, group_col="group", layer="auto")
    assert ds.layer == "metabolomics"
    assert list(ds.X.columns) == samples           # the class-average column is not a sample
    assert ds.samples.loc["CASE_02", "_group"] == "CASE" and ds.samples.loc["QC_01", "_type"] == "qc"
    assert ds.samples.loc["Blank_01", "_type"] == "blank" and ds.samples.loc["CASE_01", "_order"] == 4
    assert np.isnan(ds.X.loc["0", "Blank_01"])     # MS-DIAL zero = not detected
    assert ds.features.loc["2", "name"] == "Citrate"


def _studenci_tables(X, inj, truth, layer="proteomics", prefix="PRO"):
    ids = {old: f"{prefix}_{k + 1:03d}" for k, old in enumerate(X.columns)}
    Xr = X.rename(columns=ids)
    Xr.index = [f"{prefix[0]}{i:05d}" for i in range(len(X))]
    inten = Xr.copy()
    inten.index.name = "feature_id"
    typemap = {"sample": "study", "qc": "pooled_qc", "blank": "process_blank"}
    injections = pd.DataFrame({
        "injection_id": list(ids.values()), "omics": layer, "batch": inj.batch.values,
        "run_order": inj.run_order.values, "run_order_within_batch": inj.run_order_within_batch.values,
        "sample_type": [typemap[t] for t in inj.type], "subject_id": inj.subject_id.values,
        "source_sample_id": ["BLANK" if t == "blank" else "POOL" if t == "qc" else s
                             for t, s in zip(inj.type, inj.subject_id, strict=True)],
        "is_spiked": 1})
    genes = [f"GENE{i}" for i in range(len(X))]
    features = pd.DataFrame({"feature_id": Xr.index, "uniprot_accession": [f"P{10000 + i}" for i in range(len(X))],
                             "gene_symbol": genes, "protein_name": [f"Protein {g}" for g in genes]})
    rng = np.random.default_rng(9)
    istd = pd.DataFrame(2.0 ** rng.normal(18, 0.05, (3, Xr.shape[1])), columns=Xr.columns,
                        index=pd.Index([f"{prefix}_IS_0{i}" for i in range(1, 4)], name="standard_id"))
    spiked = [Xr.index[list(X.index).index(f)] for f in truth["spiked"]]
    return inten, injections, features, istd, spiked


def _write_studenci_zip(path, n_feat=300):
    # 12 changed features out of 300: median normalisation assumes most features do not change
    X, samples, inj, truth = synth(n_feat=n_feat, n_per_group=12, groups=("CONTROL_RISK", "HFpEF"), qc_every=2,
                                   drift=0.6, batch_shift=0.5, spiked=12)
    inten, injections, features, istd, spiked = _studenci_tables(X, inj, truth)
    subj = inj.dropna(subset=["subject_id"]).drop_duplicates("subject_id")
    rng = np.random.default_rng(4)
    subjects = pd.DataFrame({"subject_id": subj.subject_id.values, "group": subj.group.values,
                             "age_years": rng.integers(40, 80, len(subj)), "sex": rng.choice(["F", "M"], len(subj))})
    buf = {}
    buf["proteomics_intensities.csv"] = inten.to_csv(na_rep="NA")
    buf["proteomics_injections.csv"] = injections.to_csv(index=False, na_rep="NA")
    buf["proteomics_features.csv"] = features.to_csv(index=False)
    buf["proteomics_internal_standards.csv"] = istd.to_csv()
    buf["proteomics_standards_metadata.csv"] = "standard_id,standard_name\nPRO_IS_01,HEAVY1\nPRO_IS_02,HEAVY2\n"
    buf["subjects.csv"] = subjects.to_csv(index=False)
    buf["data_dictionary.csv"] = "table,field,dtype\nsubjects.csv,group,category\nsubjects.csv,sex,category\n"
    with zipfile.ZipFile(path, "w") as z:
        for name, text in buf.items():
            z.writestr("STUDENCI/" + name, text)
    return spiked


def test_parse_studenci_layout_zip(tmp_path):
    _write_studenci_zip(tmp_path / "dane.zip")
    [ds] = parse.load_datasets(tmp_path, group_col="group", layer="auto")
    assert ds.layer == "proteomics"
    assert ds.istd is not None and ds.istd.shape[0] == 3
    s = ds.samples
    assert set(s._type) == {"sample", "qc", "blank"}
    assert set(s.loc[s._type == "sample", "_group"]) == {"CONTROL_RISK", "HFpEF"}
    assert "age_years" in s.columns and not ds.inferred["group"]
    assert ds.features["gene"].notna().all() and ds.features["accession"].str.startswith("P").all()
    assert s.loc["PRO_001", "_raw_type"] == "process_blank"


# ---------------------------------------------------------------- network steps (mocked)

UNIPROT_TSV = ("Entry\tProtein names\tGene Names (primary)\tFunction [CC]\tGene Ontology (biological process)\t"
               "Gene Ontology (molecular function)\tGene Ontology (cellular component)\tSubcellular location [CC]\n"
               "P04637\tCellular tumor antigen p53\tTP53\tFUNCTION: Tumor suppressor.\tapoptotic process [GO:0006915]"
               "\tDNA binding [GO:0003677]\tnucleus [GO:0005634]\tSUBCELLULAR LOCATION: Nucleus.\n")


def test_uniprot_annotation_batches_and_caches(tmp_path):
    calls = []

    def fetch(url, data=None, timeout=30):
        calls.append(url)
        return UNIPROT_TSV.encode()

    cache = tmp_path / "cache.json"
    tab, warn = annotate.uniprot(accessions=["P04637", "Q00000"], genes=None, organism="hsapiens",
                                 cache_path=cache, fetch=fetch)
    assert warn is None and len(calls) == 1 and "accessions=" in calls[0]
    assert tab.loc["P04637", "protein_name"] == "Cellular tumor antigen p53"
    assert "GO:0006915" in tab.loc["P04637", "go_bp"]
    tab2, _ = annotate.uniprot(accessions=["P04637", "Q00000"], genes=None, organism="hsapiens",
                               cache_path=cache, fetch=fetch)
    assert len(calls) == 1 and tab2.equals(tab)   # second time everything comes from the cache


def test_uniprot_by_gene_and_offline(tmp_path):
    def fetch(url, data=None, timeout=30):
        assert "gene_exact" in url and "organism_id%3A9606" in url
        return UNIPROT_TSV.encode()

    tab, warn = annotate.uniprot(accessions=None, genes=["TP53"], organism="hsapiens",
                                 cache_path=tmp_path / "c.json", fetch=fetch)
    assert tab.loc["TP53", "accession"] == "P04637"

    def offline(url, data=None, timeout=30):
        raise OSError("no network")

    tab, warn = annotate.uniprot(accessions=["P04637"], genes=None, organism="hsapiens",
                                 cache_path=tmp_path / "d.json", fetch=offline)
    assert tab.empty and "UniProt" in warn


def test_gprofiler_enrichment_mocked(tmp_path):
    sent = {}

    def fetch(url, data=None, timeout=30):
        sent.update(json.loads(data))
        return json.dumps({"result": [
            {"source": "GO:BP", "native": "GO:0006915", "name": "apoptotic process", "p_value": 1e-4,
             "term_size": 100, "query_size": 5, "intersection_size": 4, "query": "wszystkie", "significant": True}
        ]}).encode()

    tab, warn = annotate.gprofiler({"wszystkie": ["TP53", "MDM2", "ATM"]}, background=["TP53", "MDM2", "ATM", "X"],
                                   organism="hsapiens", cache_path=tmp_path / "g.json", fetch=fetch)
    assert warn is None and tab.iloc[0]["native"] == "GO:0006915"
    assert sent["domain_scope"] == "custom" and sent["background"] == ["TP53", "MDM2", "ATM", "X"]

    def offline(url, data=None, timeout=30):
        raise TimeoutError

    tab, warn = annotate.gprofiler({"wszystkie": ["TP53", "MDM2", "ATM"]}, background=["TP53"], organism="hsapiens",
                                   cache_path=tmp_path / "h.json", fetch=offline)
    assert tab.empty and "g:Profiler" in warn


# ---------------------------------------------------------------- end to end, like the runner

def run_recipe(work, **given):
    params = RECIPES["omics"].validate({k: str(v) for k, v in given.items()})
    (work / "params.json").write_text(json.dumps(params))
    env = dict(os.environ, PYTHONPATH=str(REPO), COMPUTE_THREADS="2", COMPUTE_BOT_OFFLINE="1", MPLBACKEND="Agg")
    return subprocess.run([sys.executable, "-m", "compute_bot.recipes.omics"], cwd=work, env=env,
                          capture_output=True, text=True, timeout=600)


def test_end_to_end_studenci_zip(tmp_path):
    spiked = _write_studenci_zip(tmp_path / "dane.zip")
    # synthetic gaps are random (MCAR): halfmin would bias them, the audit says so and knn is the right choice
    r = run_recipe(tmp_path, covariates="age_years,sex", impute="knn")
    assert r.returncode == 0, r.stderr[-3000:]
    assert "PROGRESS:" in r.stdout
    summary = (tmp_path / "summary.txt").read_text()
    assert len(summary) <= 2500 and "istotn" in summary and "FDR" in summary
    assert "HFpEF" in summary and "CONTROL_RISK" in summary
    assert "brak sieci" in summary.lower() or "offline" in summary.lower()
    assert "MCAR" in summary
    w = tmp_path / "wyniki"
    for f in ["statystyka.csv", "macierz_po_obrobce.csv", "qc_cechy.csv", "qc_iniekcje.csv", "plsda_vip.csv",
              "kroki.txt", "volcano.png", "heatmapa.png", "pca_przed_po.png", "qc_dryf.png"]:
        assert (w / f).exists(), f
    res = pd.read_csv(w / "statystyka.csv", index_col=0)
    assert res.attrs == {} and "q" in res.columns
    found = set(res.index[res.q < 0.05])
    assert len(found & set(spiked)) >= 0.8 * len(spiked)
    assert len(found - set(spiked)) <= 5
    qcf = pd.read_csv(w / "qc_cechy.csv", index_col=0)
    assert qcf["cv_qc_po"].median() < qcf["cv_qc_przed"].median()
    steps = (w / "kroki.txt").read_text()
    assert "blank" in steps.lower() and "dryf" in steps.lower()
    hl = (tmp_path / "highlights.txt").read_text().split()
    assert 1 <= len(hl) <= 4 and all((tmp_path / p).exists() for p in hl)
    # a second run in the same directory gives the same result
    first = (w / "statystyka.csv").read_text()
    r2 = run_recipe(tmp_path, covariates="age_years,sex", impute="knn")
    assert r2.returncode == 0 and (w / "statystyka.csv").read_text() == first


def test_end_to_end_gg_matrix_inferred_groups(tmp_path):
    rng = np.random.default_rng(0)
    names = [f"P{p}_{r}" for p in (1, 2) for r in (1, 2, 3, 4)] + [f"QC{i}" for i in range(3)] + ["Blank_1"]
    cols = [WIN + n + ".wiff" for n in names]
    lines = ["\t".join(["Genes", "N.Sequences", "N.Proteotypic.Sequences"] + cols)]
    for i in range(80):
        base = rng.uniform(5, 12)
        vals = []
        for n in names:
            v = base + rng.normal(0, 0.15) + (1.5 if (n.startswith("P2") and i < 10) else 0)
            vals.append("" if n.startswith("Blank") else f"{2 ** v:.3f}")
        lines.append("\t".join([f"G{i}", "2", "2"] + vals))
    (tmp_path / "report.gg_matrix (2).tsv").write_text("\n".join(lines) + "\n")
    r = run_recipe(tmp_path)
    assert r.returncode == 0, r.stderr[-3000:]
    summary = (tmp_path / "summary.txt").read_text()
    assert "nazw" in summary and "P1" in summary and "P2" in summary
    res = pd.read_csv(tmp_path / "wyniki" / "statystyka.csv", index_col=0)
    assert (res.loc[[f"G{i}" for i in range(10)], "q"] < 0.05).mean() >= 0.8


def test_end_to_end_bad_input_reports_error(tmp_path):
    (tmp_path / "notatki.txt").write_text("to nie jest tabela\n")
    r = run_recipe(tmp_path)
    assert r.returncode != 0
    assert r.stderr.strip().splitlines()[-1].startswith("ERROR:")


def test_end_to_end_unknown_ref_group(tmp_path):
    _write_studenci_zip(tmp_path / "dane.zip", n_feat=60)
    r = run_recipe(tmp_path, ref="NIEMA")
    assert r.returncode != 0
    last = r.stderr.strip().splitlines()[-1]
    assert last.startswith("ERROR:") and "HFpEF" in last


@pytest.mark.slow
def test_live_uniprot_and_gprofiler(tmp_path):
    tab, warn = annotate.uniprot(accessions=["P04637"], genes=None, organism="hsapiens",
                                 cache_path=tmp_path / "c.json")
    assert warn is None and tab.loc["P04637", "gene"] == "TP53"
    enr, warn = annotate.gprofiler({"wszystkie": ["TP53", "MDM2", "CDKN1A", "ATM", "CHEK2"]}, background=None,
                                   organism="hsapiens", cache_path=tmp_path / "g.json")
    assert warn is None and len(enr) > 0

