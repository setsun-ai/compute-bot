"""Tests for the QSAR/QSPR recipe: pure maths on data with a known answer, then the recipe run like the runner does."""
import json
import os
import shutil
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("openpyxl")

from compute_bot.recipes import RECIPES  # noqa: E402
from compute_bot.recipes.qsar import prefilter, selection, split, stats, table  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------- helpers

def synthetic(n=40, p_noise=8, seed=0, noise=0.1):
    """y = 2*x1 - x2 + small noise, plus descriptors that are pure noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2 + p_noise))
    x[:, 0] = x[:, 0] * 1.5 + 3.0  # not standardised on purpose: coefficients are in original units
    y = 2 * x[:, 0] - x[:, 1] + rng.normal(scale=noise, size=n)
    cols = ["x1", "x2"] + [f"n{i}" for i in range(p_noise)]
    df = pd.DataFrame(x, columns=cols)
    df.insert(0, "name", [f"c{i:02d}" for i in range(n)])
    df["logK"] = y
    return df


def run_recipe(tmp_path, files, timeout=900, **params):
    """Start the recipe as the runner does: python -m in the work dir with a validated params.json."""
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    for fname, writer in files.items():
        writer(work / fname)
    given = {k: (",".join(v) if isinstance(v, list | tuple) else str(v)) for k, v in params.items()}
    (work / "params.json").write_text(json.dumps(RECIPES["qsar"].validate(given)))
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    env["COMPUTE_THREADS"] = "2"
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env["MPLBACKEND"] = "Agg"
    proc = subprocess.run([sys.executable, "-m", "compute_bot.recipes.qsar"], cwd=work, env=env,
                          capture_output=True, text=True, timeout=timeout)
    return proc, work


def csv_writer(df, **kw):
    return lambda path: df.to_csv(path, index=False, **kw)


def two_header_xlsx(path, df_desc, names, y, extra_rows=()):
    """The course spreadsheet shape: two header rows, junk cells, \\xa0 in names, an empty split column."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "PROJEKT"
    first = ["Organic compounds", "Wzór sumaryczny", "Struktura", "EXP LogKda", "Zbiór"]
    ws.append(first + [None] * df_desc.shape[1])
    ws.append([None] * len(first) + [c + "\xa0" for c in df_desc.columns])
    for nm, yy, row in zip(names, y, df_desc.itertuples(index=False), strict=True):
        ws.append([nm, "C6H6", "#VALUE!", yy, None] + list(row))
    for nm, row in extra_rows:
        ws.append([nm, None, None, None, None] + list(row))
    wb.save(path)


# ---------------------------------------------------------------- metrics, hand-computed

def test_external_q2_and_ccc_match_hand_values():
    y_train = np.array([1, 2, 3, 4, 5], float)  # mean 3, SS = 10
    y_test = np.array([2, 5], float)
    y_pred = np.array([2.5, 4.5])  # PRESS = 0.5
    assert stats.q2_f1(y_test, y_pred, y_train.mean()) == pytest.approx(1 - 0.5 / 5)
    assert stats.q2_f2(y_test, y_pred) == pytest.approx(1 - 0.5 / 4.5)
    assert stats.q2_f3(y_test, y_pred, y_train) == pytest.approx(1 - (0.5 / 2) / (10 / 5))
    assert stats.ccc(y_test, y_pred) == pytest.approx(6 / 6.5)
    assert stats.rmse(y_test, y_pred) == pytest.approx(0.5)
    assert stats.r2(y_train, y_train) == pytest.approx(1.0)


def test_rm2_and_golbraikh_tropsha_on_perfect_and_biased_predictions():
    y = np.array([1.0, 2.0, 3.5, 4.0, 6.0, 7.5])
    perfect = stats.rm2(y, y)
    assert perfect["rm2_avg"] == pytest.approx(1.0)
    assert perfect["rm2_delta"] == pytest.approx(0.0, abs=1e-12)
    gt = stats.golbraikh_tropsha(y, y, q2_loo=0.9)
    assert gt["pass"] and gt["k"] == pytest.approx(1.0)
    # predictions that correlate perfectly but are twice too large: r2 = 1 yet slopes fail
    gt_bad = stats.golbraikh_tropsha(y, 2 * y, q2_loo=0.9)
    assert gt_bad["r2"] == pytest.approx(1.0)
    assert not gt_bad["pass"]
    assert gt_bad["k"] == pytest.approx(0.5)


def test_leverage_and_warning_leverage():
    x_train = np.array([[0.0], [1.0], [2.0]])
    h = stats.leverage(x_train, x_train)
    assert h == pytest.approx([1 / 3 + 1 / 2, 1 / 3, 1 / 3 + 1 / 2])
    assert h.sum() == pytest.approx(2.0)  # trace of the hat matrix = p' + 1
    assert stats.leverage(x_train, np.array([[4.0]]))[0] == pytest.approx(1 / 3 + (4 - 1) ** 2 / 2)
    assert stats.h_star(p=1, n=3) == pytest.approx(2.0)
    assert stats.h_star(p=4, n=30) == pytest.approx(0.5)


def test_closed_form_loo_equals_explicit_refits():
    df = synthetic(n=25, p_noise=2, seed=3, noise=0.5)
    x = df[["x1", "x2", "n0"]].to_numpy()
    y = df["logK"].to_numpy()
    q2, y_cv = stats.q2_loo_ols(x, y)
    manual = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        a = np.column_stack([np.ones(keep.sum()), x[keep]])
        beta = np.linalg.lstsq(a, y[keep], rcond=None)[0]
        manual.append(np.r_[1.0, x[i]] @ beta)
    assert y_cv == pytest.approx(np.array(manual))
    assert q2 == pytest.approx(1 - ((y - y_cv) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    batch = selection.q2loo_batch((x - x.mean(0)) / x.std(0, ddof=1), y, np.array([[0, 1, 2]]))
    assert batch[0][0] == pytest.approx(q2)


def test_classification_metrics_hand_values():
    y_true = np.array(["a", "a", "a", "a", "b", "b"])
    y_pred = np.array(["a", "a", "a", "b", "b", "a"])
    m = stats.classification_metrics(y_true, y_pred, positive="b")
    assert m["sensitivity"] == pytest.approx(0.5)  # 1 of 2 b found
    assert m["specificity"] == pytest.approx(0.75)  # 3 of 4 a kept
    assert m["balanced_accuracy"] == pytest.approx(0.625)
    tp, tn, fp, fn = 1, 3, 1, 1
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    assert m["mcc"] == pytest.approx(mcc)
    m2 = stats.classification_metrics(y_true, y_pred, positive="b", score=np.array([.1, .2, .3, .6, .9, .4]))
    assert m2["roc_auc"] == pytest.approx(7 / 8)


# ---------------------------------------------------------------- splitting

def test_sorted_split_keeps_extremes_in_training_and_every_kth_in_test():
    rng = np.random.default_rng(1)
    y = rng.permutation(np.arange(20, dtype=float))
    test = split.sorted_split(y, every=3)
    order = np.argsort(y, kind="stable")
    expected = {order[i] for i in range(2, 19, 3)}  # 3rd, 6th, ... in sorted order, never the last one
    assert set(np.flatnonzero(test)) == expected
    assert not test[np.argmin(y)] and not test[np.argmax(y)]
    # when the k-th position lands on the maximum it still stays in training
    y2 = np.arange(9, dtype=float)
    t2 = split.sorted_split(y2, every=3)
    assert not t2[8] and set(np.flatnonzero(t2)) == {2, 5}


def test_kennard_stone_on_tiny_known_set():
    x = np.array([[0.0], [1.0], [2.0], [3.0], [10.0]])
    assert split.kennard_stone(x, 5) == [0, 4, 3, 1, 2]
    assert sorted(split.kennard_stone(x, 3)) == [0, 3, 4]
    square = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5]], float)
    first = split.kennard_stone(square, 4)
    assert 4 not in first  # the centre is the least informative point


def test_random_split_is_seeded():
    a = split.random_split(30, 0.25, seed=7)
    b = split.random_split(30, 0.25, seed=7)
    assert (a == b).all() and a.sum() == 8


def test_split_column_labels_in_polish_and_english():
    labels = split.parse_split_labels(["T", "V", "uczący", "walidacyjny", "train", "test", "U", "W", "Testowy"])
    assert list(labels) == ["train", "test", "train", "test", "train", "test", "train", "test", "test"]
    with pytest.raises(ValueError):
        split.parse_split_labels(["T", "banana"])


# ---------------------------------------------------------------- pre-filter, selection

def test_prefilter_drops_constant_and_keeps_better_of_correlated_pair():
    rng = np.random.default_rng(0)
    n = 30
    x1 = rng.normal(size=n)
    y = x1 + rng.normal(scale=0.3, size=n)
    # extra noise orthogonal to x1 and y: corr(copy, good) ~ 0.97 > 0.95 and copy is strictly worse with y
    basis = np.column_stack([np.ones(n), x1, y])
    e = rng.normal(size=n)
    e -= basis @ np.linalg.lstsq(basis, e, rcond=None)[0]
    df = pd.DataFrame({
        "good": x1,
        "copy": x1 + 0.25 * e / e.std(),
        "const": np.ones(n),
        "almost": np.r_[np.zeros(n - 1), 1.0],
        "other": rng.normal(size=n),
    })
    kept, report = prefilter.prefilter(df, y, corr_max=0.95)
    assert "good" in kept and "copy" not in kept and "other" in kept
    assert "const" not in kept and "almost" not in kept
    assert report["correlated"] == 1 and report["constant"] == 2


@pytest.mark.parametrize("method", ["exhaustive", "forward", "ga", "enet"])
def test_selection_finds_the_true_descriptors(method):
    df = synthetic(n=40, p_noise=8, seed=0)
    cols = [c for c in df.columns if c not in ("name", "logK")]
    x = df[cols].to_numpy()
    xs = (x - x.mean(0)) / x.std(0, ddof=1)
    res = selection.select(xs, df["logK"].to_numpy(), cols, method=method, k_max=4, seed=1)
    assert sorted(res.chosen) == ["x1", "x2"]
    assert res.q2 > 0.98


def test_auto_selection_uses_exhaustive_when_small():
    df = synthetic(n=30, p_noise=4)
    cols = [c for c in df.columns if c not in ("name", "logK")]
    x = df[cols].to_numpy()
    res = selection.select((x - x.mean(0)) / x.std(0, ddof=1), df["logK"].to_numpy(), cols, method="auto",
                           k_max=3, seed=0)
    assert res.method == "exhaustive"


# ---------------------------------------------------------------- reading tables

def test_two_header_row_xlsx_is_merged_and_cleaned(tmp_path):
    df = synthetic(n=12, p_noise=1)
    path = tmp_path / "dane.xlsx"
    two_header_xlsx(path, df[["x1", "x2", "n0"]].rename(columns={"x1": "log D", "x2": "Mw'", "n0": "εα"}),
                    df["name"], df["logK"], extra_rows=[("naphthalene", [3.3, 0.1, 0.2])])
    raw = table.read_table(path)
    assert list(raw.columns[:8]) == ["Organic compounds", "Wzór sumaryczny", "Struktura", "EXP LogKda", "Zbiór",
                                     "log D", "Mw'", "εα"]
    assert len(raw) == 13
    cols = table.detect_columns(raw)
    assert cols.target == "EXP LogKda" and cols.name == "Organic compounds"
    assert cols.smiles is None and cols.task == "regression"
    assert cols.given == ["log D", "Mw'", "εα"]  # junk 'Struktura' and empty 'Zbiór' are not descriptors
    assert cols.split == "Zbiór"


def test_csv_with_semicolons_and_decimal_commas(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("Nazwa;logP;MW;pIC50\nA;1,5;100,2;5,1\nB;2,5;120,0;6,0\nC;#DIV/0!;130;\nD;3,0;140;7,2\n",
                    encoding="utf-8")
    raw = table.read_table(path)
    cols = table.detect_columns(raw)
    assert cols.target == "pIC50" and cols.name == "Nazwa"
    assert raw["logP"].iloc[0] == pytest.approx(1.5)
    assert np.isnan(raw["logP"].iloc[2]) and np.isnan(raw["pIC50"].iloc[2])


def test_ambiguous_target_asks_for_target_param():
    raw = pd.DataFrame({"name": ["a", "b", "c"], "p1": [1.0, 2, 3], "p2": [3.0, 1, 2]})
    with pytest.raises(table.InputError, match="target="):
        table.detect_columns(raw)
    assert table.detect_columns(raw, target="p2").target == "p2"


def test_descriptor_names_ending_in_y_are_not_the_target():
    raw = pd.DataFrame({"Lp.": [1.0, 2, 3, 4, 5], "name": list("abcde"), "EXP LogKd": [1.0, 2, 3, 4, 5],
                        "ETA_Shape_Y": [3.0, 1, 2, 2, 1], "MOMI-Y": [1.0, 5, 2, 3, 3], "nR": [1.0, 2, 2, 3, 1],
                        "nO": [0.0, 0, 1, 2, 0]})
    cols = table.detect_columns(raw)
    assert cols.target == "EXP LogKd"
    assert cols.given == ["ETA_Shape_Y", "MOMI-Y", "nR", "nO"]  # row numbers out, PaDEL counts nR/nO stay


def test_smiles_column_found_by_content():
    pytest.importorskip("rdkit")
    raw = pd.DataFrame({"związek": ["etanol", "benzen", "fenol"], "struct": ["CCO", "c1ccccc1", "Oc1ccccc1"],
                        "y": [0.1, 2.1, 1.5]})
    cols = table.detect_columns(raw)
    assert cols.smiles == "struct" and cols.name == "związek" and cols.target == "y"


def test_text_target_means_classification():
    raw = pd.DataFrame({"name": list("abcdef"), "d": [1.0, 2, 3, 4, 5, 6],
                        "aktywność": ["aktywny", "nieaktywny"] * 3})
    cols = table.detect_columns(raw)
    assert cols.task == "classification" and cols.target == "aktywność"


def test_parse_mopac_output():
    from compute_bot.recipes.qsar import descriptors
    text = """
          FINAL HEAT OF FORMATION =        -21.17527 KCAL/MOL =     -88.59734 KJ/MOL
          COSMO AREA              =        130.16 SQUARE ANGSTROMS
          COSMO VOLUME            =        119.81 CUBIC ANGSTROMS
          IONIZATION POTENTIAL    =          9.098340 EV
          HOMO LUMO ENERGIES (EV) =         -9.098  0.128
          MOLECULAR WEIGHT        =         94.1128         POINT GROUP:  Cs
              NET ATOMIC CHARGES AND DIPOLE CONTRIBUTIONS

  ATOM NO.   TYPE          CHARGE      No. of ELECS.   s-Pop       p-Pop
    1          C          -0.375846        4.3758     1.10812     3.26773
    2          O          -0.476164        6.4762     1.80640     4.66977
    3          H           0.322087        0.6779     0.67791
    4          H           0.166608        0.8334     0.83339
 DIPOLE           X         Y         Z       TOTAL
 POINT-CHG.    -0.558     1.139     0.256     1.294
 SUM            0.071     1.322     0.306     1.359
 ETOT (EONE + ETWO)                      -1107.3385 EV
 * JOB ENDED NORMALLY *
"""
    d = descriptors.parse_mopac(text)
    assert d["PM6_HOF"] == pytest.approx(-21.17527)
    assert d["PM6_HOMO"] == pytest.approx(-9.098) and d["PM6_LUMO"] == pytest.approx(0.128)
    assert d["PM6_gap"] == pytest.approx(9.226)
    assert d["PM6_dipole"] == pytest.approx(1.359)
    assert d["PM6_IP"] == pytest.approx(9.09834)
    assert d["PM6_Etot"] == pytest.approx(-1107.3385)
    assert d["PM6_COSMO_area"] == pytest.approx(130.16) and d["PM6_COSMO_volume"] == pytest.approx(119.81)
    assert d["PM6_qmax"] == pytest.approx(0.322087) and d["PM6_qmin"] == pytest.approx(-0.476164)
    assert d["PM6_qHmax"] == pytest.approx(0.322087)
    with pytest.raises(ValueError):
        descriptors.parse_mopac("nothing here")


# ---------------------------------------------------------------- end to end, as the runner starts it

def _read(work, rel):
    return (work / rel).read_text(encoding="utf-8")


def test_recipe_on_synthetic_csv_recovers_model(tmp_path):
    df = synthetic(n=40, p_noise=8, seed=0)
    new = df.iloc[:1].copy()
    new["name"] = "nowy"
    new["logK"] = np.nan
    data = pd.concat([df, new], ignore_index=True)
    proc, work = run_recipe(tmp_path, {"dane.csv": csv_writer(data)}, models=["mlr", "knn", "rf"], scramble=30,
                            select="exhaustive")
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "PROGRESS:" in proc.stdout
    summary = _read(work, "summary.txt")
    assert len(summary) <= 2500
    assert "x1" in summary and "x2" in summary and "Q²LOO" in summary
    selected = _read(work, "wyniki/selected.txt")
    assert "x1" in selected and "x2" in selected
    metrics = pd.read_csv(work / "wyniki/metrics.csv")
    assert set(metrics["model"]) == {"mlr", "knn", "rf"}
    mlr = metrics.set_index("model").loc["mlr"]
    assert mlr["Q2_LOO"] > 0.98 and mlr["Q2_F3"] > 0.95 and mlr["CCC"] > 0.95
    assert mlr["scr_R2_mean"] < 0.3 and mlr["cR2p"] > 0.8
    coefs = pd.read_csv(work / "wyniki/mlr_coefficients.csv").set_index("term")
    assert coefs.loc["x1", "coef"] == pytest.approx(2.0, abs=0.05)
    assert coefs.loc["x2", "coef"] == pytest.approx(-1.0, abs=0.05)
    assert {"SE", "t", "p", "VIF"} <= set(coefs.columns)
    pred = pd.read_csv(work / "wyniki/predictions.csv")
    assert set(pred["model"]) == {"mlr", "knn", "rf"}
    row = pred[(pred["compound"] == "nowy") & (pred["model"] == "mlr")].iloc[0]
    assert row["set"] == "prognoza" and row["y_pred"] == pytest.approx(data["logK"].iloc[0], abs=0.3)
    assert bool(row["in_ad"])
    assert set(pred["set"]) == {"uczący", "testowy", "prognoza"}
    assert (work / "wyniki/descriptors.csv").exists()
    for png in ("williams.png", "scrambling.png", "pca.png", "corr_heatmap.png", "obs_vs_pred_mlr.png"):
        assert (work / "wyniki" / png).exists(), png
    hl = [h for h in _read(work, "highlights.txt").splitlines() if h.strip()]
    assert 1 <= len(hl) <= 4 and all((work / h).exists() for h in hl)


def test_recipe_on_two_header_xlsx_with_prediction_row(tmp_path):
    df = synthetic(n=30, p_noise=4, seed=5)
    desc = df[["x1", "x2", "n0", "n1", "n2", "n3"]].rename(columns={"x1": "log D", "x2": "Mw'"})
    proc, work = run_recipe(
        tmp_path,
        {"Dane.xlsx": lambda p: two_header_xlsx(p, desc, df["name"], df["logK"],
                                               extra_rows=[("naphthalene", list(desc.iloc[3]))])},
        models=["mlr", "pls", "ridge"], scramble=20)
    assert proc.returncode == 0, proc.stderr[-3000:]
    summary = _read(work, "summary.txt")
    assert "log D" in summary and "naphthalene" in summary
    pred = pd.read_csv(work / "wyniki/predictions.csv")
    nap = pred[pred["compound"] == "naphthalene"]
    assert len(nap) == 3 and nap["y_pred"].notna().all()


def test_recipe_with_rdkit_descriptors_from_smiles(tmp_path):
    pytest.importorskip("rdkit")
    from rdkit import Chem
    from rdkit.Chem import Crippen
    smiles = ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO", "c1ccccc1", "Cc1ccccc1", "CCc1ccccc1", "Oc1ccccc1",
              "Clc1ccccc1", "Clc1ccc(Cl)cc1", "Brc1ccccc1", "c1ccc2ccccc2c1", "CC(C)O", "CC(=O)O", "CCC(=O)O",
              "CCCCC(=O)O", "OCC(O)CO", "CCOCC", "CCCCCC", "CCCCCCCC", "C1CCCCC1", "ClC(Cl)Cl", "ClC(Cl)(Cl)Cl",
              "CC(C)(C)O", "Nc1ccccc1", "O=[N+]([O-])c1ccccc1", "CC#N", "CCN(CC)CC", "OC(=O)c1ccccc1",
              "COc1ccccc1", "Cc1ccc(C)cc1"]
    y = [Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in smiles]
    data = pd.DataFrame({"Związek": [f"m{i}" for i in range(len(smiles))], "SMILES": smiles, "logKow": y})
    data.loc[len(data)] = ["naftalen-2-ol", "Oc1ccc2ccccc2c1", np.nan]
    data.loc[len(data)] = ["zły", "C1CC(", np.nan]  # unparseable: reported, run goes on
    proc, work = run_recipe(tmp_path, {"mols.csv": csv_writer(data)}, descriptors=["rdkit"],
                            models=["mlr", "svr"], scramble=10, max_desc=2, split="ks")
    assert proc.returncode == 0, proc.stderr[-3000:]
    summary = _read(work, "summary.txt")
    assert "zły" in summary  # the failed molecule is named
    desc = pd.read_csv(work / "wyniki/descriptors.csv")
    assert "MolLogP" in desc.columns and len(desc) == len(data)
    metrics = pd.read_csv(work / "wyniki/metrics.csv").set_index("model")
    assert metrics.loc["mlr", "Q2_LOO"] > 0.9
    pred = pd.read_csv(work / "wyniki/predictions.csv")
    row = pred[(pred["compound"] == "naftalen-2-ol") & (pred["model"] == "mlr")].iloc[0]
    true = Crippen.MolLogP(Chem.MolFromSmiles("Oc1ccc2ccccc2c1"))
    assert row["y_pred"] == pytest.approx(true, abs=0.6)


def test_recipe_classification_mode(tmp_path):
    rng = np.random.default_rng(4)
    n = 60
    x = rng.normal(size=(n, 5))
    label = np.where(x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.3, size=n) > 0.3, "aktywny", "nieaktywny")
    data = pd.DataFrame(x, columns=[f"d{i}" for i in range(5)])
    data.insert(0, "name", [f"z{i}" for i in range(n)])
    data["klasa"] = label
    data.loc[n] = ["nowy", 2.0, 1.0, 0, 0, 0, None]
    proc, work = run_recipe(tmp_path, {"k.csv": csv_writer(data)}, models=["mlr", "knn", "rf", "svr"],
                            split="random", scramble=10)
    assert proc.returncode == 0, proc.stderr[-3000:]
    summary = _read(work, "summary.txt")
    assert "klasyfikacja" in summary.lower()
    metrics = pd.read_csv(work / "wyniki/metrics.csv")
    assert {"logistic", "knn", "rf", "svm"} <= set(metrics["model"])
    for col in ("test_balanced_accuracy", "test_mcc", "test_sensitivity", "test_specificity", "test_roc_auc"):
        assert col in metrics.columns
    assert metrics["cv_balanced_accuracy"].max() > 0.75
    pred = pd.read_csv(work / "wyniki/predictions.csv")
    assert pred[(pred["compound"] == "nowy")]["y_pred"].eq("aktywny").all()


def test_recipe_split_from_column_and_rerun_gives_same_result(tmp_path):
    df = synthetic(n=24, p_noise=3, seed=9)
    df["Zbiór"] = ["T", "T", "V"] * 8
    proc, work = run_recipe(tmp_path, {"d.csv": csv_writer(df)}, split="column", models=["mlr", "dt"], scramble=10)
    assert proc.returncode == 0, proc.stderr[-3000:]
    pred = pd.read_csv(work / "wyniki/predictions.csv")
    sets = pred[pred["model"] == "mlr"].set_index("compound")["set"]
    assert (sets[df["name"][df["Zbiór"] == "V"]] == "testowy").all()
    first = _read(work, "wyniki/metrics.csv")
    env = dict(os.environ, PYTHONPATH=ROOT, COMPUTE_THREADS="2")
    again = subprocess.run([sys.executable, "-m", "compute_bot.recipes.qsar"], cwd=work, env=env,
                           capture_output=True, text=True, timeout=600)
    assert again.returncode == 0, again.stderr[-2000:]
    assert _read(work, "wyniki/metrics.csv") == first


def test_recipe_reports_owner_error_for_missing_column(tmp_path):
    df = synthetic(n=20, p_noise=2)
    proc, _ = run_recipe(tmp_path, {"d.csv": csv_writer(df)}, target="nie_ma_takiej")
    assert proc.returncode != 0
    assert proc.stderr.strip().splitlines()[-1].startswith("ERROR:")
    assert "nie_ma_takiej" in proc.stderr


# ---------------------------------------------------------------- slow: mordred, MOPAC

SMALL = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1", "Clc1ccccc1", "CC(=O)O", "CCOCC", "CCCCCC",
         "Nc1ccccc1", "CC#N", "CCCl", "CCBr", "CC(C)O", "COC"]


@pytest.mark.slow
def test_recipe_with_mordred(tmp_path):
    pytest.importorskip("mordred")
    from rdkit import Chem
    from rdkit.Chem import Descriptors as D
    y = [D.TPSA(Chem.MolFromSmiles(s)) / 10 for s in SMALL]
    data = pd.DataFrame({"name": [f"m{i}" for i in range(len(SMALL))], "smiles": SMALL, "y": y})
    proc, work = run_recipe(tmp_path, {"m.csv": csv_writer(data)}, descriptors=["mordred"], models=["mlr"],
                            scramble=5, max_desc=2)
    assert proc.returncode == 0, proc.stderr[-3000:]
    desc = pd.read_csv(work / "wyniki/descriptors.csv")
    assert desc.shape[1] > 500


@pytest.mark.slow
def test_recipe_with_mopac_and_cache_on_restart(tmp_path):
    if not shutil.which("mopac", path=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")):
        pytest.skip("no mopac binary")
    pytest.importorskip("rdkit")
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"name": [f"m{i}" for i in range(len(SMALL))], "smiles": SMALL,
                         "y": rng.normal(size=len(SMALL))})
    proc, work = run_recipe(tmp_path, {"m.csv": csv_writer(data)}, descriptors=["mopac"], models=["mlr"],
                            scramble=5, max_desc=2)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "MOPAC" in proc.stdout
    desc = pd.read_csv(work / "wyniki/descriptors.csv")
    for col in ("PM6_HOMO", "PM6_LUMO", "PM6_gap", "PM6_dipole", "PM6_HOF", "PM6_IP", "PM6_COSMO_area", "PM6_Etot"):
        assert desc[col].notna().all(), col
    # benzene HOMO in PM6 is around -9.7 eV
    assert desc.loc[desc["smiles"] == "c1ccccc1", "PM6_HOMO"].iloc[0] == pytest.approx(-9.7, abs=0.5)
    cache = sorted((work / "cache" / "mopac").glob("*.json"))
    assert len(cache) == len(SMALL)
    stamps = {p: p.stat().st_mtime_ns for p in cache}
    env = dict(os.environ, PYTHONPATH=ROOT, COMPUTE_THREADS="2",
               PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", ""))
    again = subprocess.run([sys.executable, "-m", "compute_bot.recipes.qsar"], cwd=work, env=env,
                           capture_output=True, text=True, timeout=600)
    assert again.returncode == 0, again.stderr[-2000:]
    assert {p: p.stat().st_mtime_ns for p in cache} == stamps
