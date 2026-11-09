"""The QSAR/QSPR workflow, step by step. `run()` reads the work directory and writes everything into it;
the steps themselves are functions of arrays and tables."""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import descriptors as dsc
from . import models as mdl
from . import prefilter, selection, split, stats, table
from .report import OUT, descriptors_only, progress, write_outputs
from .table import InputError

INPUT_EXT = (".xlsx", ".xlsm", ".xls", ".csv", ".tsv", ".txt", ".smi")
OWN_FILES = {"summary.txt", "highlights.txt", "params.json"}
GAP_MAX = 0.3  # R² − Q²LOO (or train − CV balanced accuracy) above this = overfitted, not eligible as best
POSITIVE_WORDS = {"1", "active", "aktywny", "aktywna", "aktywne", "yes", "tak", "pos", "positive", "toxic",
                  "toksyczny", "true", "+"}


# ---------------------------------------------------------------- input

def find_input(workdir):
    files = [p for p in Path(workdir).iterdir() if p.is_file() and p.suffix.lower() in INPUT_EXT
             and p.name not in OWN_FILES and not p.name.startswith((".", "~$"))]
    for ext in INPUT_EXT:
        found = sorted(p for p in files if p.suffix.lower() == ext)
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise InputError(f"Kilka tabel ({', '.join(p.name for p in found)}) – wyślij jedną.")
    raise InputError("Nie znalazłem tabeli (xlsx, csv, tsv, txt albo smi).")


@dataclass
class Data:
    names: np.ndarray
    smiles: list | None
    status: np.ndarray  # label / predict / excluded
    y: np.ndarray  # float (regression) or object labels (classification)
    task: str
    target: str | None
    cols: table.Columns
    raw: pd.DataFrame
    notes: list = field(default_factory=list)


def load(path, params):
    raw = table.read_table(path)
    cols = table.detect_columns(raw, target=params["target"], smiles=params["smiles"], name=params["name"],
                                split_col=params["split_col"])
    n = len(raw)
    if cols.name:
        names = [str(v) if v is not None and str(v) != "nan" else f"związek {i + 1}"
                 for i, v in enumerate(raw[cols.name].tolist())]
    else:
        names = [f"związek {i + 1}" for i in range(n)]
    seen = {}
    for i, nm in enumerate(names):
        seen[nm] = seen.get(nm, 0) + 1
        if seen[nm] > 1:
            names[i] = f"{nm} #{seen[nm]}"
    smiles = raw[cols.smiles].tolist() if cols.smiles else None
    status = np.array(["predict"] * n, dtype=object)
    notes = list(cols.notes)
    if cols.target is None:
        y = np.full(n, np.nan)
    elif cols.task == "regression":
        y = raw[cols.target].to_numpy(dtype=float)
        status[np.isfinite(y)] = "label"
        bad = raw.attrs.get("unparsed", {}).get(cols.target, [])
        if bad:
            status[bad] = "excluded"
            notes.append(f"Odpowiedź nie jest liczbą u {len(bad)} związków (pominięte): "
                         + ", ".join(names[i] for i in bad[:5]))
    else:
        col = raw[cols.target]
        vals = []
        for v in col.tolist():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                vals.append(None)
            elif isinstance(v, float) and v == int(v):
                vals.append(str(int(v)))
            else:
                vals.append(str(v))
        y = np.array(vals, dtype=object)
        status[np.array([v is not None for v in vals])] = "label"
    return Data(np.array(names, dtype=object), smiles, status, y, cols.task, cols.target, cols, raw, notes)


# ---------------------------------------------------------------- descriptors

def compute_descriptors(data, sources, workdir, threads, seed):
    """All requested descriptor blocks side by side; rows whose molecule failed are listed in `failed`."""
    sources = list(dict.fromkeys(sources))
    notes, failed = [], {}
    computed = [s for s in sources if s != "given"]
    if "given" in sources and not data.cols.given:
        if data.smiles is not None and not computed:
            notes.append("W pliku nie ma deskryptorów liczbowych – liczę deskryptory RDKit ze SMILES.")
            computed = ["rdkit"]
        elif not computed:
            raise InputError("W pliku nie ma deskryptorów (kolumn liczbowych) ani SMILES. Dodaj kolumnę SMILES "
                             "i descriptors=rdkit albo kolumny z deskryptorami.")
    if computed and data.smiles is None:
        raise InputError(f"descriptors={','.join(computed)} wymaga kolumny ze SMILES – nie znalazłem jej. "
                         "Podaj smiles=<kolumna>.")
    blocks = []
    if "given" in sources and data.cols.given:
        blocks.append(("given", data.raw[data.cols.given].reset_index(drop=True).astype(float)))
    if computed:
        progress("przygotowanie cząsteczek")
        mols, errors = dsc.prepare_mols(data.smiles)
        failed.update(errors)
        cache = Path(workdir) / "cache"
        for src in computed:
            if src == "rdkit":
                progress("deskryptory RDKit")
                blocks.append(("rdkit", dsc.rdkit_block(mols)))
            elif src == "morgan":
                blocks.append(("morgan", dsc.morgan_block(mols)))
            elif src == "mordred":
                progress("deskryptory Mordred")
                blocks.append(("mordred", dsc.mordred_block(mols, cache / "mordred", threads, progress)))
            elif src == "mopac":
                progress("MOPAC PM6")
                try:
                    block, errs = dsc.mopac_block(mols, cache / "mopac", threads, progress, seed)
                except FileNotFoundError:
                    notes.append("Brak programu mopac – pominięto deskryptory PM6.")
                    continue
                for i, e in errs.items():
                    failed.setdefault(i, e)
                blocks.append(("mopac", block))
    if not blocks:
        raise InputError("Nie udało się policzyć żadnych deskryptorów.")
    used, parts = set(), []
    for src, block in blocks:
        ren = {c: (f"{c}_{src}" if c in used else c) for c in block.columns}
        used.update(ren.values())
        parts.append(block.rename(columns=ren))
    desc = pd.concat(parts, axis=1)
    desc = desc.replace([np.inf, -np.inf], np.nan)
    return desc, failed, notes, [s for s, _ in blocks]


def drop_incomplete(desc, rows, names):
    """Descriptors with any missing value among the modelling compounds go; say which compounds caused it."""
    sub = desc.loc[rows]
    missing = sub.isna().any()
    culprits = sub.loc[:, missing].isna().sum(1)
    culprits = culprits[culprits > 0].sort_values(ascending=False)
    worst = [f"{names[i]} ({int(c)})" for i, c in culprits.head(5).items()]
    return list(desc.columns[~missing]), int(missing.sum()), worst


# ---------------------------------------------------------------- split

def make_split(params, data, rows, desc_cols, desc, seed):
    y = data.y[rows]
    method = params["split"]
    if method == "sorted":
        key = y if data.task == "regression" else pd.factorize(y, sort=True)[0]
        test = split.sorted_split(key, params["every"])
        how = f"sortowanie po y, co {params['every']}. związek do testu (skrajne w uczącym)"
    elif method == "ks":
        x = desc.loc[rows, desc_cols].to_numpy(float)
        sd = x.std(0)
        x = (x[:, sd > 0] - x[:, sd > 0].mean(0)) / sd[sd > 0]
        test = split.ks_split(x, params["test"])
        how = f"Kennard–Stone na deskryptorach, test {100 * params['test']:.0f}%"
    elif method == "random":
        test = split.random_split(len(rows), params["test"], seed, strata=y if data.task != "regression" else None)
        how = f"losowy (ziarno {seed}), test {100 * params['test']:.0f}%"
    else:
        col = data.cols.split
        if col is None:
            raise InputError("split=column, ale nie ma kolumny z podziałem. Podaj split_col=<kolumna>.")
        values = data.raw[col].to_numpy(dtype=object)[rows]
        try:
            labels = split.parse_split_labels(values)
        except ValueError as exc:
            raise InputError(f"Nie rozumiem wartości „{exc}” w kolumnie {col}. Użyj T/V, train/test, "
                             "uczący/testowy.") from None
        empty = [data.names[r] for r, lab in zip(rows, labels, strict=True) if lab is None]
        if len(empty) == len(rows):
            raise InputError(f"Kolumna {col} jest pusta – wypełnij ją albo użyj split=sorted.")
        if empty:
            raise InputError(f"Brak przydziału w kolumnie {col} dla: {', '.join(empty[:8])}.")
        test = labels == "test"
        how = f"z kolumny {col}"
    if (~test).sum() < 5:
        raise InputError("Za mało związków w zbiorze uczącym (< 5).")
    return test, how


# ---------------------------------------------------------------- validation helpers

def regression_metrics(y_tr, p_tr, cv_tr, y_te, p_te, n_desc):
    m = {"R2": stats.r2(y_tr, p_tr), "RMSE_C": stats.rmse(y_tr, p_tr), "MAE_C": stats.mae(y_tr, p_tr),
         "Q2_LOO": stats.r2(y_tr, cv_tr), "RMSE_CV": stats.rmse(y_tr, cv_tr), "MAE_CV": stats.mae(y_tr, cv_tr)}
    n = len(y_tr)
    m["R2_adj"] = 1 - (1 - m["R2"]) * (n - 1) / (n - n_desc - 1) if n - n_desc - 1 > 0 else np.nan
    m["R2_minus_Q2"] = m["R2"] - m["Q2_LOO"]
    m["Q2_F1"] = stats.q2_f1(y_te, p_te, np.mean(y_tr))
    m["Q2_F2"] = stats.q2_f2(y_te, p_te)
    m["Q2_F3"] = stats.q2_f3(y_te, p_te, y_tr)
    m["CCC"] = stats.ccc(y_te, p_te)
    m["R2_ext"] = stats.pearson_r2(y_te, p_te)
    m["RMSE_P"] = stats.rmse(y_te, p_te)
    m["MAE_P"] = stats.mae(y_te, p_te)
    rm = stats.rm2(y_te, p_te) if len(y_te) >= 3 else {"rm2_avg": np.nan, "rm2_delta": np.nan}
    m["rm2_avg"], m["rm2_delta"] = rm["rm2_avg"], rm["rm2_delta"]
    gt = stats.golbraikh_tropsha(y_te, p_te, m["Q2_LOO"])
    m.update(GT_k=gt["k"], GT_k_prime=gt["k_prime"], GT_r0_2=gt["r0_2"], GT_r0p_2=gt["r0p_2"],
             GT_passed=gt["passed"], GT_pass=bool(gt["pass"]))
    return m


def scramble_mlr(x, y, n_perm, seed):
    """Y-scrambling of MLR with the same descriptors: R² and Q²LOO of every permutation, closed form."""
    rng = np.random.default_rng(seed)
    a = np.column_stack([np.ones(len(y)), x])
    hat = a @ np.linalg.pinv(a.T @ a) @ a.T
    h = np.clip(1 - np.diag(hat), 1e-10, None)
    yp = np.column_stack([rng.permutation(y) for _ in range(n_perm)])
    resid = yp - hat @ yp
    ss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum(0) / ss
    q2 = 1 - ((resid / h[:, None]) ** 2).sum(0) / ss
    r = np.array([abs(np.corrcoef(y, yp[:, i])[0, 1]) for i in range(n_perm)])
    return r, r2, q2


def scramble_reselect(x, y, k, n_perm, seed, progress=None):
    """Y-scrambling with the variable selection repeated on every permuted y (forward, same size k).

    Fixed-descriptor scrambling cannot see the chance correlation that comes from picking k out of many
    descriptors; this does. Returns the Q²LOO of the best subset found for each permutation."""
    rng = np.random.default_rng(seed + 1)
    out = []
    for i in range(n_perm):
        yp = rng.permutation(y)
        per_k = selection.forward(selection.RegressionScorer(x, yp), x.shape[1], k, [""] * x.shape[1])
        out.append(max(r["score"] for r in per_k))
        if progress and (i + 1) % 20 == 0:
            progress(f"Y-scrambling z ponownym wyborem {i + 1}/{n_perm}")
    return np.array(out)


def scramble_generic(name, est, x, y, n_perm, seed, task, threads):
    """Refit the tuned model on permuted y: training R² (BA) and 5-fold CV Q² (BA). Also the unpermuted 5-fold."""
    from sklearn.base import clone
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

    rng = np.random.default_rng(seed)
    if task == "regression":
        cv = KFold(5, shuffle=True, random_state=seed)
    else:
        _, counts = np.unique(y, return_counts=True)
        cv = StratifiedKFold(max(2, min(5, counts.min())), shuffle=True, random_state=seed)

    def score(yy):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kw = mdl._fit_params(name, yy, task)
            fitted = clone(est).fit(x, yy, **kw)
            cvp = np.asarray(cross_val_predict(clone(est), x, yy, cv=cv, n_jobs=threads, params=kw or None)).ravel()
        fit_p = np.asarray(fitted.predict(x)).ravel()
        if task == "regression":
            return stats.r2(yy, fit_p), stats.r2(yy, cvp)
        return balanced_accuracy_score(yy, fit_p), balanced_accuracy_score(yy, cvp)

    base = score(y)
    rs, a, b = [], [], []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        sa, sb = score(yp)
        a.append(sa)
        b.append(sb)
        rs.append(abs(np.corrcoef(y, yp)[0, 1]) if task == "regression" else float((yp == y).mean()))
    return np.array(rs), np.array(a), np.array(b), base


def choose_best_regression(metrics, sq_err):
    """One-standard-error rule on the cross-validated error of the training set.

    Eligible: not overfitted (R² − Q²LOO ≤ GAP_MAX). Among them the lowest CV mean squared error wins, but every
    model whose CV MSE is within one standard error of that minimum counts as equally good, and the simplest of
    those (order in models.SIMPLICITY) is chosen. `sq_err[name]` = squared CV residuals of the training compounds.
    Returns (best, all_overfitted, tolerance in Q² units).
    """
    df = metrics[np.isfinite(metrics["Q2_LOO"])]
    ok = df[df["R2_minus_Q2"] <= GAP_MAX]
    pool = ok if len(ok) else df
    mse = {m: float(np.mean(sq_err[m])) for m in pool.index}
    top = min(mse, key=mse.get)
    se = float(np.std(sq_err[top], ddof=1) / np.sqrt(len(sq_err[top])))
    near = [m for m in pool.index if mse[m] <= mse[top] + se]
    rank = {m: i for i, m in enumerate(mdl.SIMPLICITY)}
    ss = mse[top] / max(1e-12, 1 - pool.loc[top, "Q2_LOO"])  # = SS_tot / n, converts MSE to Q² units
    return min(near, key=lambda m: rank.get(m, 99)), len(ok) == 0, se / ss


def choose_best_classification(metrics, n_train):
    """Same idea for classes: within one binomial SE of the best CV balanced accuracy the simplest model wins."""
    df = metrics[np.isfinite(metrics["cv_balanced_accuracy"])]
    gap = df["train_balanced_accuracy"] - df["cv_balanced_accuracy"]
    ok = df[gap <= GAP_MAX]
    pool = ok if len(ok) else df
    top = pool["cv_balanced_accuracy"].max()
    se = float(np.sqrt(max(top * (1 - top), 1e-6) / n_train))
    near = pool[pool["cv_balanced_accuracy"] >= top - se]
    rank = {m: i for i, m in enumerate(mdl.SIMPLICITY)}
    return min(near.index, key=lambda m: rank.get(m, 99)), len(ok) == 0, se


def positive_class(classes, y):
    for c in classes:
        if table.normkey(c) in POSITIVE_WORDS:
            return c
    counts = {c: int((y == c).sum()) for c in classes}
    return min(classes, key=lambda c: (counts[c], -classes.index(c)))


# ---------------------------------------------------------------- the run

def run(workdir, params, threads=1):
    workdir = Path(workdir)
    out = workdir / OUT
    out.mkdir(exist_ok=True)
    seed = params["seed"]
    progress("wczytywanie tabeli")
    path = find_input(workdir)
    data = load(path, params)
    notes = list(data.notes)

    desc, failed, dnotes, sources = compute_descriptors(data, params["descriptors"], workdir, threads, seed)
    notes += dnotes
    info = {"file": path.name, "sources": sources, "n_desc_start": desc.shape[1], "gap_max": GAP_MAX}
    head = pd.DataFrame({"compound": data.names})
    if data.smiles is not None:
        head["smiles"] = data.smiles
    if data.target:
        head[data.target] = data.y
    pd.concat([head, desc], axis=1).to_csv(out / "descriptors.csv", index=False)
    if failed:
        notes.append(f"Nie policzono deskryptorów dla {len(failed)} związków: "
                     + "; ".join(f"{data.names[i]} ({e})" for i, e in list(failed.items())[:6]))
    if data.target is None:
        return descriptors_only(workdir, data, desc, info, notes)

    status = data.status.copy()
    for i in failed:
        if status[i] == "label":
            status[i] = "excluded"
    rows = np.flatnonzero(status == "label")
    if len(rows) < 8:
        raise InputError(f"Za mało związków z odpowiedzią i deskryptorami ({len(rows)}); potrzeba co najmniej 8.")
    if data.task == "classification":
        _, counts = np.unique(data.y[rows].astype(str), return_counts=True)
        if counts.min() < 3:
            raise InputError("W klasyfikacji każda klasa potrzebuje co najmniej 3 związków.")

    usable, n_na, worst = drop_incomplete(desc, rows, data.names)
    info.update(n_na=n_na, na_worst=worst)
    if not usable:
        raise InputError("Każdy deskryptor ma braki wśród związków z odpowiedzią – uzupełnij tabelę.")

    test_mask, split_how = make_split(params, data, rows, usable, desc, seed)
    tr, te = rows[~test_mask], rows[test_mask]
    info["split"] = split_how
    if params["split"] != "column" and data.cols.split and data.raw[data.cols.split].notna().any():
        notes.append(f"Plik ma kolumnę {data.cols.split} z podziałem – użyj split=column, żeby z niej skorzystać.")

    y_all = data.y
    if data.task == "regression":
        y_tr, y_te = y_all[tr].astype(float), y_all[te].astype(float)
    else:
        classes = sorted({str(v) for v in y_all[rows]})
        code = {c: i for i, c in enumerate(classes)}
        y_tr = np.array([code[v] for v in y_all[tr]])
        y_te = np.array([code[v] for v in y_all[te]])

    progress("odsiew deskryptorów")
    kept, frep = prefilter.prefilter(desc.loc[tr, usable].reset_index(drop=True), y_tr, params["corr_max"],
                                     task=data.task)
    info["prefilter"] = frep
    if not kept:
        raise InputError("Po odsiewie nie został żaden deskryptor (wszystkie stałe).")

    n_tr = len(tr)
    k_max = max(1, min(params["max_desc"], n_tr // 5, len(kept)))
    info["k_max"] = k_max
    if n_tr // 5 < params["max_desc"]:
        notes.append(f"Topliss: przy {n_tr} związkach uczących najwyżej {max(1, n_tr // 5)} deskryptory (n/p ≥ 5).")
    (xs_tr,), mu, sd = prefilter.autoscale(desc.loc[tr, kept].to_numpy(float))
    progress("wybór deskryptorów")
    sel = selection.select(xs_tr, y_tr, kept, method=params["select"], k_max=k_max, seed=seed, task=data.task,
                           progress=progress)
    chosen = sel.chosen
    notes += sel.notes
    p = len(chosen)
    info["ratio"] = n_tr / p
    if sel.method == "all" and n_tr / p < 5:
        notes.append(f"select=all: {p} deskryptorów na {n_tr} związków uczących – Topliss (n/p ≥ 5) niespełniony.")

    # scaled matrices of the chosen descriptors for every row that has them
    raw_sel = desc[chosen].to_numpy(float)
    finite = np.isfinite(raw_sel).all(1)
    (s_tr,), mu, sd = prefilter.autoscale(raw_sel[tr])
    s_all = np.full_like(raw_sel, np.nan)
    s_all[finite] = (raw_sel[finite] - mu) / sd
    s_te = s_all[te]

    names = list(params["models"])
    if data.task == "classification":
        names = list(dict.fromkeys(mdl.CLASS_MAP[m] for m in names))
    fitted, mrows, preds = {}, {}, {}
    for i, name in enumerate(names, 1):
        if name in ("mlr",) and p >= n_tr - 1:
            notes.append(f"MLR pominięty: {p} deskryptorów na {n_tr} związków.")
            continue
        progress(f"model {name} ({i}/{len(names)})")
        try:
            est, best_params = mdl.tune(name, s_tr, y_tr, data.task, seed, threads)
        except ImportError:
            notes.append(f"Brak pakietu dla modelu {name} – pominięty.")
            continue
        fitted[name] = (est, best_params)
        if data.task == "regression":
            p_tr = mdl.predict(est, s_tr)
            cv_tr = (stats.q2_loo_ols(s_tr, y_tr)[1] if name == "mlr"
                     else mdl.cv_predict(name, est, s_tr, y_tr, data.task, seed, threads))
            p_te = mdl.predict(est, s_te) if len(te) else np.array([])
            p_all = np.full(len(desc), np.nan)
            p_all[finite] = mdl.predict(est, s_all[finite])
            m = regression_metrics(y_tr, p_tr, cv_tr, y_te, p_te, p)
            preds[name] = {"fit": p_tr, "cv": cv_tr, "test": p_te, "all": p_all}
        else:
            proba_cv = mdl.cv_predict(name, est, s_tr, y_tr, data.task, seed, threads, method="predict_proba")
            pr_tr, pr_te = est.predict_proba(s_tr), est.predict_proba(s_te) if len(te) else np.zeros((0, len(classes)))
            pr_all = np.full((len(desc), len(classes)), np.nan)
            pr_all[finite] = est.predict_proba(s_all[finite])
            m = {}
            pos = positive_class(classes, np.array(classes)[y_tr])
            for tag, yy, pr in (("train", y_tr, pr_tr), ("cv", y_tr, proba_cv), ("test", y_te, pr_te)):
                yt = np.array(classes, dtype=object)[yy] if len(yy) else np.array([], dtype=object)
                yp = np.array(classes, dtype=object)[np.argmax(pr, 1)] if len(pr) else np.array([], dtype=object)
                score = pr[:, code[pos]] if len(classes) == 2 else pr
                cm = stats.classification_metrics(yt, yp, positive=pos, score=score if len(pr) else None,
                                                  classes=classes)
                m.update({f"{tag}_{k}": v for k, v in cm.items()})
            preds[name] = {"fit": pr_tr, "cv": proba_cv, "test": pr_te, "all": pr_all, "positive": pos}
        m.update(model=name, params=json.dumps(best_params, sort_keys=True), n_desc=p, n_train=n_tr, n_test=len(te),
                 cv_scheme=mdl.validation_cv(y_tr, data.task, seed)[1] if name != "mlr" else "LOO")
        mrows[name] = m
    if not mrows:
        raise InputError("Żaden model nie dał się dopasować.")
    metrics = pd.DataFrame(mrows.values()).set_index("model", drop=False)

    if data.task == "regression":
        sq = {nm: (y_tr - pr["cv"]) ** 2 for nm, pr in preds.items()}
        best, all_overfit, tol = choose_best_regression(metrics, sq)
    else:
        best, all_overfit, tol = choose_best_classification(metrics, n_tr)
    info["tolerance"] = tol
    if all_overfit:
        notes.append("Wszystkie modele są przeuczone (duża różnica między dopasowaniem a walidacją krzyżową).")
    metrics["best"] = metrics.index == best

    # applicability domain: leverage (linear models) and distance to k nearest training compounds
    h_all = np.full(len(desc), np.nan)
    hs = stats.h_star(p, n_tr)
    if p < n_tr - 1:
        h_all[finite] = stats.leverage(s_tr, s_all[finite])
    pos_in_train = np.full(len(desc), -1)
    pos_in_train[tr] = np.arange(n_tr)
    d_all = np.full(len(desc), np.nan)
    d_vals, d_thr = stats.knn_distance_ad(s_tr, s_all[finite], k=5, self_index=pos_in_train[finite])
    d_all[finite] = d_vals

    # Y-scrambling: MLR (closed form) and the best model
    scr = {}
    n_perm = params["scramble"]
    if n_perm > 0:
        progress("Y-scrambling")
        targets = ([b for b in dict.fromkeys(["mlr", best]) if b in fitted] if data.task == "regression"
                   else [best])
        for name in targets:
            if name == "mlr":
                r, a, b = scramble_mlr(s_tr, y_tr, n_perm, seed)
                base = (mrows["mlr"]["R2"], mrows["mlr"]["Q2_LOO"])
                lab = "Q²LOO"
            else:
                r, a, b, base = scramble_generic(name, fitted[name][0], s_tr, y_tr, n_perm, seed, data.task, threads)
                lab = "Q²(5-fold)" if data.task == "regression" else "BA(CV 5)"
            scr[name] = (r, a, b, base, lab)
            if data.task == "regression":
                r2m = mrows[name]["R2"]
                metrics.loc[name, "scr_R2_max"] = a.max()
                metrics.loc[name, "scr_R2_mean"] = a.mean()
                metrics.loc[name, "scr_Q2_max"] = b.max()
                metrics.loc[name, "scr_Q2_mean"] = b.mean()
                metrics.loc[name, "cR2p"] = np.sqrt(max(r2m, 0)) * np.sqrt(max(0.0, r2m - a.mean()))
            else:
                metrics.loc[name, "scr_BA_train_max"] = a.max()
                metrics.loc[name, "scr_BA_cv_max"] = b.max()
                metrics.loc[name, "scr_BA_cv_mean"] = b.mean()
        if data.task == "regression" and "mlr" in fitted and sel.method != "all" and len(kept) > p:
            (xk_tr,), _, _ = prefilter.autoscale(desc.loc[tr, kept].to_numpy(float))
            q_sel = scramble_reselect(xk_tr, y_tr, p, n_perm, seed, progress)
            metrics.loc["mlr", "scr_sel_Q2_max"] = q_sel.max()
            metrics.loc["mlr", "scr_sel_Q2_mean"] = q_sel.mean()
            info["reselect_p"] = len(kept)
        pd.DataFrame([{"model": nm, "perm": i + 1, "abs_r_or_same": v[0][i], "train": v[1][i], "cv": v[2][i]}
                      for nm, v in scr.items() for i in range(len(v[0]))]).to_csv(out / "scrambling.csv", index=False)

    ctx = dict(workdir=workdir, out=out, data=data, desc=desc, status=status, tr=tr, te=te, rows=rows,
               chosen=chosen, sel=sel, metrics=metrics, best=best, preds=preds, fitted=fitted, h_all=h_all, hs=hs,
               d_all=d_all, d_thr=d_thr, finite=finite, kept=kept, s_all=s_all, s_tr=s_tr, y_tr=y_tr, y_te=y_te,
               scr=scr, info=info, notes=notes, params=params, failed=failed, p=p, n_tr=n_tr)
    if data.task == "classification":
        ctx["classes"] = classes
    write_outputs(ctx)


# ---------------------------------------------------------------- outputs
