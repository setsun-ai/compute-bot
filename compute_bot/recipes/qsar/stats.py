"""Validation statistics for QSAR models (OECD principle 4) and the applicability domain (principle 3).

Definitions follow Consonni, Ballabio & Todeschini (Q²F3), Schüürmann (Q²F2), Chirico & Gramatica (CCC),
Roy et al. (rm²), Golbraikh & Tropsha (external criteria), Gramatica (leverage, h* = 3(p'+1)/n).
All functions take plain arrays and return numbers.
"""
from __future__ import annotations

import numpy as np


def _arr(a):
    return np.asarray(a, dtype=float).ravel()


def r2(y, y_hat):
    """Coefficient of determination 1 - SSres/SStot around the mean of y."""
    y, y_hat = _arr(y), _arr(y_hat)
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - y_hat) ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")


def rmse(y, y_hat):
    y, y_hat = _arr(y), _arr(y_hat)
    return float(np.sqrt(((y - y_hat) ** 2).mean())) if len(y) else float("nan")


def mae(y, y_hat):
    y, y_hat = _arr(y), _arr(y_hat)
    return float(np.abs(y - y_hat).mean()) if len(y) else float("nan")


def q2_f1(y_test, y_pred, y_train_mean):
    """Shi et al.: 1 - PRESS / Σ(y_test - ȳ_train)²."""
    y, p = _arr(y_test), _arr(y_pred)
    den = ((y - y_train_mean) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / den) if len(y) > 1 and den > 0 else float("nan")


def q2_f2(y_test, y_pred):
    """Schüürmann et al.: 1 - PRESS / Σ(y_test - ȳ_test)²."""
    y, p = _arr(y_test), _arr(y_pred)
    den = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / den) if len(y) > 1 and den > 0 else float("nan")


def q2_f3(y_test, y_pred, y_train):
    """Consonni et al.: 1 - [PRESS/n_test] / [Σ(y_train - ȳ_train)²/n_train]."""
    y, p, t = _arr(y_test), _arr(y_pred), _arr(y_train)
    den = ((t - t.mean()) ** 2).sum() / len(t)
    return float(1 - (((y - p) ** 2).sum() / len(y)) / den) if len(y) and den > 0 else float("nan")


def ccc(y_obs, y_pred):
    """Lin's concordance correlation coefficient (Chirico & Gramatica threshold ≥ 0.85)."""
    x, y = _arr(y_obs), _arr(y_pred)
    if len(x) < 2:
        return float("nan")
    sxy = ((x - x.mean()) * (y - y.mean())).sum()
    den = ((x - x.mean()) ** 2).sum() + ((y - y.mean()) ** 2).sum() + len(x) * (x.mean() - y.mean()) ** 2
    return float(2 * sxy / den) if den > 0 else float("nan")


def pearson_r2(y, y_hat):
    y, y_hat = _arr(y), _arr(y_hat)
    if len(y) < 2 or y.std() == 0 or y_hat.std() == 0:
        return float("nan")
    return float(np.corrcoef(y, y_hat)[0, 1] ** 2)


def _through_origin(y, p):
    """Golbraikh–Tropsha regressions through the origin: y = k·p and p = k'·y with their r0² and r'0²."""
    k = (y * p).sum() / (p * p).sum()
    kp = (y * p).sum() / (y * y).sum()
    r0 = 1 - ((y - k * p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    r0p = 1 - ((p - kp * y) ** 2).sum() / ((p - p.mean()) ** 2).sum()
    return float(k), float(kp), float(r0), float(r0p)


def rm2(y_obs, y_pred):
    """Roy's rm² metrics: average of rm² and r'm², and their difference (criteria avg > 0.5, Δ < 0.2)."""
    y, p = _arr(y_obs), _arr(y_pred)
    r2_ = pearson_r2(y, p)
    if np.isnan(r2_):
        return {"rm2": np.nan, "rm2_prime": np.nan, "rm2_avg": np.nan, "rm2_delta": np.nan}
    _, _, r0, r0p = _through_origin(y, p)
    a = r2_ * (1 - np.sqrt(abs(r2_ - r0)))
    b = r2_ * (1 - np.sqrt(abs(r2_ - r0p)))
    return {"rm2": float(a), "rm2_prime": float(b), "rm2_avg": float((a + b) / 2), "rm2_delta": float(abs(a - b))}


def golbraikh_tropsha(y_obs, y_pred, q2_loo):
    """External predictivity criteria of Golbraikh & Tropsha (2002) on the test set."""
    y, p = _arr(y_obs), _arr(y_pred)
    r2_ = pearson_r2(y, p)
    out = {"r2": r2_, "r0_2": np.nan, "r0p_2": np.nan, "k": np.nan, "k_prime": np.nan, "pass": False, "passed": 0}
    if len(y) < 3 or np.isnan(r2_):
        return out
    k, kp, r0, r0p = _through_origin(y, p)
    crit = [
        q2_loo > 0.5,
        r2_ > 0.6,
        (r2_ - r0) / r2_ < 0.1 or (r2_ - r0p) / r2_ < 0.1,
        0.85 <= k <= 1.15 or 0.85 <= kp <= 1.15,
        abs(r0 - r0p) < 0.3,
    ]
    out.update(r0_2=r0, r0p_2=r0p, k=k, k_prime=kp, passed=int(sum(crit)), total=len(crit), **{"pass": all(crit)})
    return out


def _design(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    return np.column_stack([np.ones(len(x)), x])


def q2_loo_ols(x, y):
    """Leave-one-out Q² of ordinary least squares in closed form: e_i / (1 - h_ii). Returns (Q², LOO predictions)."""
    a, y = _design(x), _arr(y)
    pinv = np.linalg.pinv(a.T @ a)
    beta = pinv @ a.T @ y
    resid = y - a @ beta
    h = np.einsum("ij,jk,ik->i", a, pinv, a)
    loo_resid = resid / np.clip(1 - h, 1e-12, None)
    y_cv = y - loo_resid
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - (loo_resid ** 2).sum() / ss), y_cv


def leverage(x_train, x):
    """Hat values h = x (XᵀX)⁻¹ xᵀ with an intercept column, X = training descriptors."""
    a = _design(x_train)
    inv = np.linalg.pinv(a.T @ a)
    b = _design(x)
    return np.einsum("ij,jk,ik->i", b, inv, b)


def h_star(p, n):
    """Warning leverage 3(p'+1)/n, p' = number of descriptors, n = training compounds."""
    return 3 * (p + 1) / n


def vif(x):
    """Variance inflation factors = diagonal of the inverse correlation matrix."""
    x = np.asarray(x, dtype=float)
    if x.shape[1] < 2:
        return np.ones(x.shape[1])
    c = np.corrcoef(x, rowvar=False)
    try:
        return np.diag(np.linalg.inv(c))
    except np.linalg.LinAlgError:
        return np.full(x.shape[1], np.inf)


def knn_distance_ad(x_train, x, k=5, quantile=0.95, self_index=None):
    """Distance-based applicability domain: mean Euclidean distance to the k nearest training compounds.

    The threshold is the given quantile of the same distance among training compounds (each without itself).
    `self_index[i]` = position of row i of x in the training set (or -1), so a training compound skips itself.
    Returns (distances of x, threshold).
    """
    x_train, x = np.asarray(x_train, float), np.asarray(x, float)
    k = max(1, min(k, len(x_train) - 1))
    d_tt = np.sqrt(((x_train[:, None, :] - x_train[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d_tt, np.inf)
    train_d = np.sort(d_tt, axis=1)[:, :k].mean(1)
    if len(x) == 0:
        return np.array([]), float(np.quantile(train_d, quantile))
    d = np.sqrt(((x[:, None, :] - x_train[None, :, :]) ** 2).sum(-1))
    if self_index is not None:
        rows = np.flatnonzero(np.asarray(self_index) >= 0)
        d[rows, np.asarray(self_index)[rows]] = np.inf
    return np.sort(d, axis=1)[:, :k].mean(1), float(np.quantile(train_d, quantile))


def classification_metrics(y_true, y_pred, positive=None, score=None, classes=None):
    """Accuracy, balanced accuracy, MCC, sensitivity, specificity (macro for >2 classes) and ROC AUC."""
    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = list(classes) if classes is not None else sorted(set(y_true) | set(y_pred))
    out = {"accuracy": float((y_true == y_pred).mean()) if len(y_true) else np.nan}
    if len(y_true) == 0:
        return dict.fromkeys(("accuracy", "balanced_accuracy", "mcc", "sensitivity", "specificity", "roc_auc"),
                             np.nan)
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    if len(classes) == 2 and positive is not None:
        pos = y_true == positive
        out["sensitivity"] = float((y_pred[pos] == positive).mean()) if pos.any() else np.nan
        out["specificity"] = float((y_pred[~pos] != positive).mean()) if (~pos).any() else np.nan
    else:
        sens, spec = [], []
        for c in classes:
            pos = y_true == c
            if pos.any():
                sens.append((y_pred[pos] == c).mean())
            if (~pos).any():
                spec.append((y_pred[~pos] != c).mean())
        out["sensitivity"] = float(np.mean(sens)) if sens else np.nan
        out["specificity"] = float(np.mean(spec)) if spec else np.nan
    out["roc_auc"] = np.nan
    if score is not None and len(set(y_true)) > 1:
        score = np.asarray(score, dtype=float)
        if score.ndim == 1:
            out["roc_auc"] = float(roc_auc_score(y_true == positive, score))
        elif len(set(y_true)) == len(classes):
            out["roc_auc"] = float(roc_auc_score(y_true, score, multi_class="ovr", labels=classes))
    return out
