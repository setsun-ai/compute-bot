"""Per-feature tests with FDR, PCA and PLS-DA (VIP, cross-validation, permutation test) on log2 data.

`L` is features x samples (log2), `groups` a Series sample -> group label for the samples to compare.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as ss

TWO_GROUP = ("ttest", "welch", "mwu")
PAIRED = ("paired_t", "wilcoxon")


def bh(p):
    """Benjamini-Hochberg q-values; NaN p stays NaN and does not count as a test."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if not m:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def _filled(x):
    return np.asarray(np.ma.filled(np.ma.asarray(x, dtype=float), np.nan), float)


def _pair_test(A, B, test):
    """p-values of A (features x n1) vs B for every feature."""
    if test in ("ttest", "welch"):
        r = ss.ttest_ind(A, B, axis=1, equal_var=(test == "ttest"), nan_policy="omit")
    elif test == "mwu":
        r = ss.mannwhitneyu(A, B, axis=1, nan_policy="omit", alternative="two-sided")
    elif test == "anova":
        return _omnibus([A, B], "anova")
    elif test == "kruskal":
        return _omnibus([A, B], "kruskal")
    else:
        raise ValueError(test)
    return _filled(r.pvalue)


def _omnibus(arrays, test):
    fn = ss.f_oneway if test == "anova" else ss.kruskal
    if not any(np.isnan(a).any() for a in arrays):
        try:
            return _filled(fn(*arrays, axis=1).pvalue)
        except ValueError:
            pass
    p = np.full(arrays[0].shape[0], np.nan)
    for i in range(len(p)):
        rows = [a[i][~np.isnan(a[i])] for a in arrays]
        rows = [r for r in rows if len(r) >= 2]
        if len(rows) >= 2:
            try:
                p[i] = fn(*rows).pvalue
            except ValueError:
                pass
    return p


def _design(groups, levels, covariates):
    D = pd.DataFrame({"const": 1.0}, index=groups.index)
    gcols = []
    for g in levels[1:]:
        name = f"grp_{g}"
        D[name] = (groups == g).astype(float)
        gcols.append(name)
    if covariates is not None and len(covariates.columns):
        cov = covariates.reindex(groups.index)
        for c in cov.columns:
            col = cov[c]
            num = pd.to_numeric(col, errors="coerce")
            if num.notna().sum() == col.notna().sum() and col.notna().any():
                D[str(c)] = num.astype(float)
            else:
                lv = sorted(col.dropna().astype(str).unique())
                for v in lv[1:]:
                    D[f"{c}_{v}"] = (col.astype(str) == v).astype(float).where(col.notna())
    return D, gcols


def _ols(Y, D, gcols):
    """Vectorised OLS for all features (complete Y). Returns coef, p per column and the F-test p for gcols."""
    X = D.to_numpy(float)
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    B = Y @ X @ XtXi                       # features x k
    resid = Y - B @ X.T
    df = n - k
    rss = (resid ** 2).sum(axis=1)
    s2 = rss / df
    se = np.sqrt(np.outer(s2, np.diag(XtXi)))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = B / se
    p = 2 * ss.t.sf(np.abs(t), df)
    idx = [list(D.columns).index(g) for g in gcols]
    Xr = np.delete(X, idx, axis=1)
    Br = Y @ Xr @ np.linalg.pinv(Xr.T @ Xr)
    rss_r = ((Y - Br @ Xr.T) ** 2).sum(axis=1)
    q = len(idx)
    with np.errstate(divide="ignore", invalid="ignore"):
        F = ((rss_r - rss) / q) / (rss / df)
    pf = ss.f.sf(F, q, df)
    return B, p, pf


def _ols_statsmodels(y, D, gcols):
    import statsmodels.api as sm
    ok = np.isfinite(y)
    if ok.sum() <= D.shape[1] + 1:
        return None
    Dk = D[ok]
    keep = [c for c in D.columns if c == "const" or Dk[c].std() > 0]
    gk = [g for g in gcols if g in keep]
    if not gk:
        return None
    fit = sm.OLS(y[ok], Dk[keep].to_numpy(float)).fit()
    coef = pd.Series(fit.params, index=keep)
    pv = pd.Series(fit.pvalues, index=keep)
    R = np.zeros((len(gk), len(keep)))
    for i, g in enumerate(gk):
        R[i, keep.index(g)] = 1
    pf = float(fit.f_test(R).pvalue)
    return coef, pv, pf


def complete_pairs(groups, pairs, levels):
    """Pair id -> (sample in levels[0], sample in levels[1]) for pairs with both members, and the ids of the
    incomplete ones (a member missing, or more than one sample of a pair in a group)."""
    p = pairs.reindex(groups.index)
    sides = []
    for g in levels:
        side = {}
        for c in groups.index[(groups == g) & p.notna()]:
            side.setdefault(p[c], []).append(c)
        sides.append(side)
    ids = sorted(set(sides[0]) | set(sides[1]), key=str)
    full = {k: (sides[0][k][0], sides[1][k][0]) for k in ids
            if len(sides[0].get(k, [])) == 1 and len(sides[1].get(k, [])) == 1}
    return full, [k for k in ids if k not in full]


def _paired_tests(L, groups, pairs, levels, test):
    """Paired t-test or Wilcoxon signed-rank on the within-pair differences (levels[1] - levels[0])."""
    full, dropped = complete_pairs(groups, pairs, levels)
    if len(full) < 2:
        raise ValueError("need at least two complete pairs")
    A = L[[a for a, _ in full.values()]].to_numpy(float)
    B = L[[b for _, b in full.values()]].to_numpy(float)
    D = B - A
    res = pd.DataFrame(index=L.index)
    with np.errstate(invalid="ignore"):
        for g, M in zip(levels, (A, B), strict=True):
            res[f"mean_{g}"] = np.nanmean(np.where(np.isfinite(D), M, np.nan), axis=1)
            res[f"n_{g}"] = np.isfinite(D).sum(axis=1)
        res["log2FC"] = np.nanmean(D, axis=1)
        if test == "paired_t":
            p = _filled(ss.ttest_rel(B, A, axis=1, nan_policy="omit").pvalue)
        else:
            p = np.full(len(L), np.nan)
            ok = (np.isfinite(D) & (D != 0)).sum(axis=1) >= 1
            if ok.any():
                p[ok] = _filled(ss.wilcoxon(D[ok], axis=1, nan_policy="omit").pvalue)
    res["p"] = p
    res.attrs.update(n_pairs=len(full), dropped_pairs=len(dropped), dropped=[str(k) for k in dropped],
                     n_used=2 * len(full), pair_samples=[c for ab in full.values() for c in ab])
    return res


def collapse_replicates(L, units, groups):
    """Mean (on the log scale) of the injections of one biological sample. `units`: injection -> unit label.
    Returns (features x units, group per unit, info with n_injections / n_units / sizes)."""
    units = units.reindex(L.columns).dropna()
    order = list(dict.fromkeys(units))
    Lu = L[units.index].T.groupby(units.to_numpy(), sort=False).mean().T[order]
    gu = pd.Series({u: groups.reindex(units.index[units == u]).dropna().iloc[0]
                    if groups.reindex(units.index[units == u]).notna().any() else None for u in order})
    sizes = units.value_counts().reindex(order)
    return Lu, gu, {"n_injections": int(len(units)), "n_units": len(order), "sizes": sizes}


def feature_tests(L, groups, ref, test="auto", covariates=None, pairs=None):
    """One test per feature: log2 fold change vs `ref`, p, BH q (attrs['test'] = test used).

    auto: paired t-test when `pairs` (sample -> subject) are given, lm when covariates are given, Welch t-test for
    2 groups, one-way ANOVA for more. With > 2 groups the omnibus p (ANOVA / Kruskal-Wallis / F-test of the group
    terms) is the main p; pairwise p_<g>/q_<g> vs ref use Welch (or Mann-Whitney / model coefficients).
    Paired tests (paired_t, wilcoxon) need two groups; pairs without both members are left out (attrs n_pairs,
    dropped_pairs).
    """
    groups = groups.dropna().astype(str)
    levels = [ref] + sorted(g for g in groups.unique() if g != ref)
    if len(levels) < 2:
        raise ValueError("need two groups")
    if test == "auto" and pairs is not None and len(levels) == 2:
        test = "paired_t"
    if test in PAIRED:
        if pairs is None or len(levels) != 2:
            raise ValueError("paired tests need pairs and exactly two groups")
        res = _paired_tests(L, groups, pairs, levels, test)
        res["q"] = bh(res["p"])
        res.attrs["test"] = test
        res.attrs["levels"] = levels
        return res
    if test == "auto":
        test = "lm" if covariates is not None and len(covariates.columns) else ("welch" if len(levels) == 2
                                                                                 else "anova")
    L = L[groups.index]
    res = pd.DataFrame(index=L.index)
    arrays = {g: L.loc[:, groups == g].to_numpy(float) for g in levels}
    for g in levels:
        res[f"mean_{g}"] = np.nanmean(arrays[g], axis=1) if arrays[g].size else np.nan
        res[f"n_{g}"] = np.isfinite(arrays[g]).sum(axis=1)
    multi = len(levels) > 2
    if test == "lm":
        D, gcols = _design(groups, levels, covariates)
        rows = D.notna().all(axis=1)
        D = D[rows]
        Y = L.loc[:, D.index].to_numpy(float)
        coef = np.full((len(L), len(gcols)), np.nan)
        pc = np.full((len(L), len(gcols)), np.nan)
        pf = np.full(len(L), np.nan)
        complete = np.isfinite(Y).all(axis=1)
        if complete.any():
            B, P, PF = _ols(Y[complete], D, gcols)
            gi = [list(D.columns).index(g) for g in gcols]
            coef[complete] = B[:, gi]
            pc[complete] = P[:, gi]
            pf[complete] = PF
        for i in np.flatnonzero(~complete):
            r = _ols_statsmodels(Y[i], D, gcols)
            if r is None:
                continue
            c, pv, f = r
            coef[i] = [c.get(g, np.nan) for g in gcols]
            pc[i] = [pv.get(g, np.nan) for g in gcols]
            pf[i] = f
        if not multi:
            res["log2FC"] = coef[:, 0]
            res["p"] = pc[:, 0]
        else:
            for j, g in enumerate(levels[1:]):
                res[f"log2FC_{g}"] = coef[:, j]
                res[f"p_{g}"] = pc[:, j]
                res[f"q_{g}"] = bh(pc[:, j])
            res["p"] = pf
        res.attrs["n_used"] = int(len(D))
    else:
        pair = test if test in TWO_GROUP else ("mwu" if test == "kruskal" else "welch")
        if not multi:
            res["log2FC"] = res[f"mean_{levels[1]}"] - res[f"mean_{ref}"]
            res["p"] = _pair_test(arrays[levels[1]], arrays[ref], test)
        else:
            omni = "kruskal" if test in ("mwu", "kruskal") else "anova"
            for g in levels[1:]:
                res[f"log2FC_{g}"] = res[f"mean_{g}"] - res[f"mean_{ref}"]
                res[f"p_{g}"] = _pair_test(arrays[g], arrays[ref], pair)
                res[f"q_{g}"] = bh(res[f"p_{g}"])
            res["p"] = _omnibus([arrays[g] for g in levels], omni)
            test = omni if test in TWO_GROUP else test
    res["q"] = bh(res["p"])
    if multi:
        fcs = res[[f"log2FC_{g}" for g in levels[1:]]]
        res["max_abs_log2FC"] = fcs.abs().max(axis=1)
    res.attrs["test"] = test
    res.attrs["levels"] = levels
    return res


# ---------------------------------------------------------------- multivariate

def _autoscale(L):
    M = L.dropna(axis=0).T                 # samples x complete features
    sd = M.std(ddof=1)
    M = M.loc[:, sd > 0]
    return (M - M.mean()) / M.std(ddof=1)


def pca(L, n=5):
    """PCA on autoscaled (unit variance) log2 data of complete features."""
    from sklearn.decomposition import PCA
    Z = _autoscale(L)
    k = max(1, min(n, Z.shape[0] - 1, Z.shape[1]))
    p = PCA(n_components=k, random_state=0).fit(Z.to_numpy())
    cols = [f"PC{i + 1}" for i in range(k)]
    scores = pd.DataFrame(p.transform(Z.to_numpy()), index=Z.index, columns=cols)
    loadings = pd.DataFrame(p.components_.T, index=Z.columns, columns=cols)
    return {"scores": scores, "loadings": loadings, "explained": p.explained_variance_ratio_}


def _plsda_class():
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.cross_decomposition import PLSRegression

    class PLSDA(ClassifierMixin, BaseEstimator):
        """PLS regression on one-hot classes; the class with the highest prediction wins."""

        def __init__(self, n_components=2):
            self.n_components = n_components

        def fit(self, X, y):
            self.classes_ = np.unique(y)
            Y = (np.asarray(y)[:, None] == self.classes_[None, :]).astype(float)
            self.pls_ = PLSRegression(n_components=self.n_components, scale=False).fit(X, Y)
            return self

        def predict(self, X):
            return self.classes_[np.argmax(self.pls_.predict(X), axis=1)]

    return PLSDA


def vip_scores(pls):
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p = w.shape[0]
    s = (t ** 2).sum(axis=0) * (q ** 2).sum(axis=0)
    wn = w / np.linalg.norm(w, axis=0)
    return np.sqrt(p * (wn ** 2 @ s) / s.sum())


def plsda(L, groups, n_perm=200, threads=1, seed=0, max_comp=5):
    """PLS-DA with autoscaling inside the CV folds: Q2 per number of components, CV accuracy, permutation p, VIP."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, permutation_test_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    groups = groups.dropna().astype(str)
    M = L[groups.index].dropna(axis=0)
    M = M[M.std(axis=1) > 0]
    Xs = M.T.to_numpy(float)
    y = groups.to_numpy()
    classes = np.unique(y)
    Y = (y[:, None] == classes[None, :]).astype(float)
    k = int(min(5, pd.Series(y).value_counts().min()))
    if k < 2 or Xs.shape[1] < 2:
        return None
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    splits = list(cv.split(Xs, y))
    n_train = min(len(tr) for tr, _ in splits)
    top = max(1, min(max_comp, n_train - 1, Xs.shape[1]))
    q2 = {}
    for a in range(1, top + 1):
        reg = make_pipeline(StandardScaler(), PLSRegression(n_components=a, scale=False))
        pred = cross_val_predict(reg, Xs, Y, cv=splits)
        q2[a] = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(axis=0)) ** 2).sum()
    best = max(q2, key=q2.get)
    PLSDA = _plsda_class()
    clf = make_pipeline(StandardScaler(), PLSDA(n_components=best))
    acc = float(cross_val_score(clf, Xs, y, cv=splits, scoring="accuracy").mean())
    perm_p, perm_scores = np.nan, np.array([])
    if n_perm > 0:
        _, perm_scores, perm_p = permutation_test_score(clf, Xs, y, cv=splits, n_permutations=n_perm,
                                                        n_jobs=threads, random_state=seed, scoring="accuracy")
    show = min(max(2, best), Xs.shape[1], len(y) - 1)
    full = make_pipeline(StandardScaler(), PLSRegression(n_components=best, scale=False)).fit(Xs, Y)
    vip = pd.Series(vip_scores(full[-1]), index=M.index)
    disp = make_pipeline(StandardScaler(), PLSRegression(n_components=show, scale=False)).fit(Xs, Y)
    scores = pd.DataFrame(disp[-1].x_scores_[:, :2], index=groups.index,
                          columns=["t1", "t2"][: min(2, disp[-1].x_scores_.shape[1])])
    return {"n_components": best, "q2": float(q2[best]), "q2_all": q2, "accuracy": acc, "perm_p": float(perm_p),
            "perm_scores": perm_scores, "n_perm": int(n_perm), "vip": vip, "scores": scores, "folds": k,
            "classes": list(classes), "chance": float(pd.Series(y).value_counts(normalize=True).max())}
