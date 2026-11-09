"""Unsupervised and correlation pre-filtering of descriptors, and autoscaling. Training set only."""
from __future__ import annotations

import numpy as np

NEAR_CONSTANT = 0.9  # drop a descriptor when more than this share of training compounds has the same value


def relevance(x, y, task="regression"):
    """|r| with y for regression and binary classes; ANOVA F for more classes."""
    x = np.asarray(x, float)
    if task == "classification":
        classes = np.unique(y)
        if len(classes) > 2:
            from sklearn.feature_selection import f_classif
            f, _ = f_classif(x, y)
            return np.nan_to_num(f)
        y = (np.asarray(y) == classes[1]).astype(float)
    y = np.asarray(y, float)
    xc, yc = x - x.mean(0), y - y.mean()
    den = np.sqrt((xc ** 2).sum(0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (xc * yc[:, None]).sum(0) / den
    return np.nan_to_num(np.abs(r))


def prefilter(x_train, y_train, corr_max=0.95, near_constant=NEAR_CONSTANT, task="regression"):
    """Return (kept column names, report). x_train: DataFrame of training compounds without missing values.

    1. constant / near-constant descriptors out;
    2. of every pair with |r| > corr_max keep the one more related to y (greedy, in order of relevance).
    """
    names = list(x_train.columns)
    x = x_train.to_numpy(dtype=float)
    n = len(x)
    keep = []
    for j in range(x.shape[1]):
        col = x[:, j]
        _, counts = np.unique(col, return_counts=True)
        if np.std(col) <= 1e-12 * max(1.0, np.abs(col).max()) or counts.max() / n > near_constant:
            continue
        keep.append(j)
    n_const = x.shape[1] - len(keep)
    rel = relevance(x[:, keep], y_train, task) if keep else np.array([])
    order = [keep[i] for i in np.argsort(-rel, kind="stable")]
    z = x[:, order]
    z = (z - z.mean(0)) / z.std(0)
    corr = np.abs(z.T @ z / n) if len(order) else np.zeros((0, 0))
    kept_pos = []
    for i in range(len(order)):
        if not kept_pos or corr[i, kept_pos].max() <= corr_max:
            kept_pos.append(i)
    kept = [names[order[i]] for i in kept_pos]
    report = {"start": len(names), "constant": n_const, "correlated": len(order) - len(kept), "kept": len(kept)}
    return kept, report


def autoscale(x_train, *others):
    """(x - mean_train) / sd_train (ddof=1) for the training block and any other blocks."""
    x_train = np.asarray(x_train, float)
    mu = x_train.mean(0)
    sd = x_train.std(0, ddof=1)
    sd[sd == 0] = 1.0
    return [(np.asarray(o, float) - mu) / sd for o in (x_train, *others)], mu, sd
