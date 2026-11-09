"""Figures (matplotlib Agg, ~150 dpi). Each function takes data and a path and writes one PNG."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TRAIN, TEST, PRED = "#1f77b4", "#ff7f0e", "#2ca02c"
DPI = 150


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def obs_vs_pred(path, title, y_train, p_train, y_test, p_test, text, ylabel="y"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_train, p_train, c=TRAIN, label=f"uczący (n={len(y_train)})", edgecolor="k", lw=0.4)
    if len(y_test):
        ax.scatter(y_test, p_test, c=TEST, marker="s", label=f"testowy (n={len(y_test)})", edgecolor="k", lw=0.4)
    allv = np.concatenate([y_train, p_train, y_test, p_test])
    lo, hi = np.nanmin(allv), np.nanmax(allv)
    pad = 0.05 * (hi - lo or 1)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"{ylabel} obserwowane")
    ax.set_ylabel(f"{ylabel} przewidziane")
    ax.set_title(title)
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", fontsize=8,
            bbox={"boxstyle": "round", "fc": "white", "alpha": 0.8})
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def williams(path, title, h, std_res, sets, names, h_star):
    """Leverage vs standardised residual; dashed lines at h* and ±3."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for s, c, m, lab in (("uczący", TRAIN, "o", "uczący"), ("testowy", TEST, "s", "testowy")):
        sel = sets == s
        ax.scatter(h[sel], std_res[sel], c=c, marker=m, label=lab, edgecolor="k", lw=0.4)
    ax.axvline(h_star, color="r", ls="--", lw=1, label=f"h* = {h_star:.2f}")
    for v in (3, -3):
        ax.axhline(v, color="gray", ls="--", lw=1)
    for hi, ri, nm in zip(h, std_res, names, strict=True):
        if np.isfinite(hi) and np.isfinite(ri) and (hi > h_star or abs(ri) > 3):
            ax.annotate(str(nm)[:25], (hi, ri), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("dźwignia h")
    ax.set_ylabel("reszta standaryzowana")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, path)


def scrambling(path, x_perm, a_perm, b_perm, a0, b0, a_label="R²", b_label="Q²", x_label="|r(y, y permutowane)|"):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(x_perm, a_perm, c=TRAIN, s=15, label=f"{a_label} po permutacji")
    ax.scatter(x_perm, b_perm, c=TEST, s=15, marker="s", label=f"{b_label} po permutacji")
    ax.scatter([1], [a0], c=TRAIN, s=90, marker="*", edgecolor="k", label=f"{a_label} modelu")
    ax.scatter([1], [b0], c=TEST, s=90, marker="*", edgecolor="k", label=f"{b_label} modelu")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{a_label}, {b_label}")
    ax.set_title(f"Y-scrambling ({len(x_perm)} permutacji)")
    low = np.nanmin(b_perm) if len(b_perm) else 0
    ax.set_ylim(max(-1.5, min(-0.2, low - 0.05)), 1.05)
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, path)


def corr_heatmap(path, corr, labels):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.0 + 0.6 * n, 0.8 + 0.6 * n))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n), [str(lab)[:18] for lab in labels], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), [str(lab)[:18] for lab in labels], fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Korelacje (zbiór uczący)")
    _save(fig, path)


def pca(path, scores, var, sets, names, color_values=None, color_label="y"):
    fig, axes = plt.subplots(1, 2 if color_values is not None else 1, figsize=(10 if color_values is not None else 5.5,
                                                                              4.5))
    axes = np.atleast_1d(axes)
    ax = axes[0]
    for s, c, m in (("uczący", TRAIN, "o"), ("testowy", TEST, "s"), ("prognoza", PRED, "^")):
        sel = sets == s
        if sel.any():
            ax.scatter(scores[sel, 0], scores[sel, 1], c=c, marker=m, label=s, edgecolor="k", lw=0.4)
    for i in np.flatnonzero(sets == "prognoza")[:15]:
        ax.annotate(str(names[i])[:20], scores[i, :2], fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({100 * var[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({100 * var[1]:.0f}%)")
    ax.set_title("Przestrzeń chemiczna (PCA)")
    ax.legend(fontsize=8)
    if color_values is not None:
        ax = axes[1]
        ok = np.isfinite(color_values)
        sc = ax.scatter(scores[ok, 0], scores[ok, 1], c=color_values[ok], cmap="viridis", edgecolor="k", lw=0.4)
        fig.colorbar(sc, ax=ax, label=color_label)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA, kolor = {color_label}")
    _save(fig, path)


def confusion(path, title, cm, classes):
    fig, ax = plt.subplots(figsize=(1.8 + 0.8 * len(classes), 1.5 + 0.8 * len(classes)))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_xticks(range(len(classes)), [str(c)[:15] for c in classes], rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), [str(c)[:15] for c in classes])
    ax.set_xlabel("przewidziana")
    ax.set_ylabel("prawdziwa")
    ax.set_title(title)
    _save(fig, path)


def roc(path, title, curves):
    """curves: list of (label, fpr, tpr, auc)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, fpr, tpr, auc in curves:
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("1 − specyficzność")
    ax.set_ylabel("czułość")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, path)
