"""Figures (matplotlib Agg, ~150 dpi). Each function takes ready data and a path and only draws."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 150
TYPE_STYLE = {"sample": ("tab:blue", "o"), "qc": ("tab:red", "*"), "replicate": ("tab:cyan", "s"),
              "blank": ("0.4", "x"), "conditioning": ("tab:orange", "^"), "other": ("tab:purple", "v")}
TYPE_PL = {"sample": "próbka", "qc": "QC", "replicate": "replikat", "blank": "blank", "conditioning": "kondycj. QC",
           "other": "inne"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _short(s, n=28):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _batch_lines(ax, samples):
    starts = samples.groupby("_batch")["_order"].min().sort_values().to_numpy()[1:]
    for x in starts:
        ax.axvline(x - 0.5, ls="--", c="k", lw=0.6)


def injections(flags, samples, path):
    s = samples.loc[flags.index]
    panels = [("na_frac", "odsetek braków"), ("log2_total", "log2 suma intensywności")]
    if flags["is_log2"].notna().any():
        panels.append(("is_log2", "średni log2 wzorców wewn."))
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.4 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (col, title) in zip(axes, panels, strict=True):
        for t, (c, m) in TYPE_STYLE.items():
            idx = s.index[s._type == t]
            if len(idx):
                ax.scatter(s.loc[idx, "_order"], flags.loc[idx, col], c=c, marker=m, s=14, label=TYPE_PL[t])
        bad = flags.index[flags.rejected.astype(bool)]
        for b in bad:
            ax.annotate(str(b), (s.loc[b, "_order"], flags.loc[b, col]), fontsize=7, color="red")
        ax.set_ylabel(title, fontsize=8)
        _batch_lines(ax, s)
    axes[0].legend(fontsize=7, ncol=6, loc="best")
    axes[0].set_title("Kontrola iniekcji (czerwone podpisy = odrzucone)")
    axes[-1].set_xlabel("kolejność iniekcji")
    _save(fig, path)


def qc_drift(L_before, L_after, samples, mask, cv_before, cv_after, path):
    s = samples.loc[L_before.columns]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [2, 2, 1.3]})
    for ax, L, title in [(axes[0], L_before, "przed: surowe log2"), (axes[1], L_after, "po normalizacji i korekcie")]:
        med = L.where(mask).median()
        for t in ("sample", "replicate", "qc"):
            idx = s.index[s._type == t]
            if len(idx):
                c, m = TYPE_STYLE[t]
                ax.scatter(s.loc[idx, "_order"], med[idx], c=c, marker=m, s=18 if t == "qc" else 10,
                           label=TYPE_PL[t], zorder=3 if t == "qc" else 2)
        _batch_lines(ax, s)
        ax.set_title(f"Mediana log2 iniekcji – {title}", fontsize=9)
        ax.set_xlabel("kolejność iniekcji")
    axes[0].legend(fontsize=7)
    a, b = cv_before.dropna(), cv_after.dropna()
    if len(a):
        bins = np.linspace(0, min(1.0, max(a.quantile(0.98), b.quantile(0.98) if len(b) else 0) + 0.02), 40)
        axes[2].hist(a.clip(upper=bins[-1]), bins=bins, alpha=0.6, label=f"przed (mediana {a.median():.1%})")
        axes[2].hist(b.clip(upper=bins[-1]), bins=bins, alpha=0.6, label=f"po (mediana {b.median():.1%})")
        axes[2].legend(fontsize=7)
    else:
        axes[2].text(0.5, 0.5, "brak ≥3 QC", ha="center", transform=axes[2].transAxes)
    axes[2].set_title("CV cech w QC", fontsize=9)
    axes[2].set_xlabel("CV")
    _save(fig, path)


def _scatter_pca(ax, scores, s, color_by, title, explained):
    if color_by == "_order":
        sc = ax.scatter(scores.iloc[:, 0], scores.iloc[:, 1], c=s.loc[scores.index, "_order"], cmap="viridis", s=14)
        plt.colorbar(sc, ax=ax, label="kolejność")
    else:
        for i, (b, idx) in enumerate(s.loc[scores.index].groupby("_batch").groups.items()):
            ax.scatter(scores.loc[idx, scores.columns[0]], scores.loc[idx, scores.columns[1]], s=14,
                       color=plt.cm.tab10(i % 10), label=f"partia {b}")
        ax.legend(fontsize=7)
    qc = [i for i in scores.index if s.loc[i, "_type"] == "qc"]
    ax.scatter(scores.loc[qc, scores.columns[0]], scores.loc[qc, scores.columns[1]], s=60, facecolors="none",
               edgecolors="red", marker="*", label="QC")
    ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%})" if len(explained) > 1 else "PC2")
    ax.set_title(title, fontsize=9)


def pca_before_after(p_before, p_after, samples, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for row, (p, lab) in enumerate([(p_before, "przed korektą"), (p_after, "po korekcie")]):
        if p["scores"].shape[1] < 2:
            continue
        _scatter_pca(axes[row, 0], p["scores"], samples, "_batch", f"PCA {lab}: partie (gwiazdki = QC)",
                     p["explained"])
        _scatter_pca(axes[row, 1], p["scores"], samples, "_order", f"PCA {lab}: kolejność iniekcji",
                     p["explained"])
    _save(fig, path)


def volcano(res, levels, labels, fdr, fc, path, n_label=10):
    comps = [(None, levels[1])] if len(levels) == 2 else [(g, g) for g in levels[1:5]]
    fig, axes = plt.subplots(1, len(comps), figsize=(6 * len(comps), 5), squeeze=False)
    for ax, (g, name) in zip(axes[0], comps, strict=True):
        x = res["log2FC"] if g is None else res[f"log2FC_{g}"]
        p = res["p"] if g is None else res[f"p_{g}"]
        q = res["q"] if g is None else res[f"q_{g}"]
        y = -np.log10(p.clip(lower=1e-300))
        sig = (q < fdr) & (x.abs() >= fc)
        sig_small = (q < fdr) & ~sig
        ax.scatter(x[~(q < fdr)], y[~(q < fdr)], s=5, c="lightgray")
        ax.scatter(x[sig_small], y[sig_small], s=7, c="tab:orange", label=f"q<{fdr:g} (n={int(sig_small.sum())})")
        up, down = sig & (x > 0), sig & (x < 0)
        ax.scatter(x[up], y[up], s=9, c="tab:red", label=f"q<{fdr:g}, log2FC≥{fc:g} (n={int(up.sum())})")
        ax.scatter(x[down], y[down], s=9, c="tab:blue", label=f"q<{fdr:g}, log2FC≤-{fc:g} (n={int(down.sum())})")
        if fc > 0:
            for v in (fc, -fc):
                ax.axvline(v, ls=":", c="k", lw=0.7)
        top = q[q < fdr].sort_values().index[:n_label]
        for f in top:
            ax.annotate(_short(labels.get(f, f), 18), (x[f], y[f]), fontsize=7)
        ax.set_xlabel(f"log2FC ({name} vs {levels[0]})")
        ax.set_ylabel("-log10(p)")
        ax.set_title(f"Volcano: {name} vs {levels[0]}", fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
    _save(fig, path)


def heatmap(L, groups, top, labels, path):
    import seaborn as sns
    Z = L.loc[top, groups.index]
    Z = Z.sub(Z.mean(axis=1), axis=0).div(Z.std(axis=1).replace(0, 1), axis=0).clip(-3, 3)
    Z = Z.fillna(0)
    Z.index = [_short(labels.get(f, f), 30) for f in Z.index]
    lev = sorted(groups.unique())
    cmap = plt.cm.tab10 if len(lev) <= 10 else plt.cm.tab20
    pal = {g: cmap(i % cmap.N) for i, g in enumerate(lev)}
    colors = groups.map(pal).rename("grupa")
    g = sns.clustermap(Z, cmap="RdBu_r", center=0, col_colors=colors, method="average", metric="euclidean",
                       xticklabels=len(Z.columns) <= 60, yticklabels=True, figsize=(min(16, 4 + 0.12 * Z.shape[1]),
                                                                                    min(16, 3 + 0.22 * len(Z))),
                       cbar_kws={"label": "z-score"}, col_cluster=Z.shape[1] > 2, row_cluster=Z.shape[0] > 2)
    g.ax_heatmap.tick_params(axis="y", labelsize=7)
    g.ax_heatmap.tick_params(axis="x", labelsize=6)
    for gname in lev:
        g.ax_col_dendrogram.bar(0, 0, color=pal[gname], label=gname, linewidth=0)
    g.ax_col_dendrogram.legend(ncol=min(6, len(lev)), fontsize=7, loc="center")
    g.fig.suptitle(f"Top {len(Z)} cech (z-score, klastrowanie hierarchiczne)", y=1.0, fontsize=10)
    g.savefig(path, dpi=DPI)
    plt.close(g.fig)


def pca_groups(p, groups, labels, path, n_load=10):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc = p["scores"].loc[groups.index]
    ex = p["explained"]
    for i, (g, idx) in enumerate(groups.groupby(groups).groups.items()):
        axes[0].scatter(sc.loc[idx, "PC1"], sc.loc[idx, "PC2"] if "PC2" in sc else 0, s=18,
                        color=plt.cm.tab10(i % 10), label=f"{g} (n={len(idx)})")
    axes[0].set_xlabel(f"PC1 ({ex[0]:.1%})")
    axes[0].set_ylabel(f"PC2 ({ex[1]:.1%})" if len(ex) > 1 else "")
    axes[0].legend(fontsize=7)
    axes[0].set_title("PCA próbek (autoskalowanie)", fontsize=10)
    ld = p["loadings"]
    if "PC2" in ld:
        mag = np.hypot(ld["PC1"], ld["PC2"])
        axes[1].scatter(ld["PC1"], ld["PC2"], s=3, c="lightgray")
        for f in mag.sort_values(ascending=False).index[:n_load]:
            axes[1].annotate("", (ld.loc[f, "PC1"], ld.loc[f, "PC2"]), (0, 0),
                             arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 0.8})
            axes[1].annotate(_short(labels.get(f, f), 18), (ld.loc[f, "PC1"], ld.loc[f, "PC2"]), fontsize=7)
        axes[1].set_xlabel("ładunek PC1")
        axes[1].set_ylabel("ładunek PC2")
    axes[1].set_title(f"Ładunki: {n_load} cech o największym wkładzie", fontsize=10)
    _save(fig, path)


def plsda(out, groups, labels, path, n_vip=15):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sc = out["scores"]
    for i, (g, idx) in enumerate(groups.groupby(groups).groups.items()):
        axes[0].scatter(sc.loc[idx, "t1"], sc.loc[idx, "t2"] if "t2" in sc else np.zeros(len(idx)), s=18,
                        color=plt.cm.tab10(i % 10), label=g)
    axes[0].set_xlabel("t1")
    axes[0].set_ylabel("t2")
    axes[0].legend(fontsize=7)
    axes[0].set_title(f"PLS-DA: {out['n_components']} skł., dokładność CV {out['accuracy']:.0%}, "
                      f"Q² {out['q2']:.2f}", fontsize=9)
    vip = out["vip"].sort_values(ascending=False)[:n_vip][::-1]
    axes[1].barh([_short(labels.get(f, f), 22) for f in vip.index], vip.values, color="tab:green")
    axes[1].axvline(1, ls=":", c="k")
    axes[1].tick_params(axis="y", labelsize=7)
    axes[1].set_title("VIP (najwyższe)", fontsize=9)
    if len(out["perm_scores"]):
        axes[2].hist(out["perm_scores"], bins=20, color="0.7", label="permutacje")
        axes[2].axvline(out["accuracy"], c="tab:red", label=f"model ({out['accuracy']:.0%})")
        axes[2].legend(fontsize=7)
        axes[2].set_title(f"Test permutacyjny: p={out['perm_p']:.3g} ({out['n_perm']} perm.)", fontsize=9)
        axes[2].set_xlabel("dokładność CV")
    _save(fig, path)


def boxplots(L, groups, top, labels, path):
    n = len(top)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    lev = sorted(groups.unique())
    for ax, f in zip(axes.flat, top, strict=False):
        data = [L.loc[f, groups.index[groups == g]].dropna().to_numpy() for g in lev]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(lev) + 1), [_short(g, 12) for g in lev], fontsize=7)
        rng = np.random.default_rng(0)
        for i, d in enumerate(data):
            ax.scatter(i + 1 + rng.uniform(-0.15, 0.15, len(d)), d, s=6, alpha=0.6)
        ax.set_title(_short(labels.get(f, f), 30), fontsize=9)
        ax.set_ylabel("log2")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    _save(fig, path)



# ---------------------------------------------------------------- lipidomics / untargeted LC-MS

def _categories(cats, index, top=9):
    """Category per point with the rare ones folded into 'inne' and the unlabelled ones last ('bez nazwy')."""
    c = cats.reindex(index).astype(object)
    c = c.where(c.notna(), "bez nazwy")
    counts = c[c != "bez nazwy"].value_counts()
    keep = list(counts.index[:top])
    c = c.where(c.isin(keep + ["bez nazwy"]), "inne")
    order = keep + [k for k in ("inne", "bez nazwy") if (c == k).any()]
    return c, order


def _scatter_by_category(ax, x, y, cats, mark=None):
    if cats is None:
        ax.scatter(x, y, s=8, c="tab:blue", alpha=0.6, linewidths=0)
    else:
        c, order = _categories(cats, x.index)
        for i, k in enumerate(order):
            idx = c.index[c == k]
            color = "0.75" if k in ("bez nazwy", "nieistotne") else "0.45" if k == "inne" else plt.cm.tab10(i % 10)
            ax.scatter(x[idx], y[idx], s=10, color=color, alpha=0.8, linewidths=0, label=f"{k} ({len(idx)})",
                       zorder=1 if k in ("bez nazwy", "nieistotne") else 2)
        ax.legend(fontsize=7, loc="best", ncol=2 if len(order) > 6 else 1)
    if mark is not None and len(mark):
        idx = [f for f in mark if f in x.index]
        ax.scatter(x[idx], y[idx], s=34, facecolors="none", edgecolors="k", linewidths=0.8, zorder=3,
                   label="istotne")


def rt_mz(rt, mz, cats, path, mark=None, mass_label="m/z", title="RT vs m/z"):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ok = rt.notna() & mz.notna()
    _scatter_by_category(ax, rt[ok], mz[ok], cats, mark)
    ax.set_xlabel("czas retencji [min]")
    ax.set_ylabel(mass_label)
    ax.set_title(title + (" (obwódka = istotne)" if mark is not None and len(mark) else ""), fontsize=10)
    _save(fig, path)


def kendrick(nominal, kmd, cats, path, mark=None):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _scatter_by_category(ax, nominal, kmd, cats, mark)
    ax.set_xlabel("nominalna masa Kendricka (baza CH₂)")
    ax.set_ylabel("defekt masy Kendricka (KMD)")
    ax.set_title("Wykres Kendricka: szeregi homologiczne (+CH₂) leżą na poziomych liniach", fontsize=10)
    _save(fig, path)


def van_krevelen(vk, cats, path, mark=None):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _scatter_by_category(ax, vk["O_C"], vk["H_C"], cats, mark)
    ax.set_xlabel("O/C")
    ax.set_ylabel("H/C")
    ax.set_title(f"Diagram Van Krevelena ({len(vk)} cech ze wzorem sumarycznym)", fontsize=10)
    _save(fig, path)


def venn(counts, groups, path, title="Cechy obecne w grupach"):
    """Two or three circles with the number of features in every region."""
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(6, 5.2))
    n = len(groups)
    centres = [(-0.5, 0.0), (0.5, 0.0)] if n == 2 else [(-0.5, 0.3), (0.5, 0.3), (0.0, -0.55)]
    for i, (g, (cx, cy)) in enumerate(zip(groups, centres, strict=True)):
        ax.add_patch(Circle((cx, cy), 0.9, color=plt.cm.tab10(i), alpha=0.3))
        total = sum(v for k, v in counts.items() if g in k)
        lx, ly = (cx * 2.6, cy + 0.95) if n == 2 or i < 2 else (cx, cy - 1.05)
        ax.text(lx, ly, f"{_short(g, 16)}\n(n={total})", ha="center", va="center", fontsize=10)
    from itertools import combinations
    regions = [k for r in range(1, n + 1) for k in combinations(groups, r)]
    for key in regions:
        v = counts.get(key, 0)
        pts = np.array([centres[groups.index(g)] for g in key])
        x, y = pts.mean(axis=0)
        if len(key) == 1:
            x, y = x * 1.7, (y * 1.5 if n == 3 else y)
        elif n == 3 and len(key) == 2:
            x, y = x * 1.5, y * 1.5 + (0.12 if y > 0 else 0)
        ax.text(x, y, str(v), ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.9, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    _save(fig, path)


def is_drift(istd, samples, path, max_panels=8):
    """Signal of every internal standard along the injection order (log2), coloured by sample type."""
    s = samples.loc[[c for c in istd.columns if c in samples.index]]
    rows = list(istd.index[:max_panels])
    cols = min(2, len(rows))
    nr = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(nr, cols, figsize=(6.5 * cols, 2.6 * nr), squeeze=False, sharex=True)
    for ax, name in zip(axes.flat, rows, strict=False):
        y = np.log2(istd.loc[name, s.index].astype(float))
        for t, (c, m) in TYPE_STYLE.items():
            idx = s.index[s._type == t]
            if len(idx):
                ax.scatter(s.loc[idx, "_order"], y[idx], c=c, marker=m, s=14, label=TYPE_PL[t])
        study = s.index[s._type.isin(["sample", "qc", "replicate"])]
        lin = istd.loc[name, study].astype(float)
        rsd = lin.std() / lin.mean() if lin.notna().sum() > 2 else np.nan
        ax.axhline(y[study].median(), ls=":", c="k", lw=0.6)
        _batch_lines(ax, s)
        ax.set_title(f"{_short(name, 40)} – RSD {rsd:.0%}", fontsize=8)
        ax.set_ylabel("log2", fontsize=8)
    for ax in list(axes.flat)[len(rows):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=6, ncol=3)
    for ax in axes[-1]:
        ax.set_xlabel("kolejność iniekcji")
    fig.suptitle("Dryf wzorców wewnętrznych w kolejności iniekcji", fontsize=10)
    _save(fig, path)


def class_composition(shares, table, path, fdr=0.05, what="klas lipidów"):
    """Stacked bars of the mean signal share per group and, when tested, the log2 fold change of each class."""
    has_fc = table is not None and "log2FC" in table and table["log2FC"].notna().any()
    fig, axes = plt.subplots(1, 2 if has_fc else 1, figsize=(11 if has_fc else 6, 5), squeeze=False)
    ax = axes[0, 0]
    order = shares.mean(axis=1).sort_values(ascending=False).index
    bottom = np.zeros(shares.shape[1])
    for i, k in enumerate(order):
        v = shares.loc[k].to_numpy(float) * 100
        ax.bar([_short(g, 14) for g in shares.columns], v, bottom=bottom, color=plt.cm.tab20(i % 20), label=k)
        for j, h in enumerate(v):
            if h >= 6:
                ax.text(j, bottom[j] + h / 2, f"{h:.0f}%", ha="center", va="center", fontsize=7)
        bottom += np.nan_to_num(v)
    ax.set_ylabel("udział w sumie sygnału lipidów [%]")
    ax.set_title(f"Skład {what} w grupach (średnia)", fontsize=10)
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    if has_fc:
        ax = axes[0, 1]
        t = table.loc[[k for k in order if k in table.index]].iloc[::-1]
        q = t["q"] if "q" in t else None
        colors = ["tab:red" if (q is not None and q[k] < fdr and t.loc[k, "log2FC"] > 0) else
                  "tab:blue" if (q is not None and q[k] < fdr) else "0.7" for k in t.index]
        ax.barh(list(t.index), t["log2FC"], color=colors)
        ax.axvline(0, c="k", lw=0.6)
        ax.set_xlabel("log2FC sumy klasy")
        ax.set_title(f"Zmiana sumy sygnału klasy (kolor = q < {fdr:g})", fontsize=10)
    _save(fig, path)


def lipid_distributions(dist, path):
    """Chain length and double bonds of the significantly changed lipids against all measured ones."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (key, title) in zip(axes, (("dlugosc_lancuchow", "łączna długość łańcuchów (C)"),
                                       ("wiazania_podwojne", "wiązania podwójne")), strict=True):
        d = dist[dist.cecha == key]
        x = np.arange(len(d))
        ax.bar(x, d["wszystkie"], color="0.85", label="wszystkie lipidy")
        ax.bar(x - 0.2, d["w_gore"], width=0.4, color="tab:red", label="istotnie wyższe")
        ax.bar(x + 0.2, d["w_dol"], width=0.4, color="tab:blue", label="istotnie niższe")
        ax.set_xticks(x, d["wartosc"].astype(int), fontsize=7, rotation=90 if len(d) > 15 else 0)
        ax.set_xlabel(title)
        ax.set_ylabel("liczba lipidów")
    axes[0].legend(fontsize=7)
    fig.suptitle("Rozkład długości łańcuchów i nasycenia istotnych lipidów", fontsize=10)
    _save(fig, path)
