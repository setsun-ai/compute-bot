"""Everything the owner sees: CSV tables, selected.txt, figures, summary.txt and highlights.txt."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import models as mdl
from . import plots, selection, stats

OUT = "wyniki"
SUMMARY_MAX = 2500
SET_TRAIN, SET_TEST, SET_PRED, SET_OUT = "uczący", "testowy", "prognoza", "wykluczony"


def progress(msg):
    print(f"PROGRESS: {msg}", flush=True)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def mlr_table(x_raw, y, names):
    """Coefficients in original units with SE, t, p (statsmodels OLS), standardised coefficients and VIF."""
    import statsmodels.api as sm
    a = sm.add_constant(np.asarray(x_raw, float), has_constant="add")
    res = sm.OLS(np.asarray(y, float), a).fit()
    sd_x = np.asarray(x_raw, float).std(0, ddof=1)
    beta_std = np.r_[np.nan, res.params[1:] * sd_x / np.std(y, ddof=1)]
    vif = np.r_[np.nan, stats.vif(x_raw)]
    tab = pd.DataFrame({"term": ["intercept"] + list(names), "coef": res.params, "SE": res.bse, "t": res.tvalues,
                        "p": res.pvalues, "beta_std": beta_std, "VIF": vif})
    extra = {"F": float(res.fvalue) if np.isfinite(res.fvalue) else np.nan, "F_p": float(res.f_pvalue),
             "s": float(np.sqrt(res.mse_resid)), "R2_adj": float(res.rsquared_adj)}
    return tab, extra


def equation(tab, target):
    parts = [f"{tab['coef'].iloc[0]:.4g}"]
    for term, c in zip(tab["term"].iloc[1:], tab["coef"].iloc[1:], strict=True):
        parts.append(f"{'−' if c < 0 else '+'} {abs(c):.4g}·{term}")
    return f"{target} = " + " ".join(parts)


def set_labels(n, tr, te, status):
    sets = np.array([SET_PRED if s == "predict" else SET_OUT for s in status], dtype=object)
    sets[tr] = SET_TRAIN
    sets[te] = SET_TEST
    return sets


def write_outputs(c):
    out, data, metrics = c["out"], c["data"], c["metrics"]
    reg = data.task == "regression"
    sets = set_labels(len(data.names), c["tr"], c["te"], c["status"])
    smiles = data.smiles if data.smiles is not None else [None] * len(data.names)
    metrics_cols = ["model"] + [k for k in metrics.columns if k != "model"]
    metrics[metrics_cols].to_csv(out / "metrics.csv", index=False, float_format="%.6g")

    # predictions: every compound × every model
    rows = []
    pos_tr = {r: i for i, r in enumerate(c["tr"])}
    for name, pr in c["preds"].items():
        linear = name in mdl.LINEAR and np.isfinite(c["h_all"]).any()
        if reg:
            resid_tr = c["y_tr"] - pr["fit"]
            dof = c["n_tr"] - c["p"] - 1
            s = np.sqrt((resid_tr ** 2).sum() / (dof if dof > 0 else c["n_tr"]))
        for i in range(len(data.names)):
            row = {"compound": data.names[i], "smiles": smiles[i], "set": sets[i], "model": name}
            note = ""
            if i in c["failed"]:
                note = c["failed"][i]
            elif not c["finite"][i]:
                note = "brak wartości wybranych deskryptorów"
            in_ad_lev = bool(c["h_all"][i] <= c["hs"]) if np.isfinite(c["h_all"][i]) else None
            in_ad_knn = bool(c["d_all"][i] <= c["d_thr"]) if np.isfinite(c["d_all"][i]) else None
            if reg:
                yo = float(data.y[i]) if np.isfinite(data.y[i]) else np.nan
                yp = pr["all"][i]
                res = yo - yp if np.isfinite(yo) and np.isfinite(yp) else np.nan
                row.update(y_obs=yo, y_pred=yp, y_cv=pr["cv"][pos_tr[i]] if i in pos_tr else np.nan, residual=res,
                           std_residual=res / s if np.isfinite(res) else np.nan)
            else:
                proba = pr["all"][i]
                ok = np.isfinite(proba).all()
                row.update(y_obs=data.y[i], y_pred=c["classes"][int(np.argmax(proba))] if ok else None,
                           p_positive=proba[c["classes"].index(pr["positive"])] if ok else np.nan)
            row.update(leverage=c["h_all"][i], h_star=c["hs"], knn_dist=c["d_all"][i], knn_threshold=c["d_thr"],
                       ad_method="leverage" if linear else "kNN",
                       in_ad=(in_ad_lev if linear else in_ad_knn), note=note)
            rows.append(row)
    pred = pd.DataFrame(rows)
    pred.to_csv(out / "predictions.csv", index=False, float_format="%.6g")

    # MLR coefficients
    mlr_eq, mlr_extra = None, None
    tab = None
    if reg and "mlr" in c["fitted"]:
        x_raw = c["desc"].loc[c["tr"], c["chosen"]].to_numpy(float)
        tab, mlr_extra = mlr_table(x_raw, c["y_tr"], c["chosen"])
        tab.to_csv(out / "mlr_coefficients.csv", index=False, float_format="%.6g")
        mlr_eq = equation(tab, data.target)

    write_selected(c, mlr_eq, tab, mlr_extra)
    pngs = make_plots(c, pred, sets)
    (c["workdir"] / "highlights.txt").write_text("\n".join(pngs[:4]) + "\n", encoding="utf-8")
    summary = regression_summary(c, pred, mlr_eq) if reg else classification_summary(c, pred)
    (c["workdir"] / "summary.txt").write_text(summary, encoding="utf-8")


def write_selected(c, mlr_eq, tab, extra):
    sel, info = c["sel"], c["info"]
    lines = [f"Metoda wyboru: {sel.method} (kryterium {sel.score_name} na zbiorze uczącym, kara za VIF > "
             f"{selection.VIF_MAX:g})",
             f"Deskryptory: {info['n_desc_start']} → braki −{info['n_na']} → po odsiewie {info['prefilter']['kept']} "
             f"(stałe −{info['prefilter']['constant']}, skorelowane −{info['prefilter']['correlated']})",
             f"Najwięcej deskryptorów w modelu: {info['k_max']} (max_desc i Topliss n/p ≥ 5)", ""]
    if sel.per_k:
        lines.append(f"k\t{sel.score_name}\tR²\tVIFmax\tdeskryptory")
        for r in sel.per_k:
            lines.append(f"{r['k']}\t{r['score']:.4f}\t{r['r2']:.4f}\t{r['vif_max']:.2f}\t{', '.join(r['cols'])}")
        lines.append(f"\nZasada: najmniejsze k, którego {sel.score_name} jest najwyżej {selection.DELTA} poniżej "
                     "najlepszego.")
    lines.append("\nWybrane: " + ", ".join(c["chosen"]))
    if mlr_eq:
        lines += ["", "MLR: " + mlr_eq,
                  f"s = {extra['s']:.4g}, F = {extra['F']:.4g} (p = {extra['F_p']:.2g}), R²adj = {extra['R2_adj']:.4f}",
                  "", tab.to_string(index=False, float_format=lambda v: f"{v:.4g}")]
    (c["out"] / "selected.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(c, pred, sets):
    """All figures; returns the ones for highlights (relative paths), most important first."""
    out, data, metrics, best = c["out"], c["data"], c["metrics"], c["best"]
    reg = data.task == "regression"
    rel = f"{OUT}/"
    hl = []
    progress("wykresy")
    if reg:
        for name, pr in c["preds"].items():
            m = metrics.loc[name]
            text = (f"R² = {fmt(m['R2'])}\nQ²LOO = {fmt(m['Q2_LOO'])}\nQ²F3 = {fmt(m['Q2_F3'])}\n"
                    f"RMSE_C = {fmt(m['RMSE_C'])}\nRMSE_P = {fmt(m['RMSE_P'])}")
            plots.obs_vs_pred(out / f"obs_vs_pred_{name}.png", f"{name}" + (" (najlepszy)" if name == best else ""),
                              c["y_tr"], pr["fit"], c["y_te"], pr["test"], text, ylabel=data.target)
        hl.append(rel + f"obs_vs_pred_{best}.png")
        # Williams plot of the best model
        sub = pred[(pred["model"] == best) & pred["set"].isin([SET_TRAIN, SET_TEST])]
        if sub["leverage"].notna().any():
            plots.williams(out / "williams.png", f"Wykres Williamsa – {best}", sub["leverage"].to_numpy(float),
                           sub["std_residual"].to_numpy(float), sub["set"].to_numpy(), sub["compound"].to_numpy(),
                           c["hs"])
            hl.append(rel + "williams.png")
    else:
        from sklearn.metrics import confusion_matrix, roc_curve
        classes = c["classes"]
        pr = c["preds"][best]
        use_test = len(c["y_te"]) > 0
        yy = c["y_te"] if use_test else c["y_tr"]
        pp = pr["test"] if use_test else pr["cv"]
        cm = confusion_matrix(yy, np.argmax(pp, 1), labels=range(len(classes)))
        plots.confusion(out / "confusion.png", f"{best}, zbiór {'testowy' if use_test else 'uczący (CV)'}", cm,
                        classes)
        hl.append(rel + "confusion.png")
        if len(classes) == 2 and use_test and len(set(c["y_te"])) == 2:
            curves = []
            for name, prm in c["preds"].items():
                k = classes.index(prm["positive"])
                fpr, tpr, _ = roc_curve(c["y_te"] == k, prm["test"][:, k])
                curves.append((name, fpr, tpr, metrics.loc[name, "test_roc_auc"]))
            plots.roc(out / "roc.png", "ROC – zbiór testowy", curves)
            hl.append(rel + "roc.png")
    for name, (r, a, b, base, lab) in c["scr"].items():
        if name == best or (reg and name == "mlr" and best not in c["scr"]):
            fname = "scrambling.png"
            if reg:
                plots.scrambling(out / fname, r, a, b, base[0], base[1], "R²", lab)
            else:
                plots.scrambling(out / fname, r, a, b, base[0], base[1], "BA uczący", lab,
                                 x_label="udział etykiet bez zmian")
            hl.append(rel + fname)
            break
    # correlations of the chosen descriptors with each other and y (training set)
    if len(c["chosen"]) >= 1:
        x = c["desc"].loc[c["tr"], c["chosen"]].to_numpy(float)
        yv = c["y_tr"].astype(float)
        mat = np.column_stack([x, yv])
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.nan_to_num(np.corrcoef(mat, rowvar=False))
        plots.corr_heatmap(out / "corr_heatmap.png", corr, list(c["chosen"]) + [data.target])
    # PCA of the chemical space on all pre-filtered descriptors (scaled on training)
    kept = c["kept"]
    xk = c["desc"][kept].to_numpy(float)
    ok = np.isfinite(xk).all(1) & np.isin(sets, [SET_TRAIN, SET_TEST, SET_PRED])
    if len(kept) >= 2 and ok.sum() >= 3:
        mu = xk[c["tr"]].mean(0)
        sd = xk[c["tr"]].std(0, ddof=1)
        sd[sd == 0] = 1
        z = (xk[ok] - mu) / sd
        u, sv, vt = np.linalg.svd(z - z[np.isin(np.flatnonzero(ok), c["tr"])].mean(0), full_matrices=False)
        scores = u[:, :2] * sv[:2]
        var = sv ** 2 / (sv ** 2).sum()
        colour = (np.array([float(v) if np.isfinite(v) else np.nan for v in data.y[ok]], float) if reg else None)
        plots.pca(out / "pca.png", scores, var, sets[ok], data.names[ok], colour, data.target)
        hl.append(rel + "pca.png")
    return hl


def _ad_counts(pred, best):
    sub = pred[pred["model"] == best]
    out_test = sub[(sub["set"] == SET_TEST) & (sub["in_ad"] == False)]  # noqa: E712
    out_pred = sub[(sub["set"] == SET_PRED) & (sub["in_ad"] == False)]  # noqa: E712
    outliers = sub[sub["set"].isin([SET_TRAIN, SET_TEST]) & (sub.get("std_residual", pd.Series(dtype=float)).abs() > 3)
                   ] if "std_residual" in sub else sub.iloc[:0]
    return out_test, out_pred, outliers


def _fit(sections):
    """Join sections, dropping optional ones from the end until the text fits SUMMARY_MAX."""
    keep = list(sections)
    text = "\n".join(s for s, _ in keep if s)
    while len(text) > SUMMARY_MAX and any(opt for _, opt in keep):
        idx = max(i for i, (_, opt) in enumerate(keep) if opt)
        keep.pop(idx)
        text = "\n".join(s for s, _ in keep if s)
    return text[:SUMMARY_MAX]


def _pred_lines(pred, best, reg):
    sub = pred[(pred["model"] == best) & (pred["set"] == SET_PRED)]
    items = []
    for _, r in sub.iterrows():
        if (reg and not np.isfinite(r["y_pred"])) or (not reg and r["y_pred"] is None):
            items.append(f"{r['compound']}: brak ({r['note'] or 'brak deskryptorów'})")
            continue
        val = fmt(r["y_pred"], 2) if reg else str(r["y_pred"])
        ad = "w domenie" if r["in_ad"] is True or r["in_ad"] == "True" else "POZA domeną"
        items.append(f"{r['compound']} {val} ({ad})")
    return items


def regression_summary(c, pred, mlr_eq):
    data, m, best, info, sel = c["data"], c["metrics"], c["best"], c["info"], c["sel"]
    b = m.loc[best]
    n_pred = int((c["status"] == "predict").sum())
    ratio = info["ratio"]
    fr = info["prefilter"]
    s = []
    s.append((f"QSPR/QSAR – regresja, odpowiedź {data.target}. Związki z odpowiedzią: {len(c['rows'])} "
              f"(uczący {c['n_tr']}, testowy {len(c['te'])}), do prognozy: {n_pred}. Plik {info['file']}.", False))
    s.append((f"Podział: {info['split']}.", False))
    na = f", braki −{info['n_na']}" if info["n_na"] else ""
    s.append((f"Deskryptory ({'+'.join(info['sources'])}): {info['n_desc_start']}{na}, stałe −{fr['constant']}, "
              f"|r|>{c['params']['corr_max']} −{fr['correlated']} → {fr['kept']}.", False))
    if info["n_na"] and info["na_worst"]:
        s.append(("Braki głównie u: " + ", ".join(info["na_worst"][:4]) + " – uzupełnij je albo usuń te związki, "
                  "jeśli chcesz tych deskryptorów.", True))
    s.append((f"Wybór ({sel.method}, {sel.score_name} tylko na zbiorze uczącym): {len(c['chosen'])} – "
              f"{', '.join(c['chosen'][:8])}{' …' if len(c['chosen']) > 8 else ''}; n/p = {ratio:.1f} "
              f"({'Topliss OK' if ratio >= 5 else 'Topliss NIESPEŁNIONY'}).", False))
    if mlr_eq and len(mlr_eq) < 400:
        s.append(("MLR: " + mlr_eq, False))
    scheme = "Q²LOO" if b["cv_scheme"] == "LOO" else "Q²CV(10)"
    s.append((f"Najlepszy: {best} – R² {fmt(b['R2'])}, {scheme} {fmt(b['Q2_LOO'])}, RMSE_C {fmt(b['RMSE_C'])}, "
              f"RMSE_CV {fmt(b['RMSE_CV'])}; test: Q²F1 {fmt(b['Q2_F1'])}, Q²F2 {fmt(b['Q2_F2'])}, Q²F3 "
              f"{fmt(b['Q2_F3'])}, CCC {fmt(b['CCC'])}, R²ext {fmt(b['R2_ext'])}, RMSE_P {fmt(b['RMSE_P'])}, "
              f"rm² {fmt(b['rm2_avg'])} (Δ {fmt(b['rm2_delta'])}).", False))
    gt_ok = "spełnione" if b["GT_pass"] else "NIESPEŁNIONE"
    ext_bad = not (b["Q2_F3"] >= 0.6 and b["CCC"] >= 0.85) if np.isfinite(b["Q2_F3"]) else False
    s.append((f"Golbraikh–Tropsha: {int(b['GT_passed'])}/5, {gt_ok}."
              + (" Uwaga: walidacja zewnętrzna słaba (Q²F3 < 0,6 lub CCC < 0,85)." if ext_bad else ""), False))
    if best != "mlr" and "mlr" in m.index:
        r = m.loc["mlr"]
        s.append((f"MLR (interpretowalny, zasada OECD 5): R² {fmt(r['R2'])}, Q²LOO {fmt(r['Q2_LOO'])}, Q²F3 "
                  f"{fmt(r['Q2_F3'])}, CCC {fmt(r['CCC'])}, RMSE_P {fmt(r['RMSE_P'])}, G–T {int(r['GT_passed'])}/5.",
                  False))
    for nm in dict.fromkeys(["mlr", best]):
        if nm in m.index and "scr_R2_max" in m.columns and np.isfinite(m.loc[nm].get("scr_R2_max", np.nan)):
            r = m.loc[nm]
            lab = c["scr"][nm][4]
            base_q2 = c["scr"][nm][3][1]
            line = (f"Y-scrambling {nm} ({c['params']['scramble']}): R²scr max {fmt(r['scr_R2_max'])} / śr. "
                    f"{fmt(r['scr_R2_mean'])}, {lab}scr max {fmt(r['scr_Q2_max'])} (model {fmt(base_q2)}), "
                    f"cR²p {fmt(r['cR2p'])}")
            chance_ok = r["scr_Q2_max"] < min(0.5, base_q2 - 0.2)
            if nm in mdl.LINEAR:
                line += " – brak przypadkowej korelacji." if r["cR2p"] > 0.5 and chance_ok else " – UWAGA: ryzyko " \
                    "przypadkowej korelacji."
            else:
                line += (" – R² po permutacji wysokie, bo model zapamiętuje zbiór uczący (cR²p i R² nic tu nie "
                         "znaczą); liczy się Q²scr: " + ("OK." if chance_ok else "UWAGA."))
            s.append((line, False))
            if nm == "mlr" and np.isfinite(r.get("scr_sel_Q2_max", np.nan)):
                verdict = "OK" if r["scr_sel_Q2_max"] < r["Q2_LOO"] - 0.1 else "UWAGA: przypadkowa korelacja możliwa"
                s.append((f"Y-scrambling z ponownym wyborem {len(c['chosen'])} z {info['reselect_p']} deskryptorów "
                          f"(forward): Q²LOO max {fmt(r['scr_sel_Q2_max'])} / śr. {fmt(r['scr_sel_Q2_mean'])} wobec "
                          f"{fmt(r['Q2_LOO'])} – {verdict}.", False))
    rule = ("Zasada wyboru: walidacja krzyżowa na samym zbiorze uczącym, modele bez przeuczenia "
            f"(R²−Q² ≤ {info['gap_max']}); kto mieści się w 1 błędzie standardowym od najlepszego "
            f"(tu ±{info['tolerance']:.3f} Q²), ten jest równie dobry – wygrywa najprostszy. "
            "Nie R², bo rośnie z samym dopasowaniem i nie mówi o prognozie; "
            "nie test, żeby został prawdziwie zewnętrzny.")
    by_test = m["Q2_F3"].idxmax() if m["Q2_F3"].notna().any() else None
    if by_test and by_test != best:
        rule += f" Po samym Q²F3 wygrałby {by_test} ({fmt(m.loc[by_test, 'Q2_F3'])})."
    s.append((rule, False))
    rank = m.sort_values("Q2_LOO", ascending=False)
    s.append(("Q²LOO / Q²F3: " + ", ".join(f"{k} {fmt(v['Q2_LOO'], 2)}/{fmt(v['Q2_F3'], 2)}"
                                           for k, v in rank.iterrows()), True))
    out_test, out_pred, outl = _ad_counts(pred, best)
    ad_m = pred[pred["model"] == best]["ad_method"].iloc[0]
    ad_txt = f"h* = {c['hs']:.3f}" if ad_m == "leverage" else f"kNN, próg {c['d_thr']:.2f}"
    sub = pred[(pred["model"] == best) & (pred["set"] == SET_TRAIN)]
    hi_tr = sub[sub["leverage"] > c["hs"]]["compound"].astype(str).tolist()
    if hi_tr:
        s.append((f"Uczące z h > h* (wpływowe, wyznaczają domenę): {', '.join(hi_tr[:4])}.", True))
    s.append((f"Domena ({ad_txt}): poza nią {len(out_test)} testowych"
              + (f" ({', '.join(out_test['compound'].astype(str)[:4])})" if len(out_test) else "")
              + f", {len(out_pred)} do prognozy; |reszta std.| > 3: {len(outl)}"
              + (f" ({', '.join(outl['compound'].astype(str)[:4])})" if len(outl) else "") + ".", False))
    items = _pred_lines(pred, best, True)
    if items:
        s.append((f"Prognozy ({best}): " + "; ".join(items[:10]) + (" …" if len(items) > 10 else "") + ".", False))
    for note in c["notes"]:
        s.append(("Uwaga: " + note, True))
    s.append(("Pliki: wyniki/metrics.csv, predictions.csv, selected.txt, descriptors.csv, wykresy PNG.", True))
    return _fit(s)


def classification_summary(c, pred):
    data, m, best, info, sel = c["data"], c["metrics"], c["best"], c["info"], c["sel"]
    b = m.loc[best]
    classes = c["classes"]
    n_pred = int((c["status"] == "predict").sum())
    counts = {cl: int((c["y_tr"] == i).sum()) for i, cl in enumerate(classes)}
    pos = c["preds"][best]["positive"]
    fr = info["prefilter"]
    s = [(f"QSAR – klasyfikacja, odpowiedź {data.target} ({', '.join(f'{k}: {v}' for k, v in counts.items())} w "
          f"uczącym). Związki z odpowiedzią: {len(c['rows'])} (uczący {c['n_tr']}, testowy {len(c['te'])}), do "
          f"prognozy: {n_pred}.",
          False),
         (f"Podział: {info['split']}. Klasa pozytywna (czułość): {pos}. Wagi klas zrównoważone (balanced).", False),
         (f"Deskryptory ({'+'.join(info['sources'])}): {info['n_desc_start']}, braki −{info['n_na']}, stałe "
          f"−{fr['constant']}, skorelowane −{fr['correlated']} → {fr['kept']}.", False),
         (f"Wybór ({sel.method}, {sel.score_name} na uczącym): {', '.join(c['chosen'][:8])}"
          f"{' …' if len(c['chosen']) > 8 else ''}; n/p = {info['ratio']:.1f}.", False),
         (f"Najlepszy: {best} – CV: BA {fmt(b['cv_balanced_accuracy'])}, MCC {fmt(b['cv_mcc'])}, czułość "
          f"{fmt(b['cv_sensitivity'])}, specyficzność {fmt(b['cv_specificity'])}, AUC {fmt(b['cv_roc_auc'])}; test: "
          f"BA {fmt(b['test_balanced_accuracy'])}, MCC {fmt(b['test_mcc'])}, czułość {fmt(b['test_sensitivity'])}, "
          f"specyficzność {fmt(b['test_specificity'])}, AUC {fmt(b['test_roc_auc'])}.", False)]
    if "scr_BA_cv_max" in m.columns and np.isfinite(b.get("scr_BA_cv_max", np.nan)):
        s.append((f"Y-scrambling ({c['params']['scramble']}): BA(CV) po permutacji max {fmt(b['scr_BA_cv_max'])}, "
                  f"śr. {fmt(b['scr_BA_cv_mean'])} (losowo ≈ {1 / len(classes):.2f}).", False))
    s.append((f"Zasada wyboru: najwyższa zrównoważona dokładność w CV na zbiorze uczącym, bez przeuczenia "
              f"(uczący − CV ≤ {info['gap_max']}); w granicach 1 błędu standardowego (±{info['tolerance']:.3f}) "
              "wygrywa najprostszy model. Nie dokładność (myli przy "
              "niezrównoważonych klasach) i nie test (zostaje zewnętrzny).", False))
    rank = m.sort_values("cv_balanced_accuracy", ascending=False)
    s.append(("BA CV/test: " + ", ".join(
        f"{k} {fmt(v['cv_balanced_accuracy'], 2)}/{fmt(v['test_balanced_accuracy'], 2)}" for k, v in rank.iterrows()),
        True))
    out_test, out_pred, _ = _ad_counts(pred, best)
    s.append((f"Domena (kNN, próg {c['d_thr']:.2f}): poza nią {len(out_test)} testowych, {len(out_pred)} do "
              "prognozy.", False))
    items = _pred_lines(pred, best, False)
    if items:
        s.append((f"Prognozy ({best}): " + "; ".join(items[:10]) + (" …" if len(items) > 10 else "") + ".", False))
    for note in c["notes"]:
        s.append(("Uwaga: " + note, True))
    return _fit(s)


def descriptors_only(workdir, data, desc, info, notes):
    """No response column: descriptors table, PCA of the chemical space and a short summary."""
    out = Path(workdir) / OUT
    hl = []
    x = desc.dropna(axis=1, how="any")
    x = x.loc[:, x.std() > 0]
    if x.shape[1] >= 2 and len(x) >= 3:
        z = ((x - x.mean()) / x.std()).to_numpy(float)
        u, sv, _ = np.linalg.svd(z - z.mean(0), full_matrices=False)
        var = sv ** 2 / (sv ** 2).sum()
        plots.pca(out / "pca.png", u[:, :2] * sv[:2], var, np.array([SET_PRED] * len(z), dtype=object), data.names)
        hl.append(f"{OUT}/pca.png")
    (Path(workdir) / "highlights.txt").write_text("\n".join(hl) + "\n", encoding="utf-8")
    text = (f"Brak kolumny z odpowiedzią – policzyłem tylko deskryptory ({'+'.join(info['sources'])}: "
            f"{desc.shape[1]}), związki: {len(desc)} – wyniki/descriptors.csv. Żeby zbudować model, dodaj kolumnę "
            "z odpowiedzią albo podaj target=.")
    for n in notes:
        text += "\nUwaga: " + n
    (Path(workdir) / "summary.txt").write_text(text[:SUMMARY_MAX], encoding="utf-8")
