"""The chain of steps for one omics layer, with a record of what each step did (wyniki/kroki.txt) and the numbers
that go into summary.txt. File output happens here; the computations are in parse/prep/stats/annotate."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import annotate, lipids, plots, prep, stats
from .parse import InputError, choose_ref

MAX_D_RATIO = 0.5
TOP_HEATMAP = 50
TOP_BOX = 6
MAX_ANNOTATE = 200
LAYER_PL = {"proteomics": "PROTEOMIKA", "metabolomics": "METABOLOMIKA"}


def progress(text):
    print(f"PROGRESS: {text}", flush=True)


class StepLog:
    def __init__(self, title):
        self.lines = [f"===== {title} ====="]

    def step(self, name, shape, text=""):
        n_f, n_s = shape
        self.lines.append(f"[{name}] {n_f} cech × {n_s} próbek" + (f" – {text}" if text else ""))

    def add(self, text):
        self.lines.append("    " + text)

    def text(self):
        return "\n".join(self.lines) + "\n"


def _fmt_q(q):
    return f"{q:.1e}" if q < 1e-3 else f"{q:.3f}"


def _labels(features):
    """Display name: metabolite/protein name or gene; unnamed features show their ID (plus m/z@RT when known)."""
    name = features["name"].astype(object).where(features["name"].notna())
    ids = pd.Series(features.index, index=features.index).astype(str)
    mz = pd.to_numeric(features.get("mz"), errors="coerce") if "mz" in features else None
    rt = pd.to_numeric(features.get("rt"), errors="coerce") if "rt" in features else None
    if mz is not None and mz.notna().any():
        where = mz.notna() & (rt.notna() if rt is not None else True)
        tag = mz.map("{:.4f}".format) + ("@" + rt.map("{:.2f}".format) if rt is not None else "")
        where &= ~ids.str.contains("@", regex=False)       # the ID already is mass@RT
        ids = ids.where(~where, ids + " (" + tag + ")")
    gene = features["gene"] if "gene" in features else None
    if gene is not None and gene.notna().mean() > 0.5:
        name = gene.where(gene.notna(), name)      # proteins: the gene symbol is the readable short name
    lab = name.where(name.notna(), ids)
    dup = lab.duplicated(keep=False) & name.notna()
    lab[dup] = [f"{n} ({i})" for n, i in zip(lab[dup], lab.index[dup], strict=True)]
    return lab.astype(str)


def _covariates(samples, text):
    names = [c.strip() for c in str(text or "").split(",") if c.strip()]
    if not names:
        return None
    low = {str(c).lower(): c for c in samples.columns if not str(c).startswith("_")}
    missing = [n for n in names if n.lower() not in low]
    if missing:
        raise InputError(f"covariates: nie ma kolumn {', '.join(missing)}. Dostępne: "
                         + ", ".join(sorted(low.values(), key=str))[:500])
    return samples[[low[n.lower()] for n in names]]


def _groups_for_stats(g, warns):
    g = g.dropna().astype(str)
    counts = g.value_counts()
    small = counts[counts < 2]
    if len(small):
        warns.append("grupy z 1 próbką pominięte w testach: " + ", ".join(small.index[:6]))
        g = g[~g.isin(small.index)]
    return g


def validate(ds, params):
    """Checks that can fail on the owner's parameters, done before any heavy step."""
    s = ds.samples
    g = s.loc[s._type == "sample", "_group"].dropna()
    ref = None
    if g.nunique() >= 2:
        ref, _ = choose_ref(g.unique(), params.get("ref", ""))
    elif params.get("ref"):
        raise InputError(f"ref={params['ref']}, ale w danych {ds.name} nie ma co najmniej dwóch grup.")
    cov = _covariates(s, params.get("covariates", ""))
    return ref, cov


def run_layer(ds, params, out, threads=1, fetch=annotate.http_fetch, cache_dir=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir or out)
    lay = ds.layer
    R = {"layer": lay, "warnings": [], "highlights": [], "files": {}}
    warns = R["warnings"]
    log = StepLog(f"{lay}: {ds.name} ({ds.source})")
    ref, cov = validate(ds, params)
    X, S = ds.X, ds.samples
    feats = ds.features.copy()
    feats["label"] = _labels(feats)
    labels = feats["label"].to_dict()

    # ---------------------------------------------------- 0-1. load and audit
    progress(f"{lay}: audyt")
    log.step("wczytanie", X.shape, f"{ds.source}")
    for n in ds.notes:
        log.add(n)
    inf = ds.inferred
    guessed = [w for w, k in (("typy próbek", "type"), ("grupy", "group"), ("kolejność", "order"))
               if inf.get(k)]
    if guessed:
        log.add("zgadnięte z nazw kolumn: " + ", ".join(guessed))
    lines, aw = prep.audit(X, S, ds.features, ds.istd)
    if params["impute"] not in ("halfmin", "min"):
        aw = [w.split(" – imputacja")[0] if "MCAR" in w else w for w in aw]
    for ln in lines:
        log.add(ln)
    warns.extend(ds.warnings)
    warns.extend(aw)
    R["n_features0"], R["n_samples0"] = X.shape
    R["types"] = S._type.value_counts().to_dict()

    # ---------------------------------------------------- 2. bad injections
    progress(f"{lay}: kontrola iniekcji")
    flags = prep.flag_bad_injections(X, S, ds.istd)
    spared = prep.spare_whole_samples(S, flags)
    if spared:
        warns.append(f"wszystkie replikaty próbek {', '.join(spared[:5])} odstają tak samo (braki / suma sygnału) – to "
                     "cecha próbki, nie błąd iniekcji; zostawione w analizie")
    rejected = flags.rejected.astype(bool)
    use, replaced = prep.choose_study_injections(S, rejected.to_dict())
    R["rejected"] = [(c, flags.loc[c, "reason"]) for c in flags.index[rejected]]
    R["replaced"] = replaced
    log.step("iniekcje", X.shape, f"odrzucone {int(rejected.sum())} (|z| > 3 dla IS, sumy intensywności, braków)")
    for c, why in R["rejected"]:
        log.add(f"{c} ({S.loc[c, '_type']}): {why}")
    for c in flags.index[flags.reason.astype(str).str.startswith("zostawiona")]:
        log.add(f"{c}: {flags.loc[c, 'reason']}")
    for a, b in replaced:
        log.add(f"{a} odrzucona – w statystyce zastępuje ją replikat {b}")
    plots.injections(flags, S, out / "qc_iniekcje.png")

    # analysis samples: passing sample/qc/replicate injections; a replicate that stands in becomes a sample
    analysed = [c for c in X.columns if S.loc[c, "_type"] in prep.STUDY_TYPES and not rejected[c]]
    Sa = S.loc[analysed].copy()
    for _, b in replaced:
        Sa.loc[b, "_type"] = "sample"
    # further injections of the same subject stay samples: they are averaged (or kept) as replicates in step 6
    blanks = [c for c in X.columns if S.loc[c, "_type"] == "blank"]
    dropped_types = S._type[~S._type.isin(list(prep.STUDY_TYPES) + ["blank"])].value_counts().to_dict()
    if dropped_types:
        log.add("pominięte w analizie typy: " + ", ".join(f"{k}={v}" for k, v in dropped_types.items()))

    # ---------------------------------------------------- 3. feature filters
    progress(f"{lay}: filtr cech")
    Sb = pd.concat([Sa, S.loc[blanks]])
    keep_b, binfo = prep.blank_filter(X[list(Sa.index) + blanks], Sb, params["blank_ratio"])
    log.step("filtr blanków", (int(keep_b.sum()), len(Sa)),
             f"usunięto {int((~keep_b).sum())}: mediana w próbkach < {params['blank_ratio']:g}× mediana w blankach"
             if blanks else "brak blanków – pominięty")
    keep_na, nafrac = prep.na_filter(X.loc[keep_b, Sa.index], Sa, params["na_max"])
    keep_na = keep_na.reindex(X.index, fill_value=False)
    keep = keep_b & keep_na
    log.step("filtr braków", (int(keep.sum()), len(Sa)),
             f"usunięto {int((keep_b & ~keep_na).sum())}: > {params['na_max']:.0%} braków w każdej grupie")
    R["removed_blank"], R["removed_na"] = int((~keep_b).sum()), int((keep_b & ~keep_na).sum())
    keep_qc, qc_na = prep.qc_presence_filter(X.loc[keep, Sa.index], Sa, params["na_max"])
    keep_qc = keep_qc.reindex(X.index, fill_value=False)
    R["removed_qc"] = int((keep & ~keep_qc).sum())
    if qc_na is not None:
        log.step("obecność w QC", (int((keep & keep_qc).sum()), len(Sa)),
                 f"usunięto {R['removed_qc']}: wykryte w mniej niż {1 - params['na_max']:.0%} próbek QC")
    not_in_qc = keep & ~keep_qc
    keep = keep & keep_qc
    Xf = X.loc[keep, Sa.index]
    if len(Xf) < 2:
        raise InputError("Po filtrach blanków i braków zostało mniej niż 2 cechy – złagodź na_max/blank_ratio.")

    # ---------------------------------------------------- 4. imputation + log2
    progress(f"{lay}: imputacja i log2")
    mask = Xf.notna()
    n_missing = int((~mask).sum().sum())
    Xi = prep.impute(Xf, params["impute"])
    L0 = np.log2(Xi)
    log.step("imputacja + log2", L0.shape, f"{params['impute']}: uzupełniono {n_missing} wartości "
             f"({n_missing / max(1, mask.size):.1%}); dalej skala log2")

    # ---------------------------------------------------- 5. normalisation, drift, batch
    progress(f"{lay}: normalizacja")
    method = params["normalize"]
    if method == "is" and (ds.istd is None or ds.istd.reindex(columns=L0.columns).notna().sum().sum() == 0):
        warns.append("normalize=is, ale brak wzorców wewnętrznych – użyto mediany")
        method = "median"
    L1, factors = prep.normalize(L0, method, Sa, istd=ds.istd, mask=mask)
    log.step("normalizacja", L1.shape, f"{method}; czynniki {factors.min():.2f}–{factors.max():.2f}")
    progress(f"{lay}: korekta dryfu")
    dmode = {"auto": "auto", "loess": "loess", "none": "none"}[params["drift"]]
    L2, dinfo = prep.drift_correct(L1, Sa, mask, mode=dmode, order_inferred=bool(inf.get("order")))
    log.step("dryf (QC-LOESS)", L2.shape, f"skorygowane partie: {dinfo['corrected_batches']}/{dinfo['batches']}")
    for n in dinfo["notes"]:
        log.add(n)
    L3, binf = prep.batch_correct(L2, Sa, mask, mode=params["batch"])
    log.step("partie", L3.shape, {"none": "bez korekty", "qc": "mediana QC w partii → mediana QC ogółem",
                                  "samples": "mediana próbek w partii"}[binf["method"]])
    for n in binf["notes"]:
        log.add(n)
        if "część efektu" in n:
            warns.append(n)
    R["drift"], R["batch"], R["normalize"] = dinfo, binf, method

    cv0 = prep.qc_cv(L0, Sa, mask)
    cv1 = prep.qc_cv(L3, Sa, mask)
    dr = prep.d_ratio(L3, Sa, mask)
    rep0, rep1 = prep.replicate_delta(L0, Sa), prep.replicate_delta(L3, Sa)
    R["cv"] = (cv0.median(), cv1.median()) if cv1.notna().any() else None
    R["dratio"] = dr.median() if dr.notna().any() else None
    R["rep"] = (rep0, rep1) if rep0 is not None else None
    if R["cv"]:
        caveat = (" (korekta liczona na tych samych QC zawsze obniża ich CV – uczciwszą miarą są replikaty)"
                  if dinfo["corrected_batches"] or binf["method"] == "qc" else "")
        log.add(f"mediana CV w QC: {R['cv'][0]:.1%} → {R['cv'][1]:.1%}{caveat}")
    if R["rep"]:
        log.add(f"mediana |Δlog2| próbka–replikat: {rep0:.3f} → {rep1:.3f}")
    progress(f"{lay}: PCA przed/po")
    pb, pa = stats.pca(L0), stats.pca(L3)
    plots.pca_before_after(pb, pa, Sa, out / "pca_przed_po.png")
    plots.qc_drift(L0, L3, Sa, mask, cv0, cv1, out / "qc_dryf.png")

    # quality filter (Broadhurst et al. 2018: CV_QC and D-ratio after correction)
    bad_cv = cv1 > params["qc_cv"]
    bad_dr = (dr > MAX_D_RATIO) & ~bad_cv
    keep_q = ~(bad_cv | bad_dr)
    Lq = L3[keep_q]
    if cv1.notna().any():
        log.step("CV QC i D-ratio", (len(Lq), len(Sa)), f"usunięto {int(bad_cv.sum())} z CV > {params['qc_cv']:.0%} "
                 f"i {int(bad_dr.sum())} z D-ratio > {MAX_D_RATIO}")
    else:
        log.step("CV QC i D-ratio", (len(Lq), len(Sa)), "mniej niż 3 QC – filtr pominięty")
        warns.append("brak (≥3) próbek QC – nie da się ocenić powtarzalności cech")
    R["removed_cv"], R["removed_dr"] = int(bad_cv.sum()), int(bad_dr.sum())
    R["n_final"] = len(Lq)

    # feature-level QC table for every input feature
    decision = pd.Series("OK", index=X.index)
    decision[~keep_b] = "tlo_blank"
    decision[keep_b & ~keep_na] = "za_duzo_brakow"
    decision[not_in_qc] = "brak_w_QC"
    decision[bad_cv.reindex(X.index, fill_value=False)] = "CV_QC"
    decision[bad_dr.reindex(X.index, fill_value=False)] = "D_ratio"
    qct = pd.DataFrame({"etykieta": feats["label"]})
    qct["blank_ratio"] = binfo["blank_ratio"].reindex(X.index).round(3)
    for g in nafrac.columns:
        qct[f"braki_{g}"] = nafrac[g].reindex(X.index).round(3)
    qct["cv_qc_przed"] = cv0.reindex(X.index).round(4)
    qct["cv_qc_po"] = cv1.reindex(X.index).round(4)
    qct["d_ratio"] = dr.reindex(X.index).round(4)
    qct["decyzja"] = decision
    qct.to_csv(out / "qc_cechy.csv")
    inj = S[[c for c in ("_type", "_raw_type", "_group", "_batch", "_order", "_subject", "_file") if c in S]].copy()
    inj.columns = [c.lstrip("_") for c in inj.columns]
    inj = inj.join(flags.round(4))
    inj["czynnik_normalizacji"] = factors.reindex(inj.index).round(4)
    Lq.round(5).to_csv(out / "macierz_po_obrobce.csv")
    R["highlights"] += [out / "pca_przed_po.png", out / "qc_dryf.png"]

    # ---------------------------------------------------- 6. statistics
    # biological samples: injections of the same subject in the same group (or marked as replicates in the names)
    # are one unit; replicates=mean averages them on the log scale, replicates=keep tests every injection
    sample_cols = [c for c in Sa.index if Sa.loc[c, "_type"] == "sample"]
    pair_given = bool(str(params.get("pair") or "").strip())
    want_paired = params["test"] in stats.PAIRED or (params["test"] == "auto" and pair_given)
    subj_col = "_pair" if "_pair" in Sa and Sa["_pair"].notna().any() else "_subject"
    subj_name = inf.get("pair_col") if subj_col == "_pair" else "osoba z arkusza/nazw"
    average = params.get("replicates", "mean") == "mean" or want_paired
    unit_cols = [c for c in Sa.index if Sa.loc[c, "_type"] == "sample" or (
        average and Sa.loc[c, "_type"] == "replicate" and pd.notna(Sa.loc[c, "_group"]))]
    units = prep.replicate_units(Sa, unit_cols) if unit_cols else pd.Series(dtype=object)
    units = units[units.isin(set(units.reindex(sample_cols)))]
    sizes = units.value_counts()
    n_multi = int((sizes > 1).sum())
    R["replicates"] = None
    if n_multi and average:
        Lu, gu, cinfo = stats.collapse_replicates(Lq, units, Sa.loc[units.index, "_group"])
        log.step("replikaty", Lu.shape, f"{cinfo['n_injections']} iniekcji → {cinfo['n_units']} próbek biologicznych: "
                 "średnia log2 iniekcji tej samej osoby/próbki w grupie (replikaty techniczne to nie nowe osoby)")
        R["replicates"] = (cinfo["n_injections"], cinfo["n_units"])
    else:
        units = units.reindex(sample_cols).dropna()
        Lu, gu = Lq[sample_cols], Sa.loc[sample_cols, "_group"]
        if n_multi:
            n_extra = int(len(units) - units.nunique())
            warns.append(f"pseudoreplikacja: replicates=keep, a {n_extra} iniekcji to powtórzenia tych samych "
                         "próbek biologicznych – test liczy je jako niezależne, p-wartości są zaniżone")
            log.add("replicates=keep: każda iniekcja osobno (replikaty techniczne NIE są niezależne)")
    first = {u: units.index[units == u][0] for u in Lu.columns} if n_multi and average else {c: c for c in Lu.columns}
    inj["w_statystyce"] = inj.index.isin(units.index if n_multi and average else list(Lu.columns))
    inj["probka_biologiczna"] = units.reindex(inj.index)
    inj.to_csv(out / "qc_iniekcje.csv")
    stat_cols = list(Lu.columns)
    groups = _groups_for_stats(gu, warns)
    R["ref"] = ref
    pair_ids = pd.Series({u: Sa.loc[first[u], subj_col] for u in groups.index}, dtype=object)
    pairs, test_param = None, params["test"]
    if want_paired and groups.nunique() >= 2:
        if not pair_ids.notna().any():
            raise InputError(f"test={params['test']} porównuje pary – podaj pair=<kolumna z osobą/parą w arkuszu "
                             "próbek> (ta sama wartość dla obu próbek pary).")
        other = sorted(g for g in groups.unique() if g != ref)
        full, lone = (stats.complete_pairs(groups, pair_ids, [ref] + other) if len(other) == 1 else ({}, []))
        if len(other) != 1:
            warns.append("testy dla par działają dla 2 grup – użyto testu niesparowanego")
        elif len(full) < 3:
            warns.append(f"tylko {len(full)} pełnych par ({subj_name}) – użyto testu niesparowanego")
        else:
            pairs = pair_ids
            groups = groups[[u for u in groups.index if u in {c for ab in full.values() for c in ab}]]
            log.add(f"pary wg {subj_name}: {len(full)} pełnych" + (f"; bez pary (pominięte): "
                    f"{', '.join(map(str, lone))}" if lone else ""))
            R["pairs"] = {"n": len(full), "lone": [str(k) for k in lone], "col": subj_name}
            if params["test"] == "wilcoxon" and len(full) < 6:
                warns.append(f"test Wilcoxona przy {len(full)} parach nie da p < {2 / 2 ** len(full):.3f} – żadna "
                             "cecha nie przejdzie FDR; użyj test=paired_t")
        if pairs is None:
            test_param = "auto" if params["test"] in stats.PAIRED else params["test"]
    elif groups.nunique() >= 2 and pair_ids.notna().any():
        both = pd.crosstab(pair_ids[pair_ids.notna()], groups[pair_ids.notna()]).gt(0).sum(axis=1).gt(1).mean()
        if both >= 0.5:
            warns.append("te same osoby są w obu grupach, a test jest niesparowany – dodaj pair=<kolumna z osobą>, "
                         "test dla par ma tu większą moc")
    R["groups"] = groups.value_counts().to_dict()
    if groups.nunique() < 2 or ref not in set(groups):
        warns.append("mniej niż 2 grupy z ≥2 próbkami – bez testów, PLS-DA i adnotacji")
        R["stats"] = None
        if len(stat_cols) >= 3:
            pg = stats.pca(Lu)
            plots.pca_groups(pg, pd.Series("wszystkie", index=stat_cols), labels, out / "pca_grupy.png")
            R["highlights"].insert(0, out / "pca_grupy.png")
        if lay == "metabolomics":
            _lcms_extras(ds, feats, X.loc[keep_b], Sa, Lq, R, log, out, params, None)
        (out / "kroki.txt").write_text(log.text())
        R["log"] = log.text()
        return R

    stat_feats = prep.cluster_representatives(Lq, feats)
    if len(stat_feats) < len(Lq):
        log.add(f"testy na {len(stat_feats)} reprezentantach klastrów analitycznych (addukty/izotopy jednego "
                f"związku) zamiast {len(Lq)} cech")
    Ls = Lu.loc[stat_feats, groups.index]
    progress(f"{lay}: testy ({len(Ls)} cech)")
    cv_cov = None
    if cov is not None and pairs is None:
        cv_cov = cov.loc[[first[u] for u in groups.index]]
        cv_cov.index = groups.index
    elif cov is not None:
        warns.append("kowariaty pominięte w teście dla par (stałe cechy osoby znoszą się w różnicy)")
    res = stats.feature_tests(Ls, groups, ref=ref, test=test_param, covariates=cv_cov, pairs=pairs)
    test, levels = res.attrs["test"], res.attrs["levels"]
    fdr, fc = params["fdr"], params["fc"]
    fc_col = "log2FC" if len(levels) == 2 else "max_abs_log2FC"
    sig = res["q"] < fdr
    R["stats"] = {"test": test, "levels": levels, "n_tests": len(res), "n_sig": int(sig.sum()),
                  "n_sig_fc": int((sig & (res[fc_col].abs() >= fc)).sum()),
                  "covariates": list(cv_cov.columns) if cv_cov is not None else [],
                  "n_used": res.attrs.get("n_used", len(groups))}
    if len(levels) == 2:
        R["stats"]["up"] = int((sig & (res.log2FC > 0)).sum())
        R["stats"]["down"] = int((sig & (res.log2FC < 0)).sum())
    log.step("testy", (len(res), len(groups)), f"{test}, BH FDR {fdr:g}: istotnych {int(sig.sum())}")
    ann_cols = [c for c in ("label", "name", "gene", "accession", "formula", "mz", "rt", "adduct") if c in feats]
    table = feats.loc[res.index, ann_cols].join(res).sort_values(["q", "p"])
    table.index.name = "feature_id"
    table.to_csv(out / "statystyka.csv")
    R["top"] = [(labels.get(f, f), table.loc[f, fc_col], table.loc[f, "q"]) for f in table.index[sig[table.index]][:8]]

    plots.volcano(res, levels, labels, fdr, fc, out / "volcano.png")
    top = list(table.index[:TOP_HEATMAP])
    if len(top) >= 2:
        plots.heatmap(Ls, groups, top, labels, out / "heatmapa.png")
    plots.boxplots(Ls, groups, list(table.index[:TOP_BOX]), labels, out / "boxploty.png")
    pg = stats.pca(Ls)
    plots.pca_groups(pg, groups, labels, out / "pca_grupy.png")
    pg["scores"].round(5).to_csv(out / "pca_wyniki.csv")
    R["highlights"] = [out / "pca_przed_po.png", out / "volcano.png", out / "heatmapa.png", out / "qc_dryf.png"]

    progress(f"{lay}: PLS-DA")
    size = Ls.shape[0] * Ls.shape[1]
    n_perm = 200 if size <= 2e6 else 100 if size <= 1e7 else 50
    pls = stats.plsda(Ls, groups, n_perm=n_perm, threads=threads, seed=0)
    R["plsda"] = pls
    if pls is not None:
        vip = pd.DataFrame({"etykieta": [labels.get(f, f) for f in pls["vip"].index], "VIP": pls["vip"].round(4)},
                           index=pls["vip"].index).sort_values("VIP", ascending=False)
        vip.index.name = "feature_id"
        vip.to_csv(out / "plsda_vip.csv")
        plots.plsda(pls, groups, labels, out / "plsda.png")
        log.step("PLS-DA", (len(pls["vip"]), len(groups)),
                 f"{pls['n_components']} skł., dokładność CV ({pls['folds']}-krotna) {pls['accuracy']:.0%}, "
                 f"Q² {pls['q2']:.2f}, permutacje p={pls['perm_p']:.3g} ({pls['n_perm']})")
    else:
        log.step("PLS-DA", Ls.shape, "pominięte (za mało próbek w grupie)")

    # ---------------------------------------------------- 7. annotation
    sig_ids = list(table.index[sig[table.index]])
    R["annotation"] = R["enrichment"] = None
    if lay == "proteomics" and sig_ids:
        accs = feats.loc[sig_ids[:MAX_ANNOTATE], "accession"].dropna()
        genes = feats.loc[sig_ids[:MAX_ANNOTATE], "gene"].dropna()
        if params["annotate"]:
            progress(f"{lay}: UniProt")
            use_acc = len(accs) >= 0.5 * min(len(sig_ids), MAX_ANNOTATE)
            ids = list(accs) if use_acc else list(genes)
            tab, w = annotate.uniprot(accessions=ids if use_acc else None, genes=None if use_acc else ids,
                                      organism=params["organism"], cache_path=cache_dir / "uniprot_cache.json",
                                      fetch=fetch)
            if w:
                warns.append(w)
            src = accs if use_acc else genes
            if len(tab):
                key = src.map(lambda v: v if v in tab.index else None).dropna()
                ann = tab.loc[key.values].copy()
                ann.index = key.index
                ann = table.loc[ann.index, ["label", fc_col, "q"]].join(ann)
                ann.index.name = "feature_id"
                ann.to_csv(out / "adnotacja_uniprot.csv")
                R["annotation"] = ann
                log.step("UniProt", (len(ann), len(groups)), f"opisano {len(ann)}/{len(src)} istotnych białek")
                if len(src) >= 10 and len(ann) < 0.2 * len(src):
                    warns.append(f"UniProt rozpoznał tylko {len(ann)}/{len(src)} białek – sprawdź organism= "
                                 f"(teraz {params['organism']}) i identyfikatory")
            else:
                log.step("UniProt", (0, len(groups)), w or "brak wyników")
        if params["enrich"]:
            progress(f"{lay}: g:Profiler")
            gcol = "gene" if feats["gene"].notna().any() else "accession"
            bg = sorted(feats.loc[res.index, gcol].dropna().unique())
            q_all = feats.loc[sig_ids, gcol].dropna().tolist()
            queries = {"wszystkie": q_all}
            if len(levels) == 2:
                queries["w_gore"] = feats.loc[[f for f in sig_ids if res.loc[f, "log2FC"] > 0], gcol].dropna().tolist()
                queries["w_dol"] = feats.loc[[f for f in sig_ids if res.loc[f, "log2FC"] < 0], gcol].dropna().tolist()
            enr, w = annotate.gprofiler(queries, background=bg, organism=params["organism"],
                                        cache_path=cache_dir / "gprofiler_cache.json", fetch=fetch)
            if w:
                warns.append(w)
            enr.to_csv(out / "wzbogacenie.csv", index=False)
            R["enrichment"] = enr
            log.step("g:Profiler", (len(q_all), len(groups)),
                     f"{len(enr)} wzbogaconych terminów (tło: {len(bg)} zmierzonych genów)" if not w else w)
    elif lay == "metabolomics" and sig_ids:
        cols = [c for c in ("label", "name", "formula", "mz", "rt", "adduct") if c in feats]
        met = feats.loc[sig_ids, cols].join(table[[fc_col, "q"]])
        met = met[met["name"].notna()] if "name" in met else met.iloc[:0]
        met.index.name = "feature_id"
        met.to_csv(out / "metabolity_istotne.csv")
        R["metabolites"] = met
        log.step("adnotacja", (len(met), len(groups)), f"{len(met)} istotnych cech z nazwą metabolitu")
    if lay == "metabolomics":
        progress(f"{lay}: wykresy LC-MS i lipidy")
        stat = {"res": res, "groups": groups, "L": Lu.loc[:, groups.index], "ref": ref, "test": test, "pairs": pairs,
                "covariates": cv_cov, "sig": sig_ids, "levels": levels, "fdr": fdr}
        _lcms_extras(ds, feats, X.loc[keep_b], Sa, Lq, R, log, out, params, stat)
    (out / "kroki.txt").write_text(log.text())
    R["log"] = log.text()
    return R


# ---------------------------------------------------------------- LC-MS plots and the lipid mode

def _class_table(lin, labels, key, stat):
    """Per class (or saturation category): number of lipids, mean share of the lipid signal in every group and the
    same test as for the features, on log2 of the class sum and of its share."""
    sums, shares = lipids.class_sums(lin, labels)
    tab = pd.DataFrame({"n_lipidow": labels.reindex(lin.index).value_counts().reindex(sums.index)})
    if stat is None:
        tab["udzial_wszystkie"] = shares.mean(axis=1)
        mean_shares = pd.DataFrame({"wszystkie": shares.mean(axis=1)})
    else:
        g = stat["groups"]
        mean_shares = pd.DataFrame({lv: shares[g.index[g == lv]].mean(axis=1) for lv in stat["levels"]})
        for lv in stat["levels"]:
            tab[f"udzial_{lv}"] = mean_shares[lv]
        with np.errstate(divide="ignore"):
            logs = {"": np.log2(sums[g.index]), "_udzialu": np.log2(shares[g.index])}
        for suffix, M in logs.items():
            r = stats.feature_tests(M.replace(-np.inf, np.nan), g, ref=stat["ref"], test=stat["test"],
                                    covariates=stat["covariates"], pairs=stat["pairs"])
            fc = "log2FC" if len(stat["levels"]) == 2 else "max_abs_log2FC"
            tab[fc + suffix], tab["p" + suffix], tab["q" + suffix] = r[fc], r["p"], r["q"]
    tab.index.name = key
    return tab, mean_shares


def _lcms_extras(ds, feats, Xb, Sa, Lq, R, log, out, params, stat):
    """RT vs m/z, Kendrick, Van Krevelen, Venn of groups, internal-standard drift and, when the names are lipids,
    class composition, saturation, chain distributions and class enrichment. `Xb` = raw matrix after the blank
    filter, `stat` = the statistics of step 6 (None when there was nothing to test)."""
    warns, files = R["warnings"], {}
    labels = feats["label"]
    fq = feats.loc[Lq.index]
    ann = lipids.annotate(feats["name"])
    lipid_on = lipids.lipid_mode(params.get("lipids", "auto"), feats["name"], ann)
    if params.get("lipids") == "tak" and not lipid_on:
        warns.append("lipids=tak, ale żadna nazwa cechy nie wygląda na lipid (PC 34:1, TG 52:2…)")
    sig = list(stat["sig"]) if stat else []
    up = [f for f in sig if stat["res"].get("log2FC", pd.Series(dtype=float)).get(f, 0) > 0] if stat else []
    down = [f for f in sig if stat["res"].get("log2FC", pd.Series(dtype=float)).get(f, 0) < 0] if stat else []
    if lipid_on:
        cats = ann["lipid_class"]
    elif stat:
        cats = pd.Series("nieistotne", index=Lq.index).astype(object)
        cats[up], cats[down] = "istotnie wyższe", "istotnie niższe"
        cats[[f for f in sig if f not in up and f not in down]] = "istotne"
    else:
        cats = None
    mark = sig if lipid_on else None

    mz = pd.to_numeric(fq["mz"], errors="coerce")
    rt = pd.to_numeric(fq["rt"], errors="coerce")
    neutral = ds.source.startswith("Agilent Profinder")
    points = pd.DataFrame({"etykieta": labels.reindex(Lq.index), "mz": mz, "rt": rt})
    if (mz.notna() & rt.notna()).sum() >= 3:
        plots.rt_mz(rt, mz, cats, out / "rt_mz.png", mark=mark, mass_label="masa neutralna [Da]" if neutral else "m/z",
                    title="RT vs masa" if neutral else "RT vs m/z")
        files["rt_mz"] = out / "rt_mz.png"
    if mz.notna().sum() >= 3:
        nominal, kmd = lipids.kendrick(mz[mz.notna()])
        points.loc[mz.notna(), "masa_Kendricka_nominalna"] = nominal
        points.loc[mz.notna(), "KMD_CH2"] = np.round(kmd, 5)
        plots.kendrick(points["masa_Kendricka_nominalna"].dropna(), points["KMD_CH2"].dropna(), cats,
                       out / "kmd.png", mark=mark)
        files["kmd"] = out / "kmd.png"
    vk = lipids.van_krevelen(fq["formula"]) if "formula" in fq else pd.DataFrame()
    if len(vk) >= 3:
        points = points.join(vk.round(4))
        plots.van_krevelen(vk, cats, out / "van_krevelen.png", mark=mark)
        files["van_krevelen"] = out / "van_krevelen.png"
    if len(files):
        if lipid_on:
            points["klasa"] = ann["lipid_class"].reindex(points.index)
        points.index.name = "feature_id"
        points.to_csv(out / "kmd_van_krevelen.csv")
        log.step("wykresy LC-MS", Lq.shape, "RT vs m/z, Kendrick (CH₂)" + (f", Van Krevelen ({len(vk)} wzorów)"
                 if len(vk) >= 3 else "") + " → kmd_van_krevelen.csv")

    st = Sa.index[(Sa._type == "sample") & Sa._group.notna()]
    g_inj = Sa.loc[st, "_group"].astype(str)
    if 2 <= g_inj.nunique() <= 3:
        present = lipids.presence(Xb[list(st)], g_inj)
        present = present[present.any(axis=1)]
        counts = lipids.venn_counts(present)
        table = present.copy()
        table.insert(0, "etykieta", labels.reindex(table.index))
        table.index.name = "feature_id"
        table.to_csv(out / "venn.csv")
        plots.venn(counts, list(present.columns), out / "venn.png",
                   title="Cechy obecne w grupach (wykryte w ≥ 50 % iniekcji grupy, po filtrze blanków)")
        R["venn"] = counts
        log.step("Venn", (len(present), len(st)), "; ".join(f"{' ∩ '.join(k)}: {v}" for k, v in counts.items()))

    if ds.istd is not None and ds.istd.notna().any().any():
        istd = ds.istd.reindex(columns=[c for c in Sa.index if c in ds.istd.columns])
        rows = []
        for name in istd.index:
            v = istd.loc[name].astype(float)
            qc, smp = v[Sa.loc[v.index, "_type"] == "qc"], v[Sa.loc[v.index, "_type"] == "sample"]
            ok = v.notna()
            rho = v[ok].corr(Sa.loc[v.index[ok], "_order"], method="spearman") if ok.sum() >= 4 else np.nan
            rows.append({"wzorzec": name, "wykryty_w": int(ok.sum()), "rsd_wszystkie": v.std() / v.mean(),
                         "rsd_qc": qc.std() / qc.mean() if qc.notna().sum() >= 3 else np.nan,
                         "rsd_probki": smp.std() / smp.mean() if smp.notna().sum() >= 3 else np.nan,
                         "trend_spearman": rho})
        ist = pd.DataFrame(rows).set_index("wzorzec").round(4)
        ist.to_csv(out / "wzorce_is.csv")
        plots.is_drift(ds.istd, pd.concat([Sa, ds.samples.loc[[c for c in ds.samples.index if c not in Sa.index]]]),
                       out / "is_dryf.png")
        R["istd"] = {"n": len(ist), "rsd": float(ist["rsd_wszystkie"].median()),
                     "rsd_qc": float(ist["rsd_qc"].median()) if ist["rsd_qc"].notna().any() else None,
                     "worst": (str(ist["rsd_wszystkie"].idxmax()), float(ist["rsd_wszystkie"].max()))}
        log.step("wzorce wewnętrzne", (len(ist), istd.shape[1]),
                 f"mediana RSD {R['istd']['rsd']:.0%}; największy dryf/rozrzut: {R['istd']['worst'][0]} "
                 f"({R['istd']['worst'][1]:.0%}) → wzorce_is.csv, is_dryf.png")

    if not lipid_on:
        return
    named = int(feats["name"].notna().sum())
    a = ann.loc[Lq.index]
    full = ann.copy()
    full.insert(0, "etykieta", labels)
    full["po_filtrach"] = full.index.isin(Lq.index)
    full.index.name = "feature_id"
    full.to_csv(out / "lipidy_adnotacja.csv")
    n_lip = int(a["lipid_class"].notna().sum())
    counts = a["lipid_class"].value_counts()
    log.step("lipidy", (n_lip, Lq.shape[1]), f"rozpoznano {int(ann['lipid_class'].notna().sum())}/{named} nazw; po "
             f"filtrach {n_lip} lipidów w {len(counts)} klasach: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    lin = 2.0 ** (stat["L"] if stat else Lq[list(st)] if len(st) else Lq)
    ctab, cshares = _class_table(lin, a["lipid_class"], "klasa", stat)
    stab, sshares = _class_table(lin, a["saturation"], "nasycenie", stat)
    stab = stab.reindex([k for k in lipids.SAT_ORDER if k in stab.index])
    ctab = ctab.sort_values("n_lipidow", ascending=False)
    ctab.round(5).to_csv(out / "lipidy_klasy.csv")
    stab.round(5).to_csv(out / "lipidy_nasycenie.csv")
    plots.class_composition(cshares, ctab if stat else None, out / "lipidy_klasy.png", fdr=params["fdr"])
    plots.class_composition(sshares.reindex(stab.index), stab if stat else None, out / "lipidy_nasycenie.png",
                            fdr=params["fdr"], what="nasycenia (SFA/MUFA/PUFA wg liczby wiązań podwójnych)")
    R["lipids"] = {"n": n_lip, "named": named, "classes": counts.to_dict(), "class_table": ctab, "sat_table": stab,
                   "levels": stat["levels"] if stat else None, "enrichment": None}
    log.add("udziały = część sumy sygnału lipidów po normalizacji (nie stężenia: klasy jonizują różnie)")
    if stat:
        tested = a.loc[[f for f in stat["res"].index if f in a.index]]
        parts = [("wszystkie", sig)] + ([("w_gore", up), ("w_dol", down)] if len(stat["levels"]) == 2 else [])
        enr = pd.concat([lipids.class_enrichment(tested, ids).assign(kierunek=k) for k, ids in parts])
        enr = enr[["kierunek", "klasa", "w_klasie", "istotne", "oczekiwane", "odds_ratio", "p", "q"]]
        enr.to_csv(out / "lipidy_wzbogacenie_klas.csv", index=False)
        R["lipids"]["enrichment"] = enr
        dist = lipids.distributions(tested, up=up if up or down else sig, down=down)
        dist.to_csv(out / "lipidy_istotne_rozklad.csv", index=False)
        plots.lipid_distributions(dist, out / "lipidy_istotne_rozklad.png")
        best = enr[(enr.kierunek == "wszystkie") & (enr.q < params["fdr"])]
        log.step("klasy lipidów", (len(ctab), lin.shape[1]),
                 f"test {stat['test']} na sumach klas; wzbogacenie klas wśród {len(sig)} istotnych (Fisher, BH): "
                 + (", ".join(f"{r.klasa} {r.istotne}/{r.w_klasie}" for r in best.itertuples()) or "brak"))
    hl = [out / "pca_przed_po.png", out / "volcano.png", out / "lipidy_klasy.png", files.get("rt_mz", out / "kmd.png")]
    R["highlights"] = [h for h in hl if h.exists()] + [h for h in R["highlights"] if h not in hl]


# ---------------------------------------------------------------- summary

def _types_text(t):
    pl = {"sample": "próbek", "qc": "QC", "blank": "blanków", "replicate": "replikatów",
          "conditioning": "kondycjonujących QC", "other": "innych"}
    return ", ".join(f"{v} {pl.get(k, k)}" for k, v in sorted(t.items(), key=lambda kv: -kv[1]))


def layer_summary(ds, R, params, budget=2400):
    """Polish summary of one layer. Lines carry a priority: when the budget is short the least important go first
    (the order on the page stays the same)."""
    items = []

    def add(prio, text):
        items.append((prio, text))

    lay = LAYER_PL.get(R["layer"], R["layer"].upper())
    add(0, f"{lay} – {ds.source} ({ds.name}): {R['n_features0']} cech × {R['n_samples0']} iniekcji "
           f"({_types_text(R['types'])}).")
    inf = ds.inferred
    guessed = [w for w, k in (("typy", "type"), ("grupy", "group"), ("kolejność", "order")) if inf.get(k)]
    if R.get("groups"):
        gs = sorted(R["groups"].items(), key=lambda kv: (kv[0] != R["ref"], kv[0]))
        txt = ", ".join(f"{g} (n={n}{', odniesienie' if g == R['ref'] else ''})" for g, n in gs[:8])
        if len(gs) > 8:
            txt += f" i {len(gs) - 8} innych"
        src = "zgadnięte z nazw próbek" if inf.get("group") else f"kolumna {inf.get('group_col')}"
        add(0, f"Grupy ({src}): {txt}.")
    if guessed:
        add(1, "Z nazw kolumn wywnioskowano: " + ", ".join(guessed) + " – arkusz próbek (csv/xlsx) to nadpisze.")
    joined = [m.group(1).split("[")[-1].rstrip("]") + " (" + m.group(2) + ")" for n in ds.notes
              if (m := re.match(r"dołączono (.+?) do arkusza próbek po kolumnie (.+?) \(", n))]
    if joined or inf.get("from_names"):
        sheet = next((re.match(r"arkusz próbek: (.+?) \(", n).group(1).split("[")[-1].rstrip("]")
                      for n in ds.notes if n.startswith("arkusz próbek: ")), "arkusz")
        txt = f"Metadane: {sheet}" + (" + " + ", ".join(joined) if joined else "")
        if inf.get("from_names"):
            txt += f"; dla {inf['from_names']} plików bez zgodnego wiersza grupę/osobę odczytano z nazw"
        add(2, txt + ".")
    if R.get("replicates"):
        add(1, f"Replikaty: {R['replicates'][0]} iniekcji → {R['replicates'][1]} próbek biologicznych (średnia log2); "
               "n w testach = próbki biologiczne.")
    if R["rejected"]:
        rj = "; ".join(f"{c} ({w})" for c, w in R["rejected"][:3])
        add(2, f"Odrzucone iniekcje: {len(R['rejected'])} – {rj}" + ("…" if len(R["rejected"]) > 3 else "") + ".")
        for a, b in R["replaced"][:2]:
            add(7, f"{a} zastąpiona replikatem {b}.")
    else:
        add(6, "Odrzucone iniekcje: 0.")
    in_qc = f" → brak w QC −{R['removed_qc']}" if R.get("removed_qc") else ""
    add(2, f"Cechy: {R['n_features0']} → blanki −{R['removed_blank']} → braki −{R['removed_na']}{in_qc} → "
           f"CV/D-ratio −{R['removed_cv'] + R['removed_dr']} → {R['n_final']}.")
    q = [f"normalizacja {R['normalize']}"]
    d = R["drift"]
    q.append(f"dryf QC-LOESS w {d['corrected_batches']}/{d['batches']} partiach" if d["corrected_batches"]
             else "bez korekty dryfu")
    q.append({"none": "bez korekty partii", "qc": "partie wyrównane medianą QC",
              "samples": "partie wyrównane medianą próbek"}[R["batch"]["method"]])
    if R["cv"]:
        q.append(f"mediana CV w QC {R['cv'][0]:.1%} → {R['cv'][1]:.1%}")
    if R["dratio"] is not None:
        q.append(f"D-ratio {R['dratio']:.2f}")
    if R["rep"]:
        q.append(f"replikaty |Δlog2| {R['rep'][0]:.3f} → {R['rep'][1]:.3f}")
    add(3, "Jakość: " + "; ".join(q) + ".")
    st = R.get("stats")
    if st:
        cov = f" z kowariatami {', '.join(map(str, st['covariates']))}" if st["covariates"] else ""
        lv = st["levels"]
        pr = R.get("pairs")
        if pr:
            lone = f"; bez pary pominięto {len(pr['lone'])}: {', '.join(pr['lone'][:4])}" if pr["lone"] else ""
            cov += f" dla par ({pr['n']} par wg {pr['col']}{lone})"
        line = f"Test {st['test']}{cov}, FDR {params['fdr']:.0%}: {st['n_sig']} istotnych z {st['n_tests']}"
        if "up" in st:
            line += f" ({st['up']}↑, {st['down']}↓ w {lv[1]} vs {lv[0]})"
        add(0, line + f"; z |log2FC| ≥ {params['fc']:g}: {st['n_sig_fc']}.")
        if R.get("top"):
            add(4, "Najsilniejsze: " + "; ".join(f"{_short(n, 26)} ({fc:+.2f}, q={_fmt_q(qv)})"
                                                for n, fc, qv in R["top"][:6]) + ".")
        p = R.get("plsda")
        if p:
            add(5, f"PLS-DA: {p['n_components']} skł., dokładność CV {p['accuracy']:.0%} (losowo {p['chance']:.0%}), "
                   f"Q² {p['q2']:.2f}, test permutacyjny p={p['perm_p']:.3g}.")
        ann = R.get("annotation")
        if ann is not None and len(ann):
            fn = [f"{r.gene or r.label}: {_short(r.protein_name, 32)}" for r in ann.head(3).itertuples()]
            add(8, "UniProt: " + "; ".join(fn) + ".")
        enr = R.get("enrichment")
        if enr is not None and len(enr):
            e = enr[enr["query"] == "wszystkie"] if "wszystkie" in set(enr["query"]) else enr
            terms = "; ".join(f"{r.source} {_short(r.name, 40)} (p={r.p_value:.1e})" for r in e.head(4).itertuples())
            add(5, f"Wzbogacenie g:Profiler ({len(enr)} terminów): {terms}.")
        elif enr is not None and not any("g:Profiler" in w for w in R["warnings"]):
            add(6, "Wzbogacenie g:Profiler: brak istotnych terminów względem zmierzonego tła.")
        met = R.get("metabolites")
        if met is not None and len(met) and not R.get("lipids"):
            names = list(dict.fromkeys(met["name"].astype(str)))[:6]
            add(5, f"Istotne nazwane metabolity ({met['name'].nunique()}): " + ", ".join(_short(n) for n in names)
                + ".")
    lp = R.get("lipids")
    if lp:
        cl = ", ".join(f"{k} {v}" for k, v in list(lp["classes"].items())[:7])
        add(1, f"Lipidy: {lp['n']} po filtrach w {len(lp['classes'])} klasach ({cl}"
               f"{'…' if len(lp['classes']) > 7 else ''}).")
        lv = lp["levels"]
        for prio, tab, head in ((2, lp["class_table"], "Klasy"), (4, lp["sat_table"], "Nasycenie")):
            if lv and len(lv) == 2 and "q" in tab:
                rows = tab.sort_values(f"udzial_{lv[0]}", ascending=False) if head == "Klasy" else tab
                bits = [f"{k} {r[f'udzial_{lv[0]}']:.1%}→{r[f'udzial_{lv[1]}']:.1%} (log2FC {r['log2FC']:+.2f}, "
                        f"q={_fmt_q(r['q'])})" for k, r in rows.head(5).iterrows() if pd.notna(r["q"])]
                if bits:
                    add(prio, f"{head} (udział sygnału {lv[0]}→{lv[1]}, zmiana sumy): " + "; ".join(bits) + ".")
        enr = lp["enrichment"]
        if enr is not None:
            hit = enr[enr.q < params["fdr"]]
            if len(hit):
                pl = {"wszystkie": "", "w_gore": "↑", "w_dol": "↓"}
                add(4, "Klasy wzbogacone wśród istotnych (Fisher): " + "; ".join(
                    f"{r.klasa}{pl[r.kierunek]} {r.istotne}/{r.w_klasie} (q={_fmt_q(r.q)})"
                    for r in hit.sort_values("q").head(5).itertuples()) + ".")
    if R.get("istd"):
        i = R["istd"]
        add(5, f"Wzorce wewnętrzne ({i['n']}, chronione przed filtrami): mediana RSD {i['rsd']:.0%}"
               + (f", w QC {i['rsd_qc']:.0%}" if i["rsd_qc"] is not None else "")
               + f"; najgorszy {_short(i['worst'][0], 24)} {i['worst'][1]:.0%}.")
    if R.get("venn"):
        add(6, "Venn (obecność w grupach): " + "; ".join(f"{'∩'.join(k)} {v}" for k, v in R["venn"].items()) + ".")
    if R["warnings"]:
        add(3, "Uwagi: " + " | ".join(dict.fromkeys(R["warnings"])) + ".")
    keep, used = set(), 0
    for i in sorted(range(len(items)), key=lambda i: (items[i][0], i)):
        n = len(items[i][1]) + 1
        if used + n <= budget:
            keep.add(i)
            used += n
    text = "\n".join(t for i, (_, t) in enumerate(items) if i in keep)
    return text[:budget]


def _short(s, n=30):
    s = re.sub(r"\s+", " ", str(s))
    return s if len(s) <= n else s[: n - 1] + "…"
