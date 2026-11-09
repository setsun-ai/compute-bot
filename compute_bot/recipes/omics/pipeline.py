"""The chain of steps for one omics layer, with a record of what each step did (wyniki/kroki.txt) and the numbers
that go into summary.txt. File output happens here; the computations are in parse/prep/stats/annotate."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import annotate, plots, prep, stats
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


def _groups_for_stats(samples, cols, warns):
    g = samples.loc[cols, "_group"].dropna().astype(str)
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
    warns.extend(aw)
    R["n_features0"], R["n_samples0"] = X.shape
    R["types"] = S._type.value_counts().to_dict()

    # ---------------------------------------------------- 2. bad injections
    progress(f"{lay}: kontrola iniekcji")
    flags = prep.flag_bad_injections(X, S, ds.istd)
    rejected = flags.rejected.astype(bool)
    use, replaced = prep.choose_study_injections(S, rejected.to_dict())
    R["rejected"] = [(c, flags.loc[c, "reason"]) for c in flags.index[rejected]]
    R["replaced"] = replaced
    log.step("iniekcje", X.shape, f"odrzucone {int(rejected.sum())} (|z| > 3 dla IS, sumy intensywności, braków)")
    for c, why in R["rejected"]:
        log.add(f"{c} ({S.loc[c, '_type']}): {why}")
    for a, b in replaced:
        log.add(f"{a} odrzucona – w statystyce zastępuje ją replikat {b}")
    plots.injections(flags, S, out / "qc_iniekcje.png")

    # analysis samples: passing sample/qc/replicate injections; a replicate that stands in becomes a sample
    analysed = [c for c in X.columns if S.loc[c, "_type"] in prep.STUDY_TYPES and not rejected[c]]
    Sa = S.loc[analysed].copy()
    for _, b in replaced:
        Sa.loc[b, "_type"] = "sample"
    for c in Sa.index[(Sa._type == "sample") & ~use.reindex(Sa.index).fillna(False).astype(bool)]:
        Sa.loc[c, "_type"] = "replicate"      # second injection of the same subject: kept for QC, not tested
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
    inj["w_statystyce"] = use.reindex(inj.index).fillna(False).astype(bool)
    inj.to_csv(out / "qc_iniekcje.csv")
    Lq.round(5).to_csv(out / "macierz_po_obrobce.csv")
    R["highlights"] += [out / "pca_przed_po.png", out / "qc_dryf.png"]

    # ---------------------------------------------------- 6. statistics
    stat_cols = [c for c in Sa.index if Sa.loc[c, "_type"] == "sample"]
    groups = _groups_for_stats(Sa, stat_cols, warns)
    R["groups"] = groups.value_counts().to_dict()
    R["ref"] = ref
    if groups.nunique() < 2 or ref not in set(groups):
        warns.append("mniej niż 2 grupy z ≥2 próbkami – bez testów, PLS-DA i adnotacji")
        R["stats"] = None
        if len(stat_cols) >= 3:
            pg = stats.pca(Lq[stat_cols])
            plots.pca_groups(pg, pd.Series("wszystkie", index=stat_cols), labels, out / "pca_grupy.png")
            R["highlights"].insert(0, out / "pca_grupy.png")
        (out / "kroki.txt").write_text(log.text())
        R["log"] = log.text()
        return R

    stat_feats = prep.cluster_representatives(Lq, feats)
    if len(stat_feats) < len(Lq):
        log.add(f"testy na {len(stat_feats)} reprezentantach klastrów analitycznych (addukty/izotopy jednego "
                f"związku) zamiast {len(Lq)} cech")
    Ls = Lq.loc[stat_feats, groups.index]
    progress(f"{lay}: testy ({len(Ls)} cech)")
    cv_cov = cov.loc[groups.index] if cov is not None else None
    res = stats.feature_tests(Ls, groups, ref=ref, test=params["test"], covariates=cv_cov)
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
    (out / "kroki.txt").write_text(log.text())
    R["log"] = log.text()
    return R


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
    if R["rejected"]:
        rj = "; ".join(f"{c} ({w})" for c, w in R["rejected"][:3])
        add(2, f"Odrzucone iniekcje: {len(R['rejected'])} – {rj}" + ("…" if len(R["rejected"]) > 3 else "") + ".")
        for a, b in R["replaced"][:2]:
            add(7, f"{a} zastąpiona replikatem {b}.")
    else:
        add(6, "Odrzucone iniekcje: 0.")
    add(2, f"Cechy: {R['n_features0']} → blanki −{R['removed_blank']} → braki −{R['removed_na']} → "
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
        if met is not None and len(met):
            names = list(dict.fromkeys(met["name"].astype(str)))[:6]
            add(5, f"Istotne nazwane metabolity ({met['name'].nunique()}): " + ", ".join(_short(n) for n in names)
                + ".")
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
