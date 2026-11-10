"""QC and preprocessing as pure functions on DataFrames.

Conventions: `X` = linear intensities, `L` = log2 intensities, both features x samples with NaN = missing.
`samples` is indexed like the columns and has `_type` (sample, qc, blank, replicate, conditioning, other),
`_batch`, `_order` and optionally `_group`, `_raw_type`, `_subject`. `mask` = True where a value was measured
(not imputed): QC metrics and drift fits use measured values only.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

MAD_SCALE = 1.4826
LOESS_SPAN = 0.75
STUDY_TYPES = ("sample", "qc", "replicate")


def robust_z(x):
    """(x - median) / (1.4826 MAD): one outlier does not move its own reference."""
    x = pd.Series(x, dtype=float)
    med = x.median()
    mad = (x - med).abs().median() * MAD_SCALE
    if not np.isfinite(mad) or mad == 0:
        sd = x.std()
        mad = sd if np.isfinite(sd) and sd > 0 else 1.0
    return (x - med) / mad


def _types(samples, cols):
    return samples.loc[cols, "_type"]


def _cols(samples, cols, kinds):
    t = _types(samples, cols)
    return [c for c in cols if t[c] in kinds]


# ---------------------------------------------------------------- audit

def audit(X, samples, features=None, istd=None):
    """Lines describing the data as it came, and warnings worth a sentence in the summary."""
    lines, warn = [], []
    s = samples.loc[X.columns]
    lines.append(f"macierz: {X.shape[0]} cech × {X.shape[1]} próbek; braki ogółem {X.isna().mean().mean():.1%}")
    lines.append("typy próbek: " + ", ".join(f"{k}={v}" for k, v in s._type.value_counts().items()))
    filled = X.columns[X.notna().sum() >= 10]       # empty blanks are "identical" without being duplicates
    dup_cols = X[filled].T.duplicated().reindex(X.columns, fill_value=False)
    if dup_cols.any():
        warn.append(f"{int(dup_cols.sum())} kolumn identycznych z inną kolumną (zdublowany plik?): "
                    + ", ".join(map(str, X.columns[dup_cols][:5])))
    dup_rows = X[X.notna().sum(axis=1) > 2].duplicated()
    if dup_rows.any():
        lines.append(f"{int(dup_rows.sum())} cech z identycznym profilem jak inna cecha")
    if features is not None and len(features):
        nf = features.reindex(X.index).isna().all(axis=1).sum()
        if nf:
            lines.append(f"{int(nf)} cech bez opisu w tabeli cech")
    na_type = X.isna().mean().groupby(s._type).mean()
    lines.append("braki wg typu próbki: " + ", ".join(f"{k} {v:.1%}" for k, v in na_type.items()))
    st = _cols(s, X.columns, ("sample",))
    if st:
        no_group = s.loc[st, "_group"].isna().sum() if "_group" in s else len(st)
        if no_group:
            warn.append(f"{int(no_group)} próbek bez grupy (pominięte w statystyce)")
    order = s._order
    if order.duplicated().any():
        warn.append("powtórzone numery kolejności iniekcji")
    for b, sb in s.groupby("_batch"):
        nq = int((sb._type == "qc").sum())
        lines.append(f"partia {b}: {len(sb)} iniekcji, {nq} QC, kolejność {sb._order.min():g}–{sb._order.max():g}")
    if "_group" in s and s._batch.nunique() > 1 and st:
        tab = pd.crosstab(s.loc[st, "_group"], s.loc[st, "_batch"])
        if len(tab) > 1 and ((tab > 0).sum(axis=1) == 1).any():
            warn.append("grupa w całości w jednej partii – efekt partii nieodróżnialny od grupy")
    if istd is not None:
        miss = istd.isna().all().sum()
        if miss:
            lines.append(f"{int(miss)} próbek bez sygnału wzorców wewnętrznych")
    m = mnar_check(X, s)
    if m is not None:
        lines.append("braki wg kwintyla intensywności (od najsłabszych): " + ", ".join(f"{v:.1%}" for v in m))
        if m[0] < 1.5 * m[-1] + 0.005 and m.mean() > 0.01:
            warn.append("braki nie zależą od intensywności (raczej losowe, MCAR) – imputacja halfmin/min zaniża "
                        "wartości; rozważ impute=knn")
    return lines, warn


def mnar_check(X, samples):
    st = _cols(samples, X.columns, ("sample",))
    if len(st) < 4 or len(X) < 50:
        return None
    sub = X[st]
    mean = np.log2(sub).mean(axis=1)
    na = sub.isna().mean(axis=1)
    ok = mean.notna()
    if ok.sum() < 50:
        return None
    q = pd.qcut(mean[ok], 5, labels=False, duplicates="drop")
    return na[ok].groupby(q).mean().to_numpy()


# ---------------------------------------------------------------- injections

def flag_bad_injections(X, samples, istd=None, z=3.0):
    """Robust z of IS response, total intensity and NA fraction across sample/QC/replicate injections.

    Rejected: |z_IS| > z, or z_total < -z, or z_NA > z (each also needs a visible absolute difference, so a very
    tight distribution does not reject harmless injections).
    """
    s = samples.loc[X.columns]
    cols = _cols(s, X.columns, STUDY_TYPES)
    out = pd.DataFrame(index=X.columns, columns=["na_frac", "log2_total", "is_log2", "z_na", "z_total", "z_is"],
                       dtype=float)
    out["na_frac"] = X.isna().mean()
    out["log2_total"] = np.log2(X.sum(min_count=1))
    if istd is not None:
        out["is_log2"] = np.log2(istd.reindex(columns=X.columns)).mean()
    out["rejected"] = False
    out["reason"] = ""
    if len(cols) < 6:
        return out
    zna = robust_z(out.loc[cols, "na_frac"])
    ztot = robust_z(out.loc[cols, "log2_total"])
    out.loc[cols, "z_na"] = zna
    out.loc[cols, "z_total"] = ztot
    na_bad = (zna > z) & (out.loc[cols, "na_frac"] - out.loc[cols, "na_frac"].median() > 0.05)
    tot_bad = (ztot < -z) & (out.loc[cols, "log2_total"].median() - out.loc[cols, "log2_total"] > 0.5)
    reasons = {c: [] for c in cols}
    for c in cols:
        if na_bad[c]:
            reasons[c].append(f"braki {out.loc[c, 'na_frac']:.0%} (z={zna[c]:.1f})")
        if tot_bad[c]:
            reasons[c].append(f"suma intensywności z={ztot[c]:.1f}")
    if istd is not None and out.loc[cols, "is_log2"].notna().sum() >= 6:
        ok = [c for c in cols if np.isfinite(out.loc[c, "is_log2"])]
        zis = robust_z(out.loc[ok, "is_log2"])
        out.loc[ok, "z_is"] = zis
        med = out.loc[ok, "is_log2"].median()
        for c in ok:
            if abs(zis[c]) > z and abs(out.loc[c, "is_log2"] - med) > 0.4:
                reasons[c].append(f"IS z={zis[c]:.1f}")
    for c, r in reasons.items():
        if r:
            out.loc[c, "rejected"] = True
            out.loc[c, "reason"] = "; ".join(r)
    return out


def choose_study_injections(samples, rejected):
    """One injection per subject for statistics: the sample if it passed, otherwise a passing technical replicate."""
    s = samples
    use = pd.Series(False, index=s.index)
    subs = {}
    for c in s.index[s._type == "sample"]:
        if rejected.get(c, False):
            continue
        subj = s.loc[c, "_subject"] if "_subject" in s else None
        if subj is None or pd.isna(subj):
            use[c] = True
        elif subj not in subs:
            subs[subj] = c
            use[c] = True
    replaced = []
    for c in s.index[(s._type == "sample")]:
        if not rejected.get(c, False) or "_subject" not in s or pd.isna(s.loc[c, "_subject"]):
            continue
        subj = s.loc[c, "_subject"]
        if subj in subs:
            continue
        reps = [r for r in s.index[(s._type == "replicate") & (s._subject == subj)] if not rejected.get(r, False)]
        if reps:
            use[reps[0]] = True
            subs[subj] = reps[0]
            replaced.append((c, reps[0]))
    return use, replaced


# ---------------------------------------------------------------- feature filters

def blank_filter(X, samples, ratio=3.0):
    """Keep a feature if its median in samples is >= ratio x its median in blanks (blank NA = not detected = 0).

    Process blanks are used when the sheet has them (solvent blanks lack the matrix and the standards)."""
    s = samples.loc[X.columns]
    blanks = list(s.index[s._type == "blank"])
    if "_raw_type" in s:
        proc = [c for c in blanks if "process" in str(s.loc[c, "_raw_type"]).lower()]
        blanks = proc or blanks
    st = _cols(s, X.columns, ("sample",)) or _cols(s, X.columns, STUDY_TYPES)
    info = pd.DataFrame(index=X.index)
    if not blanks or ratio <= 0 or not st:
        info["blank_ratio"] = np.nan
        return pd.Series(True, index=X.index), info
    blank_level = X[blanks].fillna(0).median(axis=1)
    sample_level = X[st].median(axis=1)
    r = sample_level / blank_level.replace(0, np.nan)
    keep = (blank_level == 0) | (r >= ratio)
    keep &= ~(sample_level.isna() & (blank_level > 0))
    info["blank_detect"] = X[blanks].notna().mean(axis=1)
    info["blank_ratio"] = r
    return keep, info


def na_filter(X, samples, na_max=0.5):
    """Keep a feature if at least one group has a missing fraction <= na_max (samples only; no groups = all)."""
    s = samples.loc[X.columns]
    st = _cols(s, X.columns, ("sample",))
    if not st:
        st = _cols(s, X.columns, STUDY_TYPES)
    g = s.loc[st, "_group"] if "_group" in s else pd.Series(None, index=st)
    g = g.dropna()
    if g.nunique() >= 2:
        frac = pd.concat({k: X[idx].isna().mean(axis=1) for k, idx in g.groupby(g).groups.items()}, axis=1)
    else:
        frac = pd.DataFrame({"wszystkie": X[st].isna().mean(axis=1)})
    keep = frac.min(axis=1) <= na_max
    return keep, frac


# ---------------------------------------------------------------- imputation

def impute(X, method="halfmin", seed=0):
    """Fill missing linear intensities per feature: halfmin = min/2, min, knn (on log2, 5 nearest samples), none."""
    if method == "none":
        return X.copy()
    if method in ("halfmin", "min"):
        fill = X.min(axis=1) * (0.5 if method == "halfmin" else 1.0)
        return X.apply(lambda col: col.fillna(fill))
    if method == "knn":
        from sklearn.impute import KNNImputer
        L = np.log2(X)
        allnan = L.isna().all(axis=1)
        k = max(1, min(5, X.shape[1] - 1))
        filled = KNNImputer(n_neighbors=k).fit_transform(L[~allnan].T.to_numpy())
        out = L.copy()
        out.loc[~allnan] = filled.T
        return 2.0 ** out
    raise ValueError(method)


# ---------------------------------------------------------------- normalisation

def _measured(L, mask):
    return L.where(mask) if mask is not None else L


def _complete(L, mask, min_features=50):
    ok = (mask if mask is not None else L.notna()).all(axis=1) & L.notna().all(axis=1)
    return L[ok] if ok.sum() >= min_features else L


def normalize(L, method, samples, istd=None, mask=None):
    """Sample-wise normalisation of log2 data. Returns (normalised L, linear factor per sample; L - log2 factor)."""
    cols = list(L.columns)
    if method == "none":
        return L.copy(), pd.Series(1.0, index=cols)
    base = _measured(L, mask)
    if method == "median":
        med = base.median()
        shift = med - med.median()
    elif method == "sum":
        tot = np.log2((2.0 ** _complete(L, mask)).sum())
        shift = tot - tot.median()
    elif method == "pqn":
        s = samples.loc[cols]
        qcs = [c for c in cols if s.loc[c, "_type"] == "qc"]
        ref_cols = qcs if len(qcs) >= 2 else ([c for c in cols if s.loc[c, "_type"] == "sample"] or cols)
        ref = base[ref_cols].median(axis=1)
        shift = base.sub(ref, axis=0).median()     # log2 of the median quotient
    elif method == "is":
        if istd is None or istd.reindex(columns=cols).notna().sum().sum() == 0:
            raise ValueError("is")
        m = np.log2(istd.reindex(columns=cols)).mean()
        shift = m - m.median()
        fallback = base.median() - base.median().median()
        shift = shift.where(shift.notna(), fallback)
    else:
        raise ValueError(method)
    return L.sub(shift, axis=1), 2.0 ** shift


# ---------------------------------------------------------------- drift

def loess_matrix(x_fit, x_eval, frac=None):
    """Linear smoother of local linear regression with tricube weights: fitted(x_eval) = S @ y(x_fit).

    The same S serves every feature measured at the same QC injections, so QC-LOESS is one matrix product.
    Span 0.75 (at least 4 QCs in a window), tricube bandwidth = distance to the k-th nearest QC, as in classic
    LOESS. Evaluation points are clipped to the QC range (flat extrapolation)."""
    x = np.asarray(x_fit, float)
    n = len(x)
    if frac is None:
        frac = LOESS_SPAN
    k = min(n, max(4, int(np.ceil(frac * n))))
    xe = np.clip(np.asarray(x_eval, float), x.min(), x.max())
    S = np.zeros((len(xe), n))
    for i, x0 in enumerate(xe):
        d = np.abs(x - x0)
        h = np.sort(d)[k - 1] * 1.0001
        if h <= 0:
            h = 1.0
        w = np.clip(1 - (d / h) ** 3, 0, None) ** 3
        A = np.column_stack([np.ones(n), x - x0])
        AtW = A.T * w
        M = AtW @ A
        if np.count_nonzero(w) < 2 or abs(np.linalg.det(M)) < 1e-10:
            S[i] = w / w.sum()
        else:
            S[i] = np.linalg.solve(M, AtW)[0]
    return S


def drift_correct(L, samples, mask, mode="auto", order_inferred=False, min_auto=5, min_loess=4):
    """QC-LOESS (QC-RSC) per batch: subtract the QC trend along run order, keep the batch's QC median level."""
    info = {"mode": mode, "corrected_batches": 0, "batches": 0, "notes": []}
    C = L.copy()
    if mode == "none":
        info["notes"].append("korekta dryfu wyłączona (drift=none)")
        return C, info
    if order_inferred and mode == "auto":
        info["notes"].append("brak kolejności iniekcji w metadanych – dryfu nie korygowano (podaj run_order)")
        return C, info
    s = samples.loc[L.columns]
    need = min_auto if mode == "auto" else min_loess
    for b, sb in s.groupby("_batch", sort=True):
        info["batches"] += 1
        qcs = list(sb.index[sb._type == "qc"])
        if len(qcs) < need:
            info["notes"].append(f"partia {b}: {len(qcs)} QC (< {need}) – bez korekty dryfu")
            continue
        cols = list(sb.index)
        xq = sb.loc[qcs, "_order"].to_numpy(float)
        xe = sb.loc[cols, "_order"].to_numpy(float)
        Y = L[qcs].where(mask[qcs]).to_numpy(float)
        miss = np.isnan(Y)
        fit = np.full((len(L), len(cols)), np.nan)
        pats, inv = np.unique(miss, axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        min_pts = max(3, int(np.ceil(0.6 * len(qcs))))
        for p, pat in enumerate(pats):
            avail = ~pat
            if avail.sum() < min_pts:
                continue
            rows = inv == p
            S = loess_matrix(xq[avail], xe)
            fit[rows] = Y[rows][:, avail] @ S.T
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            target = np.nanmedian(Y, axis=1)
        corr = fit - target[:, None]
        corr[~np.isfinite(corr)] = 0.0
        C[cols] = L[cols].to_numpy(float) - corr
        info["corrected_batches"] += 1
    return C, info


def batch_correct(L, samples, mask, mode="auto"):
    """Median centring per batch: on QC medians when every batch has >= 2 QCs, otherwise on sample medians."""
    s = samples.loc[L.columns]
    batches = list(pd.unique(s._batch))
    if mode == "none" or len(batches) < 2:
        why = "wyłączona (batch=none)" if mode == "none" else "jedna partia"
        return L.copy(), {"method": "none", "notes": [f"korekta partii: {why}"]}
    M = L.where(mask)
    notes = []
    if "?" in batches and len(batches) > 2:     # injections with no batch in the sheet cannot be aligned
        batches = [b for b in batches if b != "?"]
        notes.append(f"{int((s._batch == '?').sum())} iniekcji bez partii – zostawione bez korekty partii")
    qc_n = s[s._type == "qc"].groupby("_batch").size().reindex(batches, fill_value=0)
    method = "qc" if (qc_n >= 2).all() else "samples"
    kind = "qc" if method == "qc" else "sample"
    ref_all = [c for c in L.columns if s.loc[c, "_type"] == kind]
    target = M[ref_all].median(axis=1)
    C = L.copy()
    for b in batches:
        cols = list(s.index[s._batch == b])
        ref = [c for c in cols if s.loc[c, "_type"] == kind]
        if not ref:
            notes.append(f"partia {b}: brak próbek odniesienia – bez korekty")
            continue
        shift = (M[ref].median(axis=1) - target).fillna(0.0)
        C[cols] = L[cols].sub(shift, axis=0)
    if method == "samples":
        notes.append("partie wyrównane medianą próbek (za mało QC) – jeśli grupy są nierówno rozłożone w partiach, "
                     "część efektu biologicznego może zniknąć")
    return C, {"method": method, "notes": notes}


# ---------------------------------------------------------------- feature quality

def qc_cv(L, samples, mask):
    """CV (SD/mean on the linear scale) of the measured QC values; NaN with fewer than 3."""
    s = samples.loc[L.columns]
    qcs = list(s.index[s._type == "qc"])
    if len(qcs) < 3:
        return pd.Series(np.nan, index=L.index)
    lin = 2.0 ** L[qcs].where(mask[qcs])
    cv = lin.std(axis=1) / lin.mean(axis=1)
    return cv.where(lin.notna().sum(axis=1) >= 3)


def _mad(df):
    med = df.median(axis=1)
    return df.sub(med, axis=0).abs().median(axis=1) * MAD_SCALE


def d_ratio(L, samples, mask):
    """D-ratio = MAD(QC) / MAD(samples) on log2 (technical vs total variation; Broadhurst et al. 2018)."""
    s = samples.loc[L.columns]
    qcs = list(s.index[s._type == "qc"])
    st = list(s.index[s._type == "sample"])
    if len(qcs) < 3 or len(st) < 3:
        return pd.Series(np.nan, index=L.index)
    M = L.where(mask)
    return (_mad(M[qcs]) / _mad(M[st]).replace(0, np.nan)).where(M[qcs].notna().sum(axis=1) >= 3)


def replicate_delta(L, samples):
    """Median |log2 difference| between a sample and its technical replicate (not used by any correction)."""
    s = samples.loc[L.columns]
    if "_subject" not in s:
        return None
    pairs = []
    for r in s.index[s._type == "replicate"]:
        subj = s.loc[r, "_subject"]
        o = s.index[(s._type == "sample") & (s._subject == subj)]
        if len(o) and pd.notna(subj):
            pairs.append((o[0], r))
    if not pairs:
        return None
    return float(np.median([(L[a] - L[b]).abs().median() for a, b in pairs]))


def qc_presence_filter(X, samples, na_max=0.5, min_qc=3):
    """Keep a feature detected in at least (1 - na_max) of the pooled QCs: what the QC does not show cannot be
    checked for repeatability. Skipped (all kept) with fewer than `min_qc` QCs."""
    s = samples.loc[X.columns]
    qcs = list(s.index[s._type == "qc"])
    if len(qcs) < min_qc:
        return pd.Series(True, index=X.index), None
    frac = X[qcs].isna().mean(axis=1)
    return frac <= na_max, frac


REPLICATE_NAME = [re.compile(r"(?i)^(.*?)[\s_\-.]*(?:replicate|repl|rep|tech|inj|r)[\s_\-]?\d+$"),
                  re.compile(r"^(.*\d)[A-Za-z]$")]
RUN_NUMBER = re.compile(r"^(.+\d[A-Za-z])[\s_]+\d{2,}$")


def _replicate_stem(name):
    """'K_1A' -> 'K_1', 'A0_r2' -> 'A0', 'DA 12_1B 1207' -> 'DA 12_1'; None when the name marks no replicate
    (a bare '_1' / '_2' is a sample number, not a replicate)."""
    text = str(name).strip()
    m = RUN_NUMBER.match(text)
    if m:
        text = m.group(1)
    for pattern in REPLICATE_NAME:
        m = pattern.match(text)
        if m and m.group(1):
            return m.group(1)
    return None


def replicate_units(samples, cols):
    """Injection -> biological sample it belongs to. The same subject (`_pair`, else `_subject`) in the same group
    is one unit; without subjects the replicate mark in the name decides (1A/1B, _rep2). A unit with one injection
    keeps the injection's name."""
    s = samples.loc[cols]
    subj = None
    for c in ("_pair", "_subject"):
        if c in s and s[c].notna().any():
            subj = s[c]
            break
    group = s["_group"] if "_group" in s else pd.Series(None, index=s.index)
    keys = {}
    for c in cols:
        g = group[c] if pd.notna(group[c]) else ""
        if subj is not None and pd.notna(subj[c]):
            keys[c] = (str(subj[c]), str(g))
        else:
            stem = _replicate_stem(c) if subj is None else None
            keys[c] = (stem, str(g)) if stem else (c, None)
    members = {}
    for c, k in keys.items():
        members.setdefault(k, []).append(c)
    labels = {}
    for k, cs in members.items():
        label = cs[0] if len(cs) == 1 else " ".join(x for x in k if x)
        for c in cs:
            labels[c] = label
    return pd.Series(labels).reindex(cols)


def spare_whole_samples(samples, flags):
    """Un-reject injections when every replicate of a biological sample (>= 2 injections) is flagged the same way
    for missing values or total signal: that is how the sample is, not a failed injection. A flag from the internal
    standards stays. Returns the spared unit labels; `flags` is changed in place."""
    cols = [c for c in flags.index if samples.loc[c, "_type"] == "sample"]
    if not cols:
        return []
    units = replicate_units(samples, cols)
    spared = []
    for u, members in units.groupby(units).groups.items():
        members = list(members)
        if len(members) < 2 or not flags.loc[members, "rejected"].all():
            continue
        if flags.loc[members, "reason"].str.contains("IS").all():
            continue
        keep = [c for c in members if "IS" not in flags.loc[c, "reason"]]
        flags.loc[keep, "rejected"] = False
        flags.loc[keep, "reason"] = "zostawiona (odstaje cała próbka): " + flags.loc[keep, "reason"]
        spared.append(str(u))
    return spared


def cluster_representatives(L, features):
    """One feature per analytical cluster (adducts/isotopes of one compound): the one with the highest median."""
    if "cluster" not in features or features["cluster"].reindex(L.index).isna().all():
        return list(L.index)
    cl = features["cluster"].reindex(L.index)
    med = L.median(axis=1)
    single = list(cl.index[cl.isna()])
    rep = med[cl.notna()].groupby(cl[cl.notna()]).idxmax().tolist()
    keep = set(single) | set(rep)
    return [f for f in L.index if f in keep]
