"""Variable selection on the training set only, scored by Q²LOO of MLR (regression) or by the cross-validated
balanced accuracy of logistic regression (classification). The test set is never seen here.

Subsets whose descriptors are collinear (VIF > 10) are penalised, so they are chosen only when nothing else exists.
The number of descriptors is the smallest one whose score is within DELTA of the best size (parsimony).
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

VIF_MAX = 10.0
DELTA = 0.02  # a bigger model has to be better by more than this
EXHAUSTIVE_BUDGET_S = 300.0  # select=auto runs exhaustive search only when the estimate is below this
EXHAUSTIVE_HARD_S = 4 * 3600.0  # even select=exhaustive gives up (falls back to GA) above this


@dataclass
class SelectionResult:
    chosen: list
    q2: float
    method: str
    per_k: list = field(default_factory=list)  # dicts: k, cols, score, r2, vif_max
    notes: list = field(default_factory=list)
    score_name: str = "Q²LOO"


# ---------------------------------------------------------------- scorers

def q2loo_batch(x, y, subsets):
    """Q²LOO, R² and max VIF of MLR for many descriptor subsets at once (closed-form LOO, batched).

    x must be autoscaled (columns mean 0, sd 1), subsets: int array (B, k). Returns three arrays of length B.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    subsets = np.asarray(subsets, dtype=int)
    n = len(y)
    b, k = subsets.shape
    xs = np.transpose(x[:, subsets], (1, 0, 2))  # (B, n, k)
    a = np.concatenate([np.ones((b, n, 1)), xs], axis=2)
    ata = np.einsum("bni,bnj->bij", a, a) + np.eye(k + 1) * 1e-9 * n
    inv = np.linalg.inv(ata)
    beta = np.einsum("bij,bnj,n->bi", inv, a, y)
    resid = y[None, :] - np.einsum("bni,bi->bn", a, beta)
    h = np.einsum("bni,bij,bnj->bn", a, inv, a)
    one_minus = 1 - h
    bad = (one_minus < 1e-8).any(1)
    press = ((resid / np.clip(one_minus, 1e-8, None)) ** 2).sum(1)
    ss = ((y - y.mean()) ** 2).sum()
    q2 = 1 - press / ss
    r2 = 1 - (resid ** 2).sum(1) / ss
    q2[bad] = -np.inf
    if k == 1:
        vif = np.ones(b)
    else:
        corr = np.einsum("bni,bnj->bij", xs, xs) / (n - 1)
        d = np.sqrt(np.einsum("bii->bi", corr))
        corr = corr / d[:, :, None] / d[:, None, :]
        try:
            vif = np.einsum("bii->bi", np.linalg.inv(corr)).max(1)
        except np.linalg.LinAlgError:
            vif = np.array([_vif_one(c) for c in corr])
    return q2, r2, vif


def _vif_one(corr):
    try:
        return float(np.diag(np.linalg.inv(corr)).max())
    except np.linalg.LinAlgError:
        return np.inf


class RegressionScorer:
    name = "Q²LOO"

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.chunk = max(50, int(4e6 / (len(y) * 6)))

    def __call__(self, subsets):
        subsets = np.asarray(subsets, dtype=int)
        out = [q2loo_batch(self.x, self.y, subsets[i:i + self.chunk]) for i in range(0, len(subsets), self.chunk)]
        q2, r2, vif = (np.concatenate(v) for v in zip(*out, strict=True))
        return np.where(vif > VIF_MAX, q2 - 1.0, q2), q2, r2, vif


class ClassificationScorer:
    """Balanced accuracy of a balanced logistic regression in stratified CV (fast enough to search with)."""
    name = "BA(CV)"

    def __init__(self, x, y, seed=0):
        from sklearn.model_selection import StratifiedKFold
        self.x, self.y = x, np.asarray(y)
        _, counts = np.unique(self.y, return_counts=True)
        self.cv = list(StratifiedKFold(n_splits=max(2, min(5, counts.min())), shuffle=True,
                                       random_state=seed).split(x, self.y))
        self.chunk = 1

    def _one(self, cols):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score
        xs = self.x[:, cols]
        pred = np.empty(len(self.y), dtype=self.y.dtype)
        for tr, te in self.cv:
            m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(xs[tr], self.y[tr])
            pred[te] = m.predict(xs[te])
        fit = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(xs, self.y)
        return balanced_accuracy_score(self.y, pred), balanced_accuracy_score(self.y, fit.predict(xs))

    def __call__(self, subsets):
        subsets = np.asarray(subsets, dtype=int)
        res = np.array([self._one(list(s)) for s in subsets]).reshape(-1, 2)
        score, train = res[:, 0], res[:, 1]
        k = subsets.shape[1]
        if k == 1:
            vif = np.ones(len(subsets))
        else:
            vif = np.array([_vif_one(np.corrcoef(self.x[:, list(s)], rowvar=False)) for s in subsets])
        return np.where(vif > VIF_MAX, score - 1.0, score), score, train, vif


# ---------------------------------------------------------------- search strategies

def _record(per_k, k, subset, names, fit, score, r2, vif):
    per_k.append({"k": k, "cols": [names[i] for i in subset], "idx": list(map(int, subset)),
                  "fitness": float(fit), "score": float(score), "r2": float(r2), "vif_max": float(vif)})


def n_subsets(p, k_max):
    return sum(math.comb(p, k) for k in range(1, k_max + 1))


def estimate_exhaustive_s(scorer, p, k_max, sample=400):
    """Seconds an exhaustive search would take, measured on a sample of the largest subsets."""
    k = min(k_max, p)
    rng = np.random.default_rng(0)
    if isinstance(scorer, ClassificationScorer):
        sample = 10
    m = min(sample, math.comb(p, k))
    subs = np.array([np.sort(rng.choice(p, k, replace=False)) for _ in range(m)])
    t0 = time.perf_counter()
    scorer(subs)
    per = (time.perf_counter() - t0) / m
    return per * n_subsets(p, k_max)


def exhaustive(scorer, p, k_max, names, progress=None):
    per_k = []
    for k in range(1, k_max + 1):
        best = (-np.inf, None, 0, 0, 0)
        it = itertools.combinations(range(p), k)
        chunk = max(1000, scorer.chunk * 4)
        done = 0
        total = math.comb(p, k)
        while True:
            block = np.fromiter(itertools.islice(it, chunk), dtype=np.dtype((np.int64, k)))
            if not len(block):
                break
            fit, sc, r2, vif = scorer(block)
            i = int(np.argmax(fit))
            if fit[i] > best[0]:
                best = (fit[i], block[i], sc[i], r2[i], vif[i])
            done += len(block)
            if progress and total > 50 * chunk and done % (10 * chunk) == 0:
                progress(f"wybór deskryptorów: {k} z {p}, {100 * done // total}%")
        _record(per_k, k, best[1], names, best[0], best[2], best[3], best[4])
    return per_k


def forward(scorer, p, k_max, names, progress=None):
    per_k, current = [], []
    for k in range(1, k_max + 1):
        cand = [j for j in range(p) if j not in current]
        if not cand:
            break
        subs = np.array([sorted(current + [j]) for j in cand])
        fit, sc, r2, vif = scorer(subs)
        i = int(np.argmax(fit))
        current = list(subs[i])
        _record(per_k, k, subs[i], names, fit[i], sc[i], r2[i], vif[i])
        if progress:
            progress(f"wybór deskryptorów (forward): {k}/{k_max}")
    return per_k


def genetic(scorer, p, k_max, names, seed=0, pop=60, generations=100, patience=25, restarts=2, progress=None):
    """Genetic algorithm for each model size k: tournament selection, crossover from the parents' union,
    point mutation, elitism. Small k (few combinations) is searched exhaustively instead."""
    per_k = []
    small = 300_000  # batched Q²LOO: this many subsets take seconds, so search them all
    if isinstance(scorer, ClassificationScorer):
        pop, generations, patience, restarts = 30, 40, 12, 1
        small = pop * 50
    for k in range(1, k_max + 1):
        if math.comb(p, k) <= small:
            best = exhaustive(scorer, p, k, names)[-1]
            per_k.append(best | {"k": k})
            continue
        cache = {}

        def evaluate(subs, cache=cache):
            new = [s for s in {tuple(s) for s in subs} if s not in cache]
            if new:
                for s, *vals in zip(new, *scorer(np.array(new)), strict=True):
                    cache[s] = vals
            return np.array([cache[tuple(s)][0] for s in subs])

        overall = None
        for r in range(restarts):
            rng = np.random.default_rng(seed * 1000 + k * 10 + r)
            population = [tuple(sorted(rng.choice(p, k, replace=False))) for _ in range(pop)]
            if per_k:  # a quarter of the start: the best subset of size k-1 plus one more descriptor
                prev = per_k[-1]["idx"]
                extra = [j for j in rng.permutation(p) if j not in prev][: pop // 4]
                population[: len(extra)] = [tuple(sorted(prev + [int(j)])) for j in extra]
            best_fit, stale = -np.inf, 0
            for _gen in range(generations):
                fit = evaluate(population)
                order = np.argsort(-fit)
                if fit[order[0]] > best_fit + 1e-12:
                    best_fit, stale = fit[order[0]], 0
                else:
                    stale += 1
                if stale >= patience:
                    break
                nxt = [population[order[0]], population[order[1]]]
                while len(nxt) < pop:
                    a = population[max(rng.choice(pop, 3, replace=False), key=lambda i: fit[i])]
                    b = population[max(rng.choice(pop, 3, replace=False), key=lambda i: fit[i])]
                    pool = sorted(set(a) | set(b))
                    child = set(rng.choice(pool, k, replace=False).tolist())
                    if rng.random() < 0.3 or tuple(sorted(child)) in (a, b):
                        out = rng.choice(sorted(child))
                        child.discard(out)
                        child.add(int(rng.choice([j for j in range(p) if j not in child])))
                    nxt.append(tuple(sorted(int(c) for c in child)))
                population = nxt
            top = max(cache, key=lambda s: cache[s][0])
            if overall is None or cache[top][0] > cache[overall][0]:
                overall = top
        if isinstance(scorer, RegressionScorer):
            overall = replacement(scorer, p, overall, cache)
        _record(per_k, k, overall, names, *cache[overall])
        if progress:
            progress(f"wybór deskryptorów (GA): {k}/{k_max}")
    return per_k


def replacement(scorer, p, subset, cache, max_rounds=20):
    """Sequential replacement: swap each descriptor for every other one while that improves the fitness."""
    best = tuple(subset)
    for _ in range(max_rounds):
        cands = {tuple(sorted(set(best) - {out} | {j})) for out in best for j in range(p) if j not in best}
        new = [c for c in cands if c not in cache]
        if new:
            for sub, *vals in zip(new, *scorer(np.array(new)), strict=True):
                cache[sub] = vals
        top = max(cands, key=lambda c: cache[c][0])
        if cache[top][0] <= cache[best][0] + 1e-12:
            break
        best = top
    return best


def enet_order(x, y, seed=0, task="regression"):
    """Descriptors ordered by |coefficient| of a cross-validated elastic net (zero coefficients dropped)."""
    if task == "regression":
        from sklearn.linear_model import ElasticNetCV
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=min(5, len(y)), shuffle=True, random_state=seed)
        m = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0], cv=cv, max_iter=100000, n_jobs=1).fit(x, y)
        coef = np.abs(m.coef_)
    else:
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.model_selection import StratifiedKFold
        _, counts = np.unique(y, return_counts=True)
        cv = StratifiedKFold(n_splits=max(2, min(5, counts.min())), shuffle=True, random_state=seed)
        m = LogisticRegressionCV(Cs=10, cv=cv, solver="saga", l1_ratios=[0.5, 0.9, 1.0], class_weight="balanced",
                                 max_iter=5000, scoring="balanced_accuracy", **_penalty_kw()).fit(x, y)
        coef = np.abs(np.atleast_2d(m.coef_)).max(0)
    order = [int(j) for j in np.argsort(-coef, kind="stable") if coef[j] > 1e-10]
    return order


def _penalty_kw():
    import inspect

    from sklearn.linear_model import LogisticRegressionCV
    default = inspect.signature(LogisticRegressionCV).parameters.get("penalty")
    return {} if default is not None and default.default == "deprecated" else {"penalty": "elasticnet"}


def choose_k(per_k):
    """Smallest model whose fitness is within DELTA of the best one."""
    best = max(r["fitness"] for r in per_k)
    return next(r for r in per_k if r["fitness"] >= best - DELTA)


def select(x, y, names, method="auto", k_max=4, seed=0, task="regression", progress=None):
    """Pick descriptors on (autoscaled) training data. Returns SelectionResult with the chosen names."""
    x = np.asarray(x, float)
    p = x.shape[1]
    k_max = max(1, min(k_max, p))
    scorer = RegressionScorer(x, y) if task == "regression" else ClassificationScorer(x, y, seed)
    notes = []
    if method == "all":
        return SelectionResult(list(names), float("nan"), "all", [], ["Bez wyboru: wszystkie deskryptory po odsiewie."],
                               scorer.name)
    used = method
    if method in ("auto", "exhaustive"):
        est = estimate_exhaustive_s(scorer, p, k_max)
        limit = EXHAUSTIVE_BUDGET_S if method == "auto" else EXHAUSTIVE_HARD_S
        if est <= limit:
            used = "exhaustive"
            if est > 30:
                notes.append(f"Przeszukanie wszystkich {n_subsets(p, k_max):,} podzbiorów (~{est / 60:.0f} min).")
        else:
            used = "ga"
            notes.append(f"Pełne przeszukanie trwałoby ~{est / 3600:.1f} h ({n_subsets(p, k_max):.2e} podzbiorów), "
                         "więc algorytm genetyczny.")
    if used == "exhaustive":
        per_k = exhaustive(scorer, p, k_max, names, progress)
    elif used == "forward":
        per_k = forward(scorer, p, k_max, names, progress)
    elif used == "ga":
        per_k = genetic(scorer, p, k_max, names, seed=seed, progress=progress)
    elif used == "enet":
        order = enet_order(x, y, seed, task)
        if not order:
            notes.append("Elastic net wyzerował wszystkie współczynniki – użyto forward.")
            used = "forward"
            per_k = forward(scorer, p, k_max, names, progress)
        else:
            per_k = []
            for k in range(1, min(k_max, len(order)) + 1):
                sub = np.array([sorted(order[:k])])
                fit, sc, r2, vif = scorer(sub)
                _record(per_k, k, sub[0], names, fit[0], sc[0], r2[0], vif[0])
    else:
        raise ValueError(method)
    best = choose_k(per_k)
    if best["vif_max"] > VIF_MAX:
        notes.append(f"Uwaga: wybrane deskryptory są współliniowe (VIF = {best['vif_max']:.0f}).")
    return SelectionResult(best["cols"], best["score"], used, per_k, notes, scorer.name)
