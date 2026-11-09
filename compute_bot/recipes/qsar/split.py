"""Splitting compounds into training and test sets. Every function returns a boolean mask: True = test."""
from __future__ import annotations

import unicodedata

import numpy as np


def sorted_split(y, every=3):
    """Sort by the response, put every k-th compound (k-th, 2k-th, ...) into the test set.

    The compounds with the lowest and highest response always stay in training, so the model is never asked to
    extrapolate the response range on the test set.
    """
    y = np.asarray(y)
    n = len(y)
    order = np.argsort(y, kind="stable")
    test = np.zeros(n, bool)
    for pos in range(every - 1, n, every):
        if 0 < pos < n - 1:
            test[order[pos]] = True
    return test


def kennard_stone(x, n_select):
    """Kennard–Stone: start with the two most distant points, then repeatedly add the point farthest from the
    already selected ones (max-min Euclidean distance). Returns indices in the order they were picked."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    n_select = min(n_select, n)
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
    i, j = np.unravel_index(np.argmax(d), d.shape)
    picked = [int(min(i, j)), int(max(i, j))][:n_select]
    min_d = np.minimum(d[picked[0]], d[picked[-1]])
    min_d[picked] = -1
    while len(picked) < n_select:
        nxt = int(np.argmax(min_d))
        picked.append(nxt)
        min_d = np.minimum(min_d, d[nxt])
        min_d[picked] = -1
    return picked


def ks_split(x, test_frac):
    """Training = the points Kennard–Stone picks first; the rest is the test set."""
    n = len(x)
    n_test = max(1, int(round(test_frac * n)))
    train = kennard_stone(x, n - n_test)
    test = np.ones(n, bool)
    test[train] = False
    return test


def random_split(n, test_frac, seed=42, strata=None):
    """Random split with a fixed seed; stratified by class when strata are given."""
    rng = np.random.default_rng(seed)
    test = np.zeros(n, bool)
    if strata is None:
        n_test = max(1, int(round(test_frac * n)))
        test[rng.permutation(n)[:n_test]] = True
        return test
    strata = np.asarray(strata)
    for value in sorted(set(strata.tolist()), key=str):
        idx = np.flatnonzero(strata == value)
        n_test = int(round(test_frac * len(idx)))
        if len(idx) >= 2:
            n_test = min(max(1, n_test), len(idx) - 1)
        else:
            n_test = 0
        test[rng.permutation(idx)[:n_test]] = True
    return test


_TRAIN_WORDS = ("t", "u", "c", "l", "tr", "train", "training", "trening", "uczacy", "kalibracyjny")
_TEST_WORDS = ("v", "w", "e", "p", "te", "ts", "test", "testowy", "walidacyjny", "validation")
_TRAIN_PREFIX = ("tr", "ucz", "learn", "cal", "kalib", "fit", "model")
_TEST_PREFIX = ("te", "wal", "val", "ext", "zewn", "pred", "progn")


def _norm(text):
    text = str(text).replace("ł", "l").replace("Ł", "L")
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower().strip()


def parse_split_labels(values):
    """Map T/V, train/test, U/W, uczący/walidacyjny/testowy ... to 'train' / 'test'; None for empty cells."""
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
            out.append(None)
            continue
        s = _norm(v)
        # whole words first (single letters are ambiguous as prefixes), then prefixes
        if s in _TRAIN_WORDS:
            out.append("train")
        elif s in _TEST_WORDS:
            out.append("test")
        elif s.startswith(_TRAIN_PREFIX):
            out.append("train")
        elif s.startswith(_TEST_PREFIX):
            out.append("test")
        else:
            raise ValueError(str(v))
    return np.array(out, dtype=object)
