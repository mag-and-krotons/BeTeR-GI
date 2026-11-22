"""The given wholes.

optdigits   data/optdigits.tes from BeTeR-Intelligence (1797 images, 8 x 8, values 0..16, UCI, CC BY 4.0), read as
            values / 16, with his stratified 70/30 split (random_state 42) for the comparison with stack6.
Webster     Project Gutenberg's Webster's Unabridged Dictionary as published in JSON by matthewreagan/
            WebstersEnglishDictionary (102,217 entries). gutenberg.org is not reachable from this build, so his exact
            29765.txt stream is replaced by this edition of the same dictionary.
Wikipedia   the reachable present-day Wikipedia text: WikiText-2 (Good and Featured English Wikipedia articles,
            2.5 million words, rare words replaced by <unk>), from pytorch/examples. Full English Wikipedia
            (dumps.wikimedia.org) is not reachable from this build.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def optdigits():
    a = np.loadtxt(os.path.join(DATA, "optdigits.tes"), delimiter=",", dtype=np.int64)
    return a[:, :64] / 16.0, a[:, 64]


def his_split(y, test_size=0.3, seed=42):
    """BeTeR-Intelligence digits/stack6.py split: per class a fixed permutation (RandomState(seed)), first share held
    out. Returns index arrays (train, test) into the full set."""
    rng = np.random.RandomState(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        idx = idx[rng.permutation(len(idx))]
        k = int(round(len(idx) * test_size))
        te += list(idx[:k])
        tr += list(idx[k:])
    return np.array(sorted(tr)), np.array(sorted(te))


WORD = re.compile(r"[a-z]+|[0-9]+")


def webster_tokens():
    d = json.load(open(os.path.join(DATA, "webster_mr.json")))
    out = []
    for k, v in d.items():
        out += WORD.findall(k.lower())
        out += WORD.findall(v.lower())
    return out


def wikipedia_tokens():
    out = []
    for part in ("train", "valid", "test"):
        with open(os.path.join(DATA, f"wt2_{part}.txt"), encoding="utf8") as f:
            for line in f:
                out += [t for t in WORD.findall(line.lower()) if t != "unk"]
    return out


def webster():
    return json.load(open(os.path.join(DATA, "webster_mr.json")))
