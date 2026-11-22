"""Letters close within their word; words enter one scale up as single steps.

A silence is not a step: it bounds the walk. A letter's arriving walk is the letters before it in its word, its
leaving walk the letters after it in its word (pi away), each with its twist (the d-th letter d twelfths of a turn),
each stopping where the silence is. One scale up, a word enters as one step: its value is the exact remainder of its
own walk (each letter twisted by its place in the word) with the number of steps to its silence. The word before
and the word after are the arriving and leaving observers of that scale.
"""
from __future__ import annotations

import re

import numpy as np

from . import walk as W

MAXLEN = 30


def book(entries):
    txt = " ".join(f"{k} {v}" for k, v in entries.items())
    return from_words(words_of(txt))


def words_of(txt, keep_unknown=False):
    """Words of a text: runs of letters. An unknown-word placeholder (<unk>) is kept as an empty word: it bounds the
    walks and, having no letters, never returns."""
    txt = txt.lower().translate(str.maketrans("éèêëàâäôöûüîïç", "eeeeaaaoouuiic"))
    pat = r"<unk>|[a-z]+" if keep_unknown else r"[a-z]+"
    return ["" if w == "<unk>" else w for w in re.findall(pat, txt) if len(w) <= MAXLEN]


def from_words(words):
    lens = np.array([len(w) for w in words])
    letters = (np.frombuffer("".join(words).encode(), np.uint8) - 97).astype(np.int64)
    wid = np.repeat(np.arange(len(words)), lens)
    start = np.concatenate([[0], np.cumsum(lens)[:-1]])
    pos = np.arange(len(letters)) - start[wid]
    return letters, wid, pos, lens[wid], lens


def mix(x):
    """splitmix64: scrambles every bit of a 64-bit value (no linear structure survives)."""
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def h(*cols):
    """Key from integer columns: every column scrambled with its own index, then folded in through a scramble.
    (An earlier version combined columns with the same multiplier that builds the word hashes: neighbouring words
    could cancel a difference in the word itself. 61 of 251,607 keys collided.)"""
    k = np.zeros(len(cols[0]), np.uint64)
    j = 0
    for c in cols:
        c = c if c.ndim == 2 else c[:, None]
        for i in range(c.shape[1]):
            j += 1
            v = c[:, i].astype(np.int64).view(np.uint64)
            k = mix(k ^ mix(v + np.uint64(j) * np.uint64(0x632BE59BD9B4E019)))
    return k


def within_word(v2, pos, wlen):
    """Arriving and leaving remainders of every letter, bounded by its word's silences."""
    n = len(v2)
    T = W.TWIST[12]
    before = np.zeros((n, 4), np.int32)
    after = np.zeros((n, 4), np.int32)
    for d in range(1, MAXLEN):
        ok = pos >= d
        idx = np.flatnonzero(ok)
        before[idx] += v2[idx - d].astype(np.int32)[:, None] * T[d % 12][None, :]
        ok = (wlen - pos - 1) >= d
        idx = np.flatnonzero(ok)
        after[idx] -= v2[idx + d].astype(np.int32)[:, None] * T[d % 12][None, :]
    return before, after


def word_values(v2, wid, pos, lens):
    """Each word as one step: the exact remainder of its own walk, and its length."""
    T = W.TWIST[12]
    rem = np.zeros((len(lens), 4), np.int64)
    np.add.at(rem, wid, v2.astype(np.int64)[:, None] * T[pos % 12])
    v = h(rem, lens)
    v[lens == 0] = np.uint64(1)                     # an unknown word: never equal to a real word's value
    return v


def partial(v2, pos, wlen, span, side):
    """The walk over at most `span` letters arriving (side -1) or leaving (side +1, pi away), never past the word's
    silence; twists of the span's own closure (4: the reading, 3: the place, 1: the single step). Returns the
    remainder and how many steps were walked before the silence."""
    n = len(v2)
    T = W.TWIST[span] if span in (3, 4) else np.array([[1]])
    out = np.zeros((n, T.shape[1]), np.int16)
    avail = pos if side < 0 else (wlen - pos - 1)
    for d in range(1, span + 1):
        idx = np.flatnonzero(avail >= d)
        out[idx] += v2[idx + side * d].astype(np.int16)[:, None] * T[d % len(T)][None, :].astype(np.int16)
    return (out if side < 0 else -out), np.minimum(avail, span).astype(np.int8)
