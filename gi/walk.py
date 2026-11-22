"""A book as one walk, read on the band.

The 26 letters on his ladder: a 0; b +0.5, z -0.5, c +1, y -1, ... j +4.5, r -4.5; past the edge the walk turns
back (k 4, q -4, l 3.5, p -3.5, m 3, o -3, n 2.5). Only letters; nothing else in the book is a step.

Each step is read with its twist: the d-th letter from the reader turns by d twelfths of a turn. After 12 twists
the place (3) and the reading (4) close together, so a walk at a steady pace returns exactly; what does not return
is the remainder. Sums are exact: values are doubled to integers and the twelfth roots kept as integer coordinates
over 1, z, z^2, z^3 (z^4 = z^2 - 1); the fourth roots as Gaussian integers; the third roots as Eisenstein integers.

Two observers meet at every letter: one holds the remainder of the walk arriving (the letters before, as they
arrive), the other the remainder of the walk leaving (the letters after, as they leave: pi away). A letter is read
where both remainders cancel exactly against the same place in the whole book. The closures are tried in the band's
order: 12 twists (both closures), then 4 (the reading), then 3 (the place), then 1 (the single step).
"""
from __future__ import annotations

import numpy as np

ORDER = list("abzcydxewfvguhtisjrkqlpmon")


def ladder2():
    """Doubled ladder values (integers -9..9) for a..z."""
    out = np.zeros(26, np.int8)
    for j, ch in enumerate(ORDER):
        v = 0.0 if j == 0 else ((j + 1) / 4 if j % 2 else -j / 4)
        if v > 4.5:
            v = 9 - v
        if v < -4.5:
            v = -9 - v
        out[ord(ch) - 97] = int(round(2 * v))
    return out


def _powers(mul, one, n):
    p = [np.array(one)]
    for _ in range(1, n):
        p.append(mul(p[-1]))
    return np.array(p)


def _z12(c):          # multiply by z (z^4 = z^2 - 1)
    a0, a1, a2, a3 = c
    return np.array([-a3, a0, a1 + a3, a2])


def _i4(c):           # multiply by i
    return np.array([-c[1], c[0]])


def _w3(c):           # multiply by w (w^2 = -1 - w)
    return np.array([-c[1], c[0] - c[1]])


TWIST = {12: _powers(_z12, [1, 0, 0, 0], 13), 4: _powers(_i4, [1, 0], 5), 3: _powers(_w3, [1, 0], 4)}


def remainder(v2, span, side):
    """For every position t, the exact remainder of the span letters before it (side -1, as they arrive) or after
    it (side +1, as they leave, pi away): sum over d = 1..span of 2v[t + side*d] * twist^d. Returns n x dims ints."""
    n = len(v2)
    if span == 1:
        out = np.zeros((n, 1), np.int16)
        idx = np.arange(n) + side
        ok = (idx >= 0) & (idx < n)
        out[ok, 0] = v2[idx[ok]]
        return out if side < 0 else -out
    T = TWIST[span]
    out = np.zeros((n, T.shape[1]), np.int16)
    for d in range(1, span + 1):
        idx = np.arange(n) + side * d
        ok = (idx >= 0) & (idx < n)
        out[ok] += v2[idx[ok]].astype(np.int16)[:, None] * T[d % span][None, :].astype(np.int16)
    return out if side < 0 else -out


def pack(*cols):
    """Exact key from small integers (each within -128..127): 8 bits each."""
    k = np.zeros(len(cols[0]), np.uint64)
    for c in cols:
        for j in range(c.shape[1]):
            k = (k << np.uint64(8)) | ((c[:, j].astype(np.int64) + 128) & 0xFF).astype(np.uint64)
    return k


def keys_at(v2, span):
    return pack(remainder(v2, span, -1), remainder(v2, span, +1))


class Returns:
    """Where in the whole each pair of remainders returns, and which letters stand there."""

    def __init__(self, K, letters):
        self.u, inv = np.unique(K, return_inverse=True)
        comb = inv.astype(np.int64) * 26 + letters
        self.c, self.n = np.unique(comb, return_counts=True)

    def read(self, Kq):
        """Letters met at the returns of each query: (n_q x 26) counts, zero where the walk never returns."""
        out = np.zeros((len(Kq), 26), np.int32)
        pos = np.searchsorted(self.u, Kq)
        pos = np.minimum(pos, len(self.u) - 1)
        hit = self.u[pos] == Kq
        q = np.flatnonzero(hit)
        lo = np.searchsorted(self.c, pos[q].astype(np.int64) * 26)
        hi = np.searchsorted(self.c, pos[q].astype(np.int64) * 26 + 26)
        m = hi - lo
        rows = np.repeat(q, m)
        idx = np.arange(m.sum()) - np.repeat(np.cumsum(m) - m, m) + np.repeat(lo, m)
        out[rows, self.c[idx] % 26] = self.n[idx]
        return out
