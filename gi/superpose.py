"""Neural superposition reader. A letter is seen by every observer whose span covers it from inside its word: a letters
before it and b after it (a, b <= 4, edges marked where they touch silence), and the whole word. Each observer closes
against the whole on its own twisted walk (exact remainders, arriving and leaving pi away). Their readings are summed,
unweighted, as a fly's lamina cartridge sums the receptors that look at one point; the letter is read where the sum
peaks."""
from __future__ import annotations

import numpy as np

from . import walk as W
from . import words as WD

T12 = W.TWIST[12].astype(np.int16)


def span_key(v2, pos, wlen, a, b):
    """Key of the observer that sees each letter with exactly a letters before and b after (within the word), or 0
    where the word is too short for it."""
    n = len(v2)
    ok = (pos >= a) & ((wlen - pos - 1) >= b)
    rb = np.zeros((n, 4), np.int16); ra = np.zeros((n, 4), np.int16)
    idx = np.flatnonzero(ok)
    for d in range(1, a + 1):
        rb[idx] += v2[idx - d].astype(np.int16)[:, None] * T12[d % 12][None, :]
    for d in range(1, b + 1):
        ra[idx] -= v2[idx + d].astype(np.int16)[:, None] * T12[d % 12][None, :]
    start = (pos == a).astype(np.int8); end = ((wlen - pos - 1) == b).astype(np.int8)
    shape = np.full(n, a * 16 + b, np.int16)
    k = WD.h(rb, ra, start, end, shape)
    k[~ok] = 0
    return k


def observers(maxspan=4):
    return [(a, b) for a in range(maxspan + 1) for b in range(maxspan + 1) if (a, b) != (0, 0)]
