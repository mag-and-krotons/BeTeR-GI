"""The reader: one act for every piece (brief, section 7).

The reader has a standing: a position on the band, or rest (no position, the centre: equally far from every
position). A piece is read by one act:

  seen      the piece is read by its two members against the whole it belongs to: the still alone, and the still
            with the turn. Each member names the closure that leaves least behind. If both name the same closure the
            piece closes once. If they name different closures the piece closes into both (8 and 1 together).
  meeting   seen and seer meet at 1/2: the seen says which closures exist, the seer says which is met first, the
            one standing nearest on the band. Neither weighs the other. At rest, or where the closures stand equally
            far, the seen's own order holds (what leaves least behind with the turn).
  remainder what the closure does not hold returns as it is. Its position, if it has one, is the output, and the
            reader stands there for the next piece. A piece with no place on the band leaves the reader at rest.

What a piece is read against depends on what it is; the act does not:
  written digit   laps written by construction from the given images (his stack6 move: three writings a, b, c;
                  still = the middle, turn = a - c), each lap at its key's position on the band
  construction    the band itself: 12 closes, the walk forward and the walk reversed are the two members
  spoken word     the given whole's words as walks on the wheel (tone = cell on the 30-wheel, each letter one twist
                  further along the band); forward and reversed walks give still and turn; the word's position is
                  the one its definitions in the whole construct
"""
from __future__ import annotations

import re

import numpy as np

from . import band

VOWEL_CELL = {"a": 5, "e": 7, "i": 11, "o": 13, "u": 17}
CONS = [c for c in "abcdefghijklmnopqrstuvwxyz" if c not in VOWEL_CELL]
FREE = [n for n in range(4, 30) if n not in VOWEL_CELL.values()]
WHEEL = {**VOWEL_CELL, **dict(zip(CONS, FREE))}          # his 30-cell wheel (BeTeR-Intelligence textcells.py)


# ------------------------------------------------------------------ the seen: digits
def write(F, keys, k, rng=None, order=None):
    """Laps from the given images (his stack6 move), written in the order of a route or of rng.permutation."""
    mem = {}
    rank = None if order is None else {v: i for i, v in enumerate(order)}
    for c in np.unique(keys):
        idx = np.flatnonzero(keys == c)
        idx = idx[rng.permutation(len(idx))] if order is None else np.array(sorted(idx, key=rank.get))
        I = F[idx]
        t = len(I) // 3
        a, b, cc = I[:t], I[t:2 * t], I[2 * t:3 * t]
        fwd, rev = (a - b) - cc, (cc - b) - a
        still = -((fwd + rev) / 2).mean(0)
        turn = (fwd - rev) / 2
        _, _, Vh = np.linalg.svd(turn - turn.mean(0), full_matrices=False)
        mem[int(c)] = (still, Vh[:k])
    return mem


def left(mem, S, turns=True):
    """What each lap leaves behind of each piece: the still alone, or the still turned in the lap's directions."""
    cs = sorted(mem)
    out = []
    for c in cs:
        m, V = mem[c]
        D = S - m
        r = np.sum(D ** 2, 1)
        if turns and len(V):
            r = r - np.sum((D @ V.T) ** 2, 1)
        out.append(r)
    return np.array(cs), np.array(out).T


def choose_k(F, keys, seed, ks=range(0, 31)):
    """His one choice in stack6: the number of turn directions, chosen inside the given whole (80/20)."""
    from .data import his_split
    rng = np.random.default_rng(seed)
    tr, te = his_split(keys, 0.2, seed)
    sc = {}
    for k in ks:
        cs, R = left(write(F[tr], keys[tr], k, rng), F[te])
        sc[k] = float((cs[np.argmin(R, 1)] == keys[te]).mean())
    return max(sc, key=sc.get), sc


# ------------------------------------------------------------------ the seen: spoken words
def walk(word):
    """A word as a walk: letter j at band slot j mod 12, sounding its wheel cell's tone exp(2 pi i c / 30)."""
    w = np.zeros(12, complex)
    for j, ch in enumerate(c for c in word if c in WHEEL):
        w[j % 12] += np.exp(2j * np.pi * WHEEL[ch] / 30)
    return w


def members(word):
    f, r = walk(word), walk(word[::-1])
    return f + r, f - r                                   # still, turn


class Words:
    """The given whole's words as walks (both members)."""

    def __init__(self, words):
        self.words = [w for w in words if re.fullmatch(r"[a-z]+", w)]
        st, tu = zip(*(members(w) for w in self.words))
        self.still, self.turn = np.array(st), np.array(tu)

    def seen(self, word):
        s, t = members(word)
        rs = np.sum(np.abs(self.still - s) ** 2, 1)
        rf = rs + np.sum(np.abs(self.turn - t) ** 2, 1)
        i_s, i_f = int(np.argmin(rs)), int(np.argmin(rf))
        return [(self.words[i_f], float(rf[i_f])), (self.words[i_s], float(rf[i_s]))]


# ------------------------------------------------------------------ keys: a word's position from its definitions
OPS_ADD = {"and", "more", "added", "plus"}
FILL = {"than", "that", "the", "a", "sum", "of", "units"}


class Keys:
    """A word closes at the position its definitions in the whole construct. Each clause of its entry that is a
    construction (number words joined by 'more than', 'and', 'less than', 'twice', 'times') is one route; the word
    closes where its routes agree, into all of them where they do not, nowhere where it has none. 'one' is the
    unit: the single step of the walk, the only thing not read from the whole."""

    def __init__(self, entries):
        self.entries = entries
        self.active = set()

    def routes(self, word, depth=0):
        txt = self.entries.get(word, "")
        out = []
        for clause in re.split(r"[;.:]", txt):
            toks = re.findall(r"[a-z]+|[0-9]+", clause.lower())
            if not toks or any(t not in band.NUMBER_WORDS and t not in OPS_ADD | FILL | {"less", "twice", "times",
                                                                                        "once"} and not t.isdigit()
                               for t in toks):
                continue
            n = self.construct(toks, depth)
            if n is not None:
                out.append(n)
        return out

    def value(self, tok, depth):
        """A quantity in a clause: a written numeral as it stands, a number word read through its own definitions
        (never through itself: a word being read cannot be a step of its own reading)."""
        if tok.isdigit():
            return int(tok)
        if tok == "one":
            return 1
        if tok in self.active or depth > 12:
            return None
        self.active.add(tok)
        r = self.routes(tok, depth + 1)
        self.active.discard(tok)
        return r[0] if r and len(set(r)) == 1 else None

    def construct(self, toks, depth):
        """Walk a clause: quantities added, 'x less than y' as y with x's lap-rest added, twice/times as repeats."""
        q = [t for t in toks if t in band.NUMBER_WORDS or t.isdigit()]
        if "twice" in toks and len(q) == 1:
            v = self.value(q[0], depth)
            return None if v is None else 2 * v
        if "times" in toks and len(q) == 2:
            a, b = self.value(q[0], depth), self.value(q[1], depth)
            return None if a is None or b is None else a * b
        if len(q) < 2:
            return None
        vals = [self.value(t, depth) for t in q]
        if any(v is None for v in vals):
            return None
        if "less" in toks:
            return vals[1] - vals[0]
        return sum(vals)

    def position(self, word):
        """Every position the word's routes reach (one when they agree)."""
        if word == "one":
            return {1}
        self.active = {word}
        r = self.routes(word, 0)
        self.active = set()
        return {n % 12 for n in r}


# ------------------------------------------------------------------ the reader
class Reader:
    def __init__(self, laps=None, words=None, keys=None):
        self.laps, self.words, self.keys = laps, words, keys
        self.standing = None                              # rest

    def meet(self, closures):
        """closures: [(name, position or None, left behind)], the seen's own order first. The seer orders them."""
        if self.standing is None or len(closures) == 1:
            return closures
        def d(c):
            return band.apart(self.standing, c[1]) if c[1] is not None else np.inf
        return sorted(closures, key=lambda c: (d(c), c[2]))

    def settle(self, first):
        self.standing = first[1]

    def seen(self, kind, piece):
        """The seen: the closures the piece's two members name, the seen's own order first."""
        if kind == "digit":
            return seen_digits(self.laps, piece[None, :])[0]
        if kind == "construction":
            fw = band.walk_phrase(piece)
            if fw is None:
                return [(piece, None, np.inf)]
            toks = re.findall(r"[a-z]+|[0-9]+|[+\-−]", piece.lower())
            cl = [(fw["n"], fw["n"], 0.0)]
            ops = [t for t in toks if t in band.ADD or t in band.TAKE]
            if len(ops) == 1 and ops[0] in band.ADD:          # read reversed: the same steps the other way
                q = [t for t in toks if band.quantity(t) is not None]
                rv = band.walk_phrase(f"{q[1]} plus {q[0]}") if len(q) == 2 else None
                if rv is not None and rv["n"] != fw["n"]:
                    cl.append((rv["n"], rv["n"], 0.0))
            return cl
        if kind == "spoken":
            cl = []
            for w, r in self.words.seen(piece):
                if any(c[0] == w for c in cl):
                    continue
                pos = self.keys.position(w)
                cl.append((w, min(pos) if len(pos) == 1 else None, r))
            return cl
        raise ValueError(kind)

    def act_seen(self, closures, move=True):
        order = self.meet(closures)
        if move:
            self.settle(order[0])
        return {"read": order[0][0], "position": order[0][1], "closes_into": [c[0] for c in order],
                "multi": len(order) > 1}

    def act(self, kind, piece, move=True):
        return self.act_seen(self.seen(kind, piece), move)


def seen_digits(laps, S):
    """Both members of every written digit at once: [(key, position, left behind with turns)] per piece."""
    cs, Rf = left(laps, S, True)
    _, Rs = left(laps, S, False)
    out = []
    for i in range(len(S)):
        kf, ks = int(cs[np.argmin(Rf[i])]), int(cs[np.argmin(Rs[i])])
        cl = [(kf, kf, float(Rf[i].min()))]
        if ks != kf:
            cl.append((ks, ks, float(Rf[i][list(cs).index(ks)])))
        out.append(cl)
    return out
