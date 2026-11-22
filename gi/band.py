"""The band: position closes after 3, the reading closes after 4 (each twist a quarter turn), both together at 12.

A quantity carried n twists stands at the pair (n mod 3, n mod 4): which of the three places, and which quarter
turn of the reading. Only 12 closes both, so the 12 pairs are the 12 positions. Nothing is subtracted: taking b away
is adding the rest of the lap, 12 - b. Six is (0, 2): the place does not move and the reading turns two quarters,
which is pi. So "- 6" and "+ 6" are one step, and that step is the pi shift of the reading.
"""
from __future__ import annotations

import re

import numpy as np

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
                "twenty"]
ADD = {"plus", "and", "past", "add", "+", "after"}
TAKE = {"minus", "less", "-", "−", "before", "take"}


def pos(n: int):
    """A quantity's position: (n mod 3, n mod 4)."""
    return (n % 3, n % 4)


def laps(n: int):
    """How many whole laps of 12 close, and what remains."""
    return n // 12, n % 12


def from_pair(p3: int, p4: int) -> int:
    """The one position in 0..11 with this pair (the two closures meet only at 12)."""
    for n in range(12):
        if n % 3 == p3 % 3 and n % 4 == p4 % 4:
            return n
    raise ValueError


def add(p, q):
    return ((p[0] + q[0]) % 3, (p[1] + q[1]) % 4)


def take(p, q):
    """p with q taken away, by addition only: add the rest of q's lap."""
    return add(p, pos(12 - from_pair(*q)))


def reading_turn(p):
    """The reading's turn at this position: the quarter turns e^{-i pi/2 * (n mod 4)}."""
    return np.exp(-1j * np.pi / 2 * p[1])


def quantity(tok: str):
    if tok.isdigit():
        return int(tok)
    if tok in NUMBER_WORDS:
        return NUMBER_WORDS.index(tok)
    return None


def walk_phrase(text: str):
    """Walk a spoken or written construction ("13 - 6", "two past five", "thirteen minus six") along the band.
    Returns the position it ends at and the laps that closed on the way. "x past y" walks y, then x; "x minus y"
    walks x, then the rest of y's lap."""
    toks = re.findall(r"[a-z]+|[0-9]+|[+\-−]", text.lower())
    vals, ops = [], []
    for t in toks:
        q = quantity(t)
        if q is not None:
            vals.append(q)
        elif t in ADD or t in TAKE:
            ops.append(t)
    if not vals:
        return None
    if ops and ops[0] in ("past", "after") and len(vals) == 2:
        vals = [vals[1], vals[0]]
        ops = ["plus"]
    p = pos(vals[0])
    closed = laps(vals[0])[0]
    for o, v in zip(ops, vals[1:]):
        p = take(p, pos(v)) if o in TAKE else add(p, pos(v))
        closed += laps(v)[0]
    return {"position": p, "n": from_pair(*p), "laps_closed": closed}


def name(n: int) -> str:
    return NUMBER_WORDS[n % 12]


def place(n: int) -> np.ndarray:
    """A position as a point on the band: its place among three and its quarter turn among four, each a unit
    phase, together of unit total. Every position is equally far from the rest (the centre)."""
    return np.array([np.exp(2j * np.pi * (n % 3) / 3), 1j ** (n % 4)]) / np.sqrt(2)


def apart(a: int, b: int) -> float:
    """How far two positions stand from each other on the band (squared)."""
    return float(np.sum(np.abs(place(a) - place(b)) ** 2))
