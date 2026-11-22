"""The trained side: a character model learned by gradient descent (Adam), numpy only.
Input: the symbols at fixed offsets around (or before) a letter, one-hot (27 symbols: space + 26 letters).
One hidden layer of 256 ReLU units, a softmax over the 26 letters. Settings chosen once, not tuned:
hidden 256, Adam lr 1e-3, batch 1024, 2 epochs over 3 million samples."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class MLP:
    def __init__(self, n_pos, n_sym=27, hidden=256, n_out=26, seed=0):
        r = np.random.default_rng(seed)
        self.n_pos, self.n_sym = n_pos, n_sym
        self.W1 = (r.standard_normal((n_pos * n_sym, hidden)) * 0.05).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.W2 = (r.standard_normal((hidden, n_out)) * 0.05).astype(np.float32)
        self.b2 = np.zeros(n_out, np.float32)
        self.P = [self.W1, self.b1, self.W2, self.b2]
        self.m = [np.zeros_like(p) for p in self.P]
        self.v = [np.zeros_like(p) for p in self.P]
        self.t = 0

    @property
    def n_parameters(self):
        return int(sum(p.size for p in self.P))

    def _fwd(self, X):
        idx = X + (np.arange(self.n_pos) * self.n_sym)[None, :]
        a = self.W1[idx].sum(1) + self.b1
        h = np.maximum(a, 0)
        return idx, a, h, h @ self.W2 + self.b2

    def logp(self, X, batch=8192):
        out = []
        for i in range(0, len(X), batch):
            _, _, _, z = self._fwd(X[i:i + batch])
            z = z - z.max(1, keepdims=True)
            out.append(z - np.log(np.exp(z).sum(1, keepdims=True)))
        return np.concatenate(out)

    def step(self, X, y, lr=1e-3):
        B = len(y)
        idx, a, h, z = self._fwd(X)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        loss = float(-np.log(p[np.arange(B), y] + 1e-12).mean())
        dz = p
        dz[np.arange(B), y] -= 1
        dz /= B
        gW2, gb2 = h.T @ dz, dz.sum(0)
        da = (dz @ self.W2.T) * (a > 0)
        M = sp.csr_matrix((np.ones(idx.size, np.float32), (np.repeat(np.arange(B), self.n_pos), idx.ravel())),
                          shape=(B, self.W1.shape[0]))
        gW1 = (M.T @ da).astype(np.float32)
        G = [gW1, da.sum(0), gW2, gb2]
        self.t += 1
        for P, m, v, g in zip(self.P, self.m, self.v, G):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            P -= lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)
        return loss

    def train(self, X, y, epochs=2, batch=1024, seed=0, log=print):
        r = np.random.default_rng(seed)
        for e in range(epochs):
            order = r.permutation(len(y))
            losses = []
            for i in range(0, len(y), batch):
                b = order[i:i + batch]
                losses.append(self.step(X[b], y[b]))
            log(f"  epoch {e + 1}: mean loss {np.mean(losses):.4f} nats")


def stream(words):
    """Symbols of a text: 0 = space (one per word boundary; an unknown word adds one more), 1..26 letters.
    Returns the symbols and, for every letter of the words in order, its index in the stream."""
    s = " ".join(words)
    arr = np.frombuffer(s.encode(), np.uint8)
    sym = np.where(arr == 32, 0, arr.astype(np.int16) - 96).astype(np.int8)
    letter_at = np.flatnonzero(sym > 0)
    return sym, letter_at


def windows(sym, at, offs):
    pad = max(abs(o) for o in offs)
    S = np.concatenate([np.zeros(pad, np.int8), sym, np.zeros(pad, np.int8)])
    return S[(at + pad)[:, None] + np.asarray(offs)[None, :]].astype(np.int64)
