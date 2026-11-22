"""The trained side at full strength: a recurrent network (GRU) over characters, trained through time by Adam.
27 symbols (space + 26 letters) in, next symbol out. Hidden 256. Settings chosen once: Adam lr 2e-3 then 5e-4,
gradient norm clipped at 5, 128 lanes x 64 steps, state carried across steps."""
from __future__ import annotations

import numpy as np

sig = lambda a: 1 / (1 + np.exp(-a))


class GRU:
    def __init__(self, H=256, S=27, seed=0, dtype=np.float32):
        r = np.random.default_rng(seed)
        s = 1 / np.sqrt(H)
        self.H, self.S = H, S
        self.Wx = (r.uniform(-s, s, (S, 3 * H))).astype(dtype)
        self.U = (r.uniform(-s, s, (H, 3 * H))).astype(dtype)
        self.b = np.zeros(3 * H, dtype); self.bu = np.zeros(H, dtype)
        self.V = (r.uniform(-s, s, (H, S))).astype(dtype); self.c = np.zeros(S, dtype)
        self.P = [self.Wx, self.U, self.b, self.bu, self.V, self.c]
        self.m = [np.zeros_like(p) for p in self.P]; self.v = [np.zeros_like(p) for p in self.P]; self.t = 0

    @property
    def n_parameters(self):
        return int(sum(p.size for p in self.P))

    def cell(self, x, h):
        H = self.H
        a = self.Wx[x] + self.b
        hu = h @ self.U
        z = sig(a[:, :H] + hu[:, :H]); r = sig(a[:, H:2 * H] + hu[:, H:2 * H])
        un = hu[:, 2 * H:] + self.bu
        n = np.tanh(a[:, 2 * H:] + r * un)
        return (1 - z) * n + z * h, (z, r, n, un)

    def loss_grad(self, X, Y, h):
        """X, Y: (B, T) symbols in and the symbols to read; h: (B, H) state. Returns loss, grads, last state."""
        B, T = X.shape; H = self.H
        cache, loss = [], 0.0
        G = [np.zeros_like(p) for p in self.P]
        for t in range(T):
            hp = h
            h, (z, r, n, un) = self.cell(X[:, t], h)
            lo = h @ self.V + self.c
            lo -= lo.max(1, keepdims=True); p = np.exp(lo); p /= p.sum(1, keepdims=True)
            loss -= np.log(p[np.arange(B), Y[:, t]] + 1e-12).sum()
            p[np.arange(B), Y[:, t]] -= 1
            cache.append((X[:, t], hp, z, r, n, un, h, p / (B * T)))
        dWx, dU, db, dbu, dV, dc = G
        dh = np.zeros((B, H), h.dtype)
        for x, hp, z, r, n, un, hn, dl in reversed(cache):
            dV += hn.T @ dl; dc += dl.sum(0)
            dh = dh + dl @ self.V.T
            dz = dh * (hp - n); dn = dh * (1 - z); dhp = dh * z
            dan = dn * (1 - n * n)
            dun = dan * r
            dar = dan * un * r * (1 - r)
            daz = dz * z * (1 - z)
            da = np.concatenate([daz, dar, dan], 1)
            dhu = np.concatenate([daz, dar, dun], 1)
            dU += hp.T @ dhu; dbu += dun.sum(0); db += da.sum(0)
            np.add.at(dWx, x, da)
            dh = dhp + dhu @ self.U.T
        return loss / (B * T), G, h

    def step(self, G, lr, clip=5.0):
        norm = np.sqrt(sum(float((g * g).sum()) for g in G))
        if norm > clip:
            G = [g * (clip / norm) for g in G]
        self.t += 1
        for P, m, v, g in zip(self.P, self.m, self.v, G):
            m *= 0.9; m += 0.1 * g; v *= 0.999; v += 0.001 * g * g
            P -= lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)

    def logp_stream(self, sym, lanes=128):
        """Log-probabilities of every next symbol of a stream, read in parallel lanes with state carried."""
        n = len(sym) - 1; L = n // lanes
        out = np.zeros((lanes * L, self.S), np.float32)
        h = np.zeros((lanes, self.H), self.Wx.dtype)
        Xs = sym[: lanes * L].reshape(lanes, L)
        for t in range(L):
            h, _ = self.cell(Xs[:, t].astype(np.int64), h)
            lo = h @ self.V + self.c
            lo = lo - lo.max(1, keepdims=True); lo = lo - np.log(np.exp(lo).sum(1, keepdims=True))
            out[np.arange(lanes) * L + t] = lo
        return out, lanes * L          # row i predicts sym[i + 1]
