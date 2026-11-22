"""The open challenge: a recurrent network trained on the same Webster, read on the same Wikipedia letters
(results/gru.json). Bits per letter with each prediction renormalised over the 26 letters, as the reader's."""
import json, time
import numpy as np
from gi import data, words as WD
from gi.gru import GRU
from gi.mlp import stream
t0 = time.time(); log = lambda *a: print(*a, f"[{time.time()-t0:.0f}s]", flush=True)
web_words = WD.words_of(" ".join(f"{k} {v}" for k, v in data.webster().items()))
qwords = WD.words_of("".join(open(f"data/wt2_{p}.txt", encoding="utf8").read() for p in ("valid", "test")), keep_unknown=True)
sym_w, _ = stream(web_words); sym_q, _ = stream(qwords)
blk = (np.arange(len(sym_w)) // 50_000) % 20 == 19
train, held = sym_w[~blk].astype(np.int64), sym_w[blk][:400_000].astype(np.int64)
def letter_bits(g, sym):
    lo, n = g.logp_stream(sym)
    tgt = sym[1:n + 1].astype(np.int64); prev = sym[:n]
    m = tgt > 0
    ll = lo[m][:, 1:]; ll = ll - np.log(np.exp(ll).sum(1, keepdims=True))
    b = -ll[np.arange(m.sum()), tgt[m] - 1] / np.log(2)
    first = prev[m] == 0
    return float(b.mean()), float(b[first].mean()), float(b[~first].mean()), float((ll.argmax(1) == tgt[m] - 1).mean() * 100), int(m.sum())
g = GRU(H=256); B, T = 128, 64
L = (len(train) - 1) // B
X = train[: B * L].reshape(B, L); Y = train[1: B * L + 1].reshape(B, L)
res = {"parameters": g.n_parameters, "train_symbols": int(B * L), "epochs": []}
log("parameters", g.n_parameters, "| steps per pass", L // T)
for ep, lr in enumerate((2e-3, 5e-4)):
    h = np.zeros((B, g.H), np.float32); losses = []
    for i in range(0, L - T + 1, T):
        loss, G, h = g.loss_grad(X[:, i:i + T], Y[:, i:i + T], h)
        g.step(G, lr); losses.append(loss)
        if len(losses) % 300 == 0: log(f"pass {ep+1} step {len(losses)} loss {np.mean(losses[-300:]) / np.log(2):.4f} bits/symbol")
    hb = letter_bits(g, held)
    res["epochs"].append({"pass": ep + 1, "train_bits_per_symbol": round(float(np.mean(losses) / np.log(2)), 4),
                          "webster_held_out_bits_per_letter": round(hb[0], 4)})
    log(json.dumps(res["epochs"][-1])); json.dump(res, open("results/gru.json", "w"), indent=1)
wb = letter_bits(g, sym_q.astype(np.int64))
c2 = json.load(open("results/c2.json"))
res.update({"wikipedia_letters": wb[4], "wikipedia_bits_per_letter": round(wb[0], 4), "first_letter_bits": round(wb[1], 3),
            "later_letter_bits": round(wb[2], 3), "top1_%": round(wb[3], 2),
            "reader": {"bits_per_letter": c2["reader_bits_per_letter"], "first": c2["first_letter_of_word"]["reader_bits"],
                       "later": c2["later_letters"]["reader_bits"], "top1_%": c2["reader_top1_%"]}})
json.dump(res, open("results/gru.json", "w"), indent=1); log(json.dumps(res, indent=1)); log("done")
