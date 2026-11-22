"""Challenge 2: the next letter, arriving observer only, bits per letter on Wikipedia (results/c2.json).
Whole: all of Webster. Reader: what the arriving walk meets at its returns, from the single step up to the whole
prefix with the word before; each level's letters met, with an escape weighted by how many distinct letters were met
(Witten-Bell: no constant) to the level below. Trained: 16 symbols before, same trainer as challenge 1."""
import json, time, gc
import numpy as np
from gi import data, walk as W, words as WD
from gi.mlp import MLP, stream, windows
t0 = time.time(); log = lambda *a: print(*a, f"[{time.time()-t0:.0f}s]", flush=True)
web_words = WD.words_of(" ".join(f"{k} {v}" for k, v in data.webster().items()))
qwords = WD.words_of("".join(open(f"data/wt2_{p}.txt", encoding="utf8").read() for p in ("valid", "test")), keep_unknown=True)
Wb, Qb = WD.from_words(web_words), WD.from_words(qwords)
def arriving(book):
    letters, wid, pos, wlen, lens = book
    v2 = W.ladder2()[letters]
    b, _ = WD.within_word(v2, pos, wlen)
    wv = WD.word_values(v2, wid, pos, lens).view(np.int64)
    pw = np.where(wid > 0, wv[np.maximum(wid - 1, 0)], 0)
    p = {s: WD.partial(v2, pos, wlen, s, -1) for s in (4, 3, 1)}
    pre = (b.astype(np.int16), pos.astype(np.int8))
    return {"1": lambda: WD.h(*p[1]), "3": lambda: WD.h(*p[3]), "4 and 3": lambda: WD.h(*p[4], *p[3]),
            "whole prefix": lambda: WD.h(*pre), "whole prefix + word before": lambda: WD.h(*pre, pw)}
cw, cq = arriving(Wb), arriving(Qb)
y = Qb[0]; n = len(y)
P = np.tile((np.bincount(Wb[0], minlength=26) / len(Wb[0])).astype(np.float64), (n, 1))
deepest = np.full(n, "unigram", dtype=object)
lev_bits = {"unigram": float(-np.log2(P[np.arange(n), y]).mean())}
for lev in cw:
    C = W.Returns(cw[lev](), Wb[0]).read(cq[lev]()).astype(np.float64)
    N = C.sum(1); u = (C > 0).sum(1); has = N > 0
    P[has] = (C[has] + u[has, None] * P[has]) / (N[has] + u[has])[:, None]
    deepest[has] = lev
    lev_bits[lev] = float(-np.log2(P[np.arange(n), y]).mean())
    del C; gc.collect(); log("up to", lev, "bits", round(lev_bits[lev], 4), "returns", round(float(has.mean()*100), 2))
assert np.abs(P.sum(1) - 1).max() < 1e-9
bits_r = -np.log2(P[np.arange(n), y])
del cw, cq; gc.collect()
sym_w, at_w = stream(web_words); sym_q, at_q = stream(qwords)
OFFS = list(range(-16, 0))
r = np.random.default_rng(0); samp = r.choice(len(at_w), 3_000_000, replace=False)
m = MLP(len(OFFS)); log("training", m.n_parameters, "parameters")
m.train(windows(sym_w, at_w[samp], OFFS), sym_w[at_w[samp]].astype(np.int64) - 1, log=log)
lp = m.logp(windows(sym_q, at_q, OFFS)); bits_t = -lp[np.arange(n), y] / np.log(2)
pos = Qb[2]
res = {"letters": n, "trained_parameters": m.n_parameters,
       "reader_bits_per_letter": round(float(bits_r.mean()), 4), "trained_bits_per_letter": round(float(bits_t.mean()), 4),
       "reader_top1_%": round(float((P.argmax(1) == y).mean()*100), 2), "trained_top1_%": round(float((lp.argmax(1) == y).mean()*100), 2),
       "reader_bits_adding_each_level": {k: round(v, 4) for k, v in lev_bits.items()},
       "by_deepest_level_that_returned": {lv: {"share_%": round(float((deepest == lv).mean()*100), 2),
                                               "reader_bits": round(float(bits_r[deepest == lv].mean()), 3),
                                               "trained_bits": round(float(bits_t[deepest == lv].mean()), 3)}
                                          for lv in ["unigram", "1", "3", "4 and 3", "whole prefix", "whole prefix + word before"] if (deepest == lv).any()},
       "first_letter_of_word": {"reader_bits": round(float(bits_r[pos == 0].mean()), 3), "trained_bits": round(float(bits_t[pos == 0].mean()), 3)},
       "later_letters": {"reader_bits": round(float(bits_r[pos > 0].mean()), 3), "trained_bits": round(float(bits_t[pos > 0].mean()), 3)},
       "letters_where_reader_has_fewer_bits_%": round(float((bits_r < bits_t).mean()*100), 2)}
log(json.dumps(res, indent=1)); json.dump(res, open("results/c2.json", "w"), indent=1)
