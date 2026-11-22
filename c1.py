"""Challenge 1: words the whole never held (results/c1.json). Every 10th word type is removed from Webster entirely;
the letters of those words are read where they occur in Wikipedia. Both sides get the same reduced whole."""
import json, time, gc
import numpy as np
from gi import data, walk as W, words as WD
from gi.mlp import MLP, stream, windows
t0 = time.time(); log = lambda *a: print(*a, f"[{time.time()-t0:.0f}s]", flush=True)
web_words = WD.words_of(" ".join(f"{k} {v}" for k, v in data.webster().items()))
types = sorted(set(web_words)); removed = set(types[9::10])
whole_words = [w for w in web_words if w not in removed]
assert not (set(whole_words) & removed)
qwords = WD.words_of("".join(open(f"data/wt2_{p}.txt", encoding="utf8").read() for p in ("valid", "test")), keep_unknown=True)
Wb, Qb = WD.from_words(whole_words), WD.from_words(qwords)
WEB = set(web_words)
qtype = np.array([0 if w == "" else (1 if w in removed else (2 if w in WEB else 3)) for w in qwords])[Qb[1]]
SETS = {"removed from the whole": qtype == 1, "never in Webster": qtype == 3}
log("removed types", len(removed), "of", len(types), "| test letters", {k: int(v.sum()) for k, v in SETS.items()})

def cores(book):
    letters, wid, pos, wlen, lens = book
    v2 = W.ladder2()[letters]
    b, a = WD.within_word(v2, pos, wlen)
    st = np.stack([pos, wlen - pos - 1], 1).astype(np.int8)
    wv = WD.word_values(v2, wid, pos, lens).view(np.int64); nw = len(lens)
    pw = np.where(wid > 0, wv[np.maximum(wid - 1, 0)], 0); nx = np.where(wid < nw - 1, wv[np.minimum(wid + 1, nw - 1)], 0)
    sp = {s: WD.partial(v2, pos, wlen, s, -1) + WD.partial(v2, pos, wlen, s, 1) for s in (4, 3, 1)}
    full = (b.astype(np.int16), a.astype(np.int16), st)
    return {"word + both neighbours": lambda: WD.h(*full, pw, nx), "word + word before": lambda: WD.h(*full, pw),
            "word + word after": lambda: WD.h(*full, nx), "word": lambda: WD.h(*full),
            "4 and 3 together": lambda: WD.h(*sp[4], *sp[3]), "3": lambda: WD.h(*sp[3]), "4": lambda: WD.h(*sp[4]),
            "1": lambda: WD.h(*sp[1])}
cw, cq = cores(Wb), cores(Qb)
LEV = list(cw)
first = np.full(len(Qb[0]), -1); S = np.zeros((len(Qb[0]), 26), bool); label = np.full(len(Qb[0]), -1)
per_level = {}
for i, lev in enumerate(LEV):
    C = W.Returns(cw[lev](), Wb[0]).read(cq[lev]()); met = C.sum(1) > 0
    new = met & (first < 0); first[new] = i; S[new] = C[new] > 0; label[new] = C[new].argmax(1)
    per_level[lev] = {k: round(float(met[m].mean() * 100), 3) for k, m in SETS.items()}
    del C; gc.collect(); log("level", lev, "closes (% of each set)", per_level[lev])
del cw, cq; gc.collect()
# the trained side
sym_w, at_w = stream(whole_words); sym_q, at_q = stream(qwords)
OFFS = list(range(-8, 0)) + list(range(1, 9))
r = np.random.default_rng(0); samp = r.choice(len(at_w), 3_000_000, replace=False)
Xtr = windows(sym_w, at_w[samp], OFFS); ytr = sym_w[at_w[samp]].astype(np.int64) - 1
m = MLP(len(OFFS)); log("training", m.n_parameters, "parameters"); m.train(Xtr, ytr, log=log); del Xtr
y = Qb[0]
res = {"removed_types": len(removed), "trained_parameters": m.n_parameters, "closes_by_level": per_level, "sets": {}}
for name, mask in SETS.items():
    idx = np.flatnonzero(mask); yy = y[idx]
    lp = m.logp(windows(sym_q, at_q[idx], OFFS)); rank = (lp > lp[np.arange(len(idx)), yy][:, None]).sum(1)
    Si = S[idx]; closed = Si.any(1); sz = Si.sum(1); tin = Si[np.arange(len(idx)), yy]
    res["sets"][name] = {
        "letters": int(len(idx)),
        "reader": {"closes_%": round(float(closed.mean()*100), 2), "truth_among_closures_%": round(float(tin[closed].mean()*100), 2),
                   "closure_size_median": float(np.median(sz[closed])), "closure_size_mean": round(float(sz[closed].mean()), 2),
                   "single_closure_%": round(float((sz == 1).mean()*100), 2), "single_right_%": round(float(tin[sz == 1].mean()*100), 2),
                   "forced_label_right_of_all_%": round(float((label[idx] == yy).mean()*100), 2),
                   "closed_at_level_%": {LEV[l]: round(float((first[idx] == l).mean()*100), 2) for l in range(len(LEV))}},
        "trained": {"top1_right_of_all_%": round(float((rank == 0).mean()*100), 2),
                    "truth_in_top_k_where_reader_closed_k_=_reader_set_size_%": round(float((rank[closed] < sz[closed]).mean()*100), 2),
                    "top1_right_where_reader_single_%": round(float((rank[sz == 1] == 0).mean()*100), 2),
                    "top1_right_where_reader_did_not_close_%": round(float((rank[~closed] == 0).mean()*100), 2),
                    "bits_per_letter": round(float(-lp[np.arange(len(idx)), yy].mean() / np.log(2)), 3)}}
    log(name, json.dumps(res["sets"][name], indent=1))
json.dump(res, open("results/c1.json", "w"), indent=1)
log("done")
