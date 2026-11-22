"""Challenge 1, both members: the word level and the walk within the word read together (results/c1b.json)."""
import json, gc
import numpy as np
exec(open("c1.py").read().split("cw, cq = cores(Wb), cores(Qb)")[0])
r0 = np.random.default_rng(1)
inw = np.flatnonzero(qtype == 2); keep_in = np.zeros(len(qtype), bool); keep_in[r0.choice(inw, 200_000, replace=False)] = True
SETS["in Webster (200,000 sampled)"] = keep_in
ALL = np.flatnonzero(np.any(np.stack(list(SETS.values())), 0))
cw, cq = cores(Wb), cores(Qb); LEV = list(cw)
L = {}
for lev in LEV:
    L[lev] = W.Returns(cw[lev](), Wb[0]).read(cq[lev]()[ALL]) > 0; gc.collect()
del cw, cq; gc.collect()
WORD, SPAN = LEV[:4], LEV[4:]
def first_of(levels):
    S = np.zeros((len(ALL), 26), bool); got = np.zeros(len(ALL), bool)
    for lev in levels:
        new = L[lev].any(1) & ~got; S[new] = L[lev][new]; got |= new
    return S, got
Sw, gw = first_of(WORD); Ss, gs = first_of(SPAN)
both = Sw | Ss
sym_w, at_w = stream(whole_words); sym_q, at_q = stream(qwords)
OFFS = list(range(-8, 0)) + list(range(1, 9))
r = np.random.default_rng(0); samp = r.choice(len(at_w), 3_000_000, replace=False)
m = MLP(len(OFFS)); m.train(windows(sym_w, at_w[samp], OFFS), sym_w[at_w[samp]].astype(np.int64) - 1, log=log)
lp = m.logp(windows(sym_q, at_q[ALL], OFFS)); y = Qb[0][ALL]
rank = (lp > lp[np.arange(len(ALL)), y][:, None]).sum(1)
res = {}
pos_in_all = {i: j for j, i in enumerate(ALL)}
for name, mask in SETS.items():
    j = np.array([pos_in_all[i] for i in np.flatnonzero(mask)]); yy = y[j]
    out = {"letters": int(len(j))}
    for rn, S in (("word level alone", Sw[j]), ("walk within the word alone", Ss[j]), ("both members", both[j])):
        c = S.any(1); sz = S.sum(1); tin = S[np.arange(len(j)), yy]
        out[rn] = {"closes_%": round(float(c.mean()*100), 2), "truth_among_%": round(float(tin[c].mean()*100), 2),
                   "size_mean": round(float(sz[c].mean()), 2), "size_median": float(np.median(sz[c])),
                   "trained_top_k_same_size_%": round(float((rank[j][c] < sz[c]).mean()*100), 2)}
    res[name] = out; log(name, json.dumps(out, indent=1))
json.dump(res, open("results/c1b.json", "w"), indent=1)
