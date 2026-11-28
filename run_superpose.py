"""The superposition reader against the trained model, on the letters where the trained model won (results/superpose.json)."""
import json, time, gc
import numpy as np
exec(open("c1.py").read().split("cw, cq = cores(Wb), cores(Qb)")[0])      # the same reduced whole, the same letters
from gi import superpose as SP
r0 = np.random.default_rng(1)
inw = np.flatnonzero(qtype == 2); keep_in = np.zeros(len(qtype), bool); keep_in[r0.choice(inw, 200_000, replace=False)] = True
SETS["in Webster (200,000 sampled)"] = keep_in
ALL = np.flatnonzero(np.any(np.stack(list(SETS.values())), 0))
v2w, v2q = W.ladder2()[Wb[0]], W.ladder2()[Qb[0]]
S = np.zeros((len(ALL), 26), np.int32); seen_by = np.zeros(len(ALL), np.int16)
obs = SP.observers(4) + ["word"]
for o in obs:
    if o == "word":
        bw, aw = WD.within_word(v2w, Wb[2], Wb[3]); kw = WD.h(bw.astype(np.int16), aw.astype(np.int16), np.stack([Wb[2], Wb[3] - Wb[2] - 1], 1).astype(np.int8)); del bw, aw
        bq, aq = WD.within_word(v2q, Qb[2], Qb[3]); kq = WD.h(bq.astype(np.int16), aq.astype(np.int16), np.stack([Qb[2], Qb[3] - Qb[2] - 1], 1).astype(np.int8))[ALL]
        okw = np.ones(len(kw), bool); okq = np.ones(len(kq), bool)
    else:
        kw = SP.span_key(v2w, Wb[2], Wb[3], *o); kq = SP.span_key(v2q, Qb[2], Qb[3], *o)[ALL]
        okw = kw != 0; okq = kq != 0
    C = W.Returns(kw[okw], Wb[0][okw]).read(kq) > 0
    C[~okq] = False
    S += C; seen_by += C.any(1)
    del kw, kq, C; gc.collect()
log("all observers read", len(obs))
# the trained side, as before
sym_w, at_w = stream(whole_words); sym_q, at_q = stream(qwords)
OFFS = list(range(-8, 0)) + list(range(1, 9))
r = np.random.default_rng(0); samp = r.choice(len(at_w), 3_000_000, replace=False)
m = MLP(len(OFFS)); m.train(windows(sym_w, at_w[samp], OFFS), sym_w[at_w[samp]].astype(np.int64) - 1, log=log)
lp = m.logp(windows(sym_q, at_q[ALL], OFFS)); y = Qb[0][ALL]
rank_t = (lp > lp[np.arange(len(ALL)), y][:, None]).sum(1)
peak = S.max(1, keepdims=True); R = (S == peak) & (peak > 0)             # read where the superposed signal peaks
rank_s = (S > S[np.arange(len(ALL)), y][:, None]).sum(1)                   # truth's place in the superposed signal
pos_in_all = {i: j for j, i in enumerate(ALL)}
prev = json.load(open("results/c1b.json"))
res = {}
for name, mask in SETS.items():
    j = np.array([pos_in_all[i] for i in np.flatnonzero(mask)]); yy = y[j]
    Rj = R[j]; c = Rj.any(1); sz = Rj.sum(1); tin = Rj[np.arange(len(j)), yy]
    one = sz == 1
    out = {"letters": int(len(j)), "seen_by_observers_mean": round(float(seen_by[j].mean()), 2),
           "superposition": {"reads_%": round(float(c.mean()*100), 2), "truth_at_peak_%": round(float(tin[c].mean()*100), 2),
                             "peak_size_mean": round(float(sz[c].mean()), 2), "single_peak_%": round(float(one.mean()*100), 2),
                             "right_where_single_%": round(float(tin[one].mean()*100), 2),
                             "truth_in_top_k_by_signal_k_=_trained_comparison": None},
           "trained_top_k_same_size_%": round(float((rank_t[j][c] < sz[c]).mean()*100), 2),
           "trained_top1_where_single_%": round(float((rank_t[j][one] == 0).mean()*100), 2),
           "previous_reader (structure order, both members)": prev.get(name, {}).get("both members")}
    for k in (1, 3, 5):
        out[f"superposition top-{k} by signal %"] = round(float((rank_s[j] < k).mean()*100), 2)
        out[f"trained top-{k} %"] = round(float((rank_t[j] < k).mean()*100), 2)
    res[name] = out; log(name, json.dumps(out, indent=1))
json.dump(res, open("results/superpose.json", "w"), indent=1)
log("done")
