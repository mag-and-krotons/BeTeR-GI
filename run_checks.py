"""Every check in CHECKS.md, in one run. python3 run_checks.py -> results/checks.json, results/checks.txt"""
import json, time, gc
import numpy as np
from gi import data, walk as W, words as WD

t0 = time.time()
LEVELS = ["letter + word before + word after", "letter + word before", "letter + word after", "letter in word alone"]
TWINS = [frozenset(p) for p in ("ik", "hl", "gm", "fn", "uo", "tp", "sq")]
P = np.uint64(1099511628211)
UNKNOWN = np.uint64(0xDEADBEEFCAFEBABE)
lines = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); lines.append(s)

# ------------------------------------------------------------------ the two books
web = WD.book(data.webster())
Wl, Wwid, Wpos, Wwlen, Wlens = web
wstart = np.concatenate([[0], np.cumsum(Wlens)[:-1]])
blk = (np.arange(len(Wl)) // 50_000) % 20 == 19
wside = blk[wstart]; nw = len(Wlens)
ok = (wside[Wwid] == wside[np.maximum(Wwid - 1, 0)]) & (wside[Wwid] == wside[np.minimum(Wwid + 1, nw - 1)])
WHOLE, TEST = np.flatnonzero(ok & ~wside[Wwid]), np.flatnonzero(ok & wside[Wwid])
qwords = WD.words_of("".join(open(f"data/wt2_{p}.txt", encoding="utf8").read() for p in ("valid", "test")),
                     keep_unknown=True)
wiki = WD.from_words(qwords)
Ql, Qwid, Qpos, Qwlen, Qlens = wiki
vocab = set(WD.words_of(" ".join(f"{k} {v}" for k, v in data.webster().items())))
q_in_vocab = np.array([w in vocab for w in qwords])[Qwid]
q_unk_next = (np.array([w == "" for w in qwords] + [False]))
q_unk_adj = q_unk_next[np.maximum(Qwid - 1, 0)] | q_unk_next[Qwid + 1]
say(f"books: Webster {len(Wl)} letters ({len(TEST)} held out); Wikipedia {len(Ql)} letters; {time.time()-t0:.0f}s")


def keys(enc, book):
    letters, wid, pos, wlen, lens = book
    n_w = len(lens)
    st = np.stack([pos, wlen - pos - 1], 1)
    if enc == "walk":
        v2 = W.ladder2()[letters]
        before, after = WD.within_word(v2, pos, wlen)
        wv = WD.word_values(v2, wid, pos, lens).view(np.int64)
        core = (before, after, st)
    else:
        pp = np.ones(32, np.uint64)
        for i in range(1, 32):
            pp[i] = pp[i - 1] * P
        H = np.zeros(n_w, np.uint64)
        np.add.at(H, wid, (letters.astype(np.uint64) + np.uint64(1)) * pp[pos])
        H[lens == 0] = UNKNOWN
        core = ((H[wid] - (letters.astype(np.uint64) + np.uint64(1)) * pp[pos]).view(np.int64), st)
        wv = H.view(np.int64)
    pw = np.where(wid > 0, wv[np.maximum(wid - 1, 0)], 0)
    nx = np.where(wid < n_w - 1, wv[np.minimum(wid + 1, n_w - 1)], 0)
    return {LEVELS[0]: lambda: WD.h(*core, pw, nx), LEVELS[1]: lambda: WD.h(*core, pw),
            LEVELS[2]: lambda: WD.h(*core, nx), LEVELS[3]: lambda: WD.h(*core)}


store = {}
for enc in ("walk", "plain"):
    kw, kq = keys(enc, web), keys(enc, wiki)
    for lev in LEVELS:
        K = kw[lev]()
        C = W.Returns(K[WHOLE], Wl[WHOLE]).read(K[TEST])
        store[("webster", enc, lev)] = (np.packbits(C > 0, 1), C.argmax(1).astype(np.int8))
        del C
        C = W.Returns(K, Wl).read(kq[lev]())
        store[("wikipedia", enc, lev)] = (np.packbits(C > 0, 1), C.argmax(1).astype(np.int8))
        del C, K; gc.collect()
    del kw, kq; gc.collect()
    say(f"{enc} read at every level, both books; {time.time()-t0:.0f}s")

freq = np.bincount(Wl, minlength=26) / len(Wl)
unpack = lambda b: np.unpackbits(b, 1)[:, :26].astype(bool)


def metrics(S, y):
    met = S.any(1); sz = S.sum(1)[met]; tin = S[met][np.arange(met.sum()), y[met]]
    one = sz == 1
    return {"closes": round(float(met.mean() * 100), 3), "truth_among": round(float(tin.mean() * 100), 3),
            "size_median": float(np.median(sz)) if len(sz) else None, "size_mean": round(float(sz.mean()), 3) if len(sz) else None,
            "size_dist_%": {k: round(float(v * 100), 2) for k, v in
                            {"1": (sz == 1).mean(), "2": (sz == 2).mean(), "3": (sz == 3).mean(),
                             "4-5": ((sz >= 4) & (sz <= 5)).mean(), "6+": (sz >= 6).mean()}.items()} if len(sz) else {},
            "single_right": round(float(tin[one].mean() * 100), 3) if one.any() else None,
            "chance_same_size_freq_weighted": round(float((S[met] * freq).sum(1).mean() * 100), 3) if len(sz) else None,
            "bits_where_truth_in": round(float(np.log2(26 / sz[tin]).mean()), 3) if tin.any() else None}


def analyse(book_name, y, wlen, pos, enc):
    sets = [unpack(store[(book_name, enc, l)][0]) for l in LEVELS]
    most = [store[(book_name, enc, l)][1] for l in LEVELS]
    out = {"levels": {l: metrics(s, y) for l, s in zip(LEVELS, sets)}}
    first = np.full(len(y), -1)
    for i, s in enumerate(sets):
        first[(first < 0) & s.any(1)] = i
    S = np.zeros_like(sets[0]); label = np.full(len(y), -1)
    for i, s in enumerate(sets):
        m = first == i; S[m] = s[m]; label[m] = most[i][m]
    out["structure_order"] = metrics(S, y)
    out["structure_order"]["count_label_right_of_all"] = round(float((label == y).mean() * 100), 3)
    both = sets[0].any(1) & sets[3].any(1)
    out["must_hold"] = {"order_closes_where_letter_scale_closes": bool(np.all(S.any(1) >= sets[3].any(1))),
                        "word_scale_set_within_letter_set": bool(np.all(~sets[0][both] | sets[3][both]))}
    out["by_word_length"] = {str(L): metrics(S[(wlen == L)], y[wlen == L]) | {"share": round(float((wlen == L).mean()*100), 2)}
                             for L in list(range(1, 11))}
    out["by_word_length"]["11+"] = metrics(S[wlen >= 11], y[wlen >= 11]) | {"share": round(float((wlen >= 11).mean()*100), 2)}
    for nm, m in (("first", pos == 0), ("middle", (pos > 0) & (pos < wlen - 1)), ("last", (pos == wlen - 1) & (wlen > 1))):
        out.setdefault("by_position", {})[nm] = metrics(S[m], y[m])
    out["by_letter"] = {chr(97 + c): {"truth_among": metrics(S[y == c], y[y == c])["truth_among"],
                                      "size_mean": metrics(S[y == c], y[y == c])["size_mean"]} for c in range(26)}
    miss = sets[0].any(1) & ~sets[0][np.arange(len(y)), y] & sets[3][np.arange(len(y)), y]
    out["word_scale_misses_by_length"] = {str(L): int((miss & (wlen == L)).sum()) for L in range(1, 9)}
    out["word_scale_misses_total"] = int(miss.sum())
    sz = S.sum(1)
    pairs = [frozenset(chr(97 + c) for c in np.flatnonzero(S[i])) for i in np.flatnonzero(sz == 2)]
    multi = np.flatnonzero(sz >= 2)
    out["ladder_twins"] = {"size_2_closures": len(pairs), "of_which_value_twins": sum(p in TWINS for p in pairs),
                           "multi_closures_holding_a_twin_pair": round(float(np.mean([any(t <= set(np.flatnonzero(S[i])) and False or
                               all(chr(97 + c) in [chr(97 + j) for j in np.flatnonzero(S[i])] for c in [ord(x) - 97 for x in t]) for t in TWINS)
                               for i in multi[:200000]]) * 100), 2) if len(multi) else None}
    out["unread_%"] = round(float((~S.any(1)).mean() * 100), 3)
    return out, S


R = {}
for enc in ("walk", "plain"):
    R[("webster", enc)], Sw = analyse("webster", Wl[TEST], Wwlen[TEST], Wpos[TEST], enc)
    R[("wikipedia", enc)], Sq = analyse("wikipedia", Ql, Qwlen, Qpos, enc)
    un = ~Sq.any(1)
    R[("wikipedia", enc)]["unread_composition"] = {"word_not_in_webster_%": round(float((~q_in_vocab[un]).mean()*100), 2),
                                                   "next_to_unknown_word_%": round(float(q_unk_adj[un].mean()*100), 2)}
    if enc == "walk":
        S_walk = {"webster": Sw, "wikipedia": Sq}
    else:
        for b, Sp in (("webster", Sw), ("wikipedia", Sq)):
            Sv = S_walk[b]; y = Wl[TEST] if b == "webster" else Ql
            bc = Sv.any(1) & Sp.any(1)
            same = np.all(Sv == Sp, 1)
            wl, pl = Sv.sum(1), Sp.sum(1)
            R[(b, "walk vs plain")] = {"both_close_%": round(float(bc.mean()*100), 3),
                                       "sets_identical_%": round(float(same[bc].mean()*100), 3),
                                       "walk_set_larger_%": round(float((wl > pl)[bc].mean()*100), 3),
                                       "plain_set_larger_%": round(float((pl > wl)[bc].mean()*100), 3),
                                       "walk_closes_plain_not_%": round(float((Sv.any(1) & ~Sp.any(1)).mean()*100), 3),
                                       "plain_closes_walk_not_%": round(float((Sp.any(1) & ~Sv.any(1)).mean()*100), 3)}
for k, v in R.items():
    say(json.dumps({" / ".join(k): v}))
json.dump({" / ".join(k): v for k, v in R.items()}, open("results/checks.json", "w"), indent=1)
open("results/checks.txt", "w").write("\n".join(lines))
say(f"done in {time.time()-t0:.0f}s")
