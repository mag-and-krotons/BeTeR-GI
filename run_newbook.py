"""Webster is the whole; a book it has never walked is read against it (results/newbook.json)."""
import json, time, gc
import numpy as np
from gi import data, walk as W, words as WD
t0 = time.time()
LEVELS = ["letter in word + word before + word after", "letter in word + word before", "letter in word + word after",
          "letter in word alone"]
def keys(letters, wid, pos, wlen, lens):
    v2 = W.ladder2()[letters]
    before, after = WD.within_word(v2, pos, wlen)
    wv = WD.word_values(v2, wid, pos, lens).view(np.int64)
    nw = len(lens)
    pw = np.where(wid > 0, wv[np.maximum(wid - 1, 0)], 0)
    nx = np.where(wid < nw - 1, wv[np.minimum(wid + 1, nw - 1)], 0)
    st = np.stack([pos, wlen - pos - 1], 1)
    return {LEVELS[0]: WD.h(before, after, st, pw, nx), LEVELS[1]: WD.h(before, after, st, pw),
            LEVELS[2]: WD.h(before, after, st, nx), LEVELS[3]: WD.h(before, after, st)}
Wl, Wwid, Wpos, Wwlen, Wlens = WD.book(data.webster())
KW = keys(Wl, Wwid, Wpos, Wwlen, Wlens); gc.collect()
txt = "".join(open(f"data/wt2_{p}.txt", encoding="utf8").read() for p in ("valid", "test"))
qwords = WD.words_of(txt, keep_unknown=True)
Ql, Qwid, Qpos, Qwlen, Qlens = WD.from_words(qwords)
KQ = keys(Ql, Qwid, Qpos, Qwlen, Qlens); gc.collect()
webster_vocab = set(WD.words_of(" ".join(f"{k} {v}" for k, v in data.webster().items())))
in_vocab = np.array([w in webster_vocab for w in qwords])[Qwid]
y = Ql
res = {"whole_letters": int(len(Wl)), "book_letters": int(len(Ql)), "book_words": int(sum(1 for w in qwords if w)),
       "letters_whose_word_is_in_webster": round(float(in_vocab.mean() * 100), 2), "levels": {}}
reads = {}
for name in LEVELS:
    C = W.Returns(KW[name], Wl).read(KQ[name]); met = C.sum(1) > 0; one = (C > 0).sum(1) == 1
    reads[name] = C
    res["levels"][name] = {"closes": round(float(met.mean()*100), 2),
                           "truth_among_closures": round(float((C[met, y[met]] > 0).mean()*100), 3),
                           "letters_per_closure": round(float((C[met] > 0).sum(1).mean()), 3),
                           "one_letter_closes": round(float(one.mean()*100), 2),
                           "right_where_one_closes": round(float((C[one].argmax(1) == y[one]).mean()*100), 3)}
    print(name, json.dumps(res["levels"][name]), flush=True)
# the complete reading in the structure's order: the first level that closes gives the closure set
first = np.full(len(y), -1)
for i, name in enumerate(LEVELS):
    met = reads[name].sum(1) > 0
    first[(first < 0) & met] = i
closed = first >= 0
S = np.zeros((len(y), 26), bool)
for i, name in enumerate(LEVELS):
    S[first == i] = reads[name][first == i] > 0
truth_in = S[np.arange(len(y)), y]
single = S.sum(1) == 1
most = np.full(len(y), -1)
for i, name in enumerate(LEVELS):
    m = first == i; most[m] = reads[name][m].argmax(1)
res["complete"] = {"closes": round(float(closed.mean()*100), 2),
                   "truth_among_closures": round(float(truth_in[closed].mean()*100), 3),
                   "of_all_letters_truth_among_closures": round(float(truth_in.mean()*100), 3),
                   "single_closure": round(float(single.mean()*100), 2),
                   "right_where_single": round(float((S[single].argmax(1) == y[single]).mean()*100), 3),
                   "count_label_(letter_met_most)_right_of_all": round(float((most == y).mean()*100), 3),
                   "word_in_webster": {"closes": round(float(closed[in_vocab].mean()*100), 2),
                                       "truth_among_closures": round(float(truth_in[in_vocab & closed].mean()*100), 3)},
                   "word_not_in_webster": {"share": round(float((~in_vocab).mean()*100), 2),
                                           "closes": round(float(closed[~in_vocab].mean()*100), 2),
                                           "truth_among_closures": round(float(truth_in[~in_vocab & closed].mean()*100), 3)}}
print("complete", json.dumps(res["complete"], indent=1), flush=True)
res["seconds"] = round(time.time() - t0)
json.dump(res, open("results/newbook.json", "w"), indent=1)
