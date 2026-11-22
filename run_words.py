"""Letters read within their word, words one scale up (results/words.json). python3 run_words.py"""
import json, time
import numpy as np
from gi import data, walk as W, words as WD
t0 = time.time()
letters, wid, pos, wlen, lens = WD.book(data.webster())
v2 = W.ladder2()[letters]
nw = len(lens)
before, after = WD.within_word(v2, pos, wlen)
wv = WD.word_values(v2, wid, pos, lens)
prev_w = np.where(wid > 0, wv[np.maximum(wid - 1, 0)], np.uint64(0))
next_w = np.where(wid < nw - 1, wv[np.minimum(wid + 1, nw - 1)], np.uint64(0))
# held out: every 20th block of 50,000 letters; a letter only if its word and both neighbours lie on its side
blk = (np.arange(len(letters)) // 50_000) % 20 == 19
wstart = np.concatenate([[0], np.cumsum(lens)[:-1]])
wside = blk[wstart]
side_ok = (wside[wid] == wside[np.maximum(wid - 1, 0)]) & (wside[wid] == wside[np.minimum(wid + 1, nw - 1)])
WHOLE = np.flatnonzero(side_ok & ~wside[wid]); TEST = np.flatnonzero(side_ok & wside[wid])
y = letters[TEST]
steps = np.stack([pos, wlen - pos - 1], 1)
levels = {"letter in word + word before + word after": WD.h(before, after, steps, prev_w.view(np.int64) % 1000003,
                                                              next_w.view(np.int64) % 1000003),
          "letter in word + word before": WD.h(before, after, steps, prev_w.view(np.int64) % 1000003),
          "letter in word + word after": WD.h(before, after, steps, next_w.view(np.int64) % 1000003),
          "letter in word alone": WD.h(before, after, steps)}
res = {"letters": int(len(letters)), "words": int(nw), "whole": int(len(WHOLE)), "held_out": int(len(TEST)), "levels": {}}
reads = {}
for name, K in levels.items():
    C = W.Returns(K[WHOLE], letters[WHOLE]).read(K[TEST]); met = C.sum(1) > 0
    one = (C > 0).sum(1) == 1
    reads[name] = (C, met)
    res["levels"][name] = {"returns": round(float(met.mean()*100), 2),
                           "one_letter_closes": round(float(one.mean()*100), 2),
                           "right_where_one_closes": round(float((C[one].argmax(1) == y[one]).mean()*100), 3),
                           "truth_among_met": round(float((C[met, y[met]] > 0).mean()*100), 3),
                           "right_letter_met_most": round(float((C[met].argmax(1) == y[met]).mean()*100), 3)}
    print(name, json.dumps(res["levels"][name]), flush=True)
for order in (list(levels), ["letter in word + word before + word after", "letter in word alone"], ["letter in word alone"]):
    pred = np.full(len(y), -1); single = np.zeros(len(y), bool)
    for s in order:
        C, met = reads[s]; new = met & (pred < 0); pred[new] = C[new].argmax(1); single[new] = (C[new] > 0).sum(1) == 1
    r = {"acc": round(float((pred == y).mean()*100), 3), "read_by_one_closing_letter": round(float(single.mean()*100), 2),
         "right_where_one_closes": round(float((pred[single] == y[single]).mean()*100), 3),
         "unread": round(float((pred < 0).mean()*100), 3)}
    res[" > ".join(order)] = r
    print(" > ".join(order), json.dumps(r), flush=True)
res["seconds"] = round(time.time() - t0)
json.dump(res, open("results/words.json", "w"), indent=1)
