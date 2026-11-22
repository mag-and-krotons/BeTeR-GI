"""The remaining doubts in one run (results/doubts.json). A: a knowledge question walked from the whole, against word
vectors of the same whole. B: a stronger trained character model for the next letter, against the reader's 2.345 bits."""
import json, re, time
from collections import Counter
import numpy as np
import nltk; nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet as wn
from gensim.models import Word2Vec
from gi import data, words as WD
from gi.mlp import MLP, stream
t0 = time.time(); RES = {}
def log(*a): print(*a, f"[{time.time()-t0:.0f}s]", flush=True)
def save(): json.dump(RES, open("results/doubts.json", "w"), indent=1)

# ------------------------------------------------------------------ A. what kind of thing is X?
web = {k.lower(): v for k, v in data.webster().items()}
CATS = {"animal": "animal.n.01", "plant": "plant.n.02", "mammal": "mammal.n.01", "bird": "bird.n.01", "fish": "fish.n.01",
        "insect": "insect.n.01", "reptile": "reptile.n.01", "mollusk": "mollusk.n.01", "tree": "tree.n.01", "shrub": "shrub.n.01",
        "fruit": "fruit.n.01", "vegetable": "vegetable.n.01", "disease": "disease.n.01", "metal": "metal.n.01",
        "mineral": "mineral.n.01", "vehicle": "vehicle.n.01", "weapon": "weapon.n.01", "tool": "tool.n.01",
        "building": "building.n.01", "fabric": "fabric.n.01", "instrument": "musical_instrument.n.01"}
CATS = {c: wn.synset(s) for c, s in CATS.items() if c in web}
tok = lambda t: re.findall(r"[a-z]+", t.lower())
STOP = {w for w, _ in Counter(t for v in web.values() for t in tok(v)).most_common(100)}
def fold(t):
    if t in web: return t
    for suf, rep in (("ies", "y"), ("es", ""), ("s", "")):
        if t.endswith(suf) and t[: -len(suf)] + rep in web: return t[: -len(suf)] + rep
    return None
def clauses(w, every):
    t = re.sub(r"\([^)]*\)", " ", web.get(w, ""))
    senses = [s for s in re.split(r"\b\d+\.\s", t) if s.strip()] or [t]
    return [re.split(r"[;.]", s, 1)[0] for s in (senses if every else senses[:1])]
cache = {}
def step(w, every):
    k = (w, every)
    if k not in cache:
        cache[k] = {f for c in clauses(w, every) for f in (fold(t) for t in tok(c)) if f and f not in STOP and f != w}
    return cache[k]
def answer(w, every, D=4):
    frontier, seen = {w}, {w}
    for d in range(1, D + 1):
        nxt = set().union(*(step(x, every) for x in frontier)) - seen
        got = nxt & set(CATS)
        if got: return d, got
        seen |= nxt; frontier = nxt
        if not frontier: break
    return None, set()
nouns = sorted({l for l in wn.all_lemma_names(pos="n") if re.fullmatch(r"[a-z]+", l) and l in web and l not in CATS})
truth = {w: {c for c in CATS if CATS[c] in {s for syn in wn.synsets(w, "n") for p in syn.hypernym_paths() for s in p}} for w in nouns}
r = np.random.default_rng(0)
pos_w = list(r.choice([w for w in nouns if truth[w]], 2000, replace=False))
neg_w = list(r.choice([w for w in nouns if not truth[w]], 2000, replace=False))
sents = [tok(f"{k} {v}") for k, v in web.items()]
models = {"word2vec 100d, 5 passes": Word2Vec(sents, vector_size=100, window=5, min_count=2, sg=1, negative=5, epochs=5, workers=1, seed=0)}
log("word2vec 100d trained")
models["word2vec 300d, 10 passes"] = Word2Vec(sents, vector_size=300, window=5, min_count=2, sg=1, negative=5, epochs=10, workers=1, seed=0)
log("word2vec 300d trained")
A = {}
for every in (False, True):
    name = "reader, every sense" if every else "reader, first sense"
    got = {w: answer(w, every) for w in pos_w + neg_w}
    ans = [w for w in pos_w if got[w][1]]
    sz = np.array([len(got[w][1]) for w in ans]); tin = np.array([bool(got[w][1] & truth[w]) for w in ans])
    A[name] = {"positives_answered_%": round(len(ans) / 20, 2), "truth_among_answer_%": round(float(tin.mean()*100), 2),
               "answer_size_mean": round(float(sz.mean()), 2), "single_answer_%": round(float((sz == 1).mean()*100), 2),
               "single_right_%": round(float(tin[sz == 1].mean()*100), 2),
               "answered_with_no_category_true_(negatives answered)_%": round(sum(bool(got[w][1]) for w in neg_w) / 20, 2),
               "depth_of_answer": dict(Counter(got[w][0] for w in ans))}
    for mname, m in models.items():
        cats = [c for c in CATS if c in m.wv]
        def cos(w): return np.array([m.wv.similarity(w, c) if w in m.wv else -1 for c in cats])
        top = {w: np.argsort(-cos(w)) for w in ans}
        A[name][f"{mname}: truth in its top-k, k = reader's answer size"] = round(float(np.mean(
            [bool({cats[i] for i in top[w][:len(got[w][1])]} & truth[w]) for w in ans]) * 100), 2)
        A[name][f"{mname}: top-1 right on all 2000 positives"] = round(float(np.mean(
            [cats[int(np.argmax(cos(w)))] in truth[w] for w in pos_w]) * 100), 2)
        best_p = np.array([cos(w).max() for w in pos_w]); best_n = np.array([cos(w).max() for w in neg_w])
        thr = np.quantile(best_p, 1 - len(ans) / 2000)
        A[name][f"{mname}: negatives answered at the same answer rate %"] = round(float((best_n >= thr).mean()*100), 2)
    log(name, json.dumps(A[name], indent=1))
RES["A: what kind of thing is X"] = A; save()

# ------------------------------------------------------------------ B. a stronger trained model for the next letter
web_words = WD.words_of(" ".join(f"{k} {v}" for k, v in data.webster().items()))
qwords = WD.words_of("".join(open(f"data/wt2_{p}.txt", encoding="utf8").read() for p in ("valid", "test")), keep_unknown=True)
sym_w, at_w = stream(web_words); sym_q, at_q = stream(qwords)
OFFS = np.arange(-24, 0)
PW = np.concatenate([np.zeros(24, np.int8), sym_w]); PQ = np.concatenate([np.zeros(24, np.int8), sym_q])
win = lambda P, at: P[(at + 24)[:, None] + OFFS[None, :]].astype(np.int64)
held = (at_w // 50_000) % 20 == 19
tr_at, ho_at = at_w[~held], at_w[held]
r = np.random.default_rng(0); samp = r.choice(tr_at, 8_000_000, replace=False); ho = r.choice(ho_at, 200_000, replace=False)
m = MLP(24, hidden=512); log("training", m.n_parameters, "parameters on", len(samp), "samples")
bits = lambda P, at, y: float(-m.logp(win(P, at))[np.arange(len(at)), y].mean() / np.log(2))
B = {"parameters": m.n_parameters, "epochs": []}
for e, lr in enumerate((1e-3, 3e-4)):
    order = r.permutation(len(samp)); L = []
    for i in range(0, len(order), 1024):
        b = samp[order[i:i + 1024]]
        L.append(m.step(win(PW, b), sym_w[b].astype(np.int64) - 1, lr))
    ep = {"epoch": e + 1, "train_bits": round(float(np.mean(L) / np.log(2)), 4),
          "webster_held_out_bits": round(bits(PW, ho, sym_w[ho].astype(np.int64) - 1), 4)}
    B["epochs"].append(ep); log(json.dumps(ep)); RES["B: stronger trained next-letter model"] = B; save()
yq = sym_q[at_q].astype(np.int64) - 1
lp = np.concatenate([m.logp(win(PQ, at_q[i:i + 200_000])) for i in range(0, len(at_q), 200_000)])
bt = -lp[np.arange(len(yq)), yq] / np.log(2)
first = np.concatenate([[True], sym_q[at_q[1:] - 1] == 0])
c2 = json.load(open("results/c2.json"))
B.update({"wikipedia_bits_per_letter": round(float(bt.mean()), 4), "wikipedia_top1_%": round(float((lp.argmax(1) == yq).mean()*100), 2),
          "first_letter_bits": round(float(bt[first].mean()), 3), "later_letters_bits": round(float(bt[~first].mean()), 3),
          "reader_bits_same_letters": c2["reader_bits_per_letter"], "reader_top1_%": c2["reader_top1_%"],
          "reader_first_letter_bits": c2["first_letter_of_word"]["reader_bits"], "reader_later_bits": c2["later_letters"]["reader_bits"],
          "small_trained_model_bits": c2["trained_bits_per_letter"]})
RES["B: stronger trained next-letter model"] = B; save(); log(json.dumps(B, indent=1)); log("done")
