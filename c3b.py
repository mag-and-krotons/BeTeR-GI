"""Challenge 3: is a noun a kind of X? From Webster's definitions alone, graded against WordNet (results/c3b.json).
Reader: walk definitions: a word's first clause (labels in parentheses dropped, up to the first ';' or '.'), its words
that are Webster headwords (plural folded: -s, -es, -ies), not among the whole's 100 most frequent words; then those
words' first clauses, and so on. "Yes" when the category is reached within D steps. Trained: word2vec (skip-gram,
negative sampling, gradient descent) on the same Webster text; score = cosine."""
import json, re, time
from collections import Counter
import numpy as np
import nltk; nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet as wn
from sklearn.metrics import roc_auc_score
from gensim.models import Word2Vec
from gi import data
t0 = time.time(); log = lambda *a: print(*a, f"[{time.time()-t0:.0f}s]", flush=True)
web = {k.lower(): v for k, v in data.webster().items()}
CATS = {"animal": "animal.n.01", "plant": "plant.n.02", "mammal": "mammal.n.01", "bird": "bird.n.01", "fish": "fish.n.01",
        "insect": "insect.n.01", "reptile": "reptile.n.01", "mollusk": "mollusk.n.01", "tree": "tree.n.01", "shrub": "shrub.n.01",
        "fruit": "fruit.n.01", "vegetable": "vegetable.n.01", "disease": "disease.n.01", "metal": "metal.n.01",
        "mineral": "mineral.n.01", "vehicle": "vehicle.n.01", "weapon": "weapon.n.01", "tool": "tool.n.01",
        "building": "building.n.01", "fabric": "fabric.n.01", "instrument": "musical_instrument.n.01"}
CATS = {c: wn.synset(s) for c, s in CATS.items() if c in web}
tok = lambda t: re.findall(r"[a-z]+", t.lower())
freq = Counter(t for v in web.values() for t in tok(v))
STOP = {w for w, _ in freq.most_common(100)}
def fold(t):
    if t in web: return t
    for suf, rep in (("ies", "y"), ("es", ""), ("s", "")):
        if t.endswith(suf) and t[: -len(suf)] + rep in web: return t[: -len(suf)] + rep
    return None
def first_clause(w):
    t = re.sub(r"\([^)]*\)", " ", web.get(w, ""))
    senses = [s for s in re.split(r"\b\d+\.\s", t) if s.strip()] or [t]
    return " ".join(re.split(r"[;.]", s, 1)[0] for s in senses)          # every sense: its first clause
step_cache = {}
def step(w):
    if w not in step_cache:
        step_cache[w] = {f for f in (fold(t) for t in tok(first_clause(w))) if f and f not in STOP and f != w}
    return step_cache[w]
def depth_to(w, cat, D=4):
    frontier, seen = {w}, {w}
    for d in range(1, D + 1):
        nxt = set().union(*(step(x) for x in frontier)) - seen
        if cat in nxt: return d
        seen |= nxt; frontier = nxt
        if not frontier: return None
    return None
nouns = sorted({l for l in wn.all_lemma_names(pos="n") if re.fullmatch(r"[a-z]+", l) and l in web and l not in CATS})
lab = {}
for w in nouns:
    anc = {s for syn in wn.synsets(w, "n") for p in syn.hypernym_paths() for s in p}     # every sense
    lab[w] = {c: CATS[c] in anc for c in CATS}
pos_w = [w for w in nouns if any(lab[w].values())]; neg_w = [w for w in nouns if not any(lab[w].values())]
r = np.random.default_rng(0)
test = list(r.choice(pos_w, 2000, replace=False)) + list(r.choice(neg_w, 2000, replace=False))
log("categories", len(CATS), "| nouns in both", len(nouns), "| positives", len(pos_w), "| test nouns", len(test))
pairs = [(w, c) for w in test for c in CATS]; y = np.array([lab[w][c] for w, c in pairs])
dep = np.array([depth_to(w, c) or 0 for w, c in pairs])
log("reader walked", len(pairs), "pairs")
sents = [tok(f"{k} {v}") for k, v in web.items()]
m = Word2Vec(sents, vector_size=100, window=5, min_count=2, sg=1, negative=5, epochs=5, workers=1, seed=0)
log("word2vec trained", m.wv.vectors.size + m.syn1neg.size, "parameters")
cos = np.array([m.wv.similarity(w, c) if w in m.wv and c in m.wv else -1.0 for w, c in pairs])
res = {"categories": list(CATS), "test_nouns": len(test), "pairs": len(pairs), "positive_pairs": int(y.sum()),
       "trained_parameters": int(m.wv.vectors.size + m.syn1neg.size), "reader": {}, "trained": {}}
score_r = np.where(dep > 0, 1.0 / np.maximum(dep, 1), 0.0)
res["reader"]["auc"] = round(float(roc_auc_score(y, score_r)), 4); res["trained"]["auc"] = round(float(roc_auc_score(y, cos)), 4)
for D in (1, 2, 3, 4):
    yes = (dep > 0) & (dep <= D)
    rec = float((yes & y).sum() / y.sum()); prec = float((yes & y).sum() / max(yes.sum(), 1))
    thr = np.sort(cos[y])[::-1][max(int(rec * y.sum()) - 1, 0)] if rec > 0 else np.inf
    tyes = cos >= thr
    res["reader"][f"D={D}"] = {"recall_%": round(rec*100, 2), "precision_%": round(prec*100, 2), "accuracy_%": round(float((yes == y).mean()*100), 2)}
    res["trained"][f"at the reader's recall, D={D}"] = {"precision_%": round(float((tyes & y).sum() / max(tyes.sum(), 1) * 100), 2),
                                                       "accuracy_%": round(float((tyes == y).mean()*100), 2)}
res["reader"]["by_category_D=4"] = {c: {"recall_%": round(float(((dep > 0) & y)[[i for i, p in enumerate(pairs) if p[1] == c]].sum() /
                                         max(y[[i for i, p in enumerate(pairs) if p[1] == c]].sum(), 1) * 100), 1)} for c in CATS}
res["examples_reader_yes"] = [f"{w} -> {c} in {d}" for (w, c), d, t in zip(pairs, dep, y) if d and t][:8]
res["examples_reader_wrong_yes"] = [f"{w} -> {c} in {d}: {first_clause(w)[:60]}" for (w, c), d, t in zip(pairs, dep, y) if d and not t][:8]
np.savez("results/c3b_scores.npz", y=y, dep=dep, cos=cos)
log(json.dumps(res, indent=1)); json.dump(res, open("results/c3b.json", "w"), indent=1)
