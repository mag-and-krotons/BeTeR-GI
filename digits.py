"""The digit reading of BeTeR-Intelligence (stack6), reproduced exactly (results/digits.json). python3 digits.py"""
import json
import numpy as np
from gi import data, reader as R
X, Y = data.optdigits(); TR, TE = data.his_split(Y)
his = json.load(open("data/stack6_published.json"))
rows = []
for seed in (0, 1, 2):
    k, _ = R.choose_k(X[TR], Y[TR], seed)
    laps = R.write(X[TR], Y[TR], k, np.random.default_rng(seed))
    cs, Rf = R.left(laps, X[TE]); p = cs[np.argmin(Rf, 1)]
    rows.append({"seed": seed, "k": k, "accuracy_%": round(float((p == Y[TE]).mean() * 100), 2),
                 "published_%": [s["test_all_ten"] for s in his["seeds"] if s["seed"] == seed][0]})
    print(json.dumps(rows[-1]))
json.dump(rows, open("results/digits.json", "w"), indent=1)
