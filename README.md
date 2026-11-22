# BeTeR General Intelligence

Reading with nothing trained: no parameters, no gradient descent, no fitted constant. The whole is given, and reading
is closure against it. Built on the BeTeR principle and the three-π singularity structure: objects are related only
through the mechanism of their construction.

## How it reads

- **Letters are positions.** The 26 letters sit on a ladder of half-steps from a: a 0; b +½, z −½; c +1, y −1; …;
  j +4.5, r −4.5. Beyond ±4.5 the walk turns back: k 4, q −4, l 3.5, p −3.5, m 3, o −3, n 2.5.
- **A text is one walk.** Each step is read with its twist: the d-th letter from the reader turns by d twelfths of a
  turn, so the place (3) and the reading (4) close together at 12. A steady walk returns exactly; what does not
  return is the remainder. All sums are exact integers.
- **Silence bounds the walk.** A letter's arriving walk is the letters before it in its word; its leaving walk is the
  letters after it, π away.
- **Two observers meet at the letter.** Both remainders must return together, at one place in the whole.
- **The word is one step, one scale up.** Its value is the remainder of its own walk; the word before and the word
  after are that scale's observers.
- **The reading is every closure.** Closures are read from the most complete level down (word scale with both
  neighbours, then the word alone, then shorter walks). Where the whole gives several letters, all are read.
- **Meaning is walked from the whole.** A word's definitions are walked to their construction: kitten → "a young
  cat" → cat → … → mammal; seven → "one more than six".

## Results

Every number is reproduced by the script named. Trained models were trained on the same whole the reader walks
(Webster's Unabridged), and read the same held-out text.

| task | reader, nothing trained | trained on the same whole |
|---|---|---|
| Letter in its word, a book never walked (Wikipedia against Webster): truth among closures (`run_newbook.py`) | **99.21 %** (median closure 2 letters) | |
| Letter in its word, Webster's own held-out text (`run_words.py`, `run_checks.py`) | **99.58 %** | |
| Next letter, 1.65 M Wikipedia letters, bits per letter (`c2.py`, `run_gru.py`, `run_doubts.py`) | **2.345** (49.7 % first guess) | GRU, 225,307 parameters: 2.439 (47.3 %); MLP, 345,626: 2.614; MLP, 117,530: 2.777 |
| Is-a from definitions, graded by WordNet, AUC (`c3.py`, `c3b.py`) | **0.806** | word2vec, 14.5 M parameters: 0.679 |
| "What kind of thing is X?", truth among the answer (`run_doubts.py`) | **70.7 %** | word2vec 300d: 34.3 % (same answer size) |
| Letters of words Webster never held, truth among closures at equal set size (`c1b.py`) | **66.4 %** | 63.7 % |
| Letters of words removed from the whole, at equal set size (`c1b.py`) | 56.7 % | 81.3 % |

The digit reading of BeTeR-Intelligence is reproduced exactly (`digits.py`: 98.14 / 97.59 / 98.52 %, zero
readings differ).

Open: words the whole does not hold, and answering "none" where no category applies.

## Reproduce

    pip install -r requirements.txt
    ./fetch_data.sh                 # optdigits, Webster's Unabridged (JSON), WikiText-2
    python3 run_newbook.py          # or any script named above; results go to results/

## Files

- `gi/walk.py`: the ladder, the twists, exact remainders, returns against the whole
- `gi/words.py`: silence-bounded walks, the word one scale up
- `gi/mlp.py`, `gi/gru.py`: the trained models compared against (gradients checked)
- `gi/reader.py`, `gi/band.py`: the digit reader and the band (positions as n mod 3, n mod 4)

Licence: CC BY 4.0 (see LICENSE).
