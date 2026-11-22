#!/bin/sh
# Public data used by every run. optdigits: UCI (CC BY 4.0) via BeTeR-Intelligence; Webster's Unabridged (Project
# Gutenberg, public domain) in JSON; WikiText-2 (CC BY-SA, from Wikipedia) via pytorch/examples.
set -e; cd "$(dirname "$0")/data"
curl -sL -o optdigits.tes https://raw.githubusercontent.com/mag-and-krotons/BeTeR-Intelligence/main/data/optdigits.tes
curl -sL -o webster_mr.json https://raw.githubusercontent.com/matthewreagan/WebstersEnglishDictionary/master/dictionary.json
for p in train valid test; do
  curl -sL -o wt2_$p.txt https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/$p.txt
done
echo "data ready"
