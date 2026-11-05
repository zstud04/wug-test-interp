#!/usr/bin/env python3
"""
Build a noun-vs-verb dev set for [wug]/[wugs] from templated [mask] sentences.

Each "noun" template uses [mask] as a direct object / PP complement (grammatical
for a noun). Each "verb" template uses [mask] as a bare/infinitive verb slot
(ungrammatical for a noun). Pairing a noun template with a verb template and
filling both with the same token ([wug] or [wugs]) gives a good/bad minimal
pair that tests only part-of-speech, never singular/plural number agreement.

Usage:
  python3 eacl-rebuttal/scripts/build_dev_set.py \
      --in eacl-rebuttal/data/dev/pos_templates_source.json \
      --out eacl-rebuttal/data/dev/dev_noun_vs_verb.csv
"""
import argparse
import csv
import json
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.in_path) as f:
        data = json.load(f)
    nouns = data["noun"]
    verbs = data["verb"]
    assert len(nouns) == len(verbs), (
        f"Expected equal counts, got {len(nouns)} noun vs {len(verbs)} verb templates"
    )

    verbs_shuffled = verbs[:]
    random.Random(args.seed).shuffle(verbs_shuffled)

    rows = []
    for pair_id, (n_tmpl, v_tmpl) in enumerate(zip(nouns, verbs_shuffled)):
        assert "[mask]" in n_tmpl and "[mask]" in v_tmpl
        for token in ("[wug]", "[wugs]"):
            rows.append({
                "good": n_tmpl.replace("[mask]", token),
                "bad": v_tmpl.replace("[mask]", token),
                "token": token,
                "pair_id": pair_id,
            })

    with open(args.out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["good", "bad", "token", "pair_id"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows ({len(nouns)} pairs x 2 tokens) to {args.out_path}")


if __name__ == "__main__":
    main()
