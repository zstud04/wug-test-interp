#!/usr/bin/env python3
"""Generate a new, independently-sourced subject-verb-agreement-under-attractors
stimulus set for the [wug]/[wugs] novel-word test.

Design goals (fixed BEFORE looking at any model's score on these items):
  - Vocabulary is not hand-picked for this task. Attractor/distractor nouns are
    the existing eacl-rebuttal/data/init/noun_init_large.txt list (530 pairs,
    already algorithmically sourced from WordNet for a different, unrelated
    purpose: embedding initialization). Verbs are drawn from WordNet's verb
    lemma inventory, ranked by corpus frequency (wordfreq), mechanically
    filtered to regular -s/bare morphology (no hand-picked verb list).
  - Sentence frame follows the standard long-distance attractor paradigm
    (subject NP + 0-3 prepositional-phrase attractors of the OPPOSITE number
    + verb), matching the established design in nb/stimuli_gen/das_stimuli_gen.ipynb
    but without decorative adjectives (fewer researcher degrees of freedom).
  - Every word is filtered to be single-token in BOTH the Qwen3-VL-4B and
    Gemma-3-4B tokenizers, so the exact same item set is usable for both model
    families without separate per-model filtering.
  - Item selection uses a single fixed random seed. This script is run once;
    the resulting CSV is not edited after inspecting any model's scores on it.
"""
import argparse
import random
import re
from collections import Counter

import nltk
import pandas as pd

nltk.data.path.append("/home/gsc685/nltk_data")  # shared-machine cache; read-only
from nltk.corpus import brown

SEED = 20260101  # fixed once, never re-rolled after seeing results
PREPOSITIONS = [
    "near", "behind", "beside", "under", "above", "below", "around",
    "past", "outside", "inside", "across from", "next to", "in front of",
]
IRREGULAR_OR_MODAL = {
    "be", "am", "is", "are", "was", "were", "have", "has", "had", "do", "does",
    "did", "can", "could", "will", "would", "shall", "should", "may", "might",
    "must", "go", "goes",
}
N_ATTRACTOR_LEVELS = 4  # 0..3
ITEMS_PER_CELL = 60     # per (target_number, attractor_count) cell


def regular_3sg(bare):
    """Deterministic English 3sg-present suffixation. Returns None if the verb
    looks irregular under this simple rule (handled by the IRREGULAR_OR_MODAL
    blocklist upstream; this only applies the spelling rule for the rest)."""
    if re.search(r"(s|x|z|ch|sh|o)$", bare):
        return bare + "es"
    if re.search(r"[^aeiou]y$", bare):
        return bare[:-1] + "ies"
    return bare + "s"


def unstem_3sg(word):
    """Reverse regular_3sg's suffix rule to recover the candidate bare form."""
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if re.search(r"(s|x|z|ch|sh|o)es$", word):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return None


def build_verb_list(n=80):
    """Verb pairs are attested usage from the Brown corpus (POS-tagged real
    text), not a dictionary lemma list — this is what actually excludes
    words that are predominantly nouns with a rare dictionary verb sense
    (e.g. 'bus', 'target', 'action'), since those simply won't show up
    tagged VBZ/VB in real running text with any frequency."""
    tag_counts = Counter()
    for word, tag in brown.tagged_words():
        w = word.lower()
        if w.isalpha():
            tag_counts[(w, tag.split("-")[0])] += 1

    vbz = Counter()
    vb = Counter()
    noun = Counter()
    for (w, tag), c in tag_counts.items():
        if tag == "VBZ":
            vbz[w] += c
        elif tag == "VB":
            vb[w] += c
        elif tag in ("NN", "NNS"):
            noun[w] += c

    candidates = []
    for w, c in vbz.items():
        if w in IRREGULAR_OR_MODAL:
            continue
        bare = unstem_3sg(w)
        if bare is None or bare not in vb or bare in IRREGULAR_OR_MODAL:
            continue
        if regular_3sg(bare) != w:
            continue
        if noun.get(w, 0) > c or noun.get(bare, 0) > vb[bare]:
            continue  # predominantly a noun in attested usage; reject
        candidates.append((c + vb[bare], w, bare))

    candidates.sort(reverse=True)
    return [(w, bare) for _, w, bare in candidates[: n * 3]]


def load_noun_pairs(path):
    pairs = []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for i in range(0, len(lines) - 1, 2):
        sg, pl = lines[i], lines[i + 1]
        pairs.append((sg, pl))
    return pairs


def single_token_filter(words, tokenizers):
    ok = set(words)
    for tok in tokenizers:
        keep = set()
        for w in ok:
            ids = tok(" " + w, add_special_tokens=False).input_ids
            if len(ids) == 1:
                keep.add(w)
        ok = keep
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noun_init_large", default="eacl-rebuttal/data/init/noun_init_large.txt")
    ap.add_argument("--out_csv", default="eacl-rebuttal/data/eval/principled_attractor_stimuli.csv")
    ap.add_argument("--models", nargs="+",
                     default=["Qwen/Qwen3-VL-4B-Instruct", "google/gemma-3-4b-it"])
    ap.add_argument("--cache_dirs", nargs="+",
                     default=["/home/shared/hf_cache", "/home/km55359/.cache/hf_gemma3"])
    args = ap.parse_args()

    rng = random.Random(SEED)

    from transformers import AutoTokenizer
    tokenizers = [AutoTokenizer.from_pretrained(m, cache_dir=c)
                  for m, c in zip(args.models, args.cache_dirs)]

    noun_pairs = load_noun_pairs(args.noun_init_large)
    noun_words = set(w for pair in noun_pairs for w in pair)
    noun_words_ok = single_token_filter(noun_words, tokenizers)
    noun_pairs = [(sg, pl) for sg, pl in noun_pairs if sg in noun_words_ok and pl in noun_words_ok]
    print(f"Usable single-token distractor noun pairs: {len(noun_pairs)}")

    verb_pairs_raw = build_verb_list(n=200)
    verb_words = set(w for pair in verb_pairs_raw for w in pair)
    verb_words_ok = single_token_filter(verb_words, tokenizers)
    verb_pairs = [(s3, bare) for s3, bare in verb_pairs_raw if s3 in verb_words_ok and bare in verb_words_ok]
    print(f"Usable single-token verb pairs: {len(verb_pairs)}")
    assert len(verb_pairs) >= 20, "too few verbs survived filtering"

    rows = []
    idx = 0
    for target_number in ["singular", "plural"]:
        subj = "[wug]" if target_number == "singular" else "[wugs]"
        for n_attr in range(N_ATTRACTOR_LEVELS):
            for _ in range(ITEMS_PER_CELL):
                s3, bare = rng.choice(verb_pairs)
                good_verb, bad_verb = (s3, bare) if target_number == "singular" else (bare, s3)

                # Attractors are the OPPOSITE number from the target (maximal
                # interference), each a distinct preposition + distinct noun.
                attractor_number = "plural" if target_number == "singular" else "singular"
                chosen_preps = rng.sample(PREPOSITIONS, k=n_attr) if n_attr else []
                chosen_nouns = rng.sample(noun_pairs, k=n_attr) if n_attr else []
                pp_chunks = []
                for prep, (sg_n, pl_n) in zip(chosen_preps, chosen_nouns):
                    noun = pl_n if attractor_number == "plural" else sg_n
                    pp_chunks.append(f"{prep} the {noun}")
                # Space-joined, no commas — matches the validated PP-chain
                # format in nb/stimuli_gen/das_stimuli_gen.ipynb exactly
                # (e.g. "near the gates beside the fences under the lamps"),
                # rather than a list-like comma-joined chain models have
                # never encountered.
                pp_str = (" " + " ".join(pp_chunks)) if pp_chunks else ""

                good = f"The {subj}{pp_str} {good_verb}."
                bad = f"The {subj}{pp_str} {bad_verb}."

                rows.append({
                    "idx": idx,
                    "target_number": target_number,
                    "attractors": n_attr,
                    "condition": f"target_{target_number}_att{n_attr}",
                    "verb_3sg": s3,
                    "verb_bare": bare,
                    "attractor_nouns": ";".join(n for _, n in [(p, pl_n if attractor_number == "plural" else sg_n)
                                                                for p, (sg_n, pl_n) in zip(chosen_preps, chosen_nouns)]),
                    "good": good,
                    "bad": bad,
                })
                idx += 1

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} items to {args.out_csv}")
    print(df.groupby(["target_number", "attractors"]).size())


if __name__ == "__main__":
    main()
