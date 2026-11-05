#!/usr/bin/env python3
"""
Generate a large, number-unbiased singular/plural noun list for [wug]/[wugs]
embedding initialization, using get_unambiguous_nouns() from
representational-analysis/embed_analysis.py (imported, not duplicated).

Writes alternating singular/plural lines (same format as data/embeddings/init/
noun_init.txt) so core/train/embed_train.py's --embed_init loader needs no changes.

Usage:
  python3 eacl-rebuttal/scripts/generate_large_noun_init.py \
      --model Qwen/Qwen3-VL-2B-Instruct \
      --cache_dir /home/shared/hf_cache \
      --out eacl-rebuttal/data/init/noun_init_large.txt
"""
import argparse
import importlib.util
import os
import pathlib
import sys

import torch
from minicons import scorer

_REPO_ROOT = pathlib.Path(__file__).resolve().parent
while not (_REPO_ROOT / "core").is_dir():
    _parent = _REPO_ROOT.parent
    if _parent == _REPO_ROOT:
        raise RuntimeError(f"Could not find repo root (no core/ dir above {__file__})")
    _REPO_ROOT = _parent


def _load_get_unambiguous_nouns():
    """Import get_unambiguous_nouns from representational-analysis/embed_analysis.py.

    That directory has a hyphen in its name, so it can't be a normal dotted
    import; load the module directly from its file path instead.
    """
    mod_path = _REPO_ROOT / "representational-analysis" / "embed_analysis.py"
    spec = importlib.util.spec_from_file_location("embed_analysis", mod_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_unambiguous_nouns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--cache_dir", default="/home/shared/hf_cache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n", type=int, default=2000, help="Max candidate pairs (same default as embed_analysis.py)")
    parser.add_argument("--min_noun_ratio", type=float, default=0.5)
    parser.add_argument("--out", default=str(_REPO_ROOT / "eacl-rebuttal" / "data" / "init" / "noun_init_large.txt"))
    args = parser.parse_args()

    get_unambiguous_nouns = _load_get_unambiguous_nouns()

    hf_token = os.getenv("HF_TOKEN")
    print(f"Loading model: {args.model} (cache_dir={args.cache_dir})")
    lm = scorer.VLMScorer(
        args.model, device=args.device, token=hf_token,
        torch_dtype=torch.bfloat16, cache_dir=args.cache_dir,
    )
    tok = lm.tokenizer.tokenizer

    print(f"Finding unambiguous noun pairs (n<={args.n}, min_noun_ratio={args.min_noun_ratio})...")
    pairs = get_unambiguous_nouns(tok, n=args.n, min_noun_ratio=args.min_noun_ratio)
    print(f"  Found {len(pairs)} singular/plural pairs ({2 * len(pairs)} words)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for p in pairs:
            f.write(p["singular"] + "\n")
            f.write(p["plural"] + "\n")

    print(f"Wrote {args.out}")
    print("Preview (first 10 pairs, by frequency):")
    for p in pairs[:10]:
        print(f"  {p['singular']:>15} / {p['plural']:<15} freq={p['freq']:.2e}")


if __name__ == "__main__":
    main()
