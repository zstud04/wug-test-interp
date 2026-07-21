#!/usr/bin/env python3
"""
Minimal-pair minicons evaluation of learned [wug]/[wugs] embeddings on the
attractor agreement stimuli, across all CI seed runs.

For each condition (text, image) and each seed found under the corresponding
CI_seed_runs directory:
  - Loads the seed's learned_embeddings.pt (produced by core.train.embed_train)
  - Injects the learned [wug]/[wugs] embeddings into the model
  - Scores every good/bad singular+plural stimulus quadruple with minicons
    sequence_score
  - Records per-item scores and per-seed accuracy (overall + by condition)

Usage:
  python3 analysis/behavioral.py \
      --model Qwen/Qwen3-VL-2B-Instruct \
      --text_dir  results/train/CI_seed_runs/lr_ci_results_Qwen3-VL-2B-Instruct_text_lr0p001_50seeds \
      --image_dir results/train/CI_seed_runs/lr_ci_results_Qwen3-VL-2B-Instruct_image_lr0p001_50seeds \
      --out_dir   results/eval/attractors/Qwen3-VL-2B-Instruct

Output files (per condition):
  attractor_scored_{text,image}.csv  — per-seed, per-item good/bad scores
  attractor_summary_{text,image}.csv — per-seed accuracy, overall and by condition
"""
import argparse
import os
import pathlib
import sys

import pandas as pd
import torch
from minicons import scorer

REPO_ROOT = pathlib.Path(__file__).resolve().parent
while not (REPO_ROOT / "core").is_dir():
    parent = REPO_ROOT.parent
    if parent == REPO_ROOT:
        raise RuntimeError(f"Could not find repo root (no core/ dir above {__file__})")
    REPO_ROOT = parent
sys.path.insert(0, str(REPO_ROOT))

from utils.chat_templates import chat_template

attractor_stimuli = "data/interp/agreement_target_wug.csv"

PAIRED_COLS = ("good_singular", "bad_singular", "good_plural", "bad_plural")


# ---------------------------------------------------------------------------
# Stimuli / seed-run helpers
# ---------------------------------------------------------------------------

def load_stimuli(path):
    df = pd.read_csv(path)
    for col in PAIRED_COLS:
        assert col in df.columns, f"Stimuli CSV {path} missing required column '{col}'"
        df[col] = df[col].astype(str).str.strip()
    return df


def find_seed_embeddings(results_dir):
    """Yield (seed, path-to-learned_embeddings.pt) for each seed_N/ subfolder."""
    prefix = "syntax" if "text" in results_dir else ""
    for folder in sorted(os.listdir(results_dir)):
        if not folder.startswith("seed_"):
            continue
        seed = int(folder.split("_")[1])
        run_dir = os.path.join(results_dir, folder,
                               f"{prefix}_alternating_lr0p001_seed{seed}_ep50")
        emb_path = os.path.join(run_dir, "learned_embeddings.pt")
        if not os.path.exists(emb_path):
            print(f"  WARNING: missing learned_embeddings.pt for seed {seed}, skipping")
            continue
        yield seed, emb_path


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def batched_sequence_score(lm, queries, batch_size=8):
    """Score queries in fixed-size batches to cap peak memory."""
    scores = []
    n = len(queries)
    for i in range(0, n, batch_size):
        batch = queries[i:i + batch_size]
        scores.extend(lm.sequence_score(batch))
        print(f"    scored {min(i + batch_size, n)}/{n}", end="\r")
    print()
    return torch.tensor(scores, dtype=torch.float32)


def setup_novel_tokens(lm):
    """Register [wug]/[wugs] tokens (if missing) and return their token ids."""
    added_tokens = [" [wug]", " [wugs]"]
    tok = lm.tokenizer.tokenizer
    existing_vocab = tok.get_vocab()
    tokens_to_add = [t for t in added_tokens if t not in existing_vocab]
    if tokens_to_add:
        tok.add_tokens(tokens_to_add)
        old_len = lm.model.resize_token_embeddings().weight.shape[0]
        lm.model.resize_token_embeddings(old_len + len(tokens_to_add))

    emb = lm.model.model.language_model.embed_tokens
    lm_head = lm.model.lm_head
    assert emb.weight.data_ptr() == lm_head.weight.data_ptr(), \
        "ERROR: emb and lm_head are NOT tied! This script requires tied weights."

    new_ids = [tok(t, add_special_tokens=False).input_ids[0] for t in added_tokens]
    wug_id, wugs_id = new_ids
    return emb, wug_id, wugs_id


def inject_embeddings(emb, wug_id, wugs_id, emb_path):
    """Copy a seed's learned [wug]/[wugs] rows into the live model."""
    rec = torch.load(emb_path, map_location="cpu")
    with torch.no_grad():
        emb.weight.data[wug_id] = rec["wug_embedding"].to(emb.weight.device, dtype=emb.weight.dtype)
        emb.weight.data[wugs_id] = rec["wugs_embedding"].to(emb.weight.device, dtype=emb.weight.dtype)


def run_condition(lm, emb, wug_id, wugs_id, results_dir, stim_df, queries, batch_size):
    """Score every seed's learned embeddings on the attractor stimuli.

    `queries` maps each of PAIRED_COLS to its pre-built list of chat-template
    query strings. Returns (scored_df, summary_df): one row per (seed, item),
    and one row per (seed, condition) with accuracy/margin, respectively.
    """
    scored_rows = []
    summary_rows = []
    for seed, emb_path in find_seed_embeddings(results_dir):
        print(f"  scoring seed {seed}...")
        inject_embeddings(emb, wug_id, wugs_id, emb_path)

        gs = batched_sequence_score(lm, queries["good_singular"], batch_size=batch_size)
        bs = batched_sequence_score(lm, queries["bad_singular"], batch_size=batch_size)
        gp = batched_sequence_score(lm, queries["good_plural"], batch_size=batch_size)
        bp = batched_sequence_score(lm, queries["bad_plural"], batch_size=batch_size)

        diff_sg = gs - bs
        diff_pl = gp - bp
        correct_sg = (gs > bs).numpy()
        correct_pl = (gp > bp).numpy()
        correct_all = correct_sg & correct_pl

        seed_df = stim_df.copy()
        seed_df["seed"] = seed
        seed_df["score_good_singular"] = gs.tolist()
        seed_df["score_bad_singular"] = bs.tolist()
        seed_df["score_good_plural"] = gp.tolist()
        seed_df["score_bad_plural"] = bp.tolist()
        seed_df["score_diff_singular"] = diff_sg.tolist()
        seed_df["score_diff_plural"] = diff_pl.tolist()
        seed_df["is_correct_singular"] = correct_sg.tolist()
        seed_df["is_correct_plural"] = correct_pl.tolist()
        seed_df["is_correct_all"] = correct_all.tolist()
        scored_rows.append(seed_df)

        def summarize(cond_label, mask):
            summary_rows.append({
                "seed": seed, "condition": cond_label,
                "accuracy_singular": correct_sg[mask].mean(),
                "accuracy_plural": correct_pl[mask].mean(),
                "accuracy_joint": correct_all[mask].mean(),
                "margin_singular": diff_sg[mask].mean().item(),
                "margin_plural": diff_pl[mask].mean().item(),
                "n": int(mask.sum()),
            })

        summarize("overall", stim_df.index.isin(stim_df.index))
        for cond, grp_idx in stim_df.groupby("condition").groups.items():
            summarize(cond, stim_df.index.isin(grp_idx))

    scored_df = pd.concat(scored_rows, ignore_index=True) if scored_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return scored_df, summary_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Minimal-pair minicons evaluation of learned [wug]/[wugs] "
                    "embeddings on attractor stimuli, across CI seed runs."
    )
    parser.add_argument("--model", required=True, help="HF model name, e.g. Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--text_dir", required=True, help="CI seed-run dir for the text/syntax condition")
    parser.add_argument("--image_dir", required=True, help="CI seed-run dir for the image condition")
    parser.add_argument("--stimuli_csv", default=str(REPO_ROOT / attractor_stimuli),
                        help="Path to attractor stimuli CSV (good_singular/bad_singular/"
                             "good_plural/bad_plural columns)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HF cache directory (defaults to the HF default cache).")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    print(f"Loading model: {args.model}")
    lm = scorer.VLMScorer(
        args.model, device=args.device, token=hf_token,
        torch_dtype=torch.bfloat16, cache_dir=args.cache_dir,
    )
    emb, wug_id, wugs_id = setup_novel_tokens(lm)
    print(f"  [wug] ID: {wug_id}  [wugs] ID: {wugs_id}")

    print(f"Loading stimuli: {args.stimuli_csv}")
    stim_df = load_stimuli(args.stimuli_csv)
    queries = {
        col: [chat_template(lm, s, noimage=True, assistant=False) for s in stim_df[col]]
        for col in PAIRED_COLS
    }
    print(f"  {len(stim_df)} minimal pair quadruples")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for condition, results_dir in [("text", args.text_dir), ("image", args.image_dir)]:
        print(f"\n--- Condition: {condition} ({results_dir}) ---")
        scored_df, summary_df = run_condition(
            lm, emb, wug_id, wugs_id, results_dir, stim_df, queries, args.batch_size,
        )
        if scored_df.empty:
            print("  No seeds found, skipping.")
            continue

        scored_path = out_dir / f"attractor_scored_{condition}.csv"
        summary_path = out_dir / f"attractor_summary_{condition}.csv"
        scored_df.to_csv(scored_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        overall = summary_df[summary_df["condition"] == "overall"]
        print(f"  Wrote {scored_path.name} ({len(scored_df)} rows)")
        print(f"  Wrote {summary_path.name} ({overall['seed'].nunique()} seeds, "
              f"mean joint acc={overall['accuracy_joint'].mean():.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main()
