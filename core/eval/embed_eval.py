import os
import argparse
import pandas as pd
import torch

from minicons import scorer
from utils.chat_templates import chat_template


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate learned [wug]/[wugs] embeddings on good/bad stimuli."
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Model name/path (should match the model used in training).")
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to a learned_embeddings.pt file produced by embed_train "
                             "(must contain 'wug_embedding', 'wugs_embedding', 'added_tokens').")
    parser.add_argument("--stimuli_csv", type=str, required=True,
                        help="Path to a stimuli CSV with at least 'good' and 'bad' columns. "
                             "All other columns are passed through unchanged.")
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Output directory for the scored copy of the CSV (default: results/).")
    parser.add_argument("--cache_dir", type=str,
                        default="/mnt/dv/wid/projects3/Rogers-muri-human-ai/zstuddiford",
                        help="HF cache directory.")
    parser.add_argument("--out_name", type=str, default=None,
                        help="Optional output filename. Defaults to "
                             "'<stimuli_basename>_scored.csv'.")
    return parser.parse_args()


def load_stimuli(path):
    """Load stimuli CSV, preserving all original columns. Requires 'good' and 'bad'."""
    df = pd.read_csv(path)
    for col in ("good", "bad"):
        assert col in df.columns, f"Stimuli CSV {path} missing required column '{col}'"
    df["good"] = df["good"].astype(str).str.strip()
    df["bad"] = df["bad"].astype(str).str.strip()
    return df


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_stimuli(args.stimuli_csv)
    n = len(df)
    print("=" * 70)
    print("EVAL CONFIG")
    print(f"  Model:        {args.model}")
    print(f"  Embeddings:   {args.embeddings}")
    print(f"  Stimuli CSV:  {args.stimuli_csv} ({n} pairs)")
    print(f"  Output dir:   {args.out_dir}")
    print("=" * 70)

    device = "cuda"
    lm = scorer.VLMScorer(
        args.model, device=device, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir
    )

    rec = torch.load(args.embeddings, map_location="cpu")
    added_tokens = rec["added_tokens"]            
    wug_embedding = rec["wug_embedding"]
    wugs_embedding = rec["wugs_embedding"]
    saved_model = rec.get("model_name", None)
    if saved_model is not None and saved_model != args.model:
        print(f"  WARNING: embeddings were trained on model '{saved_model}' "
              f"but loading '{args.model}'. Token ids may not align.")
    print(f"  Loaded embeddings (added tokens: {added_tokens}, "
          f"saved_epoch={rec.get('saved_epoch', 'n/a')})")

    existing_vocab = lm.tokenizer.tokenizer.get_vocab()
    tokens_to_add = [t for t in added_tokens if t not in existing_vocab]
    if len(tokens_to_add) > 0:
        lm.tokenizer.tokenizer.add_tokens(tokens_to_add)
        old_len = lm.model.resize_token_embeddings().weight.shape[0]
        lm.model.resize_token_embeddings(old_len + len(tokens_to_add))

    emb = lm.model.model.language_model.embed_tokens
    lm_head = lm.model.lm_head
    tok = lm.tokenizer.tokenizer

    assert emb.weight.data_ptr() == lm_head.weight.data_ptr(), \
        "ERROR: emb and lm_head are NOT tied! This script requires tied weights."
    print("✓ emb.weight and lm_head.weight are tied (same tensor)")

    new_ids = [tok(t, add_special_tokens=False).input_ids[0] for t in added_tokens]
    wug_id, wugs_id = new_ids
    print(f"  [wug] ID: {wug_id}  [wugs] ID: {wugs_id}")

    with torch.no_grad():
        emb.weight.data[wug_id] = wug_embedding.to(emb.weight.device, dtype=emb.weight.dtype)
        emb.weight.data[wugs_id] = wugs_embedding.to(emb.weight.device, dtype=emb.weight.dtype)
    print(f"  Injected embeddings  [wug] norm={emb.weight[wug_id].norm().item():.4f}  "
          f"[wugs] norm={emb.weight[wugs_id].norm().item():.4f}")

    # ----- Build queries (text-only, no assistant turn) like the training eval -----
    good_queries = [chat_template(lm, s, noimage=True, assistant=False) for s in df["good"].tolist()]
    bad_queries = [chat_template(lm, s, noimage=True, assistant=False) for s in df["bad"].tolist()]

    # ----- Score -----
    good_scores = torch.tensor(lm.sequence_score(good_queries), dtype=torch.float32)
    bad_scores = torch.tensor(lm.sequence_score(bad_queries), dtype=torch.float32)
    is_correct = (good_scores > bad_scores)

    out_df = df.copy()
    out_df["score_good"] = good_scores.tolist()
    out_df["score_bad"] = bad_scores.tolist()
    out_df["is_correct"] = is_correct.tolist()

    acc = is_correct.float().mean().item()
    print(f"\nOverall accuracy: {acc:.4f} ({int(is_correct.sum().item())}/{n})")

    out_name = args.out_name or (
        os.path.splitext(os.path.basename(args.stimuli_csv))[0] + "_scored.csv"
    )
    out_path = os.path.join(args.out_dir, out_name)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote scored CSV to {out_path}")


if __name__ == "__main__":
    main()