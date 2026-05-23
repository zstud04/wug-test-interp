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
                        help="Path to a stimuli CSV. In default mode requires 'good' and 'bad' "
                             "columns. With --paired requires 'good_singular', 'bad_singular', "
                             "'good_plural', 'bad_plural'. All other columns are passed through.")
    parser.add_argument("--paired", action="store_true",
                        help="Treat each row as a singular/plural tuple. Reads "
                             "good_singular/bad_singular/good_plural/bad_plural and writes "
                             "per-number scores, diffs, and correctness flags.")
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Output directory for the scored copy of the CSV (default: results/).")
    parser.add_argument("--cache_dir", type=str,
                        default="/mnt/dv/wid/projects3/Rogers-muri-human-ai/zstuddiford",
                        help="HF cache directory.")
    parser.add_argument("--out_name", type=str, default=None,
                        help="Optional output filename. Defaults to "
                             "'<stimuli_basename>_scored.csv'.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for sequence scoring (default: 8).")
    return parser.parse_args()


# Column groups for each mode. Each entry maps a logical name -> CSV column.
SINGLE_COLS = ("good", "bad")
PAIRED_COLS = ("good_singular", "bad_singular", "good_plural", "bad_plural")


def load_stimuli(path, paired):
    """Load stimuli CSV, preserving all original columns.

    Default mode requires 'good'/'bad'. Paired mode requires the four
    good_*/bad_* tuple columns. The active text columns are stripped to str.
    """
    df = pd.read_csv(path)
    required = PAIRED_COLS if paired else SINGLE_COLS
    for col in required:
        assert col in df.columns, f"Stimuli CSV {path} missing required column '{col}'"
    for col in required:
        df[col] = df[col].astype(str).str.strip()
    return df


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


def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_stimuli(args.stimuli_csv, args.paired)
    n = len(df)
    print("=" * 70)
    print("EVAL CONFIG")
    print(f"  Model:        {args.model}")
    print(f"  Embeddings:   {args.embeddings}")
    print(f"  Stimuli CSV:  {args.stimuli_csv} ({n} {'tuples' if args.paired else 'pairs'})")
    print(f"  Mode:         {'paired (sg+pl)' if args.paired else 'single'}")
    print(f"  Output dir:   {args.out_dir}")
    print(f"  Batch size:   {args.batch_size}")
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

    # ----- Sanity check: warn if a stimulus doesn't contain a novel token -----
    active_cols = PAIRED_COLS if args.paired else SINGLE_COLS
    for col in active_cols:
        for i, s in enumerate(df[col].tolist()):
            ids = tok(s, add_special_tokens=False).input_ids
            if wug_id not in ids and wugs_id not in ids:
                print(f"  WARNING: row {i} '{col}' contains neither [wug] nor [wugs]: {s!r}")

    # ----- Build queries (text-only, no assistant turn) like the training eval -----
    def build_queries(col):
        return [chat_template(lm, s, noimage=True, assistant=False) for s in df[col].tolist()]

    out_df = df.copy()

    if not args.paired:
        # ---------------- single mode (original behavior) ----------------
        good_queries = build_queries("good")
        bad_queries = build_queries("bad")

        print("Scoring good queries...")
        good_scores = batched_sequence_score(lm, good_queries, batch_size=args.batch_size)
        print("Scoring bad queries...")
        bad_scores = batched_sequence_score(lm, bad_queries, batch_size=args.batch_size)
        is_correct = (good_scores > bad_scores)

        out_df["score_good"] = good_scores.tolist()
        out_df["score_bad"] = bad_scores.tolist()
        out_df["is_correct"] = is_correct.tolist()

        acc = is_correct.float().mean().item()
        print(f"\nOverall accuracy: {acc:.4f} ({int(is_correct.sum().item())}/{n})")

    else:
        # ---------------- paired mode (sg + pl) ----------------
        print("Scoring good_singular queries...")
        gs = batched_sequence_score(lm, build_queries("good_singular"), batch_size=args.batch_size)
        print("Scoring bad_singular queries...")
        bs = batched_sequence_score(lm, build_queries("bad_singular"), batch_size=args.batch_size)
        print("Scoring good_plural queries...")
        gp = batched_sequence_score(lm, build_queries("good_plural"), batch_size=args.batch_size)
        print("Scoring bad_plural queries...")
        bp = batched_sequence_score(lm, build_queries("bad_plural"), batch_size=args.batch_size)

        diff_sg = gs - bs
        diff_pl = gp - bp
        correct_sg = (gs > bs)
        correct_pl = (gp > bp)
        correct_all = correct_sg & correct_pl

        out_df["score_good_singular"] = gs.tolist()
        out_df["score_bad_singular"] = bs.tolist()
        out_df["score_good_plural"] = gp.tolist()
        out_df["score_bad_plural"] = bp.tolist()
        out_df["score_diff_singular"] = diff_sg.tolist()
        out_df["score_diff_plural"] = diff_pl.tolist()
        out_df["is_correct_singular"] = correct_sg.tolist()
        out_df["is_correct_plural"] = correct_pl.tolist()
        out_df["is_correct_all"] = correct_all.tolist()

        acc_sg = correct_sg.float().mean().item()
        acc_pl = correct_pl.float().mean().item()
        acc_all = correct_all.float().mean().item()
        print(f"\nSingular accuracy: {acc_sg:.4f} ({int(correct_sg.sum().item())}/{n})")
        print(f"Plural accuracy:   {acc_pl:.4f} ({int(correct_pl.sum().item())}/{n})")
        print(f"Joint (both) acc:  {acc_all:.4f} ({int(correct_all.sum().item())}/{n})")

    out_name = args.out_name or (
        os.path.splitext(os.path.basename(args.stimuli_csv))[0] + "_scored.csv"
    )
    out_path = os.path.join(args.out_dir, out_name)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote scored CSV to {out_path}")


if __name__ == "__main__":
    main()