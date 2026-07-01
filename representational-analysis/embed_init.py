#!/usr/bin/env python3
"""
Initialize [wug] and [wugs] embeddings for each seed run directory.

For each seed folder found under --results_dirs, computes a reproducible
initial embedding (mean of noun_init words + scaled random noise) and saves
it as initial_embeddings.pt alongside the learned_embeddings.pt.

Skips seeds that already have initial_embeddings.pt unless --force is set.
"""
import argparse
import os
import random

import numpy as np
import torch
from minicons import scorer
from transformers import set_seed as hf_set_seed


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def load_lines(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def safe_token_id(word, tok):
    ids = tok(" " + word, add_special_tokens=False).input_ids
    return ids[0] if len(ids) == 1 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name, e.g. Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--results_dirs", nargs="+", required=True,
                        help="One or more seed-run directories (each contains seed_N/ subfolders)")
    parser.add_argument("--noun_init", default="data/embeddings/init/noun_init.txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing initial_embeddings.pt")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    print(f"Loading model: {args.model}")
    lm = scorer.VLMScorer(args.model, device=args.device, token=hf_token,
                          torch_dtype=torch.bfloat16)

    added_tokens = [" [wug]", " [wugs]"]
    tok = lm.tokenizer.tokenizer
    existing_vocab = tok.get_vocab()
    tokens_to_add = [t for t in added_tokens if t not in existing_vocab]
    if tokens_to_add:
        tok.add_tokens(tokens_to_add)
        old_len = lm.model.resize_token_embeddings().weight.shape[0]
        lm.model.resize_token_embeddings(old_len + len(tokens_to_add))

    emb = lm.model.model.language_model.embed_tokens
    new_ids = [tok(t, add_special_tokens=False).input_ids[0] for t in added_tokens]
    wug_id, wugs_id = new_ids

    base_emb_matrix = emb.weight.detach().clone()

    embed_init_words = load_lines(args.noun_init)
    init_ids = [safe_token_id(w, tok) for w in embed_init_words]
    init_ids = [t for t in init_ids if t is not None and t < emb.weight.shape[0]]
    print(f"Using {len(init_ids)} init token ids from {args.noun_init}")

    for results_dir in args.results_dirs:
        prefix = "syntax" if "text" in results_dir else ""
        print(f"\nProcessing: {results_dir}  (prefix='{prefix}')")
        for folder in sorted(os.listdir(results_dir)):
            if not folder.startswith("seed_"):
                continue
            seed = int(folder.split("_")[1])
            run_dir = os.path.join(results_dir, folder,
                                   f"{prefix}_alternating_lr0p001_seed{seed}_ep50")
            out_path = os.path.join(run_dir, "initial_embeddings.pt")

            if os.path.exists(out_path) and not args.force:
                print(f"  skip seed {seed} (exists)")
                continue

            set_all_seeds(seed)

            with torch.no_grad():
                emb.weight.data.copy_(
                    base_emb_matrix.to(emb.weight.device, dtype=emb.weight.dtype)
                )
                target_norm = emb.weight.norm(dim=1).float().mean().item()

                pair_distances = []
                for i in range(0, len(init_ids) - 1, 2):
                    sid, pid = init_ids[i], init_ids[i + 1]
                    d = (emb.weight[sid] - emb.weight[pid]).norm().item()
                    pair_distances.append(d)
                noise_scale = float(np.mean(pair_distances)) if pair_distances else 1.0

                init_embs = emb.weight[init_ids].float()
                mean_emb_vec = init_embs.mean(dim=0)

                nw = torch.randn_like(mean_emb_vec)
                nw = nw / nw.norm() * noise_scale
                nws = torch.randn_like(mean_emb_vec)
                nws = nws / nws.norm() * noise_scale

                emb.weight.data[wug_id] = (
                    (mean_emb_vec + nw) / (mean_emb_vec + nw).norm() * target_norm
                ).to(emb.weight.dtype)
                emb.weight.data[wugs_id] = (
                    (mean_emb_vec + nws) / (mean_emb_vec + nws).norm() * target_norm
                ).to(emb.weight.dtype)

                rec = {
                    "epoch": 0,
                    "wug_id": wug_id,
                    "wugs_id": wugs_id,
                    "wug_embedding": emb.weight.data[wug_id].detach().cpu().clone(),
                    "wugs_embedding": emb.weight.data[wugs_id].detach().cpu().clone(),
                    "added_tokens": added_tokens,
                    "vocab_size": emb.weight.shape[0],
                    "model_name": args.model,
                }
                torch.save(rec, out_path)
                print(f"  saved seed {seed} -> {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
