#!/usr/bin/env python3
"""
Wug/Wugs novel-word embedding training for VLMs.

Usage:
    # IMAGE condition
    python train_wug.py --training_condition image \
        --singular_train data/singular_neutral.txt \
        --plural_train data/plural_neutral.txt \
        --goods_file data/goods.txt \
        --bads_file data/bads.txt \
        --lr 0.001 --seed 0 --epochs 9 --batch_mode alternating

    # IMAGE condition with noun-prior embedding init
    python train_wug.py --training_condition image \
        --singular_train data/singular_neutral.txt \
        --plural_train data/plural_neutral.txt \
        --goods_file data/goods.txt \
        --bads_file data/bads.txt \
        --embed_init data/noun_prior_words.txt \
        --lr 0.001 --seed 0 --epochs 9

    # SYNTAX condition (text-only, no images)
    python train_wug.py --training_condition syntax \
        --syntax_set mixed-1 \
        --singular_train data/singular_neutral.txt \
        --plural_train data/plural_neutral.txt \
        --goods_file data/goods.txt \
        --bads_file data/bads.txt \
        --lr 0.001 --seed 0 --epochs 9
"""

import os
import sys
import json
import random
import argparse
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

from minicons import scorer
from transformers import set_seed as hf_set_seed
from utils.chat_templates import (
    chat_template,
    train_chat_template,
    train_chat_template_noimage,
    train_chat_template_filler,
)

# ==========================================================
# PARSE ARGS
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Wug/Wugs embedding training")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--batch_mode", type=str, default="alternating",
                        choices=["alternating", "joint"],
                        help="'alternating': separate singular/plural steps. "
                             "'joint': one step with summed per-example CE.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of singular/plural pairs per optimizer step (default: 1)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct",
                        help="Model name/path")
    parser.add_argument("--cache_dir", type=str,
                        default="/mnt/dv/wid/projects3/Rogers-muri-human-ai/zstuddiford",
                        help="HF cache directory")
    parser.add_argument("--image_dir", type=str, default="../images/snarples",
                        help="Directory containing images (for image condition)")
    parser.add_argument("--out_dir", type=str, default="wug_lr_sweep_results",
                        help="Output directory")
    parser.add_argument("--write_embeddings", action="store_true",
                        help="Save learned [wug]/[wugs] embeddings to disk after training")
    parser.add_argument("--training_condition", type=str, required=True,
                        choices=["image", "syntax"],
                        help="'image': train with images. 'syntax': train with text-only.")
    parser.add_argument("--syntax_set", type=str, default=None,
                        help="Which syntax stimulus set to use. "
                             "Required when --training_condition=syntax.")
    parser.add_argument("--syntax_file", type=str, default="../config/syntax_training_stimuli.json",
                        help="Path to JSON file with syntax training stimuli.")
    parser.add_argument("--filler_image_dir", type=str, default=None,
                        help="Directory of filler images for syntax condition.")

    # --- Training & eval sentence files ---
    parser.add_argument("--singular_train", type=str, required=True,
                        help="Path to txt file with singular training sentences (one per line).")
    parser.add_argument("--plural_train", type=str, required=True,
                        help="Path to txt file with plural training sentences (one per line).")
    parser.add_argument("--goods_file", type=str, required=True,
                        help="Path to txt file with grammatical eval sentences (one per line).")
    parser.add_argument("--bads_file", type=str, required=True,
                        help="Path to txt file with ungrammatical eval sentences (one per line).")

    # --- Embedding init ---
    parser.add_argument("--embed_init", type=str, default=None,
                        help="Path to txt file with noun words (one per line) for embedding init. "
                             "If supplied, [wug]/[wugs] are initialized near the mean of these "
                             "embeddings with noise scaled to the mean singular-plural distance. "
                             "If omitted, uses default random init.")

    return parser.parse_args()


def load_lines(path):
    """Load non-empty lines from a text file."""
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    args = parse_args()

    LR          = args.lr
    GLOBAL_SEED = args.seed
    EPOCHS      = args.epochs
    BATCH_MODE  = args.batch_mode
    BATCH_SIZE  = args.batch_size
    WD          = 0.0
    TRAINING_CONDITION = args.training_condition

    if TRAINING_CONDITION == "syntax":
        if args.syntax_set is None:
            print("ERROR: --syntax_set is required when --training_condition=syntax")
            sys.exit(1)
        SYNTAX_SET = args.syntax_set

    # Norm control
    RENORM_EVERY_STEP = True
    MAX_NORM_MULT     = 1.10

    # Build run name
    lr_str = f"{LR:.10f}".rstrip("0").rstrip(".").replace(".", "p")
    if TRAINING_CONDITION == "image":
        img_tag = os.path.basename(args.image_dir)
        run_name = f"{img_tag}_{BATCH_MODE}_lr{lr_str}_seed{GLOBAL_SEED}_ep{EPOCHS}"
    else:
        run_name = f"syntax_{SYNTAX_SET}_{BATCH_MODE}_lr{lr_str}_seed{GLOBAL_SEED}_ep{EPOCHS}"

    OUT_ROOT = args.out_dir
    os.makedirs(OUT_ROOT, exist_ok=True)
    run_dir = os.path.join(OUT_ROOT, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # --- Load training sentences ---
    singular_sentences = load_lines(args.singular_train)
    plural_sentences = load_lines(args.plural_train)
    assert len(singular_sentences) > 0, f"No sentences in {args.singular_train}"
    assert len(plural_sentences) > 0, f"No sentences in {args.plural_train}"

    # --- Load eval sentences ---
    goods = load_lines(args.goods_file)
    bads = load_lines(args.bads_file)
    assert len(goods) == len(bads), f"Mismatch: {len(goods)} goods vs {len(bads)} bads"

    # --- Load embed_init words (if supplied) ---
    embed_init_words = None
    if args.embed_init:
        embed_init_words = load_lines(args.embed_init)
        assert len(embed_init_words) >= 4, f"Need at least 4 words in {args.embed_init}, got {len(embed_init_words)}"

    print("=" * 70)
    print("RUN CONFIG")
    print(f"  Condition:      {TRAINING_CONDITION}")
    if TRAINING_CONDITION == "syntax":
        print(f"  Syntax set:     {SYNTAX_SET}")
        print(f"  Syntax file:    {args.syntax_file}")
        if args.filler_image_dir:
            print(f"  Filler imgs:    {args.filler_image_dir}")
    print(f"  Batch mode:     {BATCH_MODE}")
    print(f"  Batch size:     {BATCH_SIZE}")
    print(f"  LR:             {LR}")
    print(f"  Seed:           {GLOBAL_SEED}")
    print(f"  Epochs:         {EPOCHS}")
    print(f"  Model:          {args.model}")
    if TRAINING_CONDITION == "image":
        print(f"  Image dir:      {args.image_dir}")
    print(f"  Singular train: {args.singular_train} ({len(singular_sentences)} sentences)")
    print(f"  Plural train:   {args.plural_train} ({len(plural_sentences)} sentences)")
    print(f"  Goods file:     {args.goods_file} ({len(goods)} sentences)")
    print(f"  Bads file:      {args.bads_file} ({len(bads)} sentences)")
    print(f"  Embed init:     {args.embed_init or 'default (random)'}")
    print(f"  Output:         {run_dir}")
    print(f"Total eval pairs: {len(goods)}")
    print("=" * 70)

    # ==========================================================
    # MODEL
    # ==========================================================
    device = "cuda"
    lm = scorer.VLMScorer(
        args.model, device=device, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir
    )

    def load_resized(path):
        return Image.open(path).convert("RGB").resize((224, 224), Image.LANCZOS)

    # ==========================================================
    # TOKENIZER + EMBEDDING SETUP
    # ==========================================================
    added_tokens = [" [wug]", " [wugs]"]
    existing_vocab = lm.tokenizer.tokenizer.get_vocab()
    tokens_to_add = [t for t in added_tokens if t not in existing_vocab]
    if len(tokens_to_add) > 0:
        lm.tokenizer.tokenizer.add_tokens(tokens_to_add)
        old_len = lm.model.resize_token_embeddings().weight.shape[0]
        lm.model.resize_token_embeddings(old_len + len(tokens_to_add))

    emb     = lm.model.language_model.embed_tokens
    lm_head = lm.model.lm_head
    tok     = lm.tokenizer.tokenizer

    new_ids = [tok(t, add_special_tokens=False).input_ids[0] for t in added_tokens]
    wug_id, wugs_id = new_ids

    assert emb.weight.data_ptr() == lm_head.weight.data_ptr(), \
        "ERROR: emb and lm_head are NOT tied! This script requires tied weights."
    print("✓ emb.weight and lm_head.weight are tied (same tensor)")

    base_emb_matrix = emb.weight.detach().clone()
    device = lm.model.device

    # ==========================================================
    # VISION ENCODER t-SNE (image condition only)
    # ==========================================================
    if TRAINING_CONDITION == "image":
        from sklearn.manifold import TSNE
        from sklearn.metrics.pairwise import cosine_similarity as csm

        def get_vision_embeddings(lm_model, images):
            embeddings = []
            lm_model.model.eval()
            with torch.no_grad():
                for img in images:
                    enc = lm_model.tokenizer(images=[img], text=[""], return_tensors="pt", padding=True)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    pv = enc.get("pixel_values", None)
                    gt = enc.get("image_grid_thw", None)
                    if pv is None:
                        raise RuntimeError("Tokenizer did not produce pixel_values.")
                    vit_out = lm_model.model.visual(pv.to(dtype=torch.bfloat16), grid_thw=gt)
                    if isinstance(vit_out, tuple):
                        vit_out = vit_out[0]
                    embeddings.append(vit_out.float().mean(dim=0).cpu().numpy())
            return np.array(embeddings, dtype=np.float32)

        def plot_vision_tsne(singular_imgs, plural_imgs, out_path):
            print("\n" + "=" * 60)
            print("VISION ENCODER t-SNE (pre-training)")
            print("=" * 60)
            sing_embs = get_vision_embeddings(lm, singular_imgs)
            plur_embs = get_vision_embeddings(lm, plural_imgs)
            all_embs = np.concatenate([sing_embs, plur_embs], axis=0)
            labels = ["singular"] * len(singular_imgs) + ["plural"] * len(plural_imgs)
            perplexity = min(5, len(all_embs) - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=GLOBAL_SEED,
                        max_iter=1000, init="pca")
            coords = tsne.fit_transform(all_embs)
            fig, ax = plt.subplots(figsize=(7, 6))
            colors = {"singular": "#2196F3", "plural": "#F44336"}
            mkrs = {"singular": "o", "plural": "s"}
            for label in ["singular", "plural"]:
                mask = [i for i, l in enumerate(labels) if l == label]
                ax.scatter(coords[mask, 0], coords[mask, 1], c=colors[label], marker=mkrs[label],
                           s=120, alpha=0.85, edgecolors="white", linewidths=0.8, label=label, zorder=3)
                for j, idx in enumerate(mask):
                    ax.annotate(str(j + 1), (coords[idx, 0], coords[idx, 1]), fontsize=8,
                                ha="center", va="center", color="white", fontweight="bold")
            ax.set_title("t-SNE of Vision Encoder Embeddings\n(mean-pooled patches, pre-training)", fontsize=12)
            ax.set_xlabel("t-SNE dim 1")
            ax.set_ylabel("t-SNE dim 2")
            ax.legend(title="Image type", fontsize=10)
            ax.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved t-SNE plot to: {out_path}")
            S = csm(sing_embs); P = csm(plur_embs); C = csm(sing_embs, plur_embs)
            ns, np_ = len(singular_imgs), len(plural_imgs)
            ws = (S.sum() - np.trace(S)) / (ns * (ns - 1)) if ns > 1 else float("nan")
            wp = (P.sum() - np.trace(P)) / (np_ * (np_ - 1)) if np_ > 1 else float("nan")
            print(f"  Within singular: {ws:.4f}  Within plural: {wp:.4f}  Across: {C.mean():.4f}")

        img_dir = args.image_dir
        singular_imgs = [load_resized(os.path.join(img_dir, f"snarple_singular_{i:01d}.png")) for i in range(1, 6)]
        plural_imgs   = [load_resized(os.path.join(img_dir, f"snarple_plural_{i:01d}.png")) for i in range(1, 6)]
        plot_vision_tsne(singular_imgs, plural_imgs, os.path.join(run_dir, "vision_encoder_tsne_pretrain.png"))

    # ==========================================================
    # DATA
    # ==========================================================
    if TRAINING_CONDITION == "image":
        singular_imgs = [load_resized(os.path.join(img_dir, f"snarple_singular_{i:01d}.png")) for i in range(1, 6)]
        plural_imgs   = [load_resized(os.path.join(img_dir, f"snarple_plural_{i:01d}.png")) for i in range(1, 6)]
        singular_templates = [train_chat_template(lm, s) for s in singular_sentences]
        plural_templates = [train_chat_template(lm, s) for s in plural_sentences]
    else:
        with open(args.syntax_file, "r") as f:
            syntax_data = json.load(f)
        if SYNTAX_SET not in syntax_data:
            print(f"ERROR: syntax_set '{SYNTAX_SET}' not found in {args.syntax_file}")
            print(f"  Available sets: {list(syntax_data.keys())}")
            sys.exit(1)
        stim = syntax_data[SYNTAX_SET]
        assert len(stim["singular"]) == 15, f"Expected 15 singular, got {len(stim['singular'])}"
        assert len(stim["plural"]) == 15, f"Expected 15 plural, got {len(stim['plural'])}"

        if args.filler_image_dir:
            filler_files = sorted([
                f for f in os.listdir(args.filler_image_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
            ])
            if len(filler_files) == 0:
                print(f"ERROR: no images found in {args.filler_image_dir}")
                sys.exit(1)
            filler_imgs = [load_resized(os.path.join(args.filler_image_dir, f)) for f in filler_files]
            print(f"Loaded {len(filler_imgs)} filler images from {args.filler_image_dir}")
            singular_templates = [train_chat_template_filler(lm, s) for s in stim["singular"]]
            plural_templates   = [train_chat_template_filler(lm, s) for s in stim["plural"]]
            singular_imgs = filler_imgs
            plural_imgs = filler_imgs
        else:
            singular_templates = [train_chat_template_noimage(lm, s) for s in stim["singular"]]
            plural_templates   = [train_chat_template_noimage(lm, s) for s in stim["plural"]]
            singular_imgs = None
            plural_imgs = None
        print(f"Loaded syntax set '{SYNTAX_SET}': {len(singular_templates)} singular, {len(plural_templates)} plural")

    good_queries = [chat_template(lm, s, noimage=True, assistant=False) for s in goods]
    bad_queries  = [chat_template(lm, s, noimage=True, assistant=False) for s in bads]
    singular_eval_idx = [i for i, s in enumerate(goods) if " [wug]" in s]
    plural_eval_idx   = [i for i, s in enumerate(goods) if " [wugs]" in s]

    # ==========================================================
    # UTILS
    # ==========================================================
    def set_all_seeds(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        hf_set_seed(seed)

    def mask_to_novel_token(input_ids):
        """Only supervise [wug]/[wugs] token positions."""
        labels = torch.full_like(input_ids, -100)
        for tid in [wug_id, wugs_id]:
            for pos in (input_ids == tid).nonzero(as_tuple=True)[0]:
                labels[pos] = tid
        return labels

    def make_batch(imgs, texts):
        if imgs is not None:
            enc = lm.tokenizer(images=imgs, text=texts, return_tensors="pt", padding=True)
        else:
            enc = lm.tokenizer(text=texts, return_tensors="pt", padding=True)
        labels = torch.stack([mask_to_novel_token(enc["input_ids"][i]) for i in range(len(texts))])
        enc = {k: v.to(device) for k, v in enc.items()}
        return enc, labels.to(device)

    def run_agreement_eval():
        gs = torch.tensor(lm.sequence_score(good_queries), dtype=torch.float32)
        bs = torch.tensor(lm.sequence_score(bad_queries), dtype=torch.float32)
        diffs = gs - bs
        acc = (diffs > 0).float()
        return {
            "overall_acc": acc.mean().item(),
            "sing_acc": acc[singular_eval_idx].mean().item(),
            "plur_acc": acc[plural_eval_idx].mean().item(),
            "diffs": diffs.cpu(),
            "good_scores": gs.cpu(),
            "bad_scores": bs.cpu(),
            "acc_mask": acc.cpu(),
        }

    reference_words = ["dog", "dogs", "cat", "cats", "bird", "birds", "bear", "bears", "rat", "rats"]
    reference_ids = [tok(f" {w}", add_special_tokens=False).input_ids[0] for w in reference_words]
    singular_plural_pairs = [
        (" cat", " cats"), (" dog", " dogs"), (" bird", " birds"), (" bear", " bears"),
        (" rat", " rats"), (" thing", " things"), (" word", " words"), (" tree", " trees"),
    ]

    def normalize_special_rows(tn):
        if TRAINING_CONDITION == "syntax":
            return
        with torch.no_grad():
            if RENORM_EVERY_STEP:
                for tid in [wug_id, wugs_id]:
                    v = emb.weight.data[tid]
                    emb.weight.data[tid] = (v / v.norm() * tn).to(emb.weight.dtype)
            else:
                mx = tn * MAX_NORM_MULT
                for tid in [wug_id, wugs_id]:
                    n = emb.weight.data[tid].norm()
                    if n > mx:
                        emb.weight.data[tid] = (emb.weight.data[tid] / n * mx).to(emb.weight.dtype)

    def get_neighbors(tid, topk=6):
        with torch.no_grad():
            ne = torch.nn.functional.normalize(emb.weight, dim=1)
            idxs = torch.topk(ne @ ne[tid], topk).indices[1:]
            return [tok.decode([i]) for i in idxs]

    def compute_sp_axis_and_traj(wt, wst):
        with torch.no_grad():
            re = np.array([emb.weight[r].detach().cpu().float().numpy() for r in reference_ids])
            pd_ = [re[i + 1] - re[i] for i in range(0, len(re), 2)]
            sp = np.mean(pd_, axis=0)
            sp = sp / np.linalg.norm(sp)
            wn = np.array([w.detach().cpu().float().numpy() for w in wt])
            wsn = np.array([w.detach().cpu().float().numpy() for w in wst])
            return sp, wn @ sp, wsn @ sp

    # ==========================================================
    # Helper: run one optimizer step
    # ==========================================================
    def do_step(loss, optimizer, target_norm):
        loss.backward()
        with torch.no_grad():
            gw = emb.weight.grad[wug_id].clone()
            gws = emb.weight.grad[wugs_id].clone()
            emb.weight.grad.zero_()
            emb.weight.grad[wug_id] = gw
            emb.weight.grad[wugs_id] = gws
            gnw = gw.norm().item()
            gnws = gws.norm().item()
            bw = emb.weight.data[wug_id].clone()
            bws = emb.weight.data[wugs_id].clone()
        optimizer.step()
        normalize_special_rows(target_norm)
        with torch.no_grad():
            sw  = (emb.weight.data[wug_id] - bw).norm().item()
            sws = (emb.weight.data[wugs_id] - bws).norm().item()
            nw  = emb.weight.data[wug_id].norm().item()
            nws = emb.weight.data[wugs_id].norm().item()
            sep = (emb.weight[wug_id] - emb.weight[wugs_id]).norm().item()
            cos = torch.nn.functional.cosine_similarity(
                emb.weight[wug_id].unsqueeze(0), emb.weight[wugs_id].unsqueeze(0)).item()
        return {"gnw": gnw, "gnws": gnws, "sw": sw, "sws": sws,
                "nw": nw, "nws": nws, "sep": sep, "cos": cos}

    def log_logit_gaps(out, labels, phase, epoch, global_step):
        rows = []
        for b in range(labels.shape[0]):
            active = labels[b] != -100
            if not active.any():
                continue
            for pos in active.nonzero(as_tuple=True)[0]:
                if pos == 0:
                    continue
                tgt = labels[b, pos]
                if tgt.item() not in [wug_id, wugs_id]:
                    continue
                with torch.no_grad():
                    lgt = out.logits[b, pos - 1]
                    lt = torch.nn.functional.cross_entropy(lgt.unsqueeze(0), tgt.unsqueeze(0)).item()
                    gap = float(lgt[wug_id].item() - lgt[wugs_id].item())
                    logit_gap_history[phase].append(gap)
                    row = {"epoch": epoch + 1, "global_step": global_step + 1, "phase": phase,
                           "token_ce": lt, "logit_wug": float(lgt[wug_id].item()),
                           "logit_wugs": float(lgt[wugs_id].item()), "logit_gap": gap}
                    rows.append(row)
                    print(f"    [{phase:8s}] {tok.decode([tgt.item()])!r}:{lt:.2f} (gap={gap:.2f})")
        return rows

    # ==========================================================
    # INITIALIZE EMBEDDINGS
    # ==========================================================
    set_all_seeds(GLOBAL_SEED)

    with torch.no_grad():
        emb.weight.data.copy_(base_emb_matrix.to(emb.weight.device, dtype=emb.weight.dtype))
        target_norm = emb.weight.norm(dim=1).float().mean().item()

        if embed_init_words is not None:
            # Use supplied noun words for init
            def _safe_token_id(w):
                ids = tok(" " + w, add_special_tokens=False).input_ids
                return ids[0] if len(ids) == 1 else None

            init_ids = [_safe_token_id(w) for w in embed_init_words]
            init_ids = [t for t in init_ids if t is not None and t < emb.weight.shape[0]]
            assert len(init_ids) >= 4, f"Only {len(init_ids)} valid token ids from embed_init file"

            # Find singular-plural pairs among init words for noise scaling
            # Try to pair consecutive words as singular/plural
            pair_distances = []
            init_pair_rows = []
            for i in range(0, len(init_ids) - 1, 2):
                sid, pid = init_ids[i], init_ids[i + 1]
                d = (emb.weight[sid] - emb.weight[pid]).norm().item()
                pair_distances.append(d)
                init_pair_rows.append({
                    "token_a": tok.decode([sid]).strip(),
                    "token_b": tok.decode([pid]).strip(),
                    "distance": d,
                })
                print(f"{tok.decode([sid]).strip():>12} -> {tok.decode([pid]).strip():<12} dist={d:.4f}")

            noise_scale = float(np.mean(pair_distances)) if pair_distances else 1.0
            print(f"Mean pair distance (noise scale): {noise_scale:.4f}")

            # Init around mean of these embeddings
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
        else:
            # Default random init (original behavior)
            pair_distances = []
            init_pair_rows = []
            for s, p in singular_plural_pairs:
                sid = tok(s, add_special_tokens=False).input_ids[0]
                pid = tok(p, add_special_tokens=False).input_ids[0]
                d = (emb.weight[sid] - emb.weight[pid]).norm().item()
                pair_distances.append(d)
                init_pair_rows.append({"token_a": s.strip(), "token_b": p.strip(), "distance": d})
                print(f"{s.strip():>8} -> {p.strip():<8} dist={d:.4f}")

            noise_scale = float(np.mean(pair_distances))
            print(f"Mean S→P distance (noise scale): {noise_scale:.4f}")

            mean_emb_vec = emb.weight.mean(dim=0)
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

        print(f"[wug] norm: {emb.weight[wug_id].norm().item():.4f}  "
              f"[wugs] norm: {emb.weight[wugs_id].norm().item():.4f}")
        print(f"Init separation: {(emb.weight[wug_id] - emb.weight[wugs_id]).norm().item():.4f}")

    pd.DataFrame(init_pair_rows).to_csv(os.path.join(run_dir, "init_pair_distances.csv"), index=False)

    # Label sanity check
    rng_chk = np.random.default_rng(0)
    if TRAINING_CONDITION == "image":
        _cs_imgs = [singular_imgs[i] for i in rng_chk.integers(0, len(singular_imgs), size=len(singular_templates))]
        _cp_imgs = [plural_imgs[i] for i in rng_chk.integers(0, len(plural_imgs), size=len(plural_templates))]
        _cs = make_batch(_cs_imgs, singular_templates)
        _cp = make_batch(_cp_imgs, plural_templates)
    else:
        _cs = make_batch(None, singular_templates)
        _cp = make_batch(None, plural_templates)
    print(f"Label check singular [0]: {tok.decode(_cs[1][0][_cs[1][0] != -100])}")
    print(f"Label check plural   [0]: {tok.decode(_cp[1][0][_cp[1][0] != -100])}")
    del _cs, _cp
    print(f"[wug] ID: {wug_id}  [wugs] ID: {wugs_id}")

    # Freeze all params, unfreeze embedding
    for p in lm.model.parameters():
        p.requires_grad = False
    emb.weight.requires_grad = True

    # ==========================================================
    # TRAINING
    # ==========================================================
    optimizer = torch.optim.Adam(
        [{"params": [emb.weight], "lr": LR}],
        betas=(0.0, 0.9), eps=1e-8, weight_decay=WD,
    )

    loss_history = []
    sing_loss_history = []
    plur_loss_history = []
    grad_norm_hist = {"wug": [], "wugs": []}
    wug_trajectory = []
    wugs_trajectory = []
    logit_gap_history = {"singular": [], "plural": []}
    eval_overall_history = []
    eval_sing_history = []
    eval_plur_history = []
    eval_diff_history = []
    step_rows = []
    epoch_rows = []
    eval_item_rows = []
    traj_rows = []
    logit_gap_rows = []
    rng = np.random.default_rng(GLOBAL_SEED)

    print(f"\n{'═' * 60}\nTRAINING (mode={BATCH_MODE}, tied weights)\n{'═' * 60}")
    global_step = 0

    for epoch in range(EPOCHS):
        si = rng.permutation(len(singular_templates))
        pi = rng.permutation(len(plural_templates))
        s_texts = [singular_templates[i] for i in si]
        p_texts = [plural_templates[i] for i in pi]

        if TRAINING_CONDITION == "image":
            sii = rng.integers(0, len(singular_imgs), size=len(singular_templates))
            pii = rng.integers(0, len(plural_imgs), size=len(plural_templates))
            s_imgs_ep = [singular_imgs[i] for i in sii]
            p_imgs_ep = [plural_imgs[i] for i in pii]
        else:
            s_imgs_ep = None
            p_imgs_ep = None

        s_order = rng.permutation(len(s_texts))
        p_order = rng.permutation(len(p_texts))
        if TRAINING_CONDITION == "image":
            s_examples = [(s_imgs_ep[i], s_texts[i]) for i in s_order]
            p_examples = [(p_imgs_ep[i], p_texts[i]) for i in p_order]
        else:
            s_examples = [(None, s_texts[i]) for i in s_order]
            p_examples = [(None, p_texts[i]) for i in p_order]

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        epoch_loss = 0.0
        steps = 0
        epoch_gn = {"wug": [], "wugs": []}

        num_pairs = len(s_examples)
        for batch_start in range(0, num_pairs, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, num_pairs)
            batch_s = s_examples[batch_start:batch_end]
            batch_p = p_examples[batch_start:batch_end]
            s_imgs_batch = [x[0] for x in batch_s] if batch_s[0][0] is not None else None
            p_imgs_batch = [x[0] for x in batch_p] if batch_p[0][0] is not None else None
            s_txts_batch = [x[1] for x in batch_s]
            p_txts_batch = [x[1] for x in batch_p]
            pair_idx = batch_start // BATCH_SIZE

            if BATCH_MODE == "joint":
                all_imgs = (s_imgs_batch + p_imgs_batch) if s_imgs_batch is not None else None
                all_txts = s_txts_batch + p_txts_batch
                enc, labels = make_batch(all_imgs, all_txts)
                optimizer.zero_grad()
                out = lm.model(**enc, labels=labels)
                logits = out.logits
                bs = len(all_txts)
                ce = torch.tensor(0.0, device=device)
                for b in range(bs):
                    ce_b = torch.nn.functional.cross_entropy(
                        logits[b][:-1].reshape(-1, logits.shape[-1]),
                        labels[b][1:].reshape(-1), ignore_index=-100)
                    ce = ce + ce_b
                ce = ce / bs
                loss = ce

                diag = do_step(loss, optimizer, target_norm)
                epoch_gn["wug"].append(diag["gnw"])
                epoch_gn["wugs"].append(diag["gnws"])

                n_s = len(s_txts_batch)
                lg_s = log_logit_gaps(out, labels[:n_s], "singular", epoch, global_step)
                lg_p = log_logit_gaps(out, labels[n_s:], "plural", epoch, global_step)
                logit_gap_rows.extend(lg_s)
                logit_gap_rows.extend(lg_p)

                print(f"    ce={ce.item():.4f} "
                      f"grad [wug]={diag['gnw']:.4f} [wugs]={diag['gnws']:.4f} "
                      f"step [wug]={diag['sw']:.4f} [wugs]={diag['sws']:.4f}")

                global_step += 1
                row = {"epoch": epoch + 1, "pair_index": pair_idx, "global_step": global_step,
                       "phase": "joint", "ce_avg": float(ce.item()),
                       "total_loss": float(loss.item()),
                       "grad_norm_wug": diag["gnw"], "grad_norm_wugs": diag["gnws"],
                       "step_wug": diag["sw"], "step_wugs": diag["sws"],
                       "norm_wug": diag["nw"], "norm_wugs": diag["nws"],
                       "separation_l2": diag["sep"], "cosine": diag["cos"]}
                step_rows.append(row)
                epoch_loss += ce.item()
                steps += 1

            else:
                for phase, imgs, txts in [("singular", s_imgs_batch, s_txts_batch),
                                           ("plural", p_imgs_batch, p_txts_batch)]:
                    enc, labels = make_batch(imgs, txts)
                    optimizer.zero_grad()
                    out = lm.model(**enc, labels=labels)
                    ce = out.loss
                    loss = ce

                    diag = do_step(loss, optimizer, target_norm)
                    epoch_gn["wug"].append(diag["gnw"])
                    epoch_gn["wugs"].append(diag["gnws"])

                    lg = log_logit_gaps(out, labels, phase, epoch, global_step)
                    logit_gap_rows.extend(lg)

                    print(f"    ce_{phase[0]}={ce.item():.4f} "
                          f"grad [wug]={diag['gnw']:.4f} [wugs]={diag['gnws']:.4f} "
                          f"step [wug]={diag['sw']:.4f} [wugs]={diag['sws']:.4f}")

                    global_step += 1
                    row = {"epoch": epoch + 1, "pair_index": pair_idx, "global_step": global_step,
                           "phase": phase, "ce_loss": float(ce.item()),
                           "total_loss": float(loss.item()),
                           "grad_norm_wug": diag["gnw"], "grad_norm_wugs": diag["gnws"],
                           "step_wug": diag["sw"], "step_wugs": diag["sws"],
                           "norm_wug": diag["nw"], "norm_wugs": diag["nws"],
                           "separation_l2": diag["sep"], "cosine": diag["cos"]}
                    step_rows.append(row)
                    epoch_loss += ce.item()
                    steps += 1

        # End-of-epoch
        avg_loss = epoch_loss / steps
        loss_history.append(avg_loss)

        if TRAINING_CONDITION == "image":
            sb = make_batch(s_imgs_ep, s_texts)
            pb = make_batch(p_imgs_ep, p_texts)
        else:
            sb = make_batch(None, s_texts)
            pb = make_batch(None, p_texts)
        with torch.no_grad():
            sl = lm.model(**sb[0], labels=sb[1]).loss.item()
            pl = lm.model(**pb[0], labels=pb[1]).loss.item()
            sing_loss_history.append(sl)
            plur_loss_history.append(pl)
            sep_val = (emb.weight[wug_id] - emb.weight[wugs_id]).norm().item()
            nwug = emb.weight[wug_id].norm().item()
            nwugs = emb.weight[wugs_id].norm().item()
            csim = torch.nn.functional.cosine_similarity(
                emb.weight[wug_id].unsqueeze(0), emb.weight[wugs_id].unsqueeze(0)).item()
            nbrs_w = get_neighbors(wug_id)
            nbrs_ws = get_neighbors(wugs_id)
            wug_trajectory.append(emb.weight.data[wug_id].detach().cpu().clone())
            wugs_trajectory.append(emb.weight.data[wugs_id].detach().cpu().clone())
            ga = float(np.mean(epoch_gn["wug"])) if epoch_gn["wug"] else 0.0
            gas = float(np.mean(epoch_gn["wugs"])) if epoch_gn["wugs"] else 0.0
            grad_norm_hist["wug"].append(ga)
            grad_norm_hist["wugs"].append(gas)

        ev = run_agreement_eval()
        eval_overall_history.append(ev["overall_acc"])
        eval_sing_history.append(ev["sing_acc"])
        eval_plur_history.append(ev["plur_acc"])
        eval_diff_history.append(ev["diffs"])

        for i in range(len(goods)):
            eval_item_rows.append({
                "epoch": epoch + 1, "item_index": i,
                "kind": "singular" if i in singular_eval_idx else "plural",
                "good_text": goods[i], "bad_text": bads[i],
                "good_score": float(ev["good_scores"][i].item()),
                "bad_score": float(ev["bad_scores"][i].item()),
                "diff": float(ev["diffs"][i].item()),
                "correct": int(ev["acc_mask"][i].item()),
            })

        print(f"\n  CE avg={avg_loss:.4f}  [s={sl:.4f} p={pl:.4f}]")
        print(f"  Grad [wug]={ga:.4e} [wugs]={gas:.4e}")
        print(f"  Norm [wug]={nwug:.4f} [wugs]={nwugs:.4f} (target={target_norm:.4f})")
        print(f"  Sep={sep_val:.4f}  Cos={csim:.4f}")
        print(f"  [wug]  nbrs: {nbrs_w}")
        print(f"  [wugs] nbrs: {nbrs_ws}")
        print(f"  Eval: overall={ev['overall_acc']:.3f} sing={ev['sing_acc']:.3f} plur={ev['plur_acc']:.3f}")

        er = {"epoch": epoch + 1, "avg_ce": avg_loss, "sing_ce": sl, "plur_ce": pl,
              "grad_wug": ga, "grad_wugs": gas, "norm_wug": nwug, "norm_wugs": nwugs,
              "target_norm": target_norm, "sep_l2": sep_val, "cos": csim,
              "nbrs_wug": " | ".join(nbrs_w), "nbrs_wugs": " | ".join(nbrs_ws),
              "eval_overall": ev["overall_acc"], "eval_sing": ev["sing_acc"], "eval_plur": ev["plur_acc"]}
        epoch_rows.append(er)

    print("\nTraining complete.")
    if args.write_embeddings:
        emb_save = {
            "wug_id": wug_id,
            "wugs_id": wugs_id,
            "wug_embedding": emb.weight.data[wug_id].detach().cpu(),
            "wugs_embedding": emb.weight.data[wugs_id].detach().cpu(),
            "added_tokens": added_tokens,
            "vocab_size": emb.weight.shape[0],
            "model_name": args.model,
        }
        emb_path = os.path.join(run_dir, "learned_embeddings.pt")
        torch.save(emb_save, emb_path)
        print(f"Saved embeddings to {emb_path}")

    # ==========================================================
    # SAVE
    # ==========================================================
    sp_dir, wug_proj, wugs_proj = compute_sp_axis_and_traj(wug_trajectory, wugs_trajectory)
    for ep in range(EPOCHS):
        traj_rows.append({
            "epoch": ep + 1,
            "wug_proj": float(wug_proj[ep]),
            "wugs_proj": float(wugs_proj[ep]),
            "dist": float(abs(wug_proj[ep] - wugs_proj[ep])),
        })

    pd.DataFrame(step_rows).to_csv(os.path.join(run_dir, "step_stats.csv"), index=False)
    pd.DataFrame(epoch_rows).to_csv(os.path.join(run_dir, "epoch_stats.csv"), index=False)
    pd.DataFrame(eval_item_rows).to_csv(os.path.join(run_dir, "eval_item_scores.csv"), index=False)
    pd.DataFrame(traj_rows).to_csv(os.path.join(run_dir, "trajectory_stats.csv"), index=False)
    pd.DataFrame(logit_gap_rows).to_csv(os.path.join(run_dir, "logit_gap_stats.csv"), index=False)
    pd.DataFrame([{
        "training_condition": TRAINING_CONDITION,
        "syntax_set": args.syntax_set,
        "batch_mode": BATCH_MODE, "batch_size": BATCH_SIZE,
        "lr": LR, "seed": GLOBAL_SEED, "epochs": EPOCHS, "model": args.model,
        "image_dir": args.image_dir if TRAINING_CONDITION == "image" else None,
        "embed_init": args.embed_init,
        "renorm": RENORM_EVERY_STEP, "betas": "(0.0, 0.9)", "wd": WD,
        "singular_train": args.singular_train, "plural_train": args.plural_train,
        "goods_file": args.goods_file, "bads_file": args.bads_file,
    }]).to_csv(os.path.join(run_dir, "run_config.csv"), index=False)

    # ==========================================================
    # PLOTS
    # ==========================================================
    fig, axes = plt.subplots(4, 2, figsize=(14, 22))
    ex = range(1, EPOCHS + 1)

    axes[0, 0].plot(ex, loss_history, lw=2, marker="o", label="avg CE")
    axes[0, 0].plot(ex, sing_loss_history, lw=1, marker=".", alpha=.7, label="sing CE")
    axes[0, 0].plot(ex, plur_loss_history, lw=1, marker=".", alpha=.7, label="plur CE")
    axes[0, 0].set_title(f"Loss (LR={LR}, {BATCH_MODE})")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=.3)

    axes[0, 1].plot(ex, grad_norm_hist["wug"], label="[wug]", marker="^", alpha=.7)
    axes[0, 1].plot(ex, grad_norm_hist["wugs"], label="[wugs]", marker="v", alpha=.7)
    axes[0, 1].set_title("Gradient Norms")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=.3)

    er_ = np.arange(1, EPOCHS + 1)
    ax = axes[1, 0]
    ax.scatter(wug_proj, er_, s=60, alpha=.8, label="[wug]", zorder=3)
    ax.scatter(wugs_proj, er_, s=60, alpha=.8, label="[wugs]", marker="s", zorder=3)
    ax.plot(wug_proj, er_, alpha=.4)
    ax.plot(wugs_proj, er_, alpha=.4)
    ax.scatter(wug_proj[0], 1, marker="x", s=200, lw=3, label="start", zorder=5)
    ax.scatter(wug_proj[-1], EPOCHS, marker="*", s=300, label="end", zorder=5)
    ax.set_xlabel("S→P projection")
    ax.set_ylabel("Epoch")
    ax.set_title("S/P Trajectory")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=.3)

    axes[1, 1].plot(er_, np.abs(wug_proj - wugs_proj), marker="o", lw=2)
    axes[1, 1].set_title("S/P Distance")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=.3)

    if logit_gap_history["singular"] and logit_gap_history["plural"]:
        axes[2, 0].plot(range(1, len(logit_gap_history["singular"]) + 1),
                        logit_gap_history["singular"], marker="o", label="sing (+)")
        axes[2, 0].plot(range(1, len(logit_gap_history["plural"]) + 1),
                        logit_gap_history["plural"], marker="s", label="plur (−)")
        axes[2, 0].axhline(0, color="k", ls="--", alpha=.5)
        axes[2, 0].set_title("Logit Gap")
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=.3)

    axes[2, 1].plot(ex, eval_overall_history, marker="o", lw=2, label="overall")
    axes[2, 1].plot(ex, eval_sing_history, marker="^", lw=1.5, label="sing")
    axes[2, 1].plot(ex, eval_plur_history, marker="s", lw=1.5, label="plur")
    axes[2, 1].set_ylim(0, 1)
    axes[2, 1].set_title("Eval Accuracy")
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=.3)

    axes[3, 0].hist(eval_diff_history[-1].numpy(), bins=20)
    axes[3, 0].axvline(0, color="k", ls="--", alpha=.5)
    axes[3, 0].set_title("Final Score Diffs")
    axes[3, 0].grid(True, alpha=.3)

    axes[3, 1].bar(["Overall", "Sing", "Plur"],
                   [eval_overall_history[-1], eval_sing_history[-1], eval_plur_history[-1]])
    axes[3, 1].set_ylim(0, 1)
    axes[3, 1].set_title("Final Accuracy")
    axes[3, 1].grid(axis="y", alpha=.3)

    plt.tight_layout()
    pp = os.path.join(run_dir, "wug_training_results.png")
    plt.savefig(pp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {pp}")

    sm = {
        "run_name": run_name,
        "training_condition": TRAINING_CONDITION,
        "syntax_set": args.syntax_set,
        "batch_mode": BATCH_MODE, "batch_size": BATCH_SIZE,
        "lr": LR, "seed": GLOBAL_SEED, "epochs": EPOCHS,
        "image_dir": args.image_dir if TRAINING_CONDITION == "image" else None,
        "embed_init": args.embed_init,
        "final_ce": loss_history[-1],
        "final_sing_ce": sing_loss_history[-1],
        "final_plur_ce": plur_loss_history[-1],
        "final_overall": eval_overall_history[-1],
        "final_sing": eval_sing_history[-1],
        "final_plur": eval_plur_history[-1],
        "final_sep_l2": epoch_rows[-1]["sep_l2"],
        "final_cos": epoch_rows[-1]["cos"],
        "final_wug_proj": float(wug_proj[-1]),
        "final_wugs_proj": float(wugs_proj[-1]),
        "run_dir": run_dir,
    }
    pd.DataFrame([sm]).to_csv(os.path.join(run_dir, "run_summary.csv"), index=False)

    print(f"\n{'#' * 70}\nRUN COMPLETE — {run_dir}\n{'#' * 70}")


if __name__ == "__main__":
    main()