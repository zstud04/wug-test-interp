import os
import sys
import glob
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
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Model name/path")
    parser.add_argument("--cache_dir", type=str,
                        default="/mnt/dv/wid/projects3/Rogers-muri-human-ai/zstuddiford",
                        help="HF cache directory")
    parser.add_argument("--image_dir", type=str, default="../images/snarples",
                        help="Directory containing images (for image condition)")
    parser.add_argument("--out_dir", type=str, default="results/wug_lr_sweep_results",
                        help="Output directory")
    parser.add_argument("--write_embeddings", action="store_true",
                        help="Save final learned [wug]/[wugs] embeddings to disk after training")
    parser.add_argument("--write_all", action="store_true",
                        help="Save [wug]/[wugs] embeddings at every epoch to the "
                             "'epoch_embs' subfolder of the run dir.")
    parser.add_argument("--training_condition", type=str, required=True,
                        choices=["image", "syntax"],
                        help="'image': train with images. 'syntax': train with text-only.")
    parser.add_argument("--filler_image_dir", type=str, default=None,
                        help="Directory of filler images for syntax condition.")
    parser.add_argument("--terminate_cond_epochs", type=int, default=5,
                        help="Early-stop patience: stop after this many consecutive "
                             "epochs with no improvement in overall eval accuracy. "
                             "Set to 0 (or >= epochs) to disable early stopping. "
                             "Default 5.")

    parser.add_argument("--train_csv", type=str, required=True,
                        help="Path to training CSV with columns 'type' (singular|plural) "
                             "and 'sentence'.")
    parser.add_argument("--eval_csv", type=str, required=True,
                        help="Path to eval CSV with columns 'good_sentence' (grammatical) "
                             "and 'bad_sentence' (ungrammatical), one pair per row.")

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


def load_train_csv(path):
    """Load training sentences from a CSV with 'type' and 'sentence' columns.

    Returns (singular_sentences, plural_sentences).
    """
    df = pd.read_csv(path)
    for col in ("type", "sentence"):
        assert col in df.columns, f"Train CSV {path} missing required column '{col}'"
    df = df.dropna(subset=["type", "sentence"]).copy()
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df = df[df["sentence"] != ""]

    valid = {"singular", "plural"}
    bad = sorted(set(df["type"]) - valid)
    assert not bad, f"Train CSV {path} has unexpected 'type' values {bad}; expected {sorted(valid)}"

    singular_sentences = df.loc[df["type"] == "singular", "sentence"].tolist()
    plural_sentences = df.loc[df["type"] == "plural", "sentence"].tolist()
    return singular_sentences, plural_sentences


def load_eval_csv(path):
    """Load eval pairs from a CSV with 'good_sentence' and 'bad_sentence' columns.

    Returns (goods, bads).
    """
    df = pd.read_csv(path)
    for col in ("good_sentence", "bad_sentence"):
        assert col in df.columns, f"Eval CSV {path} missing required column '{col}'"
    df = df.dropna(subset=["good_sentence", "bad_sentence"]).copy()
    df["good_sentence"] = df["good_sentence"].astype(str).str.strip()
    df["bad_sentence"] = df["bad_sentence"].astype(str).str.strip()
    df = df[(df["good_sentence"] != "") & (df["bad_sentence"] != "")]

    goods = df["good_sentence"].tolist()
    bads = df["bad_sentence"].tolist()
    return goods, bads


def main():
    args = parse_args()

    LR          = args.lr
    GLOBAL_SEED = args.seed
    EPOCHS      = args.epochs
    BATCH_MODE  = args.batch_mode
    BATCH_SIZE  = args.batch_size
    WD          = 0.0
    TRAINING_CONDITION = args.training_condition
    PATIENCE    = args.terminate_cond_epochs

    # Norm control
    RENORM_EVERY_STEP = True
    MAX_NORM_MULT     = 1.10

    lr_str = f"{LR:.10f}".rstrip("0").rstrip(".").replace(".", "p")
    if TRAINING_CONDITION == "image":
        img_tag = os.path.basename(args.image_dir)
        run_name = f"{img_tag}_{BATCH_MODE}_lr{lr_str}_seed{GLOBAL_SEED}_ep{EPOCHS}"
    else:
        run_name = f"syntax_{BATCH_MODE}_lr{lr_str}_seed{GLOBAL_SEED}_ep{EPOCHS}"

    OUT_ROOT = args.out_dir
    os.makedirs(OUT_ROOT, exist_ok=True)
    run_dir = os.path.join(OUT_ROOT, run_name)
    os.makedirs(run_dir, exist_ok=True)

    epoch_embs_dir = os.path.join(run_dir, "epoch_embs")
    if args.write_all:
        os.makedirs(epoch_embs_dir, exist_ok=True)

    singular_sentences, plural_sentences = load_train_csv(args.train_csv)
    assert len(singular_sentences) > 0, f"No singular sentences in {args.train_csv}"
    assert len(plural_sentences) > 0, f"No plural sentences in {args.train_csv}"


    goods, bads = load_eval_csv(args.eval_csv)
    assert len(goods) == len(bads), f"Mismatch: {len(goods)} goods vs {len(bads)} bads"

    embed_init_words = None
    if args.embed_init:
        embed_init_words = load_lines(args.embed_init)
        assert len(embed_init_words) >= 4, f"Need at least 4 words in {args.embed_init}, got {len(embed_init_words)}"

    print("=" * 70)
    print("RUN CONFIG")
    print(f"  Condition:      {TRAINING_CONDITION}")
    if TRAINING_CONDITION == "syntax":
        if args.filler_image_dir:
            print(f"  Filler imgs:    {args.filler_image_dir}")
    print(f"  Batch mode:     {BATCH_MODE}")
    print(f"  Batch size:     {BATCH_SIZE}")
    print(f"  LR:             {LR}")
    print(f"  Seed:           {GLOBAL_SEED}")
    print(f"  Epochs:         {EPOCHS}")
    print(f"  Terminate@:     {PATIENCE} epochs no overall-acc improvement"
          f"{' (disabled)' if (PATIENCE <= 0 or PATIENCE >= EPOCHS) else ''}")
    print(f"  Write all:      {args.write_all}")
    print(f"  Model:          {args.model}")
    if TRAINING_CONDITION == "image":
        print(f"  Image dir:      {args.image_dir}")
    print(f"  Train CSV:      {args.train_csv} "
          f"({len(singular_sentences)} singular, {len(plural_sentences)} plural)")
    print(f"  Eval CSV:       {args.eval_csv} ({len(goods)} pairs)")
    print(f"  Embed init:     {args.embed_init or 'default (random)'}")
    print(f"  Output:         {run_dir}")
    print(f"Total eval pairs: {len(goods)}")
    print("=" * 70)


    device = "cuda"
    lm = scorer.VLMScorer(
        args.model, device=device, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir
    )

    def load_resized(path):
        return Image.open(path).convert("RGB").resize((224, 224), Image.LANCZOS)


    added_tokens = [" [wug]", " [wugs]"]
    existing_vocab = lm.tokenizer.tokenizer.get_vocab()
    tokens_to_add = [t for t in added_tokens if t not in existing_vocab]
    if len(tokens_to_add) > 0:
        lm.tokenizer.tokenizer.add_tokens(tokens_to_add)
        old_len = lm.model.resize_token_embeddings().weight.shape[0]
        lm.model.resize_token_embeddings(old_len + len(tokens_to_add))

    emb     = lm.model.model.language_model.embed_tokens
    lm_head = lm.model.lm_head
    tok     = lm.tokenizer.tokenizer

    new_ids = [tok(t, add_special_tokens=False).input_ids[0] for t in added_tokens]
    wug_id, wugs_id = new_ids

    assert emb.weight.data_ptr() == lm_head.weight.data_ptr(), \
        "ERROR: emb and lm_head are NOT tied! This script requires tied weights."
    print("✓ emb.weight and lm_head.weight are tied (same tensor)")

    base_emb_matrix = emb.weight.detach().clone()
    device = lm.model.device

    img_dir = args.image_dir
    singular_imgs = [load_resized(os.path.join(img_dir, f"singular{i:01d}.png")) for i in range(1, 6)]
    plural_imgs   = [load_resized(os.path.join(img_dir, f"plural{i:01d}.png")) for i in range(1, 6)]


    if TRAINING_CONDITION == "image":
        singular_imgs = [load_resized(os.path.join(img_dir, f"singular{i:01d}.png")) for i in range(1, 6)]
        plural_imgs   = [load_resized(os.path.join(img_dir, f"plural{i:01d}.png")) for i in range(1, 6)]
        singular_templates = [train_chat_template(lm, s) for s in singular_sentences]
        plural_templates = [train_chat_template(lm, s) for s in plural_sentences]
    else:
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
            singular_templates = [train_chat_template_filler(lm, s) for s in singular_sentences]
            plural_templates   = [train_chat_template_filler(lm, s) for s in plural_sentences]
            singular_imgs = filler_imgs
            plural_imgs = filler_imgs
        else:
            singular_templates = [train_chat_template_noimage(lm, s) for s in singular_sentences]
            plural_templates   = [train_chat_template_noimage(lm, s) for s in plural_sentences]
            singular_imgs = None
            plural_imgs = None
        print(f"Loaded syntax stimuli: {len(singular_templates)} singular, {len(plural_templates)} plural")

    good_queries = [chat_template(lm, s, noimage=True, assistant=False) for s in goods]
    bad_queries  = [chat_template(lm, s, noimage=True, assistant=False) for s in bads]
    singular_eval_idx = [i for i, s in enumerate(goods) if " [wug]" in s]
    plural_eval_idx   = [i for i, s in enumerate(goods) if " [wugs]" in s]


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

    def save_epoch_embedding(epoch_num):
        """Write current [wug]/[wugs] embeddings to epoch_embs/epoch_{N}.pt."""
        rec = {
            "epoch": epoch_num,
            "wug_id": wug_id,
            "wugs_id": wugs_id,
            "wug_embedding": emb.weight.data[wug_id].detach().cpu().clone(),
            "wugs_embedding": emb.weight.data[wugs_id].detach().cpu().clone(),
            "added_tokens": added_tokens,
            "vocab_size": emb.weight.shape[0],
            "model_name": args.model,
        }
        path = os.path.join(epoch_embs_dir, f"epoch_{epoch_num:03d}.pt")
        torch.save(rec, path)
        return path

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

    set_all_seeds(GLOBAL_SEED)

    with torch.no_grad():
        emb.weight.data.copy_(base_emb_matrix.to(emb.weight.device, dtype=emb.weight.dtype))
        target_norm = emb.weight.norm(dim=1).float().mean().item()

        if embed_init_words is not None:
            def _safe_token_id(w):
                ids = tok(" " + w, add_special_tokens=False).input_ids
                return ids[0] if len(ids) == 1 else None

            init_ids = [_safe_token_id(w) for w in embed_init_words]
            init_ids = [t for t in init_ids if t is not None and t < emb.weight.shape[0]]
            assert len(init_ids) >= 4, f"Only {len(init_ids)} valid token ids from embed_init file"

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

    for p in lm.model.parameters():
        p.requires_grad = False
    emb.weight.requires_grad = True

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

    EARLY_STOP_ENABLED = (PATIENCE is not None and PATIENCE > 0 and PATIENCE < EPOCHS)
    EPS_IMPROVE   = 1e-6                
    best_overall  = -float("inf")
    best_epoch    = 0                    
    epochs_no_improve = 0
    best_wug_emb  = None                 
    best_wugs_emb = None
    stopped_early = False

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

        # --- Write per-epoch embedding (--write_all) ---
        if args.write_all:
            ep_path = save_epoch_embedding(epoch + 1)
            print(f"  [write_all] saved {ep_path}")

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

        if ev["overall_acc"] > best_overall + EPS_IMPROVE:
            best_overall = ev["overall_acc"]
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_wug_emb = emb.weight.data[wug_id].detach().cpu().clone()
            best_wugs_emb = emb.weight.data[wugs_id].detach().cpu().clone()
        else:
            epochs_no_improve += 1
            print(f"  [early-stop] no overall improvement for "
                  f"{epochs_no_improve}/{PATIENCE} epoch(s) "
                  f"(best={best_overall:.3f} @ epoch {best_epoch})")

        if EARLY_STOP_ENABLED and epochs_no_improve >= PATIENCE:
            stopped_early = True
            print(f"\n>>> EARLY STOP at epoch {epoch + 1}: overall acc has not improved "
                  f"for {PATIENCE} epochs. Best overall={best_overall:.3f} @ epoch {best_epoch}.")
            print(f">>> Retroactively truncating all outputs to epoch {best_epoch}.")
            break

    print("\nTraining complete.")

  
    if stopped_early and best_epoch >= 1:
        keep = best_epoch  

        loss_history          = loss_history[:keep]
        sing_loss_history     = sing_loss_history[:keep]
        plur_loss_history     = plur_loss_history[:keep]
        grad_norm_hist["wug"]  = grad_norm_hist["wug"][:keep]
        grad_norm_hist["wugs"] = grad_norm_hist["wugs"][:keep]
        wug_trajectory        = wug_trajectory[:keep]
        wugs_trajectory       = wugs_trajectory[:keep]
        eval_overall_history  = eval_overall_history[:keep]
        eval_sing_history     = eval_sing_history[:keep]
        eval_plur_history     = eval_plur_history[:keep]
        eval_diff_history     = eval_diff_history[:keep]
        epoch_rows            = [r for r in epoch_rows if r["epoch"] <= keep]
        eval_item_rows        = [r for r in eval_item_rows if r["epoch"] <= keep]
        step_rows             = [r for r in step_rows if r["epoch"] <= keep]
        logit_gap_rows        = [r for r in logit_gap_rows if r["epoch"] <= keep]

        # Logit-gap history is a flat per-step list; rebuild from retained rows
        logit_gap_history = {"singular": [], "plural": []}
        for r in logit_gap_rows:
            logit_gap_history[r["phase"]].append(r["logit_gap"])

        # Delete per-epoch embedding files past best epoch
        if args.write_all and os.path.isdir(epoch_embs_dir):
            for f in sorted(glob.glob(os.path.join(epoch_embs_dir, "epoch_*.pt"))):
                base = os.path.basename(f)
                try:
                    ep_num = int(base.replace("epoch_", "").replace(".pt", ""))
                except ValueError:
                    continue
                if ep_num > keep:
                    os.remove(f)
                    print(f"  [truncate] removed {f}")

        
        if best_wug_emb is not None:
            with torch.no_grad():
                emb.weight.data[wug_id] = best_wug_emb.to(emb.weight.device, dtype=emb.weight.dtype)
                emb.weight.data[wugs_id] = best_wugs_emb.to(emb.weight.device, dtype=emb.weight.dtype)
            print(f"  [truncate] restored best-epoch ({best_epoch}) embeddings into model.")

    EFFECTIVE_EPOCHS = len(eval_overall_history)

   
    if args.write_embeddings:
        emb_save = {
            "wug_id": wug_id,
            "wugs_id": wugs_id,
            "wug_embedding": emb.weight.data[wug_id].detach().cpu(),
            "wugs_embedding": emb.weight.data[wugs_id].detach().cpu(),
            "added_tokens": added_tokens,
            "vocab_size": emb.weight.shape[0],
            "model_name": args.model,
            "saved_epoch": best_epoch if stopped_early else EFFECTIVE_EPOCHS,
        }
        emb_path = os.path.join(run_dir, "learned_embeddings.pt")
        torch.save(emb_save, emb_path)
        print(f"Saved embeddings to {emb_path} (epoch {emb_save['saved_epoch']})")

    sp_dir, wug_proj, wugs_proj = compute_sp_axis_and_traj(wug_trajectory, wugs_trajectory)
    for ep in range(EFFECTIVE_EPOCHS):
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
        "filler_image_dir": args.filler_image_dir,
        "batch_mode": BATCH_MODE, "batch_size": BATCH_SIZE,
        "lr": LR, "seed": GLOBAL_SEED, "epochs": EPOCHS, "model": args.model,
        "image_dir": args.image_dir if TRAINING_CONDITION == "image" else None,
        "embed_init": args.embed_init,
        "renorm": RENORM_EVERY_STEP, "betas": "(0.0, 0.9)", "wd": WD,
        "train_csv": args.train_csv, "eval_csv": args.eval_csv,
        "write_all": args.write_all,
        "terminate_cond_epochs": PATIENCE,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_overall_acc": best_overall if best_epoch >= 1 else None,
        "effective_epochs": EFFECTIVE_EPOCHS,
    }]).to_csv(os.path.join(run_dir, "run_config.csv"), index=False)

    sm = {
        "run_name": run_name,
        "training_condition": TRAINING_CONDITION,
        "filler_image_dir": args.filler_image_dir,
        "batch_mode": BATCH_MODE, "batch_size": BATCH_SIZE,
        "lr": LR, "seed": GLOBAL_SEED, "epochs": EPOCHS,
        "image_dir": args.image_dir if TRAINING_CONDITION == "image" else None,
        "embed_init": args.embed_init,
        "write_all": args.write_all,
        "terminate_cond_epochs": 5,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "effective_epochs": EFFECTIVE_EPOCHS,
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