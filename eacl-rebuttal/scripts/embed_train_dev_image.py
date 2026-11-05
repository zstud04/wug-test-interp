import os
import sys
import copy
import glob
import random
import argparse
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np

from transformers import set_seed as hf_set_seed

# Fork of core/train/embed_train.py (kept unmodified): the dev-set eval is
# replaced with a held-out-image logit-gap metric (see build_qa_prompt and
# run_agreement_eval below) — see eacl-rebuttal/README.md. Path shim below lets
# this run from any cwd.
import pathlib
_REPO_ROOT = pathlib.Path(__file__).resolve().parent
while not (_REPO_ROOT / "core").is_dir():
    _parent = _REPO_ROOT.parent
    if _parent == _REPO_ROOT:
        raise RuntimeError(f"Could not find repo root (no core/ dir above {__file__})")
    _REPO_ROOT = _parent
sys.path.insert(0, str(_REPO_ROOT))

from utils.chat_templates import (
    train_chat_template,
    train_chat_template_noimage,
    train_chat_template_filler,
)
from utils.vlm_loading import get_input_output_embeddings, load_vlm


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
    parser.add_argument("--image_dir", type=str,
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
                        help="Path to eval CSV with columns 'good' (grammatical) "
                             "and 'bad' (ungrammatical), one pair per row.")

    parser.add_argument("--embed_init", type=str, default=None,
                        help="Path to txt file with noun words (one per line) for embedding init. "
                             "If supplied, [wug]/[wugs] are initialized near the mean of these "
                             "embeddings with noise scaled to the mean singular-plural distance. "
                             "If omitted, uses default random init.")
    parser.add_argument("--init_mode",
                        choices=["independent", "symmetric", "spherical_symmetric"],
                        default="independent",
                        help="How to perturb the shared noun centroid. 'independent' reproduces "
                             "the original two random directions; 'symmetric' uses +epsilon and "
                             "-epsilon before normalization; 'spherical_symmetric' constructs "
                             "equal-norm vectors with an exact final separation.")
    parser.add_argument("--symmetric_separation_scale", type=float, default=1.0,
                        help="For spherical_symmetric init, exact final separation as a multiple "
                             "of the mean singular-plural noun-pair distance (default: 1.0).")
    parser.add_argument("--selection_metric",
                        choices=["raw_margin", "balanced_logistic"],
                        default="raw_margin",
                        help="Held-out-image metric used for best-checkpoint selection and early "
                             "stopping. This does not change the training loss.")

    args = parser.parse_args()
    if args.symmetric_separation_scale < 0:
        parser.error("--symmetric_separation_scale must be non-negative")
    return args


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


def load_image_eval_csv(path):
    """Load the held-out image dev set: a CSV with 'image_path' and
    'image_class' ('singular'/'plural') columns. Returns (image_paths, image_classes).
    """
    df = pd.read_csv(path)
    for col in ("image_path", "image_class"):
        assert col in df.columns, f"Eval CSV {path} missing required column '{col}'"
    df = df.dropna(subset=["image_path", "image_class"]).copy()
    df["image_class"] = df["image_class"].astype(str).str.strip().str.lower()
    bad = sorted(set(df["image_class"]) - {"singular", "plural"})
    assert not bad, f"Eval CSV {path} has unexpected image_class values {bad}"

    image_paths = df["image_path"].tolist()
    image_classes = df["image_class"].tolist()
    return image_paths, image_classes


# Diverse, number-neutral prefixes ending right before the number-bearing noun
# (crossed with every held-out image below) — a single fixed phrasing would
# risk measuring "does the model handle this exact sentence" rather than the
# general image->number competency. Every prefix's own grammar is fixed
# regardless of what follows (subject is "I"/"This"/"It"/"That", never the
# noun itself), so none of them can leak singular/plural about wug/wugs.
QA_TEMPLATES = [
    ("Caption this image.", "The"),
    ("What do you see in this picture?", "I see the"),
    ("Describe this image.", "This shows the"),
    ("What creature is in the image?", "It looks like the"),
    ("Can you identify what's in the picture?", "That is the"),
    ("Tell me about this picture.", "This picture shows the"),
]


def build_qa_prompt(lm, question="Caption this image.", prefix="The"):
    """Prompt ending right after `prefix`, identical for every image, so the
    next-token logits at the final position give P(next token | image,
    question, prefix) directly comparable across images. Mirrors the
    training-time "Caption this image." framing (train_chat_template) rather
    than the noimage=True text-only convention used elsewhere in this script.
    """
    context = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
        {"role": "assistant", "content": [{"type": "text", "text": prefix}]},
    ]
    return lm.tokenizer.apply_chat_template(context, continue_final_message=True)


def resolve_single_token_id(tok, word):
    """Return the token id for `word` as it appears after another word in
    running text, or None if it isn't a single token for this tokenizer.

    BPE tokenizers (Qwen, Gemma, Llama 3, Mistral, SmolLM2, ...) encode a
    word differently depending on whether it's preceded by a space, so a
    leading space must be added to get the word-initial ("mid-sentence")
    form. Legacy SentencePiece tokenizers (LLaMA 1/2, e.g. llava-hf's
    llava-1.5) already insert that leading-space marker internally, so
    manually adding one produces a spurious extra token instead. Try the
    BPE convention first (it's what every model we've used so far needs),
    then fall back to the bare word.
    """
    for candidate in (" " + word, word):
        ids = tok(candidate, add_special_tokens=False).input_ids
        if len(ids) == 1:
            return ids[0]
    return None


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


    eval_image_paths, eval_image_classes = load_image_eval_csv(args.eval_csv)
    assert len(eval_image_paths) > 0, f"No rows in {args.eval_csv}"

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
    print(f"  Eval CSV:       {args.eval_csv} ({len(eval_image_paths)} held-out images: "
          f"{eval_image_classes.count('singular')} singular, {eval_image_classes.count('plural')} plural)")
    print(f"  Embed init:     {args.embed_init or 'default (random)'}")
    print(f"  Init mode:      {args.init_mode}")
    if args.init_mode == "spherical_symmetric":
        print(f"  Separation:     {args.symmetric_separation_scale:g}x noun-pair distance")
    print(f"  Selection:      {args.selection_metric}")
    print(f"  Output:         {run_dir}")
    print("=" * 70)


    device = "cuda"
    lm = load_vlm(
        args.model, device=device, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir
    )

    def load_resized(path):
        return Image.open(path).convert("RGB").resize((224, 224), Image.LANCZOS)


    tok = lm.tokenizer.tokenizer

    def choose_added_token_strings(base_words):
        """BPE tokenizers (Qwen/Gemma/SmolLM2) represent a word-initial token
        by including a literal leading space in the string passed to
        add_tokens. Legacy SentencePiece fast tokenizers (Llama-1/2/LLaVA-1.5)
        insert their own '▁' marker internally and never match a literal
        leading space against surrounding text, so the added token must be
        registered without one — otherwise it never matches inside a real
        caption and silently gets zero supervision. Probe on a throwaway
        deep copy instead of assuming either convention."""
        probe = "zzzwugprobezzz"
        # Context has exactly one literal space before the probe (matching how
        # captions actually look, e.g. "the [wug]") for BOTH candidates, so
        # only the registered token string varies, not the surrounding text.
        context = f"I saw the {probe} today."
        for candidate in (" " + probe, probe):
            test_tok = copy.deepcopy(tok)
            test_tok.add_tokens([candidate])
            pid = test_tok(candidate, add_special_tokens=False).input_ids[0]
            ctx_ids = test_tok(context, add_special_tokens=False).input_ids
            if pid in ctx_ids:
                prefix = " " if candidate.startswith(" ") else ""
                return [prefix + w for w in base_words]
        raise RuntimeError("Could not determine added-token space convention for this tokenizer")

    added_tokens = choose_added_token_strings(["[wug]", "[wugs]"])
    existing_vocab = tok.get_vocab()
    tokens_to_add = [t for t in added_tokens if t not in existing_vocab]
    if len(tokens_to_add) > 0:
        tok.add_tokens(tokens_to_add)
        old_len = lm.model.resize_token_embeddings().weight.shape[0]
        lm.model.resize_token_embeddings(old_len + len(tokens_to_add))

    emb, lm_head, IS_TIED = get_input_output_embeddings(lm.model)

    new_ids = [tok(t, add_special_tokens=False).input_ids[0] for t in added_tokens]
    wug_id, wugs_id = new_ids

    if IS_TIED:
        print("✓ emb.weight and lm_head.weight are tied (same tensor)")
    else:
        print("  emb.weight and lm_head.weight are NOT tied — training will "
              "update lm_head via gradient each step, then copy the result "
              "into the input embedding row (see do_step) so both roles "
              "reflect the trained value even though gradient can only "
              "reach lm_head under this next-token-only loss.")

    base_emb_matrix = emb.weight.detach().clone()
    device = lm.model.device




    if TRAINING_CONDITION == "image":
        img_dir = args.image_dir
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

    eval_images_by_path = {p: load_resized(p) for p in set(eval_image_paths)}
    eval_prompts_per_template = [build_qa_prompt(lm, q, prefix) for q, prefix in QA_TEMPLATES]

    # Full cross: every template x every held-out image (120 = 6 x 20 stimuli).
    # Images are loaded once (eval_images_by_path) and referenced, not duplicated.
    eval_images, eval_prompts, eval_classes, eval_template_ids, eval_paths = [], [], [], [], []
    for t_id, prompt in enumerate(eval_prompts_per_template):
        for path, cls in zip(eval_image_paths, eval_image_classes):
            eval_images.append(eval_images_by_path[path])
            eval_prompts.append(prompt)
            eval_classes.append(cls)
            eval_template_ids.append(t_id)
            eval_paths.append(path)

    singular_eval_idx = [i for i, c in enumerate(eval_classes) if c == "singular"]
    plural_eval_idx   = [i for i, c in enumerate(eval_classes) if c == "plural"]
    print(f"  Dev stimuli:    {len(QA_TEMPLATES)} templates x {len(eval_image_paths)} images "
          f"= {len(eval_images)} stimuli")


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
            # Nested (one sublist per example) rather than flat: Qwen's processor
            # accepts either, but Gemma3's requires nesting to match one image
            # per text example instead of miscounting the whole batch as one.
            enc = lm.tokenizer(images=[[im] for im in imgs], text=texts, return_tensors="pt", padding=True)
        else:
            enc = lm.tokenizer(text=texts, return_tensors="pt", padding=True)
        labels = torch.stack([mask_to_novel_token(enc["input_ids"][i]) for i in range(len(texts))])
        enc = {k: v.to(device) for k, v in enc.items()}
        return enc, labels.to(device)

    def full_batch_loss(imgs, texts):
        """Token-averaged loss over all examples, chunked to BATCH_SIZE so this
        once-per-epoch diagnostic (independent of the training batch size)
        doesn't spike peak memory above what the actual training step uses —
        on Gemma3's heavier per-image cost this full-batch forward pass (not
        the training step itself) was the actual OOM trigger."""
        total_loss_x_tokens = 0.0
        total_tokens = 0
        for start in range(0, len(texts), BATCH_SIZE):
            chunk_imgs = imgs[start:start + BATCH_SIZE] if imgs is not None else None
            chunk_texts = texts[start:start + BATCH_SIZE]
            cb = make_batch(chunk_imgs, chunk_texts)
            n_tokens = (cb[1] != -100).sum().item()
            with torch.no_grad():
                loss = lm.model(**cb[0], labels=cb[1]).loss.item()
            total_loss_x_tokens += loss * n_tokens
            total_tokens += n_tokens
        return total_loss_x_tokens / total_tokens

    def run_agreement_eval(eval_batch_size=min(40, BATCH_SIZE * 20)):
        # Held-out-image dev metric, crossing len(QA_TEMPLATES) diverse prefixes
        # with every held-out image. lm.sequence_score's whole-sequence mean-
        # logprob (used by the noun-vs-verb margin fork) doesn't work here: an
        # image contributes far more tokens than the caption text, so the
        # 1-token wug/wugs signal gets averaged away (confirmed empirically —
        # identical scores to 4 decimal places regardless of image or token).
        # Instead, run a forward pass per (template, image) stimulus and read
        # the raw logit gap logit[wug_id]-logit[wugs_id] at the final position
        # — the log-odds of wug vs wugs given that image, unaffected by
        # sequence length. signed_gap flips sign for plural images so higher is
        # always "more correct" for both classes, matching how
        # overall_acc/sing_acc/plur_acc drive early stopping elsewhere unchanged.
        gaps_chunks = []
        for start in range(0, len(eval_images), eval_batch_size):
            batch_imgs = eval_images[start:start + eval_batch_size]
            batch_prompts = eval_prompts[start:start + eval_batch_size]
            enc = lm.tokenizer(text=batch_prompts, images=[[im] for im in batch_imgs],
                                return_tensors="pt", padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = lm.model(**enc)
            last_pos = enc["attention_mask"].sum(dim=1) - 1
            idx = torch.arange(len(batch_imgs))
            final_logits = out.logits[idx, last_pos].float()
            gaps_chunks.append((final_logits[:, wug_id] - final_logits[:, wugs_id]).cpu())
        gaps = torch.cat(gaps_chunks)
        is_singular = torch.tensor([c == "singular" for c in eval_classes])
        signed = torch.where(is_singular, gaps, -gaps)
        acc = (signed > 0).float()
        return {
            "overall_acc": signed.mean().item(),
            "sing_acc": signed[singular_eval_idx].mean().item(),
            "plur_acc": signed[plural_eval_idx].mean().item(),
            "balanced_logistic": -0.5 * (
                torch.nn.functional.softplus(-signed[singular_eval_idx]).mean()
                + torch.nn.functional.softplus(-signed[plural_eval_idx]).mean()
            ).item(),
            "diffs": signed,
            "good_scores": gaps,
            "acc_mask": acc,
        }

    reference_word_pairs = [("dog", "dogs"), ("cat", "cats"), ("bird", "birds"),
                             ("bear", "bears"), ("rat", "rats")]
    # Keep only pairs where BOTH words are single tokens for this tokenizer,
    # since compute_sp_axis_and_traj assumes consecutive (sg, pl) pairs —
    # not every reference word survives in every tokenizer's vocabulary
    # (e.g. llava-1.5's SentencePiece vocab splits "cats"/"bears"/"rats").
    reference_words, reference_ids = [], []
    for s, p in reference_word_pairs:
        sid, pid = resolve_single_token_id(tok, s), resolve_single_token_id(tok, p)
        if sid is not None and pid is not None:
            reference_words.extend([s, p])
            reference_ids.extend([sid, pid])
    assert len(reference_ids) >= 4, (
        "Fewer than 2 usable singular/plural reference pairs for this tokenizer"
    )
    singular_plural_pairs = [
        ("cat", "cats"), ("dog", "dogs"), ("bird", "birds"), ("bear", "bears"),
        ("rat", "rats"), ("thing", "things"), ("word", "words"), ("tree", "trees"),
    ]

    def normalize_special_rows(tn, weight_module=None):
        if TRAINING_CONDITION == "syntax":
            return
        weight_module = weight_module if weight_module is not None else emb
        with torch.no_grad():
            if RENORM_EVERY_STEP:
                for tid in [wug_id, wugs_id]:
                    v = weight_module.weight.data[tid]
                    weight_module.weight.data[tid] = (v / v.norm() * tn).to(weight_module.weight.dtype)
            else:
                mx = tn * MAX_NORM_MULT
                for tid in [wug_id, wugs_id]:
                    n = weight_module.weight.data[tid].norm()
                    if n > mx:
                        weight_module.weight.data[tid] = (weight_module.weight.data[tid] / n * mx).to(weight_module.weight.dtype)

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
        # The loss is masked to only the wug/wugs token position itself, so
        # gradient only ever reaches whichever matrix computes the logit
        # there — lm_head. When tied, lm_head IS emb (same tensor), so this
        # is moot. When untied, emb (the input-embedding role) never gets
        # its own gradient under this objective; we sync it to the trained
        # lm_head row after every step instead of leaving it at init forever.
        driver = lm_head if not IS_TIED else emb
        loss.backward()
        with torch.no_grad():
            gw = driver.weight.grad[wug_id].clone()
            gws = driver.weight.grad[wugs_id].clone()
            driver.weight.grad.zero_()
            driver.weight.grad[wug_id] = gw
            driver.weight.grad[wugs_id] = gws
            gnw = gw.norm().item()
            gnws = gws.norm().item()
            bw = driver.weight.data[wug_id].clone()
            bws = driver.weight.data[wugs_id].clone()
        optimizer.step()
        normalize_special_rows(target_norm, driver)
        if not IS_TIED:
            with torch.no_grad():
                emb.weight.data[wug_id] = driver.weight.data[wug_id].to(emb.weight.dtype)
                emb.weight.data[wugs_id] = driver.weight.data[wugs_id].to(emb.weight.dtype)
        with torch.no_grad():
            sw  = (driver.weight.data[wug_id] - bw).norm().item()
            sws = (driver.weight.data[wugs_id] - bws).norm().item()
            nw  = driver.weight.data[wug_id].norm().item()
            nws = driver.weight.data[wugs_id].norm().item()
            sep = (driver.weight[wug_id] - driver.weight[wugs_id]).norm().item()
            cos = torch.nn.functional.cosine_similarity(
                driver.weight[wug_id].unsqueeze(0), driver.weight[wugs_id].unsqueeze(0)).item()
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

        def _spherical_symmetric_pair(center, pair_distance):
            """Return equal-norm vectors with an exact chord distance on the norm sphere."""
            center = center.float()
            center = center / center.norm()
            separation = args.symmetric_separation_scale * pair_distance
            max_separation = 2.0 * target_norm
            if separation > max_separation:
                raise ValueError(
                    f"Requested separation {separation:.6f} exceeds the maximum "
                    f"{max_separation:.6f} for vectors of norm {target_norm:.6f}"
                )
            direction = torch.randn_like(center)
            direction = direction - torch.dot(direction, center) * center
            direction = direction / direction.norm()
            sin_theta = separation / max_separation
            cos_theta = float(np.sqrt(max(0.0, 1.0 - sin_theta ** 2)))
            first = target_norm * (cos_theta * center + sin_theta * direction)
            second = target_norm * (cos_theta * center - sin_theta * direction)
            return first, second

        if embed_init_words is not None:
            assert len(embed_init_words) % 2 == 0, (
                f"{args.embed_init} must contain an even number of lines "
                f"(alternating singular/plural pairs), got {len(embed_init_words)}"
            )
            init_ids = []
            for i in range(0, len(embed_init_words), 2):
                sw, pw = embed_init_words[i], embed_init_words[i + 1]
                sid = resolve_single_token_id(tok, sw)
                pid = resolve_single_token_id(tok, pw)
                if (sid is not None and sid < emb.weight.shape[0]
                        and pid is not None and pid < emb.weight.shape[0]):
                    init_ids.extend([sid, pid])
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

            if args.init_mode == "spherical_symmetric":
                wug_init, wugs_init = _spherical_symmetric_pair(mean_emb_vec, noise_scale)
            elif args.init_mode == "symmetric":
                # The original independent perturbations each have norm d, so
                # their expected separation is sqrt(2)*d.  Opposite vectors
                # have separation 2*r; r=d/sqrt(2) matches that baseline while
                # eliminating an arbitrary initial preference for either row.
                eps = torch.randn_like(mean_emb_vec)
                eps = eps / eps.norm() * (noise_scale / np.sqrt(2.0))
                nw, nws = eps, -eps
            else:
                nw = torch.randn_like(mean_emb_vec)
                nw = nw / nw.norm() * noise_scale
                nws = torch.randn_like(mean_emb_vec)
                nws = nws / nws.norm() * noise_scale
                wug_init = (mean_emb_vec + nw) / (mean_emb_vec + nw).norm() * target_norm
                wugs_init = (mean_emb_vec + nws) / (mean_emb_vec + nws).norm() * target_norm
            if args.init_mode == "symmetric":
                wug_init = (mean_emb_vec + nw) / (mean_emb_vec + nw).norm() * target_norm
                wugs_init = (mean_emb_vec + nws) / (mean_emb_vec + nws).norm() * target_norm

            emb.weight.data[wug_id] = wug_init.to(emb.weight.dtype)
            emb.weight.data[wugs_id] = wugs_init.to(emb.weight.dtype)
            if not IS_TIED:
                lm_head.weight.data[wug_id] = wug_init.to(lm_head.weight.dtype)
                lm_head.weight.data[wugs_id] = wugs_init.to(lm_head.weight.dtype)
        else:
            pair_distances = []
            init_pair_rows = []
            for s, p in singular_plural_pairs:
                sid = resolve_single_token_id(tok, s)
                pid = resolve_single_token_id(tok, p)
                assert sid is not None and pid is not None, (
                    f"'{s}'/'{p}' aren't single tokens for this tokenizer"
                )
                d = (emb.weight[sid] - emb.weight[pid]).norm().item()
                pair_distances.append(d)
                init_pair_rows.append({"token_a": s.strip(), "token_b": p.strip(), "distance": d})
                print(f"{s.strip():>8} -> {p.strip():<8} dist={d:.4f}")

            noise_scale = float(np.mean(pair_distances))
            print(f"Mean S→P distance (noise scale): {noise_scale:.4f}")

            mean_emb_vec = emb.weight.mean(dim=0)
            if args.init_mode == "spherical_symmetric":
                wug_init, wugs_init = _spherical_symmetric_pair(mean_emb_vec, noise_scale)
            elif args.init_mode == "symmetric":
                eps = torch.randn_like(mean_emb_vec)
                eps = eps / eps.norm() * (noise_scale / np.sqrt(2.0))
                nw, nws = eps, -eps
            else:
                nw = torch.randn_like(mean_emb_vec)
                nw = nw / nw.norm() * noise_scale
                nws = torch.randn_like(mean_emb_vec)
                nws = nws / nws.norm() * noise_scale

            if args.init_mode != "spherical_symmetric":
                wug_init = (mean_emb_vec + nw) / (mean_emb_vec + nw).norm() * target_norm
                wugs_init = (mean_emb_vec + nws) / (mean_emb_vec + nws).norm() * target_norm

            emb.weight.data[wug_id] = wug_init.to(emb.weight.dtype)
            emb.weight.data[wugs_id] = wugs_init.to(emb.weight.dtype)
            if not IS_TIED:
                lm_head.weight.data[wug_id] = wug_init.to(lm_head.weight.dtype)
                lm_head.weight.data[wugs_id] = wugs_init.to(lm_head.weight.dtype)

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
    # Gradient only ever reaches the matrix that computes the wug/wugs
    # logit (see do_step) — lm_head when untied, or emb itself when tied
    # (same tensor). The other role is kept in sync explicitly, not via
    # autograd, since this next-token-only loss can't reach it directly.
    trainable = emb if IS_TIED else lm_head
    trainable.weight.requires_grad = True

    optimizer = torch.optim.Adam(
        [{"params": [trainable.weight], "lr": LR}],
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

        with torch.no_grad():
            if TRAINING_CONDITION == "image":
                sl = full_batch_loss(s_imgs_ep, s_texts)
                pl = full_batch_loss(p_imgs_ep, p_texts)
            else:
                sl = full_batch_loss(None, s_texts)
                pl = full_batch_loss(None, p_texts)
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

        for i in range(len(eval_images)):
            eval_item_rows.append({
                "epoch": epoch + 1, "item_index": i,
                "kind": eval_classes[i],
                "image_path": eval_paths[i],
                "template_id": eval_template_ids[i],
                "logit_gap_wug_minus_wugs": float(ev["good_scores"][i].item()),
                "signed_gap": float(ev["diffs"][i].item()),
                "correct": int(ev["acc_mask"][i].item()),
            })

        print(f"\n  CE avg={avg_loss:.4f}  [s={sl:.4f} p={pl:.4f}]")
        print(f"  Grad [wug]={ga:.4e} [wugs]={gas:.4e}")
        print(f"  Norm [wug]={nwug:.4f} [wugs]={nwugs:.4f} (target={target_norm:.4f})")
        print(f"  Sep={sep_val:.4f}  Cos={csim:.4f}")
        print(f"  [wug]  nbrs: {nbrs_w}")
        print(f"  [wugs] nbrs: {nbrs_ws}")
        selection_score = (ev["balanced_logistic"]
                           if args.selection_metric == "balanced_logistic"
                           else ev["overall_acc"])
        print(f"  Eval[logit-gap]: overall={ev['overall_acc']:.3f} sing={ev['sing_acc']:.3f} "
              f"plur={ev['plur_acc']:.3f} balanced_logistic={ev['balanced_logistic']:.3f} "
              f"selection={selection_score:.3f}")

        er = {"epoch": epoch + 1, "avg_ce": avg_loss, "sing_ce": sl, "plur_ce": pl,
              "grad_wug": ga, "grad_wugs": gas, "norm_wug": nwug, "norm_wugs": nwugs,
              "target_norm": target_norm, "sep_l2": sep_val, "cos": csim,
              "nbrs_wug": " | ".join(nbrs_w), "nbrs_wugs": " | ".join(nbrs_ws),
              "eval_overall": ev["overall_acc"], "eval_sing": ev["sing_acc"],
              "eval_plur": ev["plur_acc"],
              "eval_balanced_logistic": ev["balanced_logistic"],
              "selection_score": selection_score}
        epoch_rows.append(er)

        if selection_score > best_overall + EPS_IMPROVE:
            best_overall = selection_score
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_wug_emb = emb.weight.data[wug_id].detach().cpu().clone()
            best_wugs_emb = emb.weight.data[wugs_id].detach().cpu().clone()
        else:
            epochs_no_improve += 1
            print(f"  [early-stop] no selection-score improvement for "
                  f"{epochs_no_improve}/{PATIENCE} epoch(s) "
                  f"(best={best_overall:.3f} @ epoch {best_epoch})")

        if EARLY_STOP_ENABLED and epochs_no_improve >= PATIENCE:
            stopped_early = True
            print(f"\n>>> EARLY STOP at epoch {epoch + 1}: selection score has not improved "
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
                if not IS_TIED:
                    # best_wug_emb/best_wugs_emb were snapshotted from emb,
                    # which do_step kept synced to lm_head at every step, so
                    # these are also the correct lm_head values at that epoch.
                    lm_head.weight.data[wug_id] = best_wug_emb.to(lm_head.weight.device, dtype=lm_head.weight.dtype)
                    lm_head.weight.data[wugs_id] = best_wugs_emb.to(lm_head.weight.device, dtype=lm_head.weight.dtype)
            print(f"  [truncate] restored best-epoch ({best_epoch}) embeddings into model.")

    # A run can reach the epoch cap without satisfying patience.  Its best
    # checkpoint may still precede the final epoch, so restore it just as we do
    # for an early-stopped run.
    if not stopped_early and best_wug_emb is not None:
        with torch.no_grad():
            emb.weight.data[wug_id] = best_wug_emb.to(emb.weight.device, dtype=emb.weight.dtype)
            emb.weight.data[wugs_id] = best_wugs_emb.to(emb.weight.device, dtype=emb.weight.dtype)
            if not IS_TIED:
                lm_head.weight.data[wug_id] = best_wug_emb.to(lm_head.weight.device, dtype=lm_head.weight.dtype)
                lm_head.weight.data[wugs_id] = best_wugs_emb.to(lm_head.weight.device, dtype=lm_head.weight.dtype)
        print(f"  [max-epochs] restored best-epoch ({best_epoch}) embeddings into model.")

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
            "saved_epoch": best_epoch if best_epoch >= 1 else EFFECTIVE_EPOCHS,
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
        "init_mode": args.init_mode,
        "symmetric_separation_scale": args.symmetric_separation_scale,
        "selection_metric": args.selection_metric,
        "renorm": RENORM_EVERY_STEP, "betas": "(0.0, 0.9)", "wd": WD,
        "train_csv": args.train_csv, "eval_csv": args.eval_csv,
        "write_all": args.write_all,
        "terminate_cond_epochs": PATIENCE,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_selection_score": best_overall if best_epoch >= 1 else None,
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
        "init_mode": args.init_mode,
        "symmetric_separation_scale": args.symmetric_separation_scale,
        "selection_metric": args.selection_metric,
        "write_all": args.write_all,
        "terminate_cond_epochs": PATIENCE,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_selection_score": best_overall,
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
