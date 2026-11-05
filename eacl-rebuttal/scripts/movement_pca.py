#!/usr/bin/env python3
"""
PCA movement analysis for the eacl-rebuttal runs — same analysis as
representational-analysis/embed_analysis.py (PCA fit on ~2000 real sg/pl noun
embeddings, [wug]/[wugs] init and final positions projected into it, movement
along the sg-pl axis), adapted to the run layout produced by
eacl-rebuttal/scripts/embed_train_dev_image.py.

These runs don't save initial_embeddings.pt, so the init is reconstructed
exactly from each run's run_config.csv by replaying the training script's init
code (same seed, same CUDA RNG call order, same init_mode). Each reconstructed
init is checked against the "Init separation" value in the run's training log.

Output files (in --out_dir):
  sg_pl_reduced.csv              — PCA positions of all reference nouns
  wug_wugs_reduced_image.csv     — PCA positions (init + final, per seed)
  wug_wugs_movement_image.csv    — axis movement per seed
  neighbors_image.csv            — top-K neighbors of learned [wug]/[wugs]
"""
import argparse
import copy
import glob
import importlib.util
import os
import pathlib
import random
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from transformers import set_seed as hf_set_seed

_REPO_ROOT = pathlib.Path(__file__).resolve().parent
while not (_REPO_ROOT / "core").is_dir():
    _REPO_ROOT = _REPO_ROOT.parent
sys.path.insert(0, str(_REPO_ROOT))

from utils.vlm_loading import get_input_output_embeddings, load_vlm

# Reuse the original analysis' noun selection, projection and CSV writers.
_spec = importlib.util.spec_from_file_location(
    "embed_analysis", _REPO_ROOT / "representational-analysis" / "embed_analysis.py")
ea = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ea)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


def resolve_single_token_id(tok, word):
    for candidate in (" " + word, word):
        ids = tok(candidate, add_special_tokens=False).input_ids
        if len(ids) == 1:
            return ids[0]
    return None


def choose_added_token_strings(tok, base_words):
    probe = "zzzwugprobezzz"
    context = f"I saw the {probe} today."
    for candidate in (" " + probe, probe):
        test_tok = copy.deepcopy(tok)
        test_tok.add_tokens([candidate])
        pid = test_tok(candidate, add_special_tokens=False).input_ids[0]
        ctx_ids = test_tok(context, add_special_tokens=False).input_ids
        if pid in ctx_ids:
            prefix = " " if candidate.startswith(" ") else ""
            return [prefix + w for w in base_words]
    raise RuntimeError("Could not determine added-token space convention")


def reconstruct_init(cfg, emb, tok, target_norm):
    """Replay embed_train_dev_image.py's init block (embed_init branch)."""
    set_all_seeds(int(cfg["seed"]))
    words = ea_load_lines(cfg["embed_init"])
    init_ids = []
    for i in range(0, len(words), 2):
        sid = resolve_single_token_id(tok, words[i])
        pid = resolve_single_token_id(tok, words[i + 1])
        if (sid is not None and sid < emb.weight.shape[0]
                and pid is not None and pid < emb.weight.shape[0]):
            init_ids.extend([sid, pid])
    pair_distances = [(emb.weight[init_ids[i]] - emb.weight[init_ids[i + 1]]).norm().item()
                      for i in range(0, len(init_ids) - 1, 2)]
    noise_scale = float(np.mean(pair_distances))
    mean_emb_vec = emb.weight[init_ids].float().mean(dim=0)

    if cfg["init_mode"] != "independent":
        raise ValueError(f"unsupported init_mode {cfg['init_mode']}")
    nw = torch.randn_like(mean_emb_vec)
    nw = nw / nw.norm() * noise_scale
    nws = torch.randn_like(mean_emb_vec)
    nws = nws / nws.norm() * noise_scale
    wug_init = (mean_emb_vec + nw) / (mean_emb_vec + nw).norm() * target_norm
    wugs_init = (mean_emb_vec + nws) / (mean_emb_vec + nws).norm() * target_norm
    # Stored in the model dtype during training, so round-trip through it.
    return (wug_init.to(emb.weight.dtype).cpu(), wugs_init.to(emb.weight.dtype).cpu())


def ea_load_lines(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def logged_init_separation(run_root, cfg, run_dir):
    """Find 'Init separation: X' in the sweep log for this run, if present."""
    seed = int(cfg["seed"])
    for log in glob.glob(os.path.join(run_root, "logs", f"*_seed{seed}.log")):
        with open(log) as f:
            m = re.search(r"Init separation: ([0-9.]+)", f.read())
        if m:
            return float(m.group(1))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run_globs", nargs="+", required=True,
                        help="Globs matching run directories (containing learned_embeddings.pt)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cache_dir", default="/home/shared/hf_cache")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    run_dirs = sorted({os.path.dirname(p) for g in args.run_globs
                       for p in glob.glob(os.path.join(g, "learned_embeddings.pt"))})
    print(f"Found {len(run_dirs)} runs")

    lm = load_vlm(args.model, device="cuda", torch_dtype=torch.bfloat16, cache_dir=args.cache_dir)
    tok = lm.tokenizer.tokenizer if hasattr(lm.tokenizer, "tokenizer") else lm.tokenizer
    base_embs = lm.model.get_input_embeddings().weight.detach().cpu().clone()

    # Same token addition / resize as training so target_norm matches.
    added_tokens = choose_added_token_strings(tok, ["[wug]", "[wugs]"])
    to_add = [t for t in added_tokens if t not in tok.get_vocab()]
    if to_add:
        tok.add_tokens(to_add)
        old_len = lm.model.resize_token_embeddings().weight.shape[0]
        lm.model.resize_token_embeddings(old_len + len(to_add))
    emb, _, _ = get_input_output_embeddings(lm.model)
    with torch.no_grad():
        target_norm = emb.weight.norm(dim=1).float().mean().item()

    records = []
    with torch.no_grad():
        for rd in run_dirs:
            cfg = pd.read_csv(os.path.join(rd, "run_config.csv")).iloc[0]
            wug_init, wugs_init = reconstruct_init(cfg, emb, tok, target_norm)
            final = torch.load(os.path.join(rd, "learned_embeddings.pt"), map_location="cpu")
            sep = (wug_init.float() - wugs_init.float()).norm().item()
            run_root = rd.split("/init_small/")[0]
            logged = logged_init_separation(run_root, cfg, rd)
            ok = logged is None or abs(sep - logged) < 0.02
            print(f"  seed {int(cfg['seed']):>6}: init sep {sep:.4f} (log {logged}) "
                  f"{'OK' if ok else 'MISMATCH'}")
            if not ok:
                raise RuntimeError(f"Init reconstruction mismatch for {rd}")
            records.append({
                "seed": int(cfg["seed"]),
                "wug_init": wug_init, "wugs_init": wugs_init,
                "wug_final": final["wug_embedding"], "wugs_final": final["wugs_embedding"],
                "wug_id": final["wug_id"], "wugs_id": final["wugs_id"],
            })

    print("Finding unambiguous noun pairs...")
    pairs = ea.get_unambiguous_nouns(tok, n=2000, min_noun_ratio=0.5)
    singulars = [p["singular"] for p in pairs]
    plurals = [p["plural"] for p in pairs]
    print(f"  Found {len(pairs)} pairs")
    sg_embs = torch.stack([ea.get_embedding(s, base_embs, tok) for s in singulars])
    pl_embs = torch.stack([ea.get_embedding(p, base_embs, tok) for p in plurals])
    combined = torch.cat([sg_embs, pl_embs], dim=0).float().numpy()
    labels = ["sg"] * len(singulars) + ["pl"] * len(plurals)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(combined)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ea.write_sg_pl_csv(str(out_dir / "sg_pl_reduced.csv"), reduced, labels)

    sg_reduced = torch.tensor(reduced[: len(singulars)])
    pl_reduced = torch.tensor(reduced[len(singulars):])

    stack = {k: torch.stack([r[k] for r in records]).float()
             for k in ["wug_init", "wugs_init", "wug_final", "wugs_final"]}
    red = {k: torch.tensor(pca.transform(v.to(torch.float16).numpy())).float()
           for k, v in stack.items()}

    move_wug = (ea.project(red["wug_final"], sg_reduced, pl_reduced)
                - ea.project(red["wug_init"], sg_reduced, pl_reduced))
    move_wugs = (ea.project(red["wugs_init"], sg_reduced, pl_reduced)
                 - ea.project(red["wugs_final"], sg_reduced, pl_reduced))

    ea.write_wug_wugs_csv(str(out_dir / "wug_wugs_reduced_image.csv"),
                          red["wug_init"], red["wug_final"],
                          red["wugs_init"], red["wugs_final"])
    ea.write_movement_csv(str(out_dir / "wug_wugs_movement_image.csv"), move_wug, move_wugs)
    ea.write_neighbors_csv(str(out_dir / "neighbors_image.csv"), records, base_embs, tok, k=args.top_k)
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
