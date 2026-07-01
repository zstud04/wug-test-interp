#!/usr/bin/env python3
"""
Movement analysis and nearest-neighbor computation for learned wug/wugs embeddings.

For each condition (text, image) and model:
  - Loads initial and learned embeddings for all seeds
  - Fits PCA on singular/plural noun embeddings
  - Projects wug/wugs embeddings into PCA space
  - Computes movement along the singular-plural axis
  - Computes top-K cosine nearest neighbors of the learned embeddings
  - Writes CSVs to --out_dir

Output files (per condition):
  wug_wugs_reduced_{text,image}.csv  — PCA positions (init + final, per seed)
  wug_wugs_movement_{text,image}.csv — axis movement per seed
  neighbors_{text,image}.csv         — top-K neighbors for [wug] and [wugs]

And once per model:
  sg_pl_reduced.csv                  — PCA positions of all reference nouns
"""
import argparse
import csv
import os
import pathlib

import inflect
import numpy as np
import torch
from minicons import scorer
from nltk.corpus import wordnet as wn
from sklearn.decomposition import PCA
from wordfreq import word_frequency


# ---------------------------------------------------------------------------
# Noun-pair helpers
# ---------------------------------------------------------------------------

def _is_single_token(word, tok):
    ids = tok(f" {word}", add_special_tokens=False)["input_ids"]
    return len(ids) == 1


def get_unambiguous_nouns(tok, n=2000, min_noun_ratio=0.5):
    p = inflect.engine()
    banned = {"physics", "subs", "roma", "pois", "arts", "tors", "ours",
              "autos", "jeans", "aras", "cocos"}
    target_categories = {
        "noun.animal", "noun.artifact", "noun.object", "noun.plant", "noun.food"
    }

    candidates = []
    for lemma_name in wn.all_lemma_names(pos=wn.NOUN):
        lemma_name = lemma_name.lower()
        if "_" in lemma_name or "-" in lemma_name or not lemma_name.isalpha():
            continue
        if lemma_name[0].isupper():
            continue

        all_senses = wn.synsets(lemma_name)
        noun_senses = [s for s in all_senses if s.pos() == "n"]
        if not all_senses or len(noun_senses) / len(all_senses) < min_noun_ratio:
            continue

        primary = noun_senses[0]
        if primary.lexname() not in target_categories:
            continue

        if p.singular_noun(lemma_name) is not False:
            continue  # already plural

        plural_form = p.plural(lemma_name)
        if lemma_name == plural_form:
            continue
        if plural_form in banned or len(lemma_name) <= 2:
            continue
        if not (_is_single_token(lemma_name, tok) and _is_single_token(plural_form, tok)):
            continue

        candidates.append({
            "singular": lemma_name,
            "plural": plural_form,
            "freq": word_frequency(lemma_name, "en"),
        })

    candidates.sort(key=lambda x: x["freq"], reverse=True)
    return candidates[:n]


def get_embedding(word, embs, tok):
    token_id = tok(f" {word}", add_special_tokens=False)["input_ids"][0]
    return embs[token_id]


# ---------------------------------------------------------------------------
# Projection / movement
# ---------------------------------------------------------------------------

def project(queries, sg, pl):
    diff_vec = sg.mean(0) - pl.mean(0)
    return queries @ diff_vec / torch.linalg.vector_norm(diff_vec)


# ---------------------------------------------------------------------------
# Nearest-neighbor helpers
# ---------------------------------------------------------------------------

def cosine_topk(query, emb_matrix, k=5, exclude_ids=None):
    """Return top-k token indices by cosine similarity (float32)."""
    q = query.float()
    q = q / q.norm().clamp(min=1e-8)
    m = emb_matrix.float()
    norms = m.norm(dim=1, keepdim=True).clamp(min=1e-8)
    m_normed = m / norms
    sims = m_normed @ q
    if exclude_ids:
        for idx in exclude_ids:
            sims[idx] = -1e9
    return sims.topk(k).indices.tolist()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_seed_embeddings(results_dir):
    """Return list of dicts with seed, wug/wugs init+final embeddings."""
    prefix = "syntax" if "text" in results_dir else ""
    records = []
    for folder in sorted(os.listdir(results_dir)):
        if not folder.startswith("seed_"):
            continue
        seed = int(folder.split("_")[1])
        run_dir = os.path.join(results_dir, folder,
                               f"{prefix}_alternating_lr0p001_seed{seed}_ep50")
        init_path = os.path.join(run_dir, "initial_embeddings.pt")
        final_path = os.path.join(run_dir, "learned_embeddings.pt")
        if not os.path.exists(init_path):
            print(f"  WARNING: missing initial_embeddings.pt for seed {seed}, skipping")
            continue
        if not os.path.exists(final_path):
            print(f"  WARNING: missing learned_embeddings.pt for seed {seed}, skipping")
            continue
        init_data = torch.load(init_path, map_location="cpu")
        final_data = torch.load(final_path, map_location="cpu")
        records.append({
            "seed": seed,
            "wug_init": init_data["wug_embedding"],
            "wugs_init": init_data["wugs_embedding"],
            "wug_final": final_data["wug_embedding"],
            "wugs_final": final_data["wugs_embedding"],
            "wug_id": final_data["wug_id"],
            "wugs_id": final_data["wugs_id"],
        })
    return records


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_sg_pl_csv(path, reduced, labels):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "label"])
        for (x, y), label in zip(reduced, labels):
            w.writerow([x, y, label])


def write_wug_wugs_csv(path, wug_init, wug_final, wugs_init, wugs_final):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "type", "stage", "pair_id"])
        for i, (ini, fin) in enumerate(zip(wug_init, wug_final)):
            w.writerow([ini[0].item(), ini[1].item(), "wug", "init", i])
            w.writerow([fin[0].item(), fin[1].item(), "wug", "final", i])
        for i, (ini, fin) in enumerate(zip(wugs_init, wugs_final)):
            w.writerow([ini[0].item(), ini[1].item(), "wugs", "init", i])
            w.writerow([fin[0].item(), fin[1].item(), "wugs", "final", i])


def write_movement_csv(path, wug_movement, wugs_movement):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["movement", "type", "number"])
        for m in wug_movement:
            w.writerow([m.item(), "wug", "sg"])
        for m in wugs_movement:
            w.writerow([m.item(), "wugs", "pl"])


def write_neighbors_csv(path, records, emb_matrix, tok, k=5):
    """
    CSV columns: seed, wug_neighbors, wugs_neighbors
    Neighbors are the top-k token strings by cosine similarity, separated by '|'.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "wug_neighbors", "wugs_neighbors"])
        for rec in records:
            exclude = [rec["wug_id"], rec["wugs_id"]]
            wug_ids = cosine_topk(rec["wug_final"], emb_matrix, k=k, exclude_ids=exclude)
            wugs_ids = cosine_topk(rec["wugs_final"], emb_matrix, k=k, exclude_ids=exclude)
            wug_tokens = [tok.decode([i]).strip() for i in wug_ids]
            wugs_tokens = [tok.decode([i]).strip() for i in wugs_ids]
            w.writerow([rec["seed"], "|".join(wug_tokens), "|".join(wugs_tokens)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text_dir", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    print(f"Loading model: {args.model}")
    lm = scorer.VLMScorer(args.model, device=args.device, token=hf_token)
    tok = lm.tokenizer.tokenizer
    # Base embedding matrix (original vocab, before any token additions)
    embs = lm.model.model.language_model.embed_tokens.weight.detach().cpu()

    print("Finding unambiguous noun pairs...")
    pairs = get_unambiguous_nouns(tok, n=2000, min_noun_ratio=0.5)
    singulars = [p["singular"] for p in pairs]
    plurals = [p["plural"] for p in pairs]
    print(f"  Found {len(pairs)} pairs")

    sg_embs = torch.stack([get_embedding(s, embs, tok) for s in singulars])
    pl_embs = torch.stack([get_embedding(p, embs, tok) for p in plurals])

    combined = torch.cat([sg_embs, pl_embs], dim=0).float().numpy()
    combined_labels = ["sg"] * len(singulars) + ["pl"] * len(plurals)

    print("Fitting PCA...")
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(combined)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_sg_pl_csv(str(out_dir / "sg_pl_reduced.csv"), reduced, combined_labels)
    print(f"Saved sg_pl_reduced.csv ({len(combined_labels)} points)")

    sg_reduced = torch.tensor(reduced[: len(singulars)])
    pl_reduced = torch.tensor(reduced[len(singulars):])

    for condition, results_dir in [("text", args.text_dir), ("image", args.image_dir)]:
        print(f"\n--- Condition: {condition} ({results_dir}) ---")
        records = load_seed_embeddings(results_dir)
        if not records:
            print("  No complete records found, skipping.")
            continue
        print(f"  Loaded {len(records)} seeds")

        wug_init_stack = torch.stack([r["wug_init"] for r in records]).float()
        wugs_init_stack = torch.stack([r["wugs_init"] for r in records]).float()
        wug_final_stack = torch.stack([r["wug_final"] for r in records]).float()
        wugs_final_stack = torch.stack([r["wugs_final"] for r in records]).float()

        sg_f = sg_embs.float()
        pl_f = pl_embs.float()
        emb_movement_wug = (project(wug_final_stack, sg_f, pl_f)
                            - project(wug_init_stack, sg_f, pl_f))
        emb_movement_wugs = (project(wugs_init_stack, sg_f, pl_f)
                             - project(wugs_final_stack, sg_f, pl_f))

        # PCA projection (pca was fit on float32 converted from bfloat16 in notebook;
        # use float16 intermediate to match notebook's pca.transform call)
        wug_reduced_init = torch.tensor(
            pca.transform(wug_init_stack.to(torch.float16).numpy())).float()
        wugs_reduced_init = torch.tensor(
            pca.transform(wugs_init_stack.to(torch.float16).numpy())).float()
        wug_reduced_final = torch.tensor(
            pca.transform(wug_final_stack.to(torch.float16).numpy())).float()
        wugs_reduced_final = torch.tensor(
            pca.transform(wugs_final_stack.to(torch.float16).numpy())).float()

        pca_movement_wug = (project(wug_reduced_final, sg_reduced, pl_reduced)
                            - project(wug_reduced_init, sg_reduced, pl_reduced))
        pca_movement_wugs = (project(wugs_reduced_init, sg_reduced, pl_reduced)
                             - project(wugs_reduced_final, sg_reduced, pl_reduced))

        write_wug_wugs_csv(
            str(out_dir / f"wug_wugs_reduced_{condition}.csv"),
            wug_reduced_init, wug_reduced_final,
            wugs_reduced_init, wugs_reduced_final,
        )
        write_movement_csv(
            str(out_dir / f"wug_wugs_movement_{condition}.csv"),
            pca_movement_wug, pca_movement_wugs,
        )
        write_neighbors_csv(
            str(out_dir / f"neighbors_{condition}.csv"),
            records, embs, tok, k=args.top_k,
        )
        print(f"  Wrote wug_wugs_reduced_{condition}.csv")
        print(f"  Wrote wug_wugs_movement_{condition}.csv")
        print(f"  Wrote neighbors_{condition}.csv")

    print("\nDone.")


if __name__ == "__main__":
    main()
