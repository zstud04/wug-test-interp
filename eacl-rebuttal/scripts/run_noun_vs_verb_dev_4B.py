#!/usr/bin/env python3
"""Does dev-selection metric (noun-vs-verb POS margin, vs. held-out-image
number recognition) change whether 4B's plural-under-attractor collapse
appears? Same seeds/LR/batch config/images/captions as the held-out-image
4B runs; only the dev script + dev CSV differ."""
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gpu_pool import run_job_pool

REPO_ROOT = pathlib.Path(__file__).resolve().parent
while not (REPO_ROOT / "core").is_dir():
    REPO_ROOT = REPO_ROOT.parent

SEEDS = [17, 42, 123, 2024, 98765]
LR = 0.01
MODEL = "Qwen/Qwen3-VL-4B-Instruct"
IMAGE_DIR = REPO_ROOT / "eacl-rebuttal" / "data" / "train" / "im" / "creature_1"
TRAIN_CSV = REPO_ROOT / "eacl-rebuttal" / "data" / "train" / "text" / "image_train_matched_v2.csv"
EVAL_CSV = REPO_ROOT / "eacl-rebuttal" / "data" / "dev" / "dev_noun_vs_verb.csv"
EMBED_INIT = REPO_ROOT / "data" / "embeddings" / "init" / "noun_init.txt"
TRAIN_SCRIPT = REPO_ROOT / "eacl-rebuttal" / "scripts" / "embed_train_dev_margin.py"
OUT_DIR = REPO_ROOT / "eacl-rebuttal" / "results" / "noun_vs_verb_dev_4B_bs8"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    manifest_rows = []

    for seed in SEEDS:
        name = f"nounverb_bs8_seed{seed}"
        run_out = OUT_DIR / name
        argv = [
            sys.executable, str(TRAIN_SCRIPT),
            "--training_condition", "image",
            "--train_csv", str(TRAIN_CSV),
            "--eval_csv", str(EVAL_CSV),
            "--image_dir", str(IMAGE_DIR),
            "--embed_init", str(EMBED_INIT),
            "--lr", str(LR),
            "--seed", str(seed),
            "--epochs", "50",
            "--terminate_cond_epochs", "5",
            "--batch_mode", "joint",
            "--batch_size", "8",
            "--model", MODEL,
            "--cache_dir", "/home/shared/hf_cache",
            "--out_dir", str(run_out),
            "--write_embeddings",
            "--write_all",
        ]
        jobs.append({"name": name, "argv": argv})
        manifest_rows.append({
            "name": name, "init_tag": "small", "init_path": str(EMBED_INIT),
            "lr": LR, "seed": seed, "separation_scale": "",
            "run_dir": str(run_out),
        })

    print(f"Noun-vs-verb dev diagnostic: {len(jobs)} seeds, lr={LR}, bs=8 joint")
    results = run_job_pool(jobs, gpu_ids=[0, 1, 2, 3], log_dir=str(OUT_DIR / "logs"), cwd=str(REPO_ROOT))

    ret_by_name = {r["name"]: r["returncode"] for r in results}
    for row in manifest_rows:
        row["returncode"] = ret_by_name[row["name"]]

    manifest_path = OUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"Wrote manifest: {manifest_path}")

    n_fail = sum(1 for r in manifest_rows if r["returncode"] != 0)
    if n_fail:
        print(f"WARNING: {n_fail}/{len(manifest_rows)} runs failed")


if __name__ == "__main__":
    main()
