#!/usr/bin/env python3
"""Single-seed diagnostic: does the plural-collapse-below-chance pattern seen
at batch_size=8 (both independent and spherical_symmetric@1.5 init) depend on
batch_size? Sweeps batch_size x init_mode at a fixed seed/LR so results are
qualitatively comparable, not a CI-producing run.
"""
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gpu_pool import run_job_pool

REPO_ROOT = pathlib.Path(__file__).resolve().parent
while not (REPO_ROOT / "core").is_dir():
    REPO_ROOT = REPO_ROOT.parent

SEED = 17
LR = 0.01
BATCH_SIZES = [1, 2, 4, 8, 16]
INIT_CONDITIONS = [
    ("independent", None),
    ("spherical_symmetric", 1.5),
]
MODEL = "Qwen/Qwen3-VL-4B-Instruct"
IMAGE_DIR = REPO_ROOT / "eacl-rebuttal" / "data" / "train" / "im" / "creature_1"
TRAIN_CSV = REPO_ROOT / "eacl-rebuttal" / "data" / "train" / "text" / "image_train_matched_v2.csv"
EVAL_CSV = REPO_ROOT / "eacl-rebuttal" / "data" / "dev" / "dev_image_number.csv"
EMBED_INIT = REPO_ROOT / "data" / "embeddings" / "init" / "noun_init.txt"
TRAIN_SCRIPT = REPO_ROOT / "eacl-rebuttal" / "scripts" / "embed_train_dev_image.py"
OUT_DIR = REPO_ROOT / "eacl-rebuttal" / "results" / "batch_init_diagnostic_4B"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    manifest_rows = []

    for init_mode, scale in INIT_CONDITIONS:
        scale_tag = "" if scale is None else f"_sep{str(scale).replace('.', 'p')}"
        for bs in BATCH_SIZES:
            name = f"{init_mode}{scale_tag}_bs{bs}_seed{SEED}"
            run_out = OUT_DIR / name
            argv = [
                sys.executable, str(TRAIN_SCRIPT),
                "--training_condition", "image",
                "--train_csv", str(TRAIN_CSV),
                "--eval_csv", str(EVAL_CSV),
                "--image_dir", str(IMAGE_DIR),
                "--embed_init", str(EMBED_INIT),
                "--lr", str(LR),
                "--seed", str(SEED),
                "--epochs", "50",
                "--terminate_cond_epochs", "5",
                "--batch_mode", "joint",
                "--batch_size", str(bs),
                "--model", MODEL,
                "--cache_dir", "/home/shared/hf_cache",
                "--out_dir", str(run_out),
                "--init_mode", init_mode,
                "--selection_metric", "balanced_logistic",
                "--write_embeddings",
                "--write_all",
            ]
            if scale is not None:
                argv.extend(["--symmetric_separation_scale", str(scale)])
            jobs.append({"name": name, "argv": argv})
            manifest_rows.append({
                "name": name, "init_tag": "small", "init_path": str(EMBED_INIT),
                "lr": LR, "seed": SEED,
                "separation_scale": scale if scale is not None else "",
                "batch_size": bs, "init_mode": init_mode,
                "run_dir": str(run_out),
            })

    print(f"Diagnostic: {len(jobs)} jobs ({len(INIT_CONDITIONS)} init conditions x "
          f"{len(BATCH_SIZES)} batch sizes), seed={SEED}, lr={LR}")
    results = run_job_pool(jobs, gpu_ids=[0, 1, 2, 3], log_dir=str(OUT_DIR / "logs"), cwd=str(REPO_ROOT))

    ret_by_name = {r["name"]: r["returncode"] for r in results}
    for row in manifest_rows:
        row["returncode"] = ret_by_name[row["name"]]

    manifest_path = OUT_DIR / "manifest.csv"
    fieldnames = list(manifest_rows[0].keys())
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"Wrote manifest: {manifest_path}")

    n_fail = sum(1 for r in manifest_rows if r["returncode"] != 0)
    if n_fail:
        print(f"WARNING: {n_fail}/{len(manifest_rows)} runs failed")


if __name__ == "__main__":
    main()
