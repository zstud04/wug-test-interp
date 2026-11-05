#!/usr/bin/env python3
"""Preview-score whichever seeds of the 45-more-seeds Gemma3 spherical run
have finished training so far, without waiting for the last one."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gpu_pool import run_job_pool

REPO_ROOT = pathlib.Path(__file__).resolve().parent
while not (REPO_ROOT / "core").is_dir():
    REPO_ROOT = REPO_ROOT.parent

RUN_DIR = REPO_ROOT / "eacl-rebuttal" / "results" / "spherical_symmetric_gemma3_4b_sep1_more45" / "init_small" / "lr_0p003_joint_bs4_sep1"
OUT_DIR = REPO_ROOT / "eacl-rebuttal" / "results" / "spherical_symmetric_gemma3_4b_sep1_more45" / "test_eval_preview"
MODEL = "google/gemma-3-4b-it"
STIMULI = REPO_ROOT / "data" / "interp" / "agreement_target_wug.csv"
CACHE_DIR = "/home/km55359/.cache/hf_gemma3"

with open("/tmp/gemma3_45_done_seeds.txt") as f:
    seeds = [int(l.strip()) for l in f if l.strip()]

jobs = []
for seed in seeds:
    matches = list((RUN_DIR / f"seed_{seed}").glob("**/learned_embeddings.pt"))
    if not matches:
        continue
    name = f"seed{seed}"
    argv = [
        sys.executable, "-m", "core.eval.embed_eval",
        "--model", MODEL,
        "--embeddings", str(matches[0]),
        "--stimuli_csv", str(STIMULI),
        "--paired",
        "--out_dir", str(OUT_DIR / name),
        "--out_name", "scored.csv",
        "--cache_dir", CACHE_DIR,
        "--batch_size", "16",
    ]
    jobs.append({"name": name, "argv": argv})

print(f"Scoring {len(jobs)} completed checkpoints")
run_job_pool(jobs, gpu_ids=[0, 1, 3], log_dir=str(OUT_DIR / "logs"), cwd=str(REPO_ROOT))
