#!/usr/bin/env python3
"""Cross-check the singular/plural-under-attractor pattern against the older,
independently-built attractor_stimuli.csv (copula 'is/are' + stacked PP
distractors), instead of the agreement_target_wug.csv set used everywhere
else, to rule out the pattern being an artifact of that one stimulus set."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gpu_pool import run_job_pool

REPO_ROOT = pathlib.Path(__file__).resolve().parent
while not (REPO_ROOT / "core").is_dir():
    REPO_ROOT = REPO_ROOT.parent

STIMULI = REPO_ROOT / "data" / "eval" / "attractors" / "attractor_stimuli.csv"
OUT_DIR = REPO_ROOT / "eacl-rebuttal" / "results" / "deprecated_attractor_crosscheck"

SEEDS = [17, 42, 123, 2024, 98765]

CONFIGS = [
    {
        "tag": "qwen4b_bs4",
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "cache_dir": "/home/shared/hf_cache",
        "ckpt_tmpl": str(REPO_ROOT / "eacl-rebuttal" / "results" / "lr_search_4B_bs4" /
                          "init_small" / "lr_0p0075_joint_bs4" / "seed_{seed}"),
    },
    {
        "tag": "gemma3_4b_sep1",
        "model": "google/gemma-3-4b-it",
        "cache_dir": "/home/km55359/.cache/hf_gemma3",
        "ckpt_tmpl": str(REPO_ROOT / "eacl-rebuttal" / "results" / "spherical_symmetric_gemma3_4b_bs4" /
                          "init_small" / "lr_0p003_joint_bs4_sep1" / "seed_{seed}"),
    },
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=[c["tag"] for c in CONFIGS])
    args = ap.parse_args()
    configs = [c for c in CONFIGS if c["tag"] == args.tag]

    jobs = []
    for cfg in configs:
        for seed in SEEDS:
            run_dir = pathlib.Path(cfg["ckpt_tmpl"].format(seed=seed))
            matches = list(run_dir.glob("**/learned_embeddings.pt"))
            if not matches:
                print(f"MISSING checkpoint for {cfg['tag']} seed {seed}: {run_dir}")
                continue
            name = f"{cfg['tag']}_seed{seed}"
            argv = [
                sys.executable, "-m", "core.eval.embed_eval",
                "--model", cfg["model"],
                "--embeddings", str(matches[0]),
                "--stimuli_csv", str(STIMULI),
                "--out_dir", str(OUT_DIR / name),
                "--out_name", "scored.csv",
                "--cache_dir", cfg["cache_dir"],
                "--batch_size", "16",
            ]
            jobs.append({"name": name, "argv": argv})

    print(f"Scoring {len(jobs)} checkpoints against the deprecated attractor set")
    run_job_pool(jobs, gpu_ids=[0, 1, 2, 3], log_dir=str(OUT_DIR / "logs"), cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
