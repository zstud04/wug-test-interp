#!/usr/bin/env python3
"""Score several epoch checkpoints (not just the dev-selected best) from the
no-early-stop 4B trajectory diagnostic, to see if plural-under-attractor
accuracy keeps improving after the held-out-image dev metric has saturated."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gpu_pool import run_job_pool

REPO_ROOT = pathlib.Path(__file__).resolve().parent
while not (REPO_ROOT / "core").is_dir():
    REPO_ROOT = REPO_ROOT.parent

SEEDS = [17, 42]
EPOCHS = [5, 10, 15, 20, 25, 30, 40, 50]
MODEL = "Qwen/Qwen3-VL-4B-Instruct"
STIMULI = REPO_ROOT / "data" / "interp" / "agreement_target_wug.csv"
OUT_DIR = REPO_ROOT / "eacl-rebuttal" / "results" / "epoch_trajectory_diagnostic_4B"


def main():
    jobs = []
    for seed in SEEDS:
        run_dir = OUT_DIR / f"seed_{seed}" / f"creature_1_joint_lr0p01_seed{seed}_ep50"
        for epoch in EPOCHS:
            emb_path = run_dir / "epoch_embs" / f"epoch_{epoch:03d}.pt"
            name = f"seed{seed}_epoch{epoch:03d}"
            run_out = OUT_DIR / "test_eval" / name
            argv = [
                sys.executable, "-m", "core.eval.embed_eval",
                "--model", MODEL,
                "--embeddings", str(emb_path),
                "--stimuli_csv", str(STIMULI),
                "--paired",
                "--out_dir", str(run_out),
                "--out_name", "scored.csv",
                "--cache_dir", "/home/shared/hf_cache",
                "--batch_size", "32",
            ]
            jobs.append({"name": name, "argv": argv})

    print(f"Scoring {len(jobs)} checkpoints ({len(SEEDS)} seeds x {len(EPOCHS)} epochs)")
    run_job_pool(jobs, gpu_ids=[0, 1, 2, 3], log_dir=str(OUT_DIR / "test_eval" / "logs"), cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
