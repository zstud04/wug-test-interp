#!/usr/bin/env python3
"""
Score every completed sweep run's learned_embeddings.pt against the held-out
verb-agreement test set (data/interp/agreement_target_wug.csv), by shelling out
to the unmodified `python3 -m core.eval.embed_eval --paired` for each one.

Usage:
  python3 eacl-rebuttal/scripts/run_test_eval.py --gpus 0,1,2
"""
import argparse
import csv
import glob
import os
import pathlib
import sys

from gpu_pool import run_job_pool

_REPO_ROOT = pathlib.Path(__file__).resolve().parent
while not (_REPO_ROOT / "core").is_dir():
    _parent = _REPO_ROOT.parent
    if _parent == _REPO_ROOT:
        raise RuntimeError(f"Could not find repo root (no core/ dir above {__file__})")
    _REPO_ROOT = _parent

DEFAULT_STIMULI = str(_REPO_ROOT / "data" / "interp" / "agreement_target_wug.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(_REPO_ROOT / "eacl-rebuttal" / "results" / "sweep" / "manifest.csv"))
    parser.add_argument("--stimuli_csv", default=DEFAULT_STIMULI)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--cache_dir", default="/home/shared/hf_cache")
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--out_dir", default=str(_REPO_ROOT / "eacl-rebuttal" / "results" / "test_eval"))
    parser.add_argument("--force", action="store_true", help="Re-score runs even if scored.csv already exists.")
    parser.add_argument("--batch_size", type=int, default=32, help="embed_eval.py scoring batch size (its own default is 8).")
    args = parser.parse_args()

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.manifest) as f:
        manifest_rows = list(csv.DictReader(f))

    jobs = []
    skipped = []
    for row in manifest_rows:
        if str(row.get("returncode")) != "0":
            skipped.append((row["name"], "training run failed"))
            continue
        run_out = out_dir / row["name"]
        if not args.force and (run_out / "scored.csv").is_file():
            skipped.append((row["name"], "already scored"))
            continue
        # embed_train_dev_margin.py nests its real output one level deeper, in an
        # auto-named subfolder of the --out_dir we passed it; glob rather than
        # hand-replicate that naming formula.
        matches = glob.glob(os.path.join(row["run_dir"], "**", "learned_embeddings.pt"), recursive=True)
        if not matches:
            skipped.append((row["name"], f"no learned_embeddings.pt under {row['run_dir']}"))
            continue
        emb_path = matches[0]
        argv = [
            sys.executable, "-m", "core.eval.embed_eval",
            "--model", args.model,
            "--embeddings", emb_path,
            "--stimuli_csv", args.stimuli_csv,
            "--paired",
            "--out_dir", str(run_out),
            "--out_name", "scored.csv",
            "--cache_dir", args.cache_dir,
            "--batch_size", str(args.batch_size),
        ]
        jobs.append({"name": row["name"], "argv": argv})

    print(f"Test-eval: {len(jobs)} runs to score, {len(skipped)} skipped")
    for name, reason in skipped:
        print(f"  SKIP {name}: {reason}")

    if not jobs:
        print("Nothing to score.")
        return

    log_dir = out_dir / "logs"
    results = run_job_pool(jobs, gpu_ids, log_dir=str(log_dir), cwd=str(_REPO_ROOT))

    n_fail = sum(1 for r in results if r["returncode"] != 0)
    if n_fail:
        print(f"WARNING: {n_fail}/{len(jobs)} scoring jobs failed — see logs under {log_dir}")


if __name__ == "__main__":
    main()
