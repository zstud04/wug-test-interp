#!/usr/bin/env python3
"""
Incrementally score sweep training runs as they finish, instead of waiting for
the whole grid. Each run writes its own run_summary.csv/learned_embeddings.pt
as soon as *that* run is done (independent of the sweep as a whole), so this
polls for newly-completed runs, scores each against the held-out test set
(unmodified `core.eval.embed_eval --paired`), and refreshes a running
results/summary_live.csv after every pass. Meant to run on a GPU not used by
run_sweep.py (default: 3) so it doesn't compete with training.

Exits once run_sweep.py's manifest.csv exists (sweep fully done) and every
discovered run has been scored.

Usage:
  python3 eacl-rebuttal/scripts/watch_and_score.py --gpu 3
"""
import argparse
import glob
import os
import pathlib
import re
import subprocess
import sys
import time

import pandas as pd

from aggregate_results import summarize_test_scores  # reuse, don't duplicate the metric

_REPO_ROOT = pathlib.Path(__file__).resolve().parent
while not (_REPO_ROOT / "core").is_dir():
    _parent = _REPO_ROOT.parent
    if _parent == _REPO_ROOT:
        raise RuntimeError(f"Could not find repo root (no core/ dir above {__file__})")
    _REPO_ROOT = _parent

DEFAULT_STIMULI = str(_REPO_ROOT / "data" / "interp" / "agreement_target_wug.csv")
NAME_RE = re.compile(r"init_(?P<init_tag>[^/]+)/lr_(?P<lr_tag>[^/]+)/seed_(?P<seed>\d+)")


def find_completed_runs(sweep_dir):
    """Return {name: {run_dir, emb_path, init_tag, lr, seed}} for every
    training run that has finished (has both run_summary.csv and
    learned_embeddings.pt), keyed the same way run_sweep.py names jobs.
    """
    runs = {}
    for summary_path in glob.glob(os.path.join(sweep_dir, "**", "run_summary.csv"), recursive=True):
        run_row_dir = os.path.dirname(summary_path)
        emb_path = os.path.join(run_row_dir, "learned_embeddings.pt")
        if not os.path.isfile(emb_path):
            continue
        m = NAME_RE.search(run_row_dir)
        if not m:
            continue
        name = f"init_{m['init_tag']}_lr{m['lr_tag']}_seed{m['seed']}"
        runs[name] = {
            "run_dir": run_row_dir, "emb_path": emb_path,
            "init_tag": m["init_tag"], "lr": float(m["lr_tag"].replace("p", ".")),
            "seed": int(m["seed"]),
        }
    return runs


def score_one(name, info, args):
    out_dir = pathlib.Path(args.test_eval_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable, "-m", "core.eval.embed_eval",
        "--model", args.model,
        "--embeddings", info["emb_path"],
        "--stimuli_csv", args.stimuli_csv,
        "--paired",
        "--out_dir", str(out_dir),
        "--out_name", "scored.csv",
        "--cache_dir", args.cache_dir,
        "--batch_size", str(args.batch_size),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu}
    log_path = out_dir / "score.log"
    with open(log_path, "w") as lf:
        ret = subprocess.run(argv, cwd=str(_REPO_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    return ret, str(log_path)


def refresh_summary(runs, args):
    rows = []
    for name, info in runs.items():
        scored_path = os.path.join(args.test_eval_dir, name, "scored.csv")
        if not os.path.isfile(scored_path):
            continue
        train_row = pd.read_csv(os.path.join(info["run_dir"], "run_summary.csv")).iloc[0].to_dict()
        metrics = summarize_test_scores(scored_path)
        rows.append({
            "name": name, "init_tag": info["init_tag"], "lr": info["lr"], "seed": info["seed"],
            "effective_epochs": train_row.get("effective_epochs"),
            "dev_overall_margin": train_row.get("final_overall"),
            **metrics,
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(["init_tag", "lr", "seed"])
    df.to_csv(args.summary_out, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", default=str(_REPO_ROOT / "eacl-rebuttal" / "results" / "sweep"))
    parser.add_argument("--test_eval_dir", default=str(_REPO_ROOT / "eacl-rebuttal" / "results" / "test_eval"))
    parser.add_argument("--summary_out", default=str(_REPO_ROOT / "eacl-rebuttal" / "results" / "summary_live.csv"))
    parser.add_argument("--stimuli_csv", default=DEFAULT_STIMULI)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--cache_dir", default="/home/shared/hf_cache")
    parser.add_argument("--gpu", default="3", help="GPU id to score on (should not overlap run_sweep.py's --gpus)")
    parser.add_argument("--interval", type=float, default=60.0, help="Seconds between polls")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="embed_eval.py scoring batch size (bumped from its default of 8 — "
                             "plenty of GPU headroom, and a single GPU here has to keep pace with "
                             "3 GPUs training in parallel).")
    args = parser.parse_args()

    def status(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    while True:
        runs = find_completed_runs(args.sweep_dir)
        to_score = sorted(n for n in runs if not os.path.isfile(
            os.path.join(args.test_eval_dir, n, "scored.csv")))

        n_newly_scored = 0
        df = None
        for name in to_score:
            ret, log_path = score_one(name, runs[name], args)
            if ret != 0:
                # Failures are rare and worth surfacing immediately, unlike routine progress.
                status(f"FAIL scoring {name} (rc={ret}) — see {log_path}")
                continue
            n_newly_scored += 1
            # Refresh the file after EVERY run, not just once the whole backlog clears
            # — training across 3 GPUs can produce new completions faster than this
            # single GPU can score them, so the backlog may never fully empty. This is
            # silent (no status()/print) on purpose: only one status line per pass,
            # not per run, or this would emit ~140 notifications over the sweep.
            df = refresh_summary(runs, args)

        if df is None:
            df = refresh_summary(runs, args)
        if df is None:
            status(f"no scored runs yet ({len(runs)} training runs completed so far)")
        else:
            best = df.loc[df["test_acc_pooled_overall"].idxmax()]
            status(f"{n_newly_scored} newly scored this pass | {len(df)}/{len(runs)} scored total | "
                   f"mean pooled acc={df['test_acc_pooled_overall'].mean():.3f} | "
                   f"best={best['test_acc_pooled_overall']:.3f} ({best['name']})")

        manifest_path = os.path.join(args.sweep_dir, "manifest.csv")
        sweep_done = os.path.isfile(manifest_path)
        if sweep_done and not to_score:
            status(f"sweep manifest present and all {len(runs)} completed runs are scored — done.")
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
