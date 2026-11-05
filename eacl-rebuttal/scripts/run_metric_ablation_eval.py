#!/usr/bin/env python3
"""Select and score raw-margin vs balanced-logistic checkpoints.

This expects full, --write_all training trajectories produced by run_sweep.py.
For each run, it independently simulates patience-based early stopping for:

  raw_margin: mean signed logit gap (the current held-out-image metric)
  logistic:   negative class-balanced logistic loss over signed gaps

Only the dev-selected epoch for each metric is evaluated on the held-out test
set. Test results therefore never participate in checkpoint selection.
"""
import argparse
import math
import os
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import t

from aggregate_results import summarize_test_scores
from gpu_pool import run_job_pool


_REPO_ROOT = pathlib.Path(__file__).resolve().parent
while not (_REPO_ROOT / "core").is_dir():
    parent = _REPO_ROOT.parent
    if parent == _REPO_ROOT:
        raise RuntimeError("Could not locate repository root")
    _REPO_ROOT = parent


def select_with_patience(epoch_scores, score_col, patience, eps=1e-6):
    best_score = -float("inf")
    best_epoch = None
    no_improve = 0
    stop_epoch = int(epoch_scores["epoch"].max())
    for row in epoch_scores.sort_values("epoch").itertuples(index=False):
        score = float(getattr(row, score_col))
        if score > best_score + eps:
            best_score = score
            best_epoch = int(row.epoch)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                stop_epoch = int(row.epoch)
                break
    return best_epoch, best_score, stop_epoch


def summarize_dev(eval_csv):
    items = pd.read_csv(eval_csv)
    rows = []
    for epoch, group in items.groupby("epoch", sort=True):
        singular = group[group["kind"] == "singular"]
        plural = group[group["kind"] == "plural"]
        if singular.empty or plural.empty:
            raise ValueError(f"Epoch {epoch} lacks one dev class in {eval_csv}")

        sing_gap = singular["signed_gap"].to_numpy(dtype=float)
        plur_gap = plural["signed_gap"].to_numpy(dtype=float)
        sing_logistic = np.logaddexp(0.0, -sing_gap).mean()
        plur_logistic = np.logaddexp(0.0, -plur_gap).mean()
        rows.append({
            "epoch": int(epoch),
            "raw_score": float(group["signed_gap"].mean()),
            "logistic_score": -0.5 * float(sing_logistic + plur_logistic),
            "sing_logistic_loss": float(sing_logistic),
            "plur_logistic_loss": float(plur_logistic),
            "sing_accuracy": float(singular["correct"].mean()),
            "plur_accuracy": float(plural["correct"].mean()),
            "macro_accuracy": 0.5 * float(
                singular["correct"].mean() + plural["correct"].mean()),
        })
    return pd.DataFrame(rows)


def parse_seed(path):
    match = re.search(r"/seed_(\d+)/", str(path))
    if not match:
        raise ValueError(f"Could not parse seed from {path}")
    return int(match.group(1))


def parse_lr(path):
    match = re.search(r"/lr_([0-9p]+)_joint_bs\d+/", str(path))
    if not match:
        raise ValueError(f"Could not parse learning rate from {path}")
    return float(match.group(1).replace("p", "."))


def lr_tag(lr):
    return f"{lr:.10f}".rstrip("0").rstrip(".").replace(".", "p")


def test_metrics(scored_csv):
    df = pd.read_csv(scored_csv)
    result = summarize_test_scores(scored_csv)
    result["test_acc_singular"] = float(df["is_correct_singular"].mean())
    result["test_acc_plural"] = float(df["is_correct_plural"].mean())
    result["test_acc_worst_class"] = min(
        result["test_acc_singular"], result["test_acc_plural"])
    for condition, group in df.groupby("condition"):
        att = re.search(r"att(\d+)", condition).group(1)
        result[f"test_acc_singular_att{att}"] = float(group["is_correct_singular"].mean())
        result[f"test_acc_plural_att{att}"] = float(group["is_correct_plural"].mean())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", required=True, action="append",
                        help="Training sweep directory; repeat to combine disjoint LR sweeps.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--cache_dir", default="/home/shared/hf_cache")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--stimuli_csv", default=str(
        _REPO_ROOT / "data" / "interp" / "agreement_target_wug.csv"))
    args = parser.parse_args()

    sweep_dirs = [pathlib.Path(path).resolve() for path in args.sweep_dir]
    out_dir = pathlib.Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_files = sorted(
        path
        for sweep_dir in sweep_dirs
        for path in sweep_dir.glob("init_small/lr_*/seed_*/**/eval_item_scores.csv")
    )
    if not eval_files:
        raise FileNotFoundError(f"No eval_item_scores.csv files found under {sweep_dirs}")

    selections = []
    for eval_csv in eval_files:
        seed = parse_seed(eval_csv)
        lr = parse_lr(eval_csv)
        epoch_scores = summarize_dev(eval_csv)
        run_dir = eval_csv.parent
        for metric, score_col in [("raw_margin", "raw_score"),
                                  ("balanced_logistic", "logistic_score")]:
            best_epoch, best_score, stop_epoch = select_with_patience(
                epoch_scores, score_col, args.patience)
            epoch_row = epoch_scores.loc[epoch_scores["epoch"] == best_epoch].iloc[0]
            emb_path = run_dir / "epoch_embs" / f"epoch_{best_epoch:03d}.pt"
            if not emb_path.is_file():
                raise FileNotFoundError(f"Missing selected epoch embedding: {emb_path}")
            selections.append({
                "lr": lr,
                "seed": seed,
                "selection_metric": metric,
                "best_epoch": best_epoch,
                "stop_epoch": stop_epoch,
                "dev_score": best_score,
                "dev_raw_score": epoch_row["raw_score"],
                "dev_logistic_score": epoch_row["logistic_score"],
                "dev_sing_logistic_loss": epoch_row["sing_logistic_loss"],
                "dev_plur_logistic_loss": epoch_row["plur_logistic_loss"],
                "dev_sing_accuracy": epoch_row["sing_accuracy"],
                "dev_plur_accuracy": epoch_row["plur_accuracy"],
                "dev_macro_accuracy": epoch_row["macro_accuracy"],
                "embeddings": str(emb_path),
            })

    selections_df = pd.DataFrame(selections).sort_values(["selection_metric", "lr", "seed"])
    selections_df.to_csv(out_dir / "selections.csv", index=False)

    dev_lr_summary = selections_df.groupby(
        ["selection_metric", "lr"]
    ).agg(
        dev_score_mean=("dev_score", "mean"),
        dev_score_std=("dev_score", "std"),
        dev_sing_accuracy=("dev_sing_accuracy", "mean"),
        dev_plur_accuracy=("dev_plur_accuracy", "mean"),
        mean_best_epoch=("best_epoch", "mean"),
        n=("seed", "size"),
    ).reset_index()
    dev_lr_summary.to_csv(out_dir / "dev_lr_summary.csv", index=False)

    jobs = []
    for row in selections:
        name = f"{row['selection_metric']}_lr{lr_tag(row['lr'])}_seed{row['seed']}"
        run_out = out_dir / "test_eval" / name
        scored = run_out / "scored.csv"
        if scored.is_file():
            continue
        jobs.append({"name": name, "argv": [
            sys.executable, "-m", "core.eval.embed_eval",
            "--model", args.model,
            "--embeddings", row["embeddings"],
            "--stimuli_csv", args.stimuli_csv,
            "--paired",
            "--out_dir", str(run_out),
            "--out_name", "scored.csv",
            "--cache_dir", args.cache_dir,
            "--batch_size", str(args.batch_size),
        ]})

    gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if jobs:
        results = run_job_pool(
            jobs, gpu_ids, log_dir=str(out_dir / "test_eval" / "logs"),
            cwd=str(_REPO_ROOT))
        failed = [r for r in results if r["returncode"] != 0]
        if failed:
            raise RuntimeError(f"{len(failed)}/{len(results)} test-eval jobs failed")

    per_run = []
    for row in selections:
        scored = (out_dir / "test_eval" /
                  f"{row['selection_metric']}_lr{lr_tag(row['lr'])}_seed{row['seed']}" /
                  "scored.csv")
        if not scored.is_file():
            raise FileNotFoundError(f"Missing test result: {scored}")
        per_run.append({**row, **test_metrics(scored)})
    per_run_df = pd.DataFrame(per_run).sort_values(["selection_metric", "lr", "seed"])
    per_run_df.to_csv(out_dir / "per_run_results.csv", index=False)

    metrics = [
        "test_acc_pooled_overall", "test_acc_singular", "test_acc_plural",
        "test_acc_worst_class",
        *[f"test_acc_singular_att{i}" for i in range(4)],
        *[f"test_acc_plural_att{i}" for i in range(4)],
    ]
    summary_rows = []
    for (selection_metric, lr), group in per_run_df.groupby(["selection_metric", "lr"]):
        for metric in metrics:
            values = group[metric].dropna()
            n = len(values)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if n > 1 else float("nan")
            half = float(t.ppf(0.975, n - 1) * std / math.sqrt(n)) if n > 1 else float("nan")
            summary_rows.append({
                "selection_metric": selection_metric,
                "lr": lr,
                "test_metric": metric,
                "n": n,
                "mean": mean,
                "std": std,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    print("\nSelected checkpoints:")
    print(selections_df[["selection_metric", "lr", "seed", "best_epoch", "stop_epoch",
                         "dev_sing_accuracy", "dev_plur_accuracy"]].to_string(index=False))
    print("\nDev LR ranking:")
    print(dev_lr_summary.sort_values(
        ["selection_metric", "dev_score_mean"], ascending=[True, False]
    ).to_string(index=False))
    print("\nTest comparison (mean across seeds):")
    print(per_run_df.groupby(["selection_metric", "lr"])[metrics[:4]].mean().to_string())
    print(f"\nWrote results under {out_dir}")


if __name__ == "__main__":
    main()
