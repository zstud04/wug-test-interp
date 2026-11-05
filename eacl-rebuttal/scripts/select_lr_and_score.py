#!/usr/bin/env python3
"""Select one LR per init from dev scores, then score only those runs on test."""

import argparse
import csv
import glob
import math
import pathlib
import re
import subprocess
import sys

import pandas as pd
from scipy.stats import t


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE
while not (REPO_ROOT / "core").is_dir():
    REPO_ROOT = REPO_ROOT.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--cache_dir", default="/home/shared/hf_cache")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--stimuli_csv", default=str(
        REPO_ROOT / "data" / "interp" / "agreement_target_wug.csv"))
    args = ap.parse_args()

    sweep_dir = pathlib.Path(args.sweep_dir).resolve()
    with open(sweep_dir / "manifest.csv", newline="") as f:
        manifest = list(csv.DictReader(f))

    failed = [row["name"] for row in manifest if str(row.get("returncode")) != "0"]
    if failed:
        raise RuntimeError(f"Refusing selection with {len(failed)} failed runs: {failed}")

    dev_rows = []
    manifest_by_name = {row["name"]: row for row in manifest}
    for row in manifest:
        if str(row.get("returncode")) != "0":
            continue
        matches = glob.glob(str(pathlib.Path(row["run_dir"]) / "**" / "run_summary.csv"),
                            recursive=True)
        if len(matches) != 1:
            raise RuntimeError(f"Expected one run_summary.csv for {row['name']}, got {matches}")
        summary = pd.read_csv(matches[0]).iloc[0]
        epoch_stats = pd.read_csv(pathlib.Path(matches[0]).with_name("epoch_stats.csv"))
        best_epoch = int(summary["best_epoch"])
        best_row = epoch_stats.loc[epoch_stats["epoch"] == best_epoch].iloc[0]
        separation_scale = row.get("separation_scale", "")
        separation_scale = (float(separation_scale)
                            if separation_scale not in (None, "") else None)
        dev_rows.append({
            "name": row["name"],
            "init_tag": row["init_tag"],
            "lr": float(row["lr"]),
            "separation_scale": separation_scale,
            "seed": int(row["seed"]),
            "best_epoch": best_epoch,
            "dev_score": float(summary["best_selection_score"]),
            "dev_raw_margin": float(best_row["eval_overall"]),
            "dev_sing_margin": float(best_row["eval_sing"]),
            "dev_plur_margin": float(best_row["eval_plur"]),
        })

    dev = pd.DataFrame(dev_rows)
    if dev.empty:
        raise RuntimeError("No successful runs found")
    dev.to_csv(sweep_dir / "dev_per_run.csv", index=False)

    config_cols = ["init_tag", "lr"]
    if dev["separation_scale"].notna().any():
        config_cols.append("separation_scale")
    dev_summary = dev.groupby(config_cols, dropna=False).agg(
        dev_score_mean=("dev_score", "mean"),
        dev_score_std=("dev_score", "std"),
        mean_best_epoch=("best_epoch", "mean"),
        n=("seed", "size"),
    ).reset_index()
    expected_n = dev["seed"].nunique()
    incomplete = dev_summary.loc[dev_summary["n"] != expected_n]
    if not incomplete.empty:
        raise RuntimeError(f"Incomplete LR cells:\n{incomplete.to_string(index=False)}")
    dev_summary.to_csv(sweep_dir / "dev_summary.csv", index=False)

    winners = (dev_summary.sort_values(
        ["init_tag", "dev_score_mean"], ascending=[True, False]
    ).groupby("init_tag", as_index=False).head(1))
    winners.to_csv(sweep_dir / "selected_configs.csv", index=False)

    selected = []
    for win in winners.itertuples(index=False):
        mask = (dev["init_tag"] == win.init_tag) & (dev["lr"] == win.lr)
        if "separation_scale" in config_cols:
            mask &= dev["separation_scale"] == win.separation_scale
        names = dev.loc[mask, "name"]
        selected.extend(manifest_by_name[name] for name in names)

    selected_manifest = sweep_dir / "selected_manifest.csv"
    with open(selected_manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
        writer.writeheader()
        writer.writerows(selected)

    test_dir = sweep_dir / "selected_test_eval"
    subprocess.run([
        sys.executable, str(HERE / "run_test_eval.py"),
        "--manifest", str(selected_manifest),
        "--stimuli_csv", args.stimuli_csv,
        "--model", args.model,
        "--cache_dir", args.cache_dir,
        "--gpus", args.gpus,
        "--out_dir", str(test_dir),
        "--batch_size", str(args.batch_size),
    ], cwd=REPO_ROOT, check=True)

    test_rows = []
    for row in selected:
        scored = pd.read_csv(test_dir / row["name"] / "scored.csv")
        result = {
            "name": row["name"], "init_tag": row["init_tag"],
            "lr": float(row["lr"]), "seed": int(row["seed"]),
            "test_pooled": float(pd.concat([
                scored["is_correct_singular"], scored["is_correct_plural"]
            ]).mean()),
            "test_singular": float(scored["is_correct_singular"].mean()),
            "test_plural": float(scored["is_correct_plural"].mean()),
        }
        if row.get("separation_scale", "") not in (None, ""):
            result["separation_scale"] = float(row["separation_scale"])
        for condition, group in scored.groupby("condition"):
            match = re.search(r"att(\d+)", condition)
            if match is None:
                raise ValueError(f"Could not parse attractor count from {condition!r}")
            att = match.group(1)
            result[f"test_singular_att{att}"] = float(group["is_correct_singular"].mean())
            result[f"test_plural_att{att}"] = float(group["is_correct_plural"].mean())
            result[f"test_pooled_att{att}"] = float(pd.concat([
                group["is_correct_singular"], group["is_correct_plural"]
            ]).mean())
        test_rows.append(result)

    per_run = pd.DataFrame(test_rows)
    per_run.to_csv(sweep_dir / "selected_test_per_run.csv", index=False)
    metric_cols = [c for c in per_run if c.startswith("test_")]
    summary_rows = []
    test_group_cols = ["init_tag", "lr"]
    if "separation_scale" in per_run.columns:
        test_group_cols.append("separation_scale")
    for config, group in per_run.groupby(test_group_cols):
        if not isinstance(config, tuple):
            config = (config,)
        config_values = dict(zip(test_group_cols, config))
        for metric in metric_cols:
            values = group[metric].dropna()
            n = len(values)
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if n > 1 else float("nan")
            half = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float("nan")
            summary_rows.append({
                **config_values,
                "metric": metric, "n": n, "mean": mean,
                "ci95_low": mean - half, "ci95_high": mean + half,
            })
    pd.DataFrame(summary_rows).to_csv(sweep_dir / "selected_test_summary.csv", index=False)
    print("Selected configurations:")
    print(winners.to_string(index=False))
    print("\nTest means:")
    print(per_run.groupby(test_group_cols)[
        ["test_pooled", "test_singular", "test_plural"]
    ].mean().to_string())


if __name__ == "__main__":
    main()
