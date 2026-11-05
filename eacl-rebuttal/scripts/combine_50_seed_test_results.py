#!/usr/bin/env python3
"""Combine the 45-seed follow-up test results with the original 5-seed
selected_test_per_run.csv to produce a single 50-seed summary at the
dev-selected configuration.

Usage:
  python3 eacl-rebuttal/scripts/combine_50_seed_test_results.py \
      --new_manifest eacl-rebuttal/results/lr_search_4B_bs8_more45/manifest.csv \
      --new_test_dir eacl-rebuttal/results/lr_search_4B_bs8_more45/test_eval \
      --existing_per_run eacl-rebuttal/results/lr_search_4B_bs8/selected_test_per_run.csv \
      --out_dir eacl-rebuttal/results/lr_search_4B_bs8_more45
"""
import argparse
import csv
import math
import pathlib
import re

import pandas as pd
from scipy.stats import t


def build_new_rows(manifest_path, test_dir):
    with open(manifest_path, newline="") as f:
        manifest = list(csv.DictReader(f))

    failed = [row["name"] for row in manifest if str(row.get("returncode")) != "0"]
    if failed:
        raise RuntimeError(f"Refusing to combine with {len(failed)} failed runs: {failed}")

    rows = []
    for row in manifest:
        scored = pd.read_csv(pathlib.Path(test_dir) / row["name"] / "scored.csv")
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
        rows.append(result)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_manifest", required=True)
    ap.add_argument("--new_test_dir", required=True)
    ap.add_argument("--existing_per_run", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    new_df = build_new_rows(args.new_manifest, args.new_test_dir)
    existing_df = pd.read_csv(args.existing_per_run)

    dup = set(new_df["seed"]) & set(existing_df["seed"])
    if dup:
        raise RuntimeError(f"Seed overlap between existing and new runs: {sorted(dup)}")

    combined = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    if combined["seed"].duplicated().any():
        raise RuntimeError("Duplicate seeds in combined result")
    if len(combined) != 50:
        raise RuntimeError(f"Expected 50 combined seeds, got {len(combined)}")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "combined_50seed_test_per_run.csv", index=False)

    metric_cols = [c for c in combined.columns if c.startswith("test_")]
    summary_rows = []
    for metric in metric_cols:
        values = combined[metric].dropna()
        n = len(values)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if n > 1 else float("nan")
        half = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else float("nan")
        summary_rows.append({
            "metric": metric, "n": n, "mean": mean,
            "ci95_low": mean - half, "ci95_high": mean + half,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "combined_50seed_test_summary.csv", index=False)

    print(f"Combined {len(combined)} seeds -> {out_dir / 'combined_50seed_test_per_run.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
