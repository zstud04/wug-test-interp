#!/usr/bin/env python3
"""
Join each sweep run's dev-set training metrics (run_summary.csv, from the
dev-margin fork) with its held-out test-set accuracy (test_eval/<name>/scored.csv,
from unmodified core/eval/embed_eval.py --paired) into one comparison table.

Usage:
  python3 eacl-rebuttal/scripts/aggregate_results.py
"""
import argparse
import glob
import os
import pathlib

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parent
while not (_REPO_ROOT / "core").is_dir():
    _parent = _REPO_ROOT.parent
    if _parent == _REPO_ROOT:
        raise RuntimeError(f"Could not find repo root (no core/ dir above {__file__})")
    _REPO_ROOT = _parent


def find_run_summary(run_dir):
    matches = glob.glob(os.path.join(run_dir, "**", "run_summary.csv"), recursive=True)
    return matches[0] if matches else None


def _cond_label(cond):
    """'target_verb_att0_opp' -> 'att0' for readability."""
    return cond.replace("target_verb_", "").replace("_opp", "")


def summarize_test_scores(scored_csv):
    """Accuracy pooled across singular+plural items (not AND'd together), broken
    out per attractor condition. Pooling sg+pl avoids the artificially strict
    "both must be correct" bar of a joint/AND metric, and per-condition (rather
    than collapsed-across-all) is what shows the expected attractor-count falloff.
    """
    df = pd.read_csv(scored_csv)
    out = {"test_n": len(df)}
    pooled_all = []
    if "condition" in df.columns:
        for cond, grp in df.groupby("condition"):
            pooled = pd.concat([grp["is_correct_singular"], grp["is_correct_plural"]])
            pooled_all.append(pooled)
            out[f"test_acc_{_cond_label(cond)}"] = pooled.mean()
    else:
        pooled_all.append(pd.concat([df["is_correct_singular"], df["is_correct_plural"]]))
    out["test_acc_pooled_overall"] = pd.concat(pooled_all).mean()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(_REPO_ROOT / "eacl-rebuttal" / "results" / "sweep" / "manifest.csv"))
    parser.add_argument("--test_eval_dir", default=str(_REPO_ROOT / "eacl-rebuttal" / "results" / "test_eval"))
    parser.add_argument("--out", default=str(_REPO_ROOT / "eacl-rebuttal" / "results" / "summary.csv"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    rows = []
    for _, m in manifest.iterrows():
        if str(m.get("returncode")) != "0":
            continue

        summary_path = find_run_summary(m["run_dir"])
        if summary_path is None:
            print(f"  WARNING: missing run_summary.csv for {m['name']}, skipping")
            continue
        train_row = pd.read_csv(summary_path).iloc[0].to_dict()

        scored_path = os.path.join(args.test_eval_dir, m["name"], "scored.csv")
        test_metrics = {}
        if os.path.isfile(scored_path):
            test_metrics = summarize_test_scores(scored_path)
        else:
            print(f"  WARNING: missing test scored.csv for {m['name']}")

        rows.append({
            "name": m["name"], "init_tag": m["init_tag"], "lr": m["lr"], "seed": m["seed"],
            "effective_epochs": train_row.get("effective_epochs"),
            "stopped_early": train_row.get("stopped_early"),
            "dev_overall_margin": train_row.get("final_overall"),
            "dev_sing_margin": train_row.get("final_sing"),
            "dev_plur_margin": train_row.get("final_plur"),
            **test_metrics,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows to {args.out}")

    if len(out_df) and "test_acc_pooled_overall" in out_df.columns:
        att_cols = sorted(c for c in out_df.columns if c.startswith("test_acc_att"))

        print("\nTop 10 runs by test_acc_pooled_overall:")
        cols = ["name", "init_tag", "lr", "seed", "dev_overall_margin", "test_acc_pooled_overall", *att_cols]
        print(out_df.sort_values("test_acc_pooled_overall", ascending=False)[cols].head(10).to_string(index=False))

        print("\nMean pooled accuracy by init_tag (overall + per attractor condition):")
        print(out_df.groupby("init_tag")[["test_acc_pooled_overall", *att_cols]].mean().to_string())

        print("\nMean test_acc_pooled_overall by (init_tag, lr):")
        print(out_df.groupby(["init_tag", "lr"])["test_acc_pooled_overall"].agg(["mean", "std", "count"]).to_string())


if __name__ == "__main__":
    main()
