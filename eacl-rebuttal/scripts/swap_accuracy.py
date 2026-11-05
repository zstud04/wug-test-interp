#!/usr/bin/env python3
"""
Bias-free "swap" accuracy on the agreement test set, from the per-run
scored.csv files written by run_test_eval.py.

An item counts as correct when score_diff_singular + score_diff_plural > 0,
i.e. (logp(sg verb | [wug]) - logp(sg verb | [wugs])) +
     (logp(pl verb | [wugs]) - logp(pl verb | [wug])) > 0.
Each verb form is compared across the two subjects, so any fixed preference
for one verb form (e.g. "is" over "are") cancels out.

Writes per-run and summary (mean + t-based 95% CI across seeds) CSVs.
"""
import argparse
import glob
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

R = "eacl-rebuttal/results"
SETS = {
    "2B": [f"{R}/lr_search_2B_bs8/selected_test_eval/*/scored.csv",
           f"{R}/lr_search_2B_bs8_more45/test_eval/*/scored.csv"],
    "4B": [f"{R}/lr_search_4B_bs8/selected_test_eval/*/scored.csv",
           f"{R}/lr_search_4B_bs8_more45/test_eval/*/scored.csv"],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=f"{R}/swap_accuracy")
    out = pathlib.Path(p.parse_args().out_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_run = []
    for model, globs in SETS.items():
        for f in sorted(x for g in globs for x in glob.glob(g)):
            d = pd.read_csv(f)
            swap = (d.score_diff_singular + d.score_diff_plural) > 0
            row = {"model": model, "run": pathlib.Path(f).parent.name, "swap_overall": swap.mean()}
            for a, g in swap.groupby(d.attractors):
                row[f"swap_att{a}"] = g.mean()
            per_run.append(row)
    per_run = pd.DataFrame(per_run)
    per_run.to_csv(out / "swap_per_run.csv", index=False)

    rows = []
    for model, g in per_run.groupby("model"):
        for c in [c for c in per_run.columns if c.startswith("swap_")]:
            x = g[c].to_numpy()
            h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
            rows.append({"model": model, "metric": c, "n": len(x), "mean": x.mean(),
                         "ci95_low": x.mean() - h, "ci95_high": x.mean() + h})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "swap_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
