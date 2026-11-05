#!/usr/bin/env python3
"""Print pooled/singular/plural accuracy by attractor for every cell in the
batch_size x init_mode diagnostic (single seed each, not for CI reporting)."""
import csv
import pathlib
import re

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent
while not (REPO_ROOT / "core").is_dir():
    REPO_ROOT = REPO_ROOT.parent

OUT_DIR = REPO_ROOT / "eacl-rebuttal" / "results" / "batch_init_diagnostic_4B"
TEST_DIR = OUT_DIR / "test_eval"


def main():
    with open(OUT_DIR / "manifest.csv", newline="") as f:
        manifest = list(csv.DictReader(f))

    rows = []
    for row in manifest:
        scored_path = TEST_DIR / row["name"] / "scored.csv"
        if not scored_path.is_file():
            print(f"MISSING scored.csv for {row['name']}")
            continue
        scored = pd.read_csv(scored_path)
        overall_pooled = pd.concat([scored["is_correct_singular"], scored["is_correct_plural"]]).mean()
        result = {
            "init_mode": row["init_mode"],
            "separation_scale": row["separation_scale"],
            "batch_size": int(row["batch_size"]),
            "pooled": overall_pooled,
            "singular": scored["is_correct_singular"].mean(),
            "plural": scored["is_correct_plural"].mean(),
        }
        for condition, group in scored.groupby("condition"):
            att = re.search(r"att(\d+)", condition).group(1)
            result[f"plural_att{att}"] = group["is_correct_plural"].mean()
            result[f"singular_att{att}"] = group["is_correct_singular"].mean()
        rows.append(result)

    df = pd.DataFrame(rows).sort_values(["init_mode", "batch_size"])
    cols = ["init_mode", "separation_scale", "batch_size", "pooled", "singular", "plural",
            "plural_att0", "plural_att1", "plural_att2", "plural_att3"]
    pd.set_option("display.width", 200)
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    df.to_csv(OUT_DIR / "diagnostic_summary.csv", index=False)


if __name__ == "__main__":
    main()
