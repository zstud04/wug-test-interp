#!/usr/bin/env python3
"""
Compact copy of the reported runs' test-set scores.

Every run's scored.csv repeats the same 2,800 test items (sentences, words,
ids); only the four sentence log-probabilities differ. This writes:

  test_scores/items.csv               — the 2,800 items, once (incl. attractors)
  test_scores/scores_<model>.csv      — seed, idx, attractors + the four
                                        log-probabilities, for all 50 seeds

and checks that raw accuracy (singular / plural / pooled, overall and by
attractor) and swap accuracy recomputed from the compact tables exactly match
the existing per-run results. The original scored.csv files are not touched.
"""
import glob
import pathlib
import re

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "test_scores"

MODELS = {
    "qwen3-vl-2b": {
        "scored": [R / "lr_search_2B_bs8/selected_test_eval", R / "lr_search_2B_bs8_more45/test_eval"],
        "per_run": R / "lr_search_2B_bs8_more45/combined_50seed_test_per_run.csv",
        "swap_model": "2B",
    },
    "qwen3-vl-4b": {
        "scored": [R / "lr_search_4B_bs8/selected_test_eval", R / "lr_search_4B_bs8_more45/test_eval"],
        "per_run": R / "lr_search_4B_bs8_more45/combined_50seed_test_per_run.csv",
        "swap_model": "4B",
    },
}
SCORE_COLS = ["score_good_singular", "score_bad_singular", "score_good_plural", "score_bad_plural"]
ITEM_COLS = ["idx", "split", "target_type", "attractors", "condition",
             "subject_word_sg", "subject_word_pl", "target_word_sg", "target_word_pl",
             "base_sentence_sg", "base_sentence_pl",
             "good_singular", "bad_singular", "good_plural", "bad_plural",
             "tok_len_sg", "tok_len_pl", "set_id", "singular_item_id", "plural_item_id"]


def accuracies(s):
    """Per-seed raw and swap accuracies from a compact scores table."""
    s = s.assign(sg=s.score_good_singular > s.score_bad_singular,
                 pl=s.score_good_plural > s.score_bad_plural,
                 swap=(s.score_good_singular - s.score_bad_singular
                       + s.score_good_plural - s.score_bad_plural) > 0)
    rows = {}
    for seed, g in s.groupby("seed"):
        r = {"test_singular": g.sg.mean(), "test_plural": g.pl.mean(), "swap_overall": g.swap.mean()}
        r["test_pooled"] = (r["test_singular"] + r["test_plural"]) / 2
        for a, h in g.groupby("attractors"):
            r[f"test_singular_att{a}"] = h.sg.mean()
            r[f"test_plural_att{a}"] = h.pl.mean()
            r[f"test_pooled_att{a}"] = (h.sg.mean() + h.pl.mean()) / 2
            r[f"swap_att{a}"] = h.swap.mean()
        rows[seed] = r
    return pd.DataFrame.from_dict(rows, orient="index")


def main():
    OUT.mkdir(exist_ok=True)
    items = None
    swap_ref = pd.read_csv(R / "swap_accuracy/swap_per_run.csv")
    for name, cfg in MODELS.items():
        files = sorted(f for d in cfg["scored"] for f in glob.glob(str(d / "*/scored.csv")))
        assert len(files) == 50, (name, len(files))
        parts = []
        for f in files:
            d = pd.read_csv(f)
            run_items = d[ITEM_COLS]
            if items is None:
                items = run_items
            else:
                pd.testing.assert_frame_equal(run_items, items)  # identical items in every run
            seed = int(re.search(r"seed(\d+)$", pathlib.Path(f).parent.name).group(1))
            parts.append(d[["idx", "attractors"] + SCORE_COLS].assign(seed=seed))
        scores = pd.concat(parts)[["seed", "idx", "attractors"] + SCORE_COLS]
        scores = scores.sort_values(["seed", "idx"]).reset_index(drop=True)
        scores.to_csv(OUT / f"scores_{name}.csv", index=False, float_format="%.6g")

        # Verify against the existing results, reading back what was written.
        acc = accuracies(pd.read_csv(OUT / f"scores_{name}.csv"))
        ref = pd.read_csv(cfg["per_run"]).set_index("seed")
        raw_cols = [c for c in acc.columns if c.startswith("test_")]
        assert np.allclose(acc.loc[ref.index, raw_cols], ref[raw_cols], atol=1e-12), name
        sw = swap_ref[swap_ref.model == cfg["swap_model"]].copy()
        sw["seed"] = sw.run.str.extract(r"seed(\d+)$").astype(int)
        sw = sw.set_index("seed")
        swap_cols = [c for c in sw.columns if c.startswith("swap_")]
        assert np.allclose(acc.loc[sw.index, swap_cols], sw[swap_cols], atol=1e-12), name
        print(f"{name}: {scores.seed.nunique()} seeds x {len(items)} items -> "
              f"scores_{name}.csv; raw + swap accuracy match existing results")

    items.to_csv(OUT / "items.csv", index=False)
    print(f"items.csv: {len(items)} items")


if __name__ == "__main__":
    main()
