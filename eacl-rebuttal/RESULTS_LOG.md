# EACL rebuttal: final results log

Permanent record of the kept results and of the superseded experiment
directories that were deleted from `results/`. Full method and results for
the kept results: `METHODS_AND_RESULTS.md`.

All experiments below use the **matched, number-neutral captions**
(`data/train/text/image_train_matched_v2.csv`: `[wug]`/`[wugs]` sentences
identical apart from the token), the training images
`data/train/im/creature_1/`, and the **held-out-image dev metric**
(`data/dev/dev_image_number.csv`: 10 singular + 10 plural images never seen in
training, crossed with 6 QA templates = 120 stimuli) for LR / checkpoint
selection, unless noted otherwise. Test accuracy is always on the untouched
held-out set `data/interp/agreement_target_wug.csv`.

## What works (kept)

Both kept results use the identical recipe — independent small-noun init,
`batch_mode=joint`, `batch_size=8`, Adam `betas=(0.0, 0.9)`, early stopping
patience 5 / max 50 epochs, balanced-logistic dev selection — and the same
procedure: 8-LR search on 5 seeds (17, 42, 123, 2024, 98765), LR chosen by
best mean dev score, then 45 more seeds (100–144 except 123, plus 200) at
that LR.

### 2B — Qwen/Qwen3-VL-2B-Instruct

Directories: `results/lr_search_2B_bs8/` (LR search; selected lr=0.0075) +
`results/lr_search_2B_bs8_more45/` (45 more seeds → 50).

| metric | mean | 95% CI |
|---|---:|---:|
| pooled | **0.745** | [0.719, 0.770] |
| singular | 0.681 | [0.652, 0.710] |
| plural | 0.808 | [0.768, 0.848] |
| swap (bias-free) | 0.894 | [0.850, 0.939] |

| attractors | singular | plural | pooled | swap |
|---:|---:|---:|---:|---:|
| 0 | 0.696 | 0.928 | 0.812 | 0.909 |
| 1 | 0.730 | 0.788 | 0.759 | 0.885 |
| 2 | 0.679 | 0.767 | 0.723 | 0.905 |
| 3 | 0.618 | 0.749 | 0.684 | 0.877 |

### 4B — Qwen/Qwen3-VL-4B-Instruct

Directories: `results/lr_search_4B_bs8/` (LR search; selected lr=0.005) +
`results/lr_search_4B_bs8_more45/` (45 more seeds → 50).

| metric | mean | 95% CI |
|---|---:|---:|
| pooled | **0.657** | [0.645, 0.669] |
| singular | 0.743 | [0.733, 0.754] |
| plural | 0.570 | [0.549, 0.592] |
| swap (bias-free) | 0.830 | [0.812, 0.848] |

| attractors | singular | plural | pooled | swap |
|---:|---:|---:|---:|---:|
| 0 | 0.621 | 0.909 | 0.765 | 0.884 |
| 1 | 0.823 | 0.615 | 0.719 | 0.923 |
| 2 | 0.786 | 0.409 | 0.598 | 0.805 |
| 3 | 0.742 | 0.348 | 0.545 | 0.708 |

Raw singular vs. plural accuracy is confounded with the model's verb-form
prior (each compares two different verb forms); swap accuracy removes that
prior. See `METHODS_AND_RESULTS.md`.

## What doesn't work / was superseded (deleted)

These directories were removed from `results/` after their key numbers were
recorded here. Same matched-caption + held-out-image-dev recipe unless noted;
pooled test accuracy at whatever seed count they reached.

| directory | model | condition | n seeds | pooled acc | why excluded |
|---|---|---|---:|---:|---|
| `lr_search_4B_bs2` | Qwen3-VL-4B | independent init, bs=2 | 5 | 0.615 | worse batch size than bs=8 |
| `lr_search_4B_bs4` | Qwen3-VL-4B | independent init, bs=4 | 5 | 0.572 | worse batch size than bs=8 |
| `lr_search_4B_bs4_more45` | Qwen3-VL-4B | independent init, bs=4 | 50 | 0.653 | bs=4; kept results use bs=8 to match 2B |
| `lr_search_4B_bs16` | Qwen3-VL-4B | independent init, bs=16 | 5 | 0.571 | worse batch size than bs=8 |
| `lr_search_4B_bs8_large_init` | Qwen3-VL-4B | large (530-pair) noun init, bs=8 | 5 | 0.668 | not the small-init recipe used for 2B |
| `lr_search_gemma3_4b_bs4` | google/gemma-3-4b-it | independent init, bs=4 | 5 | 0.548 | weaker than Qwen3-VL |
| `lr_search_internvl35_4b_bs4` | OpenGVLab/InternVL3_5-4B-HF | independent init, bs=4 | 5 | 0.538 | weaker than Qwen3-VL |
| `lr_search_llava15_bs1` | llava-hf/llava-1.5-7b-hf | independent init, bs=1 | 5 | 0.465 | weaker than Qwen3-VL |
| `lr_search_smolvlm2_bs2` | HuggingFaceTB/SmolVLM2-2.2B-Instruct | independent init, bs=2 | 5 | 0.425 | weaker than Qwen3-VL |
| `lr_search_2B_syntax_imagedev` / `_more45` | Qwen3-VL-2B | **leaky** captions (`syntax_train_1.csv`, number-marked text) + image dev | 5 / 50 | 0.778 / 0.765 | deliberate leaky-caption control, not the image-only condition: with textual number cues the model does only slightly better (0.765 vs 0.745 at 50 seeds) |
| `noun_vs_verb_dev_4B_bs8` | Qwen3-VL-4B | noun-vs-verb **text** dev metric | — | no test summary | superseded dev-metric design |
| `sweep/`, `test_eval/`, `summary.csv`, `summary_live.csv` (top-level) | Qwen3-VL-2B | original 140-run sweep with the noun-vs-verb dev metric, earlier captions | — | — | first-generation approach, superseded by the held-out-image dev metric + bs=8 |
| `batch_init_diagnostic_2B`, `batch_init_diagnostic_4B`, `epoch0_diagnostic_4B`, `epoch_trajectory_diagnostic_4B`, `large_init_diagnostic_4B` | Qwen3-VL-2B/4B | one-off diagnostics (init bias, epoch-0, trajectories) | — | — | diagnostics only |
| `deprecated_attractor_crosscheck`, `principled_attractor_crosscheck` | Qwen3-VL | attractor-confound scratch work | — | — | exploratory; the verb-type confound is described in `METHODS_AND_RESULTS.md` |
| `gemma3_4b_smoke_bs2` | gemma-3-4b-it | smoke test | — | — | pipeline sanity check only |

## Code and data used by the kept results

- `scripts/embed_train_dev_image.py` — training (held-out-image dev metric,
  init, `selection_metric=balanced_logistic`).
- `scripts/run_sweep.py`, `gpu_pool.py`, `select_lr_and_score.py`,
  `run_test_eval.py`, `combine_50_seed_test_results.py`,
  `run_4B_independent_more45.sh` — sweeps, LR selection, test scoring,
  50-seed combination.
- `scripts/swap_accuracy.py`, `movement_pca.py`, `plot_movement_pca.py`,
  `analysis/movement-rebuttal.R` — swap accuracy, PCA movement, figures.
- `data/train/text/image_train_matched_v2.csv`, `data/train/im/creature_1/`,
  `data/dev/dev_image_number.csv` + `data/dev/held_out_images/`,
  `../data/embeddings/init/noun_init.txt`.
