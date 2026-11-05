# Learning novel-word number from images alone: method and results

Final results for the EACL rebuttal: Qwen3-VL-2B and Qwen3-VL-4B, matched
captions, held-out-image dev selection, 50 seeds each. Both models use the
identical recipe; the only per-model choice is the learning rate, selected on
the dev set. All paths are relative to the repo root (`wug-test-interp/`).

## Research question

Can a vision-language model learn that a novel word `[wug]` refers to one
entity and `[wugs]` to more than one, using **only the paired images** as the
source of number information — with no textual cue (determiners, verb
agreement, quantifiers) that could leak the distinction?

## Method

### Training data

- **Images**: `eacl-rebuttal/data/train/im/creature_1/` — 5 singular images
  (one novel creature) and 5 plural images (two of the same creature),
  `singular{1..5}.png` / `plural{1..5}.png`.
- **Captions**: `eacl-rebuttal/data/train/text/image_train_matched_v2.csv` —
  16 caption templates, each used once with `[wug]` (paired with singular
  images) and once with `[wugs]` (paired with plural images). Apart from the
  token itself, the singular and plural caption sets are **identical**
  (verified: same 16 sentences). The token never governs a verb and never
  follows a number-marked word, e.g. "Look at the [wug]." / "Look at the
  [wugs]." Only the image can tell the model which token fits.

### Model and training

- Models: `Qwen/Qwen3-VL-2B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct` (bf16).
- `[wug]` and `[wugs]` are added as new tokens. All model parameters are
  frozen except these two embedding rows (input and output embeddings are
  tied). After every optimizer step both rows are renormalized to the model's
  mean embedding norm, so training learns a direction, not a magnitude.
- Loss: ordinary next-token cross-entropy, supervised only at the
  `[wug]`/`[wugs]` positions of the caption.
- Optimizer: Adam, `betas=(0.0, 0.9)`, no weight decay.
- Batching: `batch_mode=joint`, `batch_size=8` — each step combines 8 singular
  and 8 plural (image, caption) examples.
- Up to 50 epochs, early stopping with patience 5 on the dev score; the
  best-dev-epoch embeddings are kept.
- Training script: `eacl-rebuttal/scripts/embed_train_dev_image.py`
  (sweeps launched with `eacl-rebuttal/scripts/run_sweep.py`).

### Embedding initialization (independent init)

Both tokens start from the centroid of 28 real singular/plural noun pairs
(`data/embeddings/init/noun_init.txt`: cat/cats, dog/dogs, man/men, ...;
singular and plural forms pooled), each plus its own independent random
Gaussian vector whose norm equals the mean singular–plural pair distance,
then renormalized to the target norm. The two random vectors are unrelated
to number, so neither token starts with a number bias — in the PCA
figures below, both tokens start between the singular and plural clusters.

### Model selection: held-out-image dev set

`eacl-rebuttal/data/dev/dev_image_number.csv` — 20 held-out images
(10 singular, 10 plural; `eacl-rebuttal/data/dev/held_out_images/`) of the
same creature in different poses, none of them training images. Each image is
crossed with 6 prompts that end just before the noun (e.g. "What do you see
in this picture?" → "I see the ___"), giving 120 stimuli.

For each stimulus the model's next-token logit gap
`g = logit([wug]) − logit([wugs])` is read off (sign flipped for plural
images, so positive = correct). The dev score is the class-balanced negative
log-loss of the binary `[wug]`/`[wugs]` choice:

```text
score = −0.5 · ( mean_singular softplus(−g) + mean_plural softplus(−g) )
```

0 = always confidently correct, −0.693 = chance. It is used only for early
stopping and learning-rate selection — never as the training loss, and the
test set is never used for selection.

**Learning-rate search**: 8 learning rates {0.0003, 0.001, 0.003, 0.005,
0.0075, 0.01, 0.015, 0.02} × 5 seeds {17, 42, 123, 2024, 98765}. The LR with
the best *mean* dev score across seeds is selected (one LR per model, not one
per seed). That LR is then run on 45 more seeds {100–144 except 123, 200},
giving 50 seeds.

| lr | 2B dev score | 4B dev score |
|---|---:|---:|
| 0.0003 | −0.100 | −0.517 |
| 0.001 | −0.076 | −0.252 |
| 0.003 | −0.038 | −0.300 |
| 0.005 | −0.037 | **−0.188** |
| 0.0075 | **−0.014** | −0.254 |
| 0.01 | −0.020 | −0.465 |
| 0.015 | −1.126 | −2.503 |
| 0.02 | −2.724 | −6.239 |

Selected: **2B lr = 0.0075**, **4B lr = 0.005**.

### Test set

`data/interp/agreement_target_wug.csv` — 2,800 subject–verb agreement items,
700 each with 0, 1, 2 or 3 attractor nouns between subject and verb (all
attractors opposite in number to the subject). Each item has four
sentences, e.g. with 2 attractors:

- singular: "The clumsy [wug] near the bulbs beside the roots **is**" (good)
  vs. "... **are**" (bad)
- plural: "The clumsy [wugs] near the bulb beside the root **are**" (good)
  vs. "... **is**" (bad)

Sentences are scored text-only (no image) by log-probability
(minicons `sequence_score`, via `core/eval/embed_eval.py --paired`, launched
by `eacl-rebuttal/scripts/run_test_eval.py`).

Two measures are reported:

- **Raw accuracy**: singular item correct if good > bad for the `[wug]`
  sentences; plural likewise for `[wugs]`; pooled = mean of the two.
- **Swap accuracy** (bias-free): item correct if
  `(logp(sg verb | [wug]) − logp(sg verb | [wugs])) +
  (logp(pl verb | [wugs]) − logp(pl verb | [wug])) > 0`.
  Each verb form is compared across the two subjects, so a fixed preference
  for one verb form cancels. This matters because the model has a verb-form
  prior (it prefers "is" with the copula and the bare form with lexical
  verbs), and the copula share differs across attractor levels (2.3% of
  0-attractor items vs. 40.1% of 1–3-attractor items). Raw singular-vs-plural
  differences — and part of the attractor trend — reflect that prior; swap
  accuracy does not.

All CIs below are t-based 95% intervals across the 50 seeds.

## Results

### Raw test accuracy

| | 2B | 4B |
|---|---:|---:|
| pooled | **0.745** [0.719, 0.770] | **0.657** [0.645, 0.669] |
| singular | 0.681 [0.652, 0.710] | 0.743 [0.733, 0.754] |
| plural | 0.808 [0.768, 0.848] | 0.570 [0.549, 0.592] |

By number of attractors:

| attractors | 2B sg | 2B pl | 2B pooled | 4B sg | 4B pl | 4B pooled |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.696 | 0.928 | 0.812 [0.782, 0.842] | 0.621 | 0.909 | 0.765 [0.748, 0.782] |
| 1 | 0.730 | 0.788 | 0.759 [0.733, 0.785] | 0.823 | 0.615 | 0.719 [0.704, 0.734] |
| 2 | 0.679 | 0.767 | 0.723 [0.698, 0.749] | 0.786 | 0.409 | 0.598 [0.587, 0.608] |
| 3 | 0.618 | 0.749 | 0.684 [0.661, 0.706] | 0.742 | 0.348 | 0.545 [0.535, 0.556] |

### Swap accuracy (bias-free)

| attractors | 2B | 4B |
|---:|---:|---:|
| overall | **0.894** [0.850, 0.939] | **0.830** [0.812, 0.848] |
| 0 | 0.909 [0.867, 0.951] | 0.884 [0.866, 0.901] |
| 1 | 0.885 [0.839, 0.932] | 0.923 [0.917, 0.929] |
| 2 | 0.905 [0.858, 0.953] | 0.805 [0.780, 0.830] |
| 3 | 0.877 [0.830, 0.924] | 0.708 [0.678, 0.738] |

### Embedding movement (PCA)

Following the original paper analysis: PCA fit on ~530 real singular/plural
noun pairs (unambiguous WordNet nouns, single-token in both forms); the
`[wug]`/`[wugs]` initial and learned embeddings are projected into it.
Movement = displacement along the singular–plural axis toward the correct
region (positive = correct direction). Initial embeddings were not saved
during training, so they are reconstructed exactly from each run's seed by
replaying the training script's init code; the reconstructed init matched
the logged init separation for every run (100/100).

| | `[wug]` → singular | `[wugs]` → plural |
|---|---:|---:|
| 2B | 0.164 [0.152, 0.175] | 0.118 [0.111, 0.125] |
| 4B | 0.097 [0.093, 0.101] | 0.058 [0.055, 0.060] |

All one-sided t-tests vs. 0: p < 10⁻³². Figures:
`eacl-rebuttal/figures/2b-4b-movement-pca-legendtop.pdf` and
`eacl-rebuttal/figures/2b-4b-movement-pca-vertical.pdf`.

### Summary

Both models learn the `[wug]`/`[wugs]` number distinction from images alone,
well above chance on every measure, with captions that carry no number
information. 2B is the stronger learner (0.745 raw / 0.894 swap vs. 0.657 /
0.830 for 4B); the gap is concentrated at 2–3 attractors, while at 0–1
attractors 4B matches 2B on the bias-free measure. In embedding space, both
tokens start between the real singular and plural clusters and move toward
the correct one in both models, with 4B moving about 60% as far as 2B. The
two models show opposite raw singular/plural asymmetries (2B plural >
singular, 4B singular > plural). Raw singular and plural accuracy each
compare two different verb forms, so they are confounded with the verb-form
prior; swap accuracy removes that prior but by construction has no
singular/plural split, so these asymmetries should not be read as differences
in how well each number is learned.

## What's in git

All of `eacl-rebuttal/results/` (≈210 MB: every run directory, per-epoch
embeddings, logs, per-run `scored.csv`, summaries) is ignored by the repo-root
`.gitignore` and exists only on the server. Everything needed to inspect or
recompute the reported results is copied outside it in compact form:

| in git | contents | size |
|---|---|---:|
| `eacl-rebuttal/embeddings/` | final (best-epoch) `[wug]`/`[wugs]` embeddings, 50 seeds × 2 models, + manifests | 1.2 MB |
| `eacl-rebuttal/test_scores/` | item-level test scores, 50 seeds × 2 models, + the 2,800 items once | 13 MB |
| `eacl-rebuttal/data/` | training images, matched captions, dev images + CSV | 153 MB |
| `eacl-rebuttal/pca_data/` | the PCA / movement CSVs the figures are drawn from | 0.1 MB |
| `eacl-rebuttal/figures/`, `scripts/`, `analysis/`, `*.md` | figures, code, docs | ~1 MB |

The compact test scores replace the per-run `scored.csv` files: those repeat
the same 2,800 items (sentences, words, ids) in every run, so 138 MB of the
size is duplicated item text. `test_scores/` stores the items once and only
the four sentence log-probabilities per run, and
`scripts/build_test_score_tables.py` checks that every reported raw and swap
accuracy recomputed from it matches the original files exactly.

**Not in git** (server only, under `results/`): the dev-score tables used for
LR selection, the 50-seed summary CSVs, `results/swap_accuracy/` and
`results/movement-analysis/` (raw output of `scripts/movement_pca.py`). The
test summaries and swap accuracy can be recomputed from `test_scores/`; the
dev tables are regenerated from the run directories on the server
(`scripts/select_lr_and_score.py`).

## Paths

Paths under `eacl-rebuttal/results/` are on the server only (see above).

### 2B (lr = 0.0075)

| what | path |
|---|---|
| LR search (8 LRs × 5 seeds) | `eacl-rebuttal/results/lr_search_2B_bs8/` |
| dev scores per LR | `.../lr_search_2B_bs8/dev_summary.csv`, `dev_per_run.csv` |
| 5 selected-LR runs | `.../lr_search_2B_bs8/init_small/lr_0p0075_joint_bs8/seed_{17,42,123,2024,98765}/` |
| 45 extra-seed runs | `eacl-rebuttal/results/lr_search_2B_bs8_more45/init_small/lr_0p0075_joint_bs8/seed_*/` |
| learned embeddings | `<run dir>/creature_1_joint_lr0p0075_seed<S>_ep50/learned_embeddings.pt` |
| per-epoch embeddings | `<same>/epoch_embs/epoch_NNN.pt` |
| per-run training stats/config | `<same>/run_config.csv`, `epoch_stats.csv`, `eval_item_scores.csv` |
| test item scores | `.../lr_search_2B_bs8/selected_test_eval/<run>/scored.csv`, `.../lr_search_2B_bs8_more45/test_eval/<run>/scored.csv` |
| 50-seed test summary | `eacl-rebuttal/results/lr_search_2B_bs8_more45/combined_50seed_test_summary.csv` (per run: `combined_50seed_test_per_run.csv`) |
| PCA / movement CSVs | `eacl-rebuttal/pca_data/Qwen_Qwen3-VL-2B-Instruct/` (raw output: `results/movement-analysis/...`) |

### 4B (lr = 0.005)

| what | path |
|---|---|
| LR search (8 LRs × 5 seeds) | `eacl-rebuttal/results/lr_search_4B_bs8/` |
| dev scores per LR | `.../lr_search_4B_bs8/dev_summary.csv`, `dev_per_run.csv` |
| 5 selected-LR runs | `.../lr_search_4B_bs8/init_small/lr_0p005_joint_bs8/seed_{17,42,123,2024,98765}/` |
| 45 extra-seed runs | `eacl-rebuttal/results/lr_search_4B_bs8_more45/init_small/lr_0p005_joint_bs8/seed_*/` |
| learned embeddings | `<run dir>/creature_1_joint_lr0p005_seed<S>_ep50/learned_embeddings.pt` |
| per-epoch embeddings | `<same>/epoch_embs/epoch_NNN.pt` |
| per-run training stats/config | `<same>/run_config.csv`, `epoch_stats.csv`, `eval_item_scores.csv` |
| test item scores | `.../lr_search_4B_bs8/selected_test_eval/<run>/scored.csv`, `.../lr_search_4B_bs8_more45/test_eval/<run>/scored.csv` |
| 50-seed test summary | `eacl-rebuttal/results/lr_search_4B_bs8_more45/combined_50seed_test_summary.csv` (per run: `combined_50seed_test_per_run.csv`) |
| PCA / movement CSVs | `eacl-rebuttal/pca_data/Qwen_Qwen3-VL-4B-Instruct/` (raw output: `results/movement-analysis/...`) |

### Final embeddings (50 seeds per model, one place)

Copies of every reported run's best-epoch `learned_embeddings.pt`, renamed by
seed. Each `manifest.csv` lists seed, lr, best epoch, pooled test accuracy and
the source run directory.

| what | path |
|---|---|
| 2B | `eacl-rebuttal/embeddings/qwen3-vl-2b/seed_<S>.pt` + `manifest.csv` |
| 4B | `eacl-rebuttal/embeddings/qwen3-vl-4b/seed_<S>.pt` + `manifest.csv` |

Each `.pt` is a dict with `wug_embedding`, `wugs_embedding`, `wug_id`,
`wugs_id`, `added_tokens`, `vocab_size`, `model_name`, `saved_epoch`.

### Test scores, compact (both models)

The same item-level test scores as the per-run `scored.csv` files, without
repeating the item text in every run (13 MB instead of 138 MB). Built by
`eacl-rebuttal/scripts/build_test_score_tables.py`, which checks that raw and
swap accuracy recomputed from these tables match the reported results exactly.

| what | path |
|---|---|
| the 2,800 test items (sentences, words, `attractors`), once | `eacl-rebuttal/test_scores/items.csv` |
| 2B scores: `seed`, `idx`, `attractors`, four sentence log-probs | `eacl-rebuttal/test_scores/scores_qwen3-vl-2b.csv` |
| 4B scores: same columns | `eacl-rebuttal/test_scores/scores_qwen3-vl-4b.csv` |

Join on `idx`. An item is correct for singular if
`score_good_singular > score_bad_singular` (plural likewise); swap-correct if
`(score_good_singular − score_bad_singular) + (score_good_plural − score_bad_plural) > 0`.

### Shared

| what | path |
|---|---|
| swap accuracy (both models) | `eacl-rebuttal/results/swap_accuracy/swap_summary.csv`, `swap_per_run.csv` |
| movement summary (both models) | `eacl-rebuttal/pca_data/movement_summary.csv` |
| figures | `eacl-rebuttal/figures/2b-4b-movement-pca-{legendtop,vertical}.pdf` |

### Code

| script | purpose |
|---|---|
| `eacl-rebuttal/scripts/embed_train_dev_image.py` | training (dev metric, init, early stopping) |
| `eacl-rebuttal/scripts/run_sweep.py` | LR × seed sweeps over GPUs |
| `eacl-rebuttal/scripts/select_lr_and_score.py` | pick LR by mean dev score, score it on test |
| `eacl-rebuttal/scripts/run_test_eval.py` | test-set scoring (`core/eval/embed_eval.py --paired`) |
| `eacl-rebuttal/scripts/combine_50_seed_test_results.py` | combine 5 + 45 seeds into the 50-seed summary |
| `eacl-rebuttal/scripts/run_4B_independent_more45.sh` | the 4B 45-seed train → score → combine pipeline |
| `eacl-rebuttal/scripts/swap_accuracy.py` | swap (bias-free) accuracy |
| `eacl-rebuttal/scripts/movement_pca.py` | PCA + movement (reconstructs initial embeddings) → `results/movement-analysis/` |
| `eacl-rebuttal/scripts/export_pca_data.py` | copies the CSVs the figures need into git-tracked `pca_data/` |
| `eacl-rebuttal/scripts/plot_movement_pca.py` | figures (matplotlib), reads `pca_data/` |
| `eacl-rebuttal/analysis/movement-rebuttal.R` | same figures + movement tables in ggplot, reads `pca_data/` |
