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

### 4B compared with its baselines

4B scores below 2B, and its plural accuracy drops sharply with attractors. The
same pattern appears in the paper's 4B embeddings learned from **text** with
explicit number cues ("The [wugs] run"), so it reflects how 4B handles novel
nouns, not a failure to learn from images. Real nouns show a milder version of
the same drop in 4B (plural 0.98 → 0.63) than in 2B (0.98 → 0.81).

Qwen3-VL-4B, test accuracy by number of attractors (singular / plural / swap):

| attractors | real nouns | language-cued (paper, best-dev seed) | vision-cued, new (best-dev seed 116) | vision-cued, new (50-seed mean) |
|---:|---|---|---|---|
| 0 | 0.80 / 0.98 / 0.98 | 0.66 / 0.88 / 0.91 | 0.58 / 0.98 / 0.93 | 0.62 / 0.91 / 0.88 |
| 1 | 0.87 / 0.80 / 0.94 | 0.84 / 0.63 / 0.94 | 0.84 / 0.62 / 0.95 | 0.82 / 0.61 / 0.92 |
| 2 | 0.83 / 0.70 / 0.95 | 0.81 / 0.41 / 0.85 | 0.80 / 0.39 / 0.86 | 0.79 / 0.41 / 0.80 |
| 3 | 0.81 / 0.63 / 0.91 | 0.77 / 0.35 / 0.74 | 0.76 / 0.33 / 0.76 | 0.74 / 0.35 / 0.71 |
| pooled | | 0.77 raw / 0.86 swap | 0.74 raw / 0.87 swap | 0.66 raw / 0.83 swap |

The best-dev seed is the selection rule the paper used for its single
reported embeddings, so the two middle columns compare like with like. Real
nouns: all 5,600 items of `results/eval/attractors/Qwen3-VL-4B-Instruct/agreement_target_natural_scored.csv`;
language-cued: `results/eval/attractors/Qwen3-VL-4B-Instruct/syntax/agreement_target_wug_scored.csv`.

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

Compared with the paper's Table 1 (same PCA space and measure; language
values are the paper's; 4B language has 49 seeds):

| | language sg | language pl | vision, new sg | vision, new pl | vision, paper (old) sg | vision, paper (old) pl |
|---|---:|---:|---:|---:|---:|---:|
| 2B | 0.088 ± 0.005 | 0.051 ± 0.004 | 0.164 ± 0.011 | 0.118 ± 0.007 | — | — |
| 4B | 0.092 ± 0.006 | 0.061 ± 0.006 | 0.097 ± 0.004 | 0.058 ± 0.002 | 0.067 | 0.040 |

(± = 95% CI margin; the paper prints plural movement with a negative sign.)
The paper's note that language-cued nouns move more than vision-cued ones no
longer holds: with the new embeddings, 4B vision movement equals language
movement, and 2B vision moves almost twice as far. Summary CSV:
`eacl-rebuttal/pca_data/movement_summary_language_vision.csv`.

All one-sided t-tests vs. 0: p < 10⁻³². Figures, in the paper's format
(Language panel: the paper's language-cued embeddings, unchanged; Vision
panel: these new embeddings; same PCA space):
`eacl-rebuttal/figures/{2b,4b}-movement-pca-{legendtop,vertical}.pdf`,
made by `eacl-rebuttal/analysis/movement-rebuttal.R`.

### Mechanistic analyses (rerun of the paper's Section 6)

The paper's four interventions — DAS, DiffMean, linear probe, attribution
patching (AtP, k = 128) — were rerun with the new vision-cued embeddings,
using the paper's own launch settings unchanged (same scripts, layers, token
positions, real-noun training items, `is_correct_all` test filter,
`n_sample`). Interventions are fit on real nouns, so they are identical to the
paper's; only the novel-noun test items differ. Embeddings follow the paper's
rule: each model's best-dev seed (2B: 130, 4B: 116). The paper's
language-cued results are reused unchanged. Launcher:
`eacl-rebuttal/scripts/run_interp_vision_new.sh` (also reruns the paper's 2B
5-seed replicate at 0–1 attractors, with the 5 LR-search seeds; not yet
analysed).

Cue-condition test, the paper's `fit2`
(`mean_odds ~ attractors + modality + model + method + (1 | layer)`, lmerTest):

| data | vision vs. language (β) | SE | p |
|---|---:|---:|---:|
| paper: language + old vision | −0.016 | 0.157 | .92 |
| language + new vision | −0.068 | 0.149 | .65 |

The first row reproduces the paper's reported value exactly, confirming the
analysis matches. With the new embeddings there is still no difference between
vision- and language-cued nouns. At each method's best layer (chosen on real
nouns), new vision tracks language closely for DiffMean, the probe and AtP.
For 4B DAS, vision is now closer to language at 0 attractors (8.8 vs. 7.1;
old vision 11.2, the exception the paper noted) and below it at 2–3
attractors (5.8 / 5.0 vs. 6.9 / 6.8), in line with 4B's weaker behaviour
there; those two cells rest on fewer test items (284 and 188, vs. 512 and 368
for old vision), because only items the embedding gets right for both numbers
are used.

Figures (paper's format, Language rows unchanged):
`eacl-rebuttal/figures/emb-interp-results-{2B,4B}.pdf`,
`eacl-rebuttal/figures/novel-vs-real-interp-{2B,4B}.pdf`, made by
`eacl-rebuttal/analysis/interp-rebuttal.R`, which also fits the model above.
Tables: `eacl-rebuttal/interp_data/` (copied from `results/interp_vision_new/compare/`).

### Behavioural results in the paper's Figure 2 format

`eacl-rebuttal/figures/behavioral-accuracies-attractors-legendtop.pdf`
(made by `eacl-rebuttal/analysis/behavioral-rebuttal.R`, the paper's
`analysis/behavioral.R` code): real-noun and language-cued lines are the
paper's, unchanged; the vision line is the new embeddings. Accuracy (%), 0 → 3
attractors (mean over seeds; language 4B has 49 seeds):

| | language (paper) | vision (new) |
|---|---|---|
| 2B | 78.5, 77.4, 72.7, 68.1 | 81.2, 75.9, 72.3, 68.4 |
| 4B | 74.9, 73.5, 60.7, 54.8 | 76.5, 71.9, 59.8, 54.5 |

The paper's 2B gap between vision and language is gone; in both models the
vision line now sits on the language line. Table:
`eacl-rebuttal/test_scores/behavioral_accuracy_language_vision.csv`.

### Blob condition (2B, 5 seeds)

Does the result depend on the creature images? Same recipe as the 2B result
(matched captions, independent init, joint bs = 8, 8-LR × 5-seed search, LR
chosen by mean dev score, same test set), with only the images changed:

- **Training images** (`data/train/im/blobs/`): one fixed, lumpy blob shape
  in every image (pink on white, the style of the circle images), varying only
  in position and rotation; 5 singular images (1 blob) and 5 plural images
  (2–4 blobs).
- **Held-out dev set** (`data/dev/held_out_images_blobs/`,
  `data/dev/dev_image_number_blobs.csv`): 10 singular + 10 plural new blob
  images × the same 6 prompts (one of which still says "What creature is in
  the image?", kept identical to the creature dev set).
- Images made by `scripts/make_blob_images.py`; runs by
  `scripts/run_lr_search_2B_bs8_blobs.sh`. Dev selected lr = 0.005 (mean dev
  score −0.006; next best −0.096).

| test accuracy | blobs (5 seeds) | creatures (same 5 seeds) | creatures (50 seeds) |
|---|---:|---:|---:|
| pooled (raw) | 0.741 [0.675, 0.807] | 0.774 [0.755, 0.793] | 0.744 [0.719, 0.770] |
| singular / plural | 0.684 / 0.798 | 0.709 / 0.839 | 0.681 / 0.808 |
| swap | 0.910 [0.868, 0.953] | 0.942 [0.934, 0.951] | 0.894 [0.850, 0.939] |
| pooled by attractors 0 / 1 / 2 / 3 | 0.77 / 0.77 / 0.73 / 0.70 | 0.85 / 0.79 / 0.75 / 0.71 | 0.81 / 0.76 / 0.72 / 0.68 |
| swap by attractors 0 / 1 / 2 / 3 | 0.87 / 0.91 / 0.95 / 0.91 | 0.95 / 0.93 / 0.95 / 0.93 | 0.91 / 0.89 / 0.91 / 0.88 |

2B learns number from abstract blobs about as well as from creatures: blob
accuracy equals the 50-seed creature result. Blobs are slightly below
creatures on the same 5 seeds, mostly at 0 attractors, but with 5 seeds the
intervals overlap, and blob seeds vary more (pooled 0.67–0.82). Test scores:
`eacl-rebuttal/test_scores/scores_qwen3-vl-2b_blobs.csv`.

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

4B's weaker scores match the paper's language-cued 4B embeddings almost
exactly, so images alone produce the same behaviour in 4B as explicit textual
number cues. The paper's mechanistic result also holds: no difference between
vision- and language-cued nouns (β = −0.068, p = .65).

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
| movement summary (both models, language + new vision) | `eacl-rebuttal/pca_data/movement_summary_language_vision.csv` |
| movement figures | `eacl-rebuttal/figures/{2b,4b}-movement-pca-{legendtop,vertical}.pdf` |
| interp figures | `eacl-rebuttal/figures/{emb-interp-results,novel-vs-real-interp}-{2B,4B}.pdf` |
| interp tables (cue-effect model, best-layer odds) | `eacl-rebuttal/interp_data/` |
| interp raw outputs | `eacl-rebuttal/results/interp_vision_new/` (server only) |
| Figure 2 (behaviour) | `eacl-rebuttal/figures/behavioral-accuracies-attractors-legendtop.pdf` |
| blob condition runs | `eacl-rebuttal/results/lr_search_2B_bs8_blobs/` (server only); scores in `test_scores/scores_qwen3-vl-2b_blobs.csv` |

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
| `eacl-rebuttal/scripts/run_interp_vision_new.sh` | reruns the paper's interventions with the new vision embeddings |
| `eacl-rebuttal/analysis/interp-rebuttal.R` | cue-condition model + interp figures in the paper's format |
| `eacl-rebuttal/analysis/behavioral-rebuttal.R` | Figure 2 in the paper's format |
| `eacl-rebuttal/scripts/make_blob_images.py` | blob training + dev images |
| `eacl-rebuttal/scripts/run_lr_search_2B_bs8_blobs.sh` | blob condition, 2B |
| `eacl-rebuttal/analysis/movement-rebuttal.R` | movement figures + tables in the paper's format (Language from `results/eval/movement-analysis/`, Vision from `pca_data/`) |
