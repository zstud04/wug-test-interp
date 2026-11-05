# EACL rebuttal: image-only wug/wugs number learning

Self-contained experiment folder testing whether Qwen3-VL models learn that
novel tokens `[wug]`/`[wugs]` mark singular/plural **from images alone**,
with captions that carry no number information. Training uses a fork of
`core/train/embed_train.py` kept in `scripts/`. Outside this folder, the work
depends on three repo-level changes: the test scorer
`core/eval/embed_eval.py` (now loads models via `utils/vlm_loading.py` and
handles untied embeddings; scoring is unchanged), the new
`utils/vlm_loading.py`, and the root `.gitignore` entry for
`eacl-rebuttal/results/`.

- **Method + final results (2B and 4B, 50 seeds each), with paths to all
  embeddings and results:** `METHODS_AND_RESULTS.md`
- **Record of kept vs. deleted experiments:** `RESULTS_LOG.md`
- **In git vs. server only:** `results/` is git-ignored (repo-root
  `.gitignore`). The final embeddings and item-level test scores are copied
  outside it, in compact form, to `embeddings/` and `test_scores/`, and the
  PCA figure inputs to `pca_data/` — see
  "What's in git" in `METHODS_AND_RESULTS.md`.

## Data

- `data/train/im/creature_1/` — training images: 5 singular (one novel
  creature) + 5 plural (two creatures), `singular{1..5}.png` /
  `plural{1..5}.png`.
- `data/train/text/image_train_matched_v2.csv` — 16 caption templates, each
  used once with `[wug]` and once with `[wugs]`, identical otherwise. The token
  never governs a verb and never follows a number-marked word.
- `data/dev/dev_image_number.csv` + `data/dev/held_out_images/` — dev set:
  10 singular + 10 plural held-out images of the same creature (none used in
  training; source archives `data/singular.zip`, `data/plural.zip`), crossed
  with 6 QA templates = 120 stimuli. Used only for early stopping and LR
  selection.
- Test set (outside this folder, unmodified):
  `data/interp/agreement_target_wug.csv`.
- Init nouns (outside this folder, unmodified):
  `data/embeddings/init/noun_init.txt`.

## Pipeline

```bash
cd wug-test-interp
PY=/home/km55359/.conda/envs/kmisra/bin/python3   # the system python3 lacks pandas

# 1. LR search: 8 LRs x 5 seeds (see scripts/run_lr_search_2B_bs8.sh / _4B_bs8.sh)
# 2. pick the LR with the best mean dev score and score it on the test set
$PY eacl-rebuttal/scripts/select_lr_and_score.py ...
# 3. 45 more seeds at that LR, test scoring, combine into 50 seeds
#    (scripts/run_lr_search_2B_bs8_more45.sh, scripts/run_4B_independent_more45.sh)
# 4. analyses
$PY eacl-rebuttal/scripts/swap_accuracy.py        # -> results/swap_accuracy/
$PY eacl-rebuttal/scripts/movement_pca.py ...     # -> results/movement-analysis/<model>/ (git-ignored)
$PY eacl-rebuttal/scripts/export_pca_data.py      # -> pca_data/ (tracked: what the figures read)
$PY eacl-rebuttal/scripts/plot_movement_pca.py    # -> figures/   (or source analysis/movement-rebuttal.R)
```

Always pass `--image_dir` / `--train_csv` explicitly or check the
`run_config.csv` of a finished run against the kept results: `run_sweep.py`
has defaults, and runs launched with different defaults are not comparable.

Model/cache note: this machine's HF cache is `/home/shared/hf_cache`
(`HF_HOME`), not the `/mnt/dv/...` default hardcoded in
`core/train/embed_train.py` / `core/eval/embed_eval.py`. Scripts in this folder
default `--cache_dir` to `/home/shared/hf_cache`.
