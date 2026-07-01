# Representational Analysis

This directory contains scripts for analyzing the geometry of learned `[wug]` and `[wugs]` embeddings after training. The analysis runs in two stages — initialization and analysis — across both Qwen3-VL model sizes (2B and 4B) and both training conditions (text and image).

## Scripts

### `embed_init.py`
Reconstructs the initial `[wug]`/`[wugs]` embeddings that were used at the start of each training run and saves them as `initial_embeddings.pt` alongside each run's `learned_embeddings.pt`.

Initialization is deterministic given the seed: it resets the embedding matrix to the pretrained baseline, computes the mean embedding of the words in `noun_init.txt`, and adds seed-scaled random noise (noise scale = mean pairwise distance between singular/plural noun pairs). This mirrors the initialization logic in `core/train/embed_train.py`.

Skips seeds that already have `initial_embeddings.pt` unless `--force` is set.

### `embed_analysis.py`
Loads the initial and learned embeddings for all seeds, projects them into a PCA space defined by real singular/plural noun embeddings, and measures how much each `[wug]`/`[wugs]` embedding moved along the singular→plural axis during training. Also computes the top-K nearest neighbors of each learned embedding by cosine similarity.

**Reference noun selection:** the PCA space is anchored by up to 2000 singular/plural noun pairs drawn from WordNet, filtered to retain only unambiguous, concrete nouns. Specifically, a word is included only if: (1) at least 50% of its WordNet senses are nouns, (2) its primary noun sense falls in one of five concrete categories (`noun.animal`, `noun.artifact`, `noun.object`, `noun.plant`, `noun.food`), (3) both the singular and its inflected plural are single tokens in the model's vocabulary, (4) it contains only alphabetic characters (no hyphens or underscores), and (5) it is not already a plural form. Candidates passing all filters are ranked by English word frequency (via `wordfreq`) and the top 2000 are kept.

**Outputs** (written to `--out_dir`, one set per model):

| File | Description |
|------|-------------|
| `sg_pl_reduced.csv` | PCA (x, y) coordinates for all reference singular and plural nouns |
| `wug_wugs_reduced_{text,image}.csv` | PCA positions for `[wug]`/`[wugs]` at init and final, per seed |
| `wug_wugs_movement_{text,image}.csv` | Scalar movement along the sg→pl PCA axis, per seed |
| `neighbors_{text,image}.csv` | Top-K cosine neighbors of the learned `[wug]`/`[wugs]` embeddings, per seed |

The `neighbors_{condition}.csv` format:
```
seed,wug_neighbors,wugs_neighbors
388,dog|cat|bird|fish|wolf,dogs|cats|birds|fish|wolves
...
```
Neighbors are from the original pretrained vocabulary (the `[wug]` and `[wugs]` tokens themselves are excluded).

## Running the pipeline

From the repo root:

```bash
bash representational-analysis/run_embed_pipeline.sh
```

This runs four steps in order:
1. Embed init — 4B model
2. Movement analysis + neighbors — 4B model
3. Embed init — 2B model
4. Movement analysis + neighbors — 2B model

Results are written to:
- `results/movement-analysis/Qwen_Qwen3-VL-4B-Instruct/`
- `results/movement-analysis/Qwen_Qwen3-VL-2B-Instruct/`

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `cuda:0` | GPU device |
| `--force` | off | Regenerate `initial_embeddings.pt` even where it exists |
| `--top_k` | `5` | Number of nearest neighbors to compute |

```bash
# Example: different GPU, 10 neighbors
bash representational-analysis/run_embed_pipeline.sh --device cuda:1 --top_k 10
```

Set `HF_TOKEN` in your environment if model access requires authentication.
