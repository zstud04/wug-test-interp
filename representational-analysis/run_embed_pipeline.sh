#!/usr/bin/env bash
# Reproduce embedding initialization and movement analysis for both Qwen3 VL models.
#
# Usage:
#   bash representational-analysis/run_embed_pipeline.sh [--device cuda:0] [--force] [--top_k 5]
#
# Set HF_TOKEN in your environment if the models require authentication.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [[ "$REPO_ROOT" != "/" && ! -d "$REPO_ROOT/core" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -d "$REPO_ROOT/core" ]] || { echo "ERROR: could not find repo root (no core/ dir above $SCRIPT_DIR)" >&2; exit 1; }
cd "$REPO_ROOT"

DEVICE="cuda:0"
FORCE=""
TOP_K=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --force)  FORCE="--force"; shift ;;
        --top_k)  TOP_K="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

CI_DIR="results/train/CI_seed_runs"
NOUN_INIT="data/embeddings/init/noun_init.txt"

# ---------------------------------------------------------------------------
# 4B model
# ---------------------------------------------------------------------------
MODEL_4B="Qwen/Qwen3-VL-4B-Instruct"
TEXT_DIR_4B="$CI_DIR/lr_ci_results_Qwen3-VL-4B-Instruct_text_lr0p001_50seeds"
IMAGE_DIR_4B="$CI_DIR/lr_ci_results_Qwen3-VL-4B-Instruct_image_lr0p001_50seeds"
OUT_DIR_4B="results/movement-analysis/Qwen_Qwen3-VL-4B-Instruct"

echo "========================================================"
echo "  Step 1/4 — Embed init: 4B"
echo "========================================================"
python3 representational-analysis/embed_init.py \
    --model "$MODEL_4B" \
    --results_dirs "$TEXT_DIR_4B" "$IMAGE_DIR_4B" \
    --noun_init "$NOUN_INIT" \
    --device "$DEVICE" \
    $FORCE

echo ""
echo "========================================================"
echo "  Step 2/4 — Movement analysis + neighbors: 4B"
echo "========================================================"
python3 representational-analysis/embed_analysis.py \
    --model "$MODEL_4B" \
    --text_dir  "$TEXT_DIR_4B" \
    --image_dir "$IMAGE_DIR_4B" \
    --out_dir   "$OUT_DIR_4B" \
    --device    "$DEVICE" \
    --top_k     "$TOP_K"

# ---------------------------------------------------------------------------
# 2B model
# ---------------------------------------------------------------------------
MODEL_2B="Qwen/Qwen3-VL-2B-Instruct"
TEXT_DIR_2B="$CI_DIR/lr_ci_results_Qwen3-VL-2B-Instruct_text_lr0p001_50seeds"
IMAGE_DIR_2B="$CI_DIR/lr_ci_results_Qwen3-VL-2B-Instruct_image_lr0p001_50seeds"
OUT_DIR_2B="results/movement-analysis/Qwen_Qwen3-VL-2B-Instruct"

echo ""
echo "========================================================"
echo "  Step 3/4 — Embed init: 2B"
echo "========================================================"
python3 representational-analysis/embed_init.py \
    --model "$MODEL_2B" \
    --results_dirs "$TEXT_DIR_2B" "$IMAGE_DIR_2B" \
    --noun_init "$NOUN_INIT" \
    --device "$DEVICE" \
    $FORCE

echo ""
echo "========================================================"
echo "  Step 4/4 — Movement analysis + neighbors: 2B"
echo "========================================================"
python3 representational-analysis/embed_analysis.py \
    --model "$MODEL_2B" \
    --text_dir  "$TEXT_DIR_2B" \
    --image_dir "$IMAGE_DIR_2B" \
    --out_dir   "$OUT_DIR_2B" \
    --device    "$DEVICE" \
    --top_k     "$TOP_K"

echo ""
echo "========================================================"
echo "  All done. Results written to:"
echo "    $OUT_DIR_4B"
echo "    $OUT_DIR_2B"
echo "========================================================"
