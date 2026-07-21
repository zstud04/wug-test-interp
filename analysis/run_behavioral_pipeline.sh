#!/usr/bin/env bash
# Run minimal-pair minicons evaluation (analysis/behavioral.py) on the attractor
# stimuli for both Qwen3 VL models, across all CI seed runs (text + image).
#
# Usage:
#   bash analysis/run_behavioral_pipeline.sh [--device cuda:0] [--batch_size 8]
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
BATCH_SIZE=8
STIMULI_CSV="data/interp/agreement_target_wug.csv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)      DEVICE="$2"; shift 2 ;;
        --batch_size)  BATCH_SIZE="$2"; shift 2 ;;
        --stimuli_csv) STIMULI_CSV="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

CI_DIR="results/train/CI_seed_runs"
MODELS=("Qwen/Qwen3-VL-4B-Instruct" "Qwen/Qwen3-VL-2B-Instruct")

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    MODEL_TAG="$(basename "$MODEL")"
    TEXT_DIR="$CI_DIR/lr_ci_results_${MODEL_TAG}_text_lr0p001_50seeds"
    IMAGE_DIR="$CI_DIR/lr_ci_results_${MODEL_TAG}_image_lr0p001_50seeds"
    OUT_DIR="results/eval/attractors-50seeds/${MODEL_TAG}"

    echo "========================================================"
    echo "  Step $((i + 1))/${#MODELS[@]} — Behavioral eval: ${MODEL_TAG}"
    echo "========================================================"
    python3 analysis/behavioral.py \
        --model "$MODEL" \
        --text_dir  "$TEXT_DIR" \
        --image_dir "$IMAGE_DIR" \
        --stimuli_csv "$STIMULI_CSV" \
        --out_dir   "$OUT_DIR" \
        --device    "$DEVICE" \
        --batch_size "$BATCH_SIZE"
    echo ""
done

echo "========================================================"
echo "  All done. Results written to:"
for MODEL in "${MODELS[@]}"; do
    echo "    results/eval/attractors-50seeds/$(basename "$MODEL")"
done
echo "========================================================"
