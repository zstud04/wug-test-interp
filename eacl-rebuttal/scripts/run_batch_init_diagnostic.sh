#!/usr/bin/env bash
set -euo pipefail

cd /home/km55359/wug-test-interp

out_dir="eacl-rebuttal/results/batch_init_diagnostic_4B"
model="Qwen/Qwen3-VL-4B-Instruct"

python3 eacl-rebuttal/scripts/run_batch_init_diagnostic.py

python3 eacl-rebuttal/scripts/run_test_eval.py \
  --manifest "$out_dir/manifest.csv" \
  --model "$model" \
  --gpus 0,1,2,3 \
  --out_dir "$out_dir/test_eval" \
  --batch_size 32

python3 eacl-rebuttal/scripts/summarize_batch_init_diagnostic.py

echo "BATCH_INIT_DIAGNOSTIC_4B_COMPLETE"
