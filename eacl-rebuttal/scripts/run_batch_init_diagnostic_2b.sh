#!/usr/bin/env bash
set -euo pipefail

cd /home/km55359/wug-test-interp

out_dir="eacl-rebuttal/results/batch_init_diagnostic_2B"
model="Qwen/Qwen3-VL-2B-Instruct"

python3 eacl-rebuttal/scripts/run_batch_init_diagnostic_2b.py

python3 eacl-rebuttal/scripts/run_test_eval.py \
  --manifest "$out_dir/manifest.csv" \
  --model "$model" \
  --gpus 0,1,2,3 \
  --out_dir "$out_dir/test_eval" \
  --batch_size 32

python3 eacl-rebuttal/scripts/summarize_batch_init_diagnostic_2b.py

echo "BATCH_INIT_DIAGNOSTIC_2B_COMPLETE"
