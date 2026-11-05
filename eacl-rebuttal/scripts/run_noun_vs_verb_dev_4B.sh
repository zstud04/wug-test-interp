#!/usr/bin/env bash
set -euo pipefail

cd /home/km55359/wug-test-interp

out_dir="eacl-rebuttal/results/noun_vs_verb_dev_4B_bs8"
model="Qwen/Qwen3-VL-4B-Instruct"

python3 eacl-rebuttal/scripts/run_noun_vs_verb_dev_4B.py

python3 eacl-rebuttal/scripts/run_test_eval.py \
  --manifest "$out_dir/manifest.csv" \
  --model "$model" \
  --gpus 0,1,2,3 \
  --out_dir "$out_dir/test_eval" \
  --batch_size 32

echo "NOUN_VS_VERB_DEV_4B_BS8_COMPLETE"
