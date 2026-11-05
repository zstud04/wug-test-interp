#!/usr/bin/env bash
set -euo pipefail

cd /home/km55359/wug-test-interp

out_dir="eacl-rebuttal/results/lr_search_2B_bs8"
model="Qwen/Qwen3-VL-2B-Instruct"
inits="small:/home/km55359/wug-test-interp/data/embeddings/init/noun_init.txt"

python3 eacl-rebuttal/scripts/run_sweep.py \
  --gpus 0,1,2,3 \
  --lrs 0.0003,0.001,0.003,0.005,0.0075,0.01,0.015,0.02 \
  --seeds 17,42,123,2024,98765 \
  --inits "$inits" \
  --epochs 50 \
  --terminate_cond_epochs 5 \
  --batch_mode joint \
  --batch_size 8 \
  --train_script eacl-rebuttal/scripts/embed_train_dev_image.py \
  --eval_csv eacl-rebuttal/data/dev/dev_image_number.csv \
  --model "$model" \
  --out_dir "$out_dir" \
  --init_mode independent \
  --selection_metric balanced_logistic \
  --write_all

python3 eacl-rebuttal/scripts/select_lr_and_score.py \
  --sweep_dir "$out_dir" \
  --model "$model" \
  --gpus 0,1,2,3 \
  --batch_size 32

echo "LR_SEARCH_2B_BS8_COMPLETE"
