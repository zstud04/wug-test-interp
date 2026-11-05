#!/usr/bin/env bash
set -euo pipefail

cd /home/km55359/wug-test-interp

export HF_HOME=/home/km55359/.cache/hf_llava15
export TRANSFORMERS_CACHE=/home/km55359/.cache/hf_llava15
export HUGGINGFACE_HUB_CACHE=/home/km55359/.cache/hf_llava15

out_dir="eacl-rebuttal/results/lr_search_llava15_bs1"
model="llava-hf/llava-1.5-7b-hf"
inits="small:/home/km55359/wug-test-interp/data/embeddings/init/noun_init.txt"

python3 eacl-rebuttal/scripts/run_sweep.py \
  --gpus 0,1,2,3 \
  --lrs 0.0003,0.001,0.003,0.005,0.0075,0.01,0.015,0.02 \
  --seeds 17,42,123,2024,98765 \
  --inits "$inits" \
  --epochs 50 \
  --terminate_cond_epochs 5 \
  --batch_mode joint \
  --batch_size 1 \
  --train_script eacl-rebuttal/scripts/embed_train_dev_image.py \
  --eval_csv eacl-rebuttal/data/dev/dev_image_number.csv \
  --model "$model" \
  --cache_dir "$HF_HOME" \
  --out_dir "$out_dir" \
  --init_mode independent \
  --selection_metric balanced_logistic \
  --write_all

python3 eacl-rebuttal/scripts/select_lr_and_score.py \
  --sweep_dir "$out_dir" \
  --model "$model" \
  --cache_dir "$HF_HOME" \
  --gpus 0,1,2,3 \
  --batch_size 16

echo "LR_SEARCH_LLAVA15_BS1_COMPLETE"
