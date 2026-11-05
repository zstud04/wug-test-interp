#!/usr/bin/env bash
set -euo pipefail

cd /home/km55359/wug-test-interp

export HF_HOME=/home/km55359/.cache/hf_smolvlm2
export TRANSFORMERS_CACHE=/home/km55359/.cache/hf_smolvlm2
export HUGGINGFACE_HUB_CACHE=/home/km55359/.cache/hf_smolvlm2

model="HuggingFaceTB/SmolVLM2-2.2B-Instruct"
inits="small:/home/km55359/wug-test-interp/data/embeddings/init/noun_init.txt"

# --- Stage 1: independent-init LR sweep (re-run; prior results were deleted) ---
indep_out="eacl-rebuttal/results/lr_search_smolvlm2_bs2"

python3 eacl-rebuttal/scripts/run_sweep.py \
  --gpus 0,1,2,3 \
  --lrs 0.0003,0.001,0.003,0.005,0.0075,0.01,0.015,0.02 \
  --seeds 17,42,123,2024,98765 \
  --inits "$inits" \
  --epochs 50 \
  --terminate_cond_epochs 5 \
  --batch_mode joint \
  --batch_size 2 \
  --train_script eacl-rebuttal/scripts/embed_train_dev_image.py \
  --eval_csv eacl-rebuttal/data/dev/dev_image_number.csv \
  --model "$model" \
  --cache_dir "$HF_HOME" \
  --out_dir "$indep_out" \
  --init_mode independent \
  --selection_metric balanced_logistic \
  --write_all

python3 eacl-rebuttal/scripts/select_lr_and_score.py \
  --sweep_dir "$indep_out" \
  --model "$model" \
  --cache_dir "$HF_HOME" \
  --gpus 0,1,2,3 \
  --batch_size 16

echo "LR_SEARCH_SMOLVLM2_BS2_COMPLETE"

# --- Stage 2: pick the best LR from stage 1, then sweep spherical_symmetric separation scale at that LR ---
best_lr=$(python3 -c "
import csv
with open('$indep_out/selected_configs.csv') as f:
    row = next(csv.DictReader(f))
    print(row['lr'])
")
echo "Selected best LR from independent sweep: $best_lr"

spherical_out="eacl-rebuttal/results/spherical_symmetric_smolvlm2_bs2"

python3 eacl-rebuttal/scripts/run_sweep.py \
  --gpus 0,1,2,3 \
  --lrs "$best_lr" \
  --seeds 17,42,123,2024,98765 \
  --inits "$inits" \
  --epochs 50 \
  --terminate_cond_epochs 5 \
  --batch_mode joint \
  --batch_size 2 \
  --train_script eacl-rebuttal/scripts/embed_train_dev_image.py \
  --eval_csv eacl-rebuttal/data/dev/dev_image_number.csv \
  --model "$model" \
  --cache_dir "$HF_HOME" \
  --out_dir "$spherical_out" \
  --init_mode spherical_symmetric \
  --symmetric_separation_scales 0,0.25,0.5,1,1.5 \
  --selection_metric balanced_logistic \
  --write_all

python3 eacl-rebuttal/scripts/select_lr_and_score.py \
  --sweep_dir "$spherical_out" \
  --model "$model" \
  --cache_dir "$HF_HOME" \
  --gpus 0,1,2,3 \
  --batch_size 16

echo "SPHERICAL_SCALE_SMOLVLM2_BS2_COMPLETE"
