#!/usr/bin/env bash
set -euo pipefail

cd /home/km55359/wug-test-interp

out_dir="eacl-rebuttal/results/epoch_trajectory_diagnostic_4B"
model="Qwen/Qwen3-VL-4B-Instruct"

gpu=0
for seed in 17 42; do
  CUDA_VISIBLE_DEVICES=$gpu python3 eacl-rebuttal/scripts/embed_train_dev_image.py \
    --training_condition image \
    --train_csv eacl-rebuttal/data/train/text/image_train_matched_v2.csv \
    --eval_csv eacl-rebuttal/data/dev/dev_image_number.csv \
    --image_dir eacl-rebuttal/data/train/im/creature_1 \
    --embed_init data/embeddings/init/noun_init.txt \
    --lr 0.01 \
    --seed $seed \
    --epochs 50 \
    --terminate_cond_epochs 999 \
    --batch_mode joint \
    --batch_size 8 \
    --model "$model" \
    --cache_dir /home/shared/hf_cache \
    --out_dir "$out_dir/seed_$seed" \
    --init_mode independent \
    --selection_metric balanced_logistic \
    --write_embeddings \
    --write_all &
  gpu=$((gpu+1))
done
wait

echo "EPOCH_TRAJECTORY_DIAGNOSTIC_4B_TRAINING_COMPLETE"
