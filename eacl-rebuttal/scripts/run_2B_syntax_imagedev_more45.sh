#!/usr/bin/env bash
set -euo pipefail

cd /home/km55359/wug-test-interp

out_dir="eacl-rebuttal/results/lr_search_2B_syntax_imagedev_more45"
model="Qwen/Qwen3-VL-2B-Instruct"
inits="small:/home/km55359/wug-test-interp/data/embeddings/init/noun_init.txt"
existing_per_run="eacl-rebuttal/results/lr_search_2B_syntax_imagedev/selected_test_per_run.csv"

python3 eacl-rebuttal/scripts/run_sweep.py \
  --gpus 0,1,2,3 \
  --lrs 0.01 \
  --seeds 100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,200 \
  --inits "$inits" \
  --epochs 50 \
  --terminate_cond_epochs 5 \
  --batch_mode joint \
  --batch_size 8 \
  --train_script eacl-rebuttal/scripts/embed_train_dev_image.py \
  --training_condition syntax \
  --train_csv data/embeddings/train/text/syntax_train_1.csv \
  --eval_csv eacl-rebuttal/data/dev/dev_image_number.csv \
  --model "$model" \
  --out_dir "$out_dir" \
  --init_mode independent \
  --selection_metric balanced_logistic \
  --write_all

python3 eacl-rebuttal/scripts/run_test_eval.py \
  --manifest "$out_dir/manifest.csv" \
  --model "$model" \
  --gpus 0,1,2,3 \
  --out_dir "$out_dir/test_eval" \
  --batch_size 32

python3 eacl-rebuttal/scripts/combine_50_seed_test_results.py \
  --new_manifest "$out_dir/manifest.csv" \
  --new_test_dir "$out_dir/test_eval" \
  --existing_per_run "$existing_per_run" \
  --out_dir "$out_dir"

echo "LR_SEARCH_2B_SYNTAX_IMAGEDEV_MORE45_COMPLETE"
