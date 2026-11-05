#!/usr/bin/env bash
# 45 more seeds for the 4B equivalent-to-2B condition (lr_search_4B_bs8, lr=0.005
# selected by dev metric). Every setting matches lr_search_2B_bs8_more45 except
# the model and its dev-selected lr. Trains, scores on the test set, then
# combines with the 5 original seeds into a 50-seed summary.
set -uo pipefail
cd /home/km55359/wug-test-interp
PY=/home/km55359/.conda/envs/kmisra/bin/python3
out_dir="eacl-rebuttal/results/lr_search_4B_bs8_more45"
seeds="100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,200"

$PY eacl-rebuttal/scripts/run_sweep.py \
  --gpus 0,1,2,3 \
  --lrs 0.005 \
  --seeds "$seeds" \
  --inits small:/home/km55359/wug-test-interp/data/embeddings/init/noun_init.txt \
  --epochs 50 --terminate_cond_epochs 5 \
  --batch_mode joint --batch_size 8 \
  --train_script eacl-rebuttal/scripts/embed_train_dev_image.py \
  --train_csv /home/km55359/wug-test-interp/eacl-rebuttal/data/train/text/image_train_matched_v2.csv \
  --eval_csv eacl-rebuttal/data/dev/dev_image_number.csv \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --out_dir "$out_dir" \
  --init_mode independent \
  --selection_metric balanced_logistic \
  --write_all || { echo "TRAINING FAILED"; exit 1; }

n=$(find "$out_dir" -name learned_embeddings.pt | wc -l)
echo "$(date) trained runs: $n / 45"
if [ "$n" -lt 45 ]; then echo "NOT ALL RUNS TRAINED - stopping before scoring"; exit 1; fi

$PY eacl-rebuttal/scripts/run_test_eval.py \
  --manifest "$out_dir/manifest.csv" \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --gpus 0,1,2,3 \
  --out_dir "$out_dir/test_eval" \
  --batch_size 32 || { echo "SCORING FAILED"; exit 1; }

$PY eacl-rebuttal/scripts/combine_50_seed_test_results.py \
  --new_manifest "$out_dir/manifest.csv" \
  --new_test_dir "$out_dir/test_eval" \
  --existing_per_run eacl-rebuttal/results/lr_search_4B_bs8/selected_test_per_run.csv \
  --out_dir "$out_dir" || { echo "COMBINE FAILED"; exit 1; }

echo "$(date) ALL DONE - see $out_dir/combined_50seed_test_summary.csv"
