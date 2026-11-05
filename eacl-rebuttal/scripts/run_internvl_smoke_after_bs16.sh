#!/usr/bin/env bash
set -euo pipefail

cd /home/km55359/wug-test-interp

# The batch-16 watcher (PID supplied by the launcher) owns the complete
# train -> checkpoint-select -> test-score chain.  Do not contend for its GPUs.
bs16_watcher_pid="${1:?usage: $0 BS16_WATCHER_PID}"
while kill -0 "$bs16_watcher_pid" 2>/dev/null; do
    sleep 30
done

# The model download is deliberately concurrent with the Qwen experiment.
while tmux has-session -t internvl_download 2>/dev/null; do
    sleep 15
done

model="OpenGVLab/InternVL3_5-2B-HF"
out_dir="eacl-rebuttal/results/internvl3_5_2b_smoke"

python3 eacl-rebuttal/scripts/run_sweep.py \
    --gpus 0 \
    --lrs 0.01 \
    --seeds 17 \
    --inits small:/home/km55359/wug-test-interp/data/embeddings/init/noun_init.txt \
    --epochs 1 \
    --terminate_cond_epochs 1 \
    --batch_mode joint \
    --batch_size 16 \
    --train_script eacl-rebuttal/scripts/embed_train_dev_image.py \
    --eval_csv eacl-rebuttal/data/dev/dev_image_number.csv \
    --model "$model" \
    --out_dir "$out_dir" \
    --write_all

python3 eacl-rebuttal/scripts/run_test_eval.py \
    --manifest "$out_dir/manifest.csv" \
    --model "$model" \
    --gpus 0 \
    --out_dir "$out_dir/test_eval" \
    --batch_size 32

echo "INTERNVL_SMOKE_COMPLETE"
