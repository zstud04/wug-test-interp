#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [[ "$REPO_ROOT" != "/" && ! -d "$REPO_ROOT/core" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -d "$REPO_ROOT/core" ]] || { echo "ERROR: no core/ above $SCRIPT_DIR" >&2; exit 1; }
cd "$REPO_ROOT"

CONDITION=""
TRAIN_CSV=""
EVAL_CSV=""
IMAGE_DIR=""
EMBED_INIT=""
LRS=""
N_SEEDS=""
EPOCHS=50
BATCH_MODE="alternating"
TERMINATE_COND_EPOCHS=5
OUT_DIR="results/lr_sweep_results"
SEEDS_CSV=""
FOREGROUND=0
DETACHED=0

ORIG_ARGS=("$@")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --condition)              CONDITION="$2"; shift 2;;
        --train_csv)              TRAIN_CSV="$2"; shift 2;;
        --eval_csv)               EVAL_CSV="$2"; shift 2;;
        --image_dir)              IMAGE_DIR="$2"; shift 2;;
        --embed_init)             EMBED_INIT="$2"; shift 2;;
        --lrs)                    LRS="$2"; shift 2;;
        --n_seeds)                N_SEEDS="$2"; shift 2;;
        --seeds)                  SEEDS_CSV="$2"; shift 2;;
        --epochs)                 EPOCHS="$2"; shift 2;;
        --batch_mode)             BATCH_MODE="$2"; shift 2;;
        --terminate_cond_epochs)  TERMINATE_COND_EPOCHS="$2"; shift 2;;
        --out_dir)                OUT_DIR="$2"; shift 2;;
        --foreground)             FOREGROUND=1; shift;;
        --__detached)             DETACHED=1; shift;;
        *) echo "Unknown arg: $1" >&2; exit 1;;
    esac
done

[[ -z "$CONDITION" ]] && { echo "ERROR: --condition required (image|text)" >&2; exit 1; }
[[ "$CONDITION" != "image" && "$CONDITION" != "text" ]] && { echo "ERROR: --condition must be image|text" >&2; exit 1; }
[[ -z "$TRAIN_CSV" ]] && { echo "ERROR: --train_csv required" >&2; exit 1; }
[[ -z "$EVAL_CSV"  ]] && { echo "ERROR: --eval_csv required" >&2; exit 1; }
[[ -z "$LRS"       ]] && { echo "ERROR: --lrs required" >&2; exit 1; }
[[ -z "$SEEDS_CSV" && -z "$N_SEEDS" ]] && { echo "ERROR: --n_seeds or --seeds required" >&2; exit 1; }
[[ "$CONDITION" == "image" && -z "$IMAGE_DIR" ]] && { echo "ERROR: --image_dir required for image" >&2; exit 1; }

[[ "$CONDITION" == "image" ]] && TC="image" || TC="syntax"

mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$OUT_DIR/sweep_master.log"

if [[ "$DETACHED" -eq 0 && "$FOREGROUND" -eq 0 ]]; then
    SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    nohup bash "$SELF" "${ORIG_ARGS[@]}" --__detached > "$MASTER_LOG" 2>&1 &
    echo "PID $! | log: $MASTER_LOG | tail -f $MASTER_LOG"
    exit 0
fi

if [[ -n "$SEEDS_CSV" ]]; then
    IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
else
    SEEDS=()
    for ((i=0; i<N_SEEDS; i++)); do
        SEEDS+=( "$(( RANDOM * RANDOM % 100000 ))" )
    done
fi
IFS=',' read -r -a LR_ARR <<< "$LRS"

echo "sweep start $(date) | cond=$CONDITION lrs=${LR_ARR[*]} seeds=${SEEDS[*]} ep=$EPOCHS"

for LR in "${LR_ARR[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN_OUT="$OUT_DIR/lr_${LR}/seed_${SEED}"
        mkdir -p "$RUN_OUT"
        LOG_FILE="$LOG_DIR/lr_${LR}_seed_${SEED}.log"
        echo ">>> lr=$LR seed=$SEED ($(date '+%H:%M:%S'))"

        CMD=( python3 -u -m core.train.embed_train
              --training_condition "$TC"
              --train_csv  "$TRAIN_CSV"
              --eval_csv   "$EVAL_CSV"
              --lr "$LR" --seed "$SEED" --epochs "$EPOCHS"
              --batch_mode "$BATCH_MODE"
              --terminate_cond_epochs "$TERMINATE_COND_EPOCHS"
              --out_dir "$RUN_OUT"
              --write_embeddings)
        [[ "$TC" == "image" ]] && CMD+=( --image_dir "$IMAGE_DIR" )
        [[ -n "$EMBED_INIT" ]] && CMD+=( --embed_init "$EMBED_INIT" )

        stdbuf -oL -eL "${CMD[@]}" 2>&1 | stdbuf -oL tee "$LOG_FILE"
    done
done

echo "sweep done $(date)"