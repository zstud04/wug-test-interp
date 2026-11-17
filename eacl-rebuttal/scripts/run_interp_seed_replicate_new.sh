#!/usr/bin/env bash
# Seed check for the mechanistic results (paper appendix F.1) with the NEW
# vision embeddings, 2B, all attractor counts. Mirrors
# pipelines/interp/run_seed_replicate.sh exactly: das/diffmean/probe at the
# paper's layers and token positions, patch/ablation k=128 at layers
# 2 7 12 17 22 27, n_sample 400 / 200, and -- as in that script -- test items
# selected (is_correct_all filter) using the MAIN embedding's test scores for
# every seed (new vision: seed 130's scored.csv).
#
# Also fills the two language-cued DAS runs at 3 attractors missing from the
# paper's results/interp/seed_replicate/ (seeds 54321, 96684), with the paper's
# own settings (main language embedding's test scores).
#
# Vision seeds: the 5 LR-search seeds (fixed in advance). Jobs for 2-3
# attractors run first.
#
# Usage (from repo root): bash eacl-rebuttal/scripts/run_interp_seed_replicate_new.sh [-g "0"] [-p 3] [-n]
set -uo pipefail

ROOT="$HOME/wug-test-interp"
cd "$ROOT"
PY="$HOME/.venvs/wug-interp/bin/python"
GPULIST="0"
PER_GPU=3
DRY_RUN=0
while getopts ":g:p:n" opt; do
  case "$opt" in
    g) GPULIST="$OPTARG" ;;
    p) PER_GPU="$OPTARG" ;;
    n) DRY_RUN=1 ;;
    *) echo "usage: $0 [-g 'GPU IDS'] [-p PER_GPU] [-n]" >&2; exit 1 ;;
  esac
done
read -ra GPUS <<< "$GPULIST"

OUT="eacl-rebuttal/results/interp_vision_new/seed_replicate_paper_items"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$ROOT/$OUT/logs/$STAMP"
JOBS="$LOGDIR/jobs.tsv"
mkdir -p "$LOGDIR"
ln -sfn "$STAMP" "$ROOT/$OUT/logs/latest"

M=Qwen3-VL-2B; FULL=Qwen3-VL-2B-Instruct
N_SAMPLE=200; N_SAMPLE_CELL=400; K=128
PL="--layers 1 5 10 15 20 25 28"
CL="--layers 2 7 12 17 22 27"
declare -A TOKS=([target_verb_att0_opp]="4 5" [target_verb_att1_opp]="7 8" [target_verb_att2_opp]="10 11" [target_verb_att3_opp]="13 14")
CONDS=(target_verb_att2_opp target_verb_att3_opp target_verb_att0_opp target_verb_att1_opp)
VISION_SEEDS=(17 42 123 2024 98765)
TRAIN_CSV="results/eval/attractors/${FULL}/agreement_target_natural_scored.csv"
VISION_TEST_CSV="$(ls eacl-rebuttal/results/lr_search_2B_bs8_more45/test_eval/*_seed130/scored.csv)"
LANG_TEST_CSV="results/eval/attractors/${FULL}/syntax/agreement_target_wug_scored.csv"

common() {  # <embedding> <test_csv> <cond>
  echo "--model_path Qwen/${FULL} --cache_dir /home/shared/hf_cache --embeddings_path $1" \
       "--train_csv ${TRAIN_CSV} --test_csv $2" \
       "--source_input_col base_sentence_sg --base_input_col base_sentence_pl" \
       "--source_completion_A good_singular --source_completion_B bad_singular" \
       "--filter condition=$3 is_correct_all=TRUE"
}

: > "$JOBS"
emit() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$JOBS"; }

for c in "${CONDS[@]}"; do
  for sd in "${VISION_SEEDS[@]}"; do
    emb="eacl-rebuttal/embeddings/qwen3-vl-2b/seed_${sd}.pt"
    A="$(common "$emb" "$VISION_TEST_CSV" "$c")"; T="--toks ${TOKS[$c]}"
    D="$OUT/${M}/vision/${c}/seed_${sd}"; tag="${M}_vision_${c}_seed${sd}"
    emit "das_${tag}" "$D" "$PY core/interp/das.py $A $PL $T --n_sample $N_SAMPLE_CELL --add_inverse --out_csv ${D}/das.csv"
    emit "diffmean_${tag}" "$D" "$PY core/interp/diffmean.py $A $PL $T --n_sample $N_SAMPLE_CELL --add_inverse_test --out_csv ${D}/diffmean.csv"
    emit "probe_${tag}" "$D" "$PY core/interp/linear_probe.py $A $PL $T --n_sample $N_SAMPLE_CELL --add_inverse_test --out_csv ${D}/probe.csv"
    emit "patch_${tag}" "$D" "$PY core/interp/circuit.py $A $CL --n_sample $N_SAMPLE --add_inverse --hook_points mlp_act --top_k $K --out_csv ${D}/patch_k${K}.csv --circuit_out ${D}/patch_nodes_k${K}.csv"
    emit "ablation_${tag}" "$D" "$PY core/interp/ablation.py $A $CL --n_sample $N_SAMPLE --add_inverse --hook_points mlp_act --top_k $K --ablation mean --out_csv ${D}/ablation_k${K}.csv --circuit_out ${D}/ablation_nodes_k${K}.csv"
  done
  if [[ "$c" == target_verb_att3_opp ]]; then
    for sd in 54321 96684; do
      A="$(common "embeddings/${M}/syntax_seeds/seed_${sd}.pt" "$LANG_TEST_CSV" "$c")"; T="--toks ${TOKS[$c]}"
      D="$OUT/${M}/syntax/${c}/seed_${sd}"
      emit "das_${M}_syntax_${c}_seed${sd}" "$D" "$PY core/interp/das.py $A $PL $T --n_sample $N_SAMPLE_CELL --add_inverse --out_csv ${D}/das.csv"
    done
  fi
done

N=$(wc -l < "$JOBS")
echo "[$(date +%T)] $N jobs, GPUs [${GPUS[*]}] x $PER_GPU" | tee -a "$LOGDIR/master.log"
if [[ "$DRY_RUN" == 1 ]]; then cut -f1 "$JOBS" | sed 's/^/  /'; exit 0; fi
cut -f2 "$JOBS" | sort -u | while read -r d; do mkdir -p "$ROOT/$d"; done
awk -v OFS='\t' '{print NR-1, $0}' "$JOBS" > "$LOGDIR/jobs_numbered.tsv"
export ROOT LOGDIR NGPU="${#GPUS[@]}" GPULIST="${GPUS[*]}"
run_one() {
  local i="$1" name="$2" cmd="$3" gpus; read -ra gpus <<< "$GPULIST"
  cd "$ROOT" || return 1
  local g="${gpus[$((i % NGPU))]}"
  echo "[$(date +%T)] START $name (gpu $g)"
  if CUDA_VISIBLE_DEVICES="$g" PYTHONPATH="$ROOT:$ROOT/core/interp" bash -c "$cmd" > "$LOGDIR/${name}.log" 2>&1; then
    echo "[$(date +%T)] OK    $name"
  else
    echo "[$(date +%T)] FAIL  $name -> $LOGDIR/${name}.log"; echo "$name" >> "$LOGDIR/failed.txt"
  fi
}
export -f run_one
xargs -d '\n' -a "$LOGDIR/jobs_numbered.tsv" -P $(( ${#GPUS[@]} * PER_GPU )) -I{} bash -c '
  line="{}"; i=$(printf "%s" "$line" | cut -f1); name=$(printf "%s" "$line" | cut -f2); cmd=$(printf "%s" "$line" | cut -f4-)
  run_one "$i" "$name" "$cmd"' 2>&1 | tee -a "$LOGDIR/master.log"
nfail=$( [ -f "$LOGDIR/failed.txt" ] && wc -l < "$LOGDIR/failed.txt" || echo 0 )
echo "[$(date +%T)] ALL DONE. $nfail failures." | tee -a "$LOGDIR/master.log"
