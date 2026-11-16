#!/usr/bin/env bash
# (-o takes a regex over job names, e.g. -o "^(patch|ablation)_")
# Re-run the paper's causal-intervention analyses (Sec. 6) on the new
# vision-cued embeddings (matched captions, held-out-image dev, bs=8).
# Mirrors pipelines/interp/run_all_exps.sh exactly — same scripts, layers,
# token positions, train/test CSV construction, filters, n_sample and flags —
# and writes the same layout as results/interp/main_runs/ and
# results/interp/seed_replicate/, for the vision stream only. The paper's
# language-cue (syntax) results are reused unchanged.
#
# Embedding choice follows the paper: main grid uses each model's best-dev
# seed among its 50 (2B: 130, 4B: 116); the 2B seed replicate uses the 5
# LR-search seeds (fixed in advance, not selected on anything).
#
# Usage (from repo root):  bash eacl-rebuttal/scripts/run_interp_vision_new.sh [-n] [-o SUBSTR]
set -uo pipefail

ROOT="$HOME/wug-test-interp"
cd "$ROOT"
PY="$HOME/.venvs/wug-interp/bin/python"   # kmisra + pyvene (kmisra itself left untouched)
GPUS=(0 1 2 3)
PER_GPU=2
DRY_RUN=0
ONLY=""
while getopts ":no:" opt; do
  case "$opt" in
    n) DRY_RUN=1 ;;
    o) ONLY="$OPTARG" ;;
    *) echo "usage: $0 [-n] [-o SUBSTR]" >&2; exit 1 ;;
  esac
done

OUT="eacl-rebuttal/results/interp_vision_new"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$ROOT/$OUT/logs/$STAMP"
JOBS="$LOGDIR/jobs.tsv"
mkdir -p "$LOGDIR"
ln -sfn "$STAMP" "$ROOT/$OUT/logs/latest"

N_SAMPLE=200        # circuit methods (patch, ablation)
N_SAMPLE_CELL=400   # per-cell methods (das, diffmean, probe)

MODELS=(Qwen3-VL-2B Qwen3-VL-4B)
CONDS=(target_verb_att0_opp target_verb_att1_opp target_verb_att2_opp target_verb_att3_opp)
LAYERS_Qwen3_VL_2B="1 5 10 15 20 25 28"
LAYERS_Qwen3_VL_4B="1 5 10 15 20 25 30 35 36"
# patch/ablation k=128 as in pipelines/interp/run_circuits_k128.sh (the source of
# results/interp/main_runs/*/patch_k128.csv): these layers, and NO --toks.
CIRCUIT_LAYERS="2 7 12 17 22 27"
TOKS_target_verb_att0_opp="4 5"
TOKS_target_verb_att1_opp="7 8"
TOKS_target_verb_att2_opp="10 11"
TOKS_target_verb_att3_opp="13 14"

MAIN_SEED_Qwen3_VL_2B=130
MAIN_SEED_Qwen3_VL_4B=116
REPLICATE_SEEDS=(17 42 123 2024 98765)
SEED_CONDS=(target_verb_att0_opp target_verb_att1_opp)

layers_for()  { local v="LAYERS_${1//-/_}"; echo "${!v}"; }
toks_for()    { local v="TOKS_$1"; echo "${!v}"; }
main_seed()   { local v="MAIN_SEED_${1//-/_}"; echo "${!v}"; }
emb_for()     { echo "eacl-rebuttal/embeddings/$(echo "$1" | tr 'A-Z' 'a-z')/seed_$2.pt"; }
scored_for()  {  # behavioural test scores of that exact embedding (for is_correct_all filter)
  local m="$1" sd="$2" tag
  [[ "$m" == Qwen3-VL-2B ]] && tag="2B_bs8" || tag="4B_bs8"
  ls eacl-rebuttal/results/lr_search_${tag}/selected_test_eval/*_seed${sd}/scored.csv \
     eacl-rebuttal/results/lr_search_${tag}_more45/test_eval/*_seed${sd}/scored.csv 2>/dev/null | head -1
}

common_args() {  # <model-short> <cond> <seed> [layers]
  local m="$1" c="$2" sd="$3" layers="${4:-$(layers_for "$1")}" full="${1}-Instruct" emb scored
  emb="$(emb_for "$m" "$sd")"; scored="$(scored_for "$m" "$sd")"
  [[ -f "$emb" && -f "$scored" ]] || { echo "MISSING emb/scored for $m seed $sd" >&2; exit 1; }
  echo "--model_path Qwen/${full} --cache_dir /home/shared/hf_cache" \
       "--embeddings_path ${emb}" \
       "--train_csv results/eval/attractors/${full}/agreement_target_natural_scored.csv" \
       "--test_csv ${scored}" \
       "--source_input_col base_sentence_sg --base_input_col base_sentence_pl" \
       "--source_completion_A good_singular --source_completion_B bad_singular" \
       "--filter condition=${c} is_correct_all=TRUE" \
       "--layers ${layers}"
}

emit() {  # emit <name> <outdir> <command...>
  local name="$1" dir="$2"; shift 2
  [[ -n "$ONLY" && ! "$name" =~ $ONLY ]] && return 0
  printf '%s\t%s\t%s\n' "$name" "$dir" "$*" >> "$JOBS"
}
: > "$JOBS"

# --- main grid (as results/interp/main_runs): das/diffmean/probe + patch/ablation k=128
for m in "${MODELS[@]}"; do
  sd="$(main_seed "$m")"
  for c in "${CONDS[@]}"; do
    A="$(common_args "$m" "$c" "$sd")" || exit 1
    AC="$(common_args "$m" "$c" "$sd" "$CIRCUIT_LAYERS")" || exit 1
    T="--toks $(toks_for "$c")"
    D="$OUT/main_runs/${m}-Instruct/vision/${c}"
    emit "das_${m}_${c}" "$D" "$PY core/interp/das.py $A $T --n_sample $N_SAMPLE_CELL --add_inverse --out_csv ${D}/das.csv"
    emit "diffmean_${m}_${c}" "$D" "$PY core/interp/diffmean.py $A $T --n_sample $N_SAMPLE_CELL --add_inverse_test --out_csv ${D}/diffmean.csv"
    emit "probe_${m}_${c}" "$D" "$PY core/interp/linear_probe.py $A $T --n_sample $N_SAMPLE_CELL --add_inverse_test --out_csv ${D}/probe.csv"
    emit "patch_${m}_${c}_k128" "$D" "$PY core/interp/circuit.py $AC --n_sample $N_SAMPLE --add_inverse --hook_points mlp_act --top_k 128 --out_csv ${D}/patch_k128.csv --circuit_out ${D}/patch_nodes_k128.csv"
    emit "ablation_${m}_${c}_k128" "$D" "$PY core/interp/ablation.py $AC --n_sample $N_SAMPLE --add_inverse --hook_points mlp_act --top_k 128 --ablation mean --out_csv ${D}/ablation_k128.csv --circuit_out ${D}/ablation_nodes_k128.csv"
  done
done

# --- seed replicate (as results/interp/seed_replicate): 2B, 0-1 attractors, das/diffmean/probe
m=Qwen3-VL-2B
for c in "${SEED_CONDS[@]}"; do
  for sd in "${REPLICATE_SEEDS[@]}"; do
    A="$(common_args "$m" "$c" "$sd")" || exit 1
    T="--toks $(toks_for "$c")"
    D="$OUT/seed_replicate/${m}/vision/${c}/seed_${sd}"
    emit "das_${m}_${c}_seed${sd}" "$D" "$PY core/interp/das.py $A $T --n_sample $N_SAMPLE_CELL --add_inverse --out_csv ${D}/das.csv"
    emit "diffmean_${m}_${c}_seed${sd}" "$D" "$PY core/interp/diffmean.py $A $T --n_sample $N_SAMPLE_CELL --add_inverse_test --out_csv ${D}/diffmean.csv"
    emit "probe_${m}_${c}_seed${sd}" "$D" "$PY core/interp/linear_probe.py $A $T --n_sample $N_SAMPLE_CELL --add_inverse_test --out_csv ${D}/probe.csv"
  done
done

N=$(wc -l < "$JOBS")
echo "[$(date +%T)] $N jobs, ${#GPUS[@]} GPUs x $PER_GPU" | tee -a "$LOGDIR/master.log"
if [[ "$DRY_RUN" == 1 ]]; then cut -f1 "$JOBS" | sed 's/^/  /'; echo "dry run"; exit 0; fi
cut -f2 "$JOBS" | sort -u | while read -r d; do mkdir -p "$ROOT/$d"; done

# Number jobs so each worker can pick a GPU round-robin.
awk -v OFS='\t' '{print NR-1, $0}' "$JOBS" > "$LOGDIR/jobs_numbered.tsv"
export ROOT LOGDIR NGPU="${#GPUS[@]}" GPULIST="${GPUS[*]}"
run_one() {
  local i="$1" name="$2" cmd="$3" gpus
  read -ra gpus <<< "$GPULIST"
  cd "$ROOT" || return 1
  echo "[$(date +%T)] START $name (gpu ${gpus[$((i % NGPU))]})"
  if CUDA_VISIBLE_DEVICES="${gpus[$((i % NGPU))]}" PYTHONPATH="$ROOT:$ROOT/core/interp" \
       bash -c "$cmd" > "$LOGDIR/${name}.log" 2>&1; then
    echo "[$(date +%T)] OK    $name"
  else
    echo "[$(date +%T)] FAIL  $name -> $LOGDIR/${name}.log"
    echo "$name" >> "$LOGDIR/failed.txt"
  fi
}
export -f run_one
xargs -d '\n' -a "$LOGDIR/jobs_numbered.tsv" -P $(( ${#GPUS[@]} * PER_GPU )) -I{} bash -c '
  line="{}"
  i=$(printf "%s" "$line" | cut -f1); name=$(printf "%s" "$line" | cut -f2); cmd=$(printf "%s" "$line" | cut -f4-)
  run_one "$i" "$name" "$cmd"' 2>&1 | tee -a "$LOGDIR/master.log"

nfail=$(wc -l < "$LOGDIR/failed.txt" 2>/dev/null || echo 0)
echo "[$(date +%T)] ALL DONE. $nfail failures." | tee -a "$LOGDIR/master.log"
