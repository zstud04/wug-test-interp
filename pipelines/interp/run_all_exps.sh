#!/usr/bin/env bash

set -uo pipefail


if [[ "${_RUN_ALL_DAEMON:-0}" != 1 && " $* " != *" -n "* && " $* " != *" -F "* ]]; then
  export _RUN_ALL_DAEMON=1
  _dir="$HOME/wug-test-interp/results/full_methods_run"
  mkdir -p "$_dir"
  _out="$_dir/nohup.out"
  setsid nohup bash "$0" "$@" > "$_out" 2>&1 &
  _pid=$!
  sleep 2
  echo "=============================================================="
  echo " detached. master PID : $_pid"
  echo " kill with            : kill -TERM -$_pid"
  echo " stdout               : $_out"
  echo " follow               : tail -f $_dir/logs/latest/master.log"
  echo "=============================================================="
  # If the daemon already died (bad env, failed preflight), show why.
  if ! kill -0 "$_pid" 2>/dev/null; then
    echo "--- daemon exited immediately; $_out says: ---"
    cat "$_out"
    exit 1
  fi
  exit 0
fi

NPROC=6
DRY_RUN=0
ONLY=""
FORCE=0
while getopts ":j:no:fF" opt; do
  case "$opt" in
    j) NPROC="$OPTARG" ;;
    n) DRY_RUN=1 ;;
    o) ONLY="$OPTARG" ;;
    f) FORCE=1 ;;
    F) : ;;                 # handled above; consumed here so getopts is happy
    *) echo "usage: $0 [-j N] [-n] [-o SUBSTR] [-f] [-F]" >&2; exit 1 ;;
  esac
done

ROOT="$HOME/wug-test-interp"
OUT="results/full_methods_run"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$ROOT/$OUT/logs/$STAMP"
JOBS="$LOGDIR/jobs.tsv"
mkdir -p "$LOGDIR"
ln -sfn "$STAMP" "$ROOT/$OUT/logs/latest"

# Own process group, so one signal takes down every child.
set -m
echo "$$" > "$LOGDIR/master.pid"
echo "=============================================================="
echo " master PID : $$"
echo " kill with  : kill -TERM -$$"
echo " logs       : $LOGDIR"
echo "=============================================================="
trap 'echo "[$(date +%T)] caught signal, killing children"; kill -TERM -$$ 2>/dev/null; exit 130' INT TERM

# --- preflight -------------------------------------------------------------
if [[ "$FORCE" == 0 && "$DRY_RUN" == 0 ]]; then
  gpu_ok=$(python3 -c 'import torch; print(int(torch.cuda.is_available()))' 2>/dev/null || echo 0)
  if [[ "$gpu_ok" != "1" ]]; then
    echo "ERROR: torch.cuda.is_available() is False in this environment." >&2
    echo "  which python3 : $(command -v python3)" >&2
    echo "  CONDA_DEFAULT_ENV = ${CONDA_DEFAULT_ENV:-<unset>}" >&2
    echo "  Activate the right env (conda activate wug_test_env), or pass -f." >&2
    exit 1
  fi
  echo "[$(date +%T)] CUDA ok: $(python3 -c 'import torch;print(torch.cuda.get_device_name(0))')"
fi

N_SAMPLE="${N_SAMPLE:-200}"            # circuit methods (patch, ablation)
N_SAMPLE_CELL="${N_SAMPLE_CELL:-400}"  # per-cell methods (das, diffmean, probe)

MODELS=(Qwen3-VL-2B Qwen3-VL-4B)
STREAMS=(syntax vision)
CONDS=(target_verb_att0_opp target_verb_att1_opp
       target_verb_att2_opp target_verb_att3_opp)
KS=(2 4 8 16 32 64 128 256)

LAYERS_Qwen3_VL_2B="1 5 10 15 20 25 28"
LAYERS_Qwen3_VL_4B="1 5 10 15 20 25 30 35 36"

# Token positions carrying the verb slot, per attractor count.
TOKS_target_verb_att0_opp="4 5"
TOKS_target_verb_att1_opp="7 8"
TOKS_target_verb_att2_opp="10 11"
TOKS_target_verb_att3_opp="13 14"

SEEDS_syntax=(31085 37722 48126 54321 96684)
SEEDS_vision=(27616 47410 68200 79404 92980)
SEED_CONDS=(target_verb_att0_opp target_verb_att1_opp)

# ---------------------------------------------------------------------------
emit() {  # emit <name> <outdir> <command...>
  local name="$1" dir="$2"; shift 2
  [[ -n "$ONLY" && "$name" != *"$ONLY"* ]] && return 0
  printf '%s\t%s\t%s\n' "$name" "$dir" "$*" >> "$JOBS"
}

layers_for() { local v="LAYERS_${1//-/_}"; echo "${!v}"; }
toks_for()   { local v="TOKS_$1";          echo "${!v}"; }

# Shared args WITHOUT --toks; callers add them (ablation must not restrict).
common_args() {  # <model-short> <stream> <cond> <embedding>
  local m="$1" s="$2" c="$3" emb="$4" full="${1}-Instruct"
  echo "--model_path Qwen/${full}" \
       "--embeddings_path ${emb}" \
       "--train_csv results/eval/interp/${full}/agreement_target_natural_scored.csv" \
       "--test_csv results/eval/interp/${full}/${s}/agreement_target_wug_scored.csv" \
       "--source_input_col base_sentence_sg --base_input_col base_sentence_pl" \
       "--source_completion_A good_singular --source_completion_B bad_singular" \
       "--filter condition=${c} is_correct_all=TRUE" \
       "--layers $(layers_for "$m")"
}

: > "$JOBS"

# --- main grid: das / diffmean / probe -------------------------------------
for m in "${MODELS[@]}"; do
  ml="$(echo "$m" | tr 'A-Z-' 'a-z_')"
  for s in "${STREAMS[@]}"; do
    for c in "${CONDS[@]}"; do
      A="$(common_args "$m" "$s" "$c" "embeddings/${m}/${s}/${ml}_${s}.pt")"
      T="--toks $(toks_for "$c")"
      D="$OUT/main/${m}/${s}/${c}"

      emit "das_${m}_${s}_${c}" "$D" \
        "python3 core/interp/das.py $A $T --n_sample $N_SAMPLE_CELL --add_inverse \
         --out_csv ${D}/das.csv"
      emit "diffmean_${m}_${s}_${c}" "$D" \
        "python3 core/interp/diffmean.py $A $T --n_sample $N_SAMPLE_CELL --add_inverse_test \
         --out_csv ${D}/diffmean.csv"
      emit "probe_${m}_${s}_${c}" "$D" \
        "python3 core/interp/linear_probe.py $A $T --n_sample $N_SAMPLE_CELL --add_inverse_test \
         --out_csv ${D}/probe.csv"
    done
  done
done

# --- k sweep: patch / ablation, k = 2^1..2^8 -------------------------------
# k=64 is in KS, so this subsumes the single-k patch/ablation requirement.
# NOTE: ablation gets NO --toks. Its complement is defined relative to the
# search space, so restricting positions would make faithfulness trivially 1.
for m in "${MODELS[@]}"; do
  ml="$(echo "$m" | tr 'A-Z-' 'a-z_')"
  for s in "${STREAMS[@]}"; do
    for c in "${CONDS[@]}"; do
      A="$(common_args "$m" "$s" "$c" "embeddings/${m}/${s}/${ml}_${s}.pt")"
      T="--toks $(toks_for "$c")"
      D="$OUT/k_sweep/${m}/${s}/${c}"
      for k in "${KS[@]}"; do
        emit "patch_${m}_${s}_${c}_k${k}" "$D" \
          "python3 core/interp/circuit.py $A $T --n_sample $N_SAMPLE --add_inverse \
           --hook_points mlp_act --top_k $k \
           --out_csv ${D}/patch_k${k}.csv --circuit_out ${D}/patch_nodes_k${k}.csv"
        emit "ablation_${m}_${s}_${c}_k${k}" "$D" \
          "python3 core/interp/ablation.py $A --n_sample $N_SAMPLE --add_inverse \
           --hook_points mlp_act --top_k $k --ablation mean \
           --out_csv ${D}/ablation_k${k}.csv --circuit_out ${D}/ablation_nodes_k${k}.csv"
      done
    done
  done
done

# --- seed sweep: 2B only, das / diffmean / probe, 0-1 attractors ------------
m=Qwen3-VL-2B
for s in "${STREAMS[@]}"; do
  eval "seeds=(\"\${SEEDS_${s}[@]}\")"
  for c in "${SEED_CONDS[@]}"; do
    for sd in "${seeds[@]}"; do
      A="$(common_args "$m" "$s" "$c" "embeddings/${m}/${s}_seeds/seed_${sd}.pt")"
      T="--toks $(toks_for "$c")"
      D="$OUT/seed_sweep/${m}/${s}/${c}/seed_${sd}"

      emit "das_${m}_${s}_${c}_seed${sd}" "$D" \
        "python3 core/interp/das.py $A $T --n_sample $N_SAMPLE_CELL --add_inverse \
         --out_csv ${D}/das.csv"
      emit "diffmean_${m}_${s}_${c}_seed${sd}" "$D" \
        "python3 core/interp/diffmean.py $A $T --n_sample $N_SAMPLE_CELL --add_inverse_test \
         --out_csv ${D}/diffmean.csv"
      emit "probe_${m}_${s}_${c}_seed${sd}" "$D" \
        "python3 core/interp/linear_probe.py $A $T --n_sample $N_SAMPLE_CELL --add_inverse_test \
         --out_csv ${D}/probe.csv"
    done
  done
done

N=$(wc -l < "$JOBS")
{
  echo "[$(date +%T)] $N jobs, $NPROC at a time, 1 gpu"
  echo "[$(date +%T)] master pid $$"
} | tee -a "$LOGDIR/master.log"

if [[ "$DRY_RUN" == 1 ]]; then
  cut -f1 "$JOBS" | sed 's/^/  /'
  echo "dry run — nothing executed."
  exit 0
fi

# Pre-create every output directory so concurrent workers never race on mkdir.
cut -f2 "$JOBS" | sort -u | while read -r d; do mkdir -p "$ROOT/$d"; done

export ROOT LOGDIR
run_one() {
  local name="$1" cmd="$2"
  cd "$ROOT" || return 1
  echo "[$(date +%T)] START $name"
  if eval "$cmd" > "$LOGDIR/${name}.log" 2>&1; then
    echo "[$(date +%T)] OK    $name"
  else
    echo "[$(date +%T)] FAIL  $name -> $LOGDIR/${name}.log"
    echo "$name" >> "$LOGDIR/failed.txt"
  fi
}
export -f run_one

xargs -d '\n' -a "$JOBS" -P "$NPROC" -I{} bash -c '
    line="{}"
    name=$(printf "%s" "$line" | cut -f1)
    cmd=$(printf  "%s" "$line" | cut -f3-)
    run_one "$name" "$cmd"
  ' 2>&1 | tee -a "$LOGDIR/master.log"

nfail=$(wc -l < "$LOGDIR/failed.txt" 2>/dev/null || echo 0)
echo "[$(date +%T)] done. $nfail failures." | tee -a "$LOGDIR/master.log"
[[ -s "$LOGDIR/failed.txt" ]] && sed 's/^/  FAILED /' "$LOGDIR/failed.txt"
exit 0