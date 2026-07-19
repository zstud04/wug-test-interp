#!/usr/bin/env bash
#
# run_circuits_k128.sh
#
#   hook point : mlp_act  (intermediate MLP activations, input to down_proj)
#   k          : 128
#   models     : Qwen3-VL-2B, then Qwen3-VL-4B
#   streams    : syntax, vision
#   attractors : 0, 1, 2, 3
#   methods    : patch (interchange) and ablation (mean)
#
#   2 models x 2 streams x 4 conds x 2 methods = 32 jobs. 2B jobs come first.
#
# NO --toks. Both are circuit methods: the layer x token grid is the SEARCH
# SPACE, not a loop. Where the 128 nodes land on the token axis is a finding
# (see the `tok` column of --circuit_out), not something to assume. It also
# matters for ablation specifically: the complement is defined relative to the
# search space, so restricting positions would make faithfulness ~1.0 for free.
#
# Self-daemonizing:
#   conda activate wug_test_env
#   bash run_circuits_k128.sh -j 8 -g 0,1
#
#   tail -f results/circuits_k128/logs/latest/master.log
#   kill -TERM -$(cat results/circuits_k128/logs/latest/master.pid)
#
# Options:
#   -j N        TOTAL concurrent jobs across all GPUs (default 8)
#   -g LIST     comma-separated GPU ids (default 0,1); job n -> gpu n % |LIST|
#   -k K        circuit size (default 128)
#   -n          dry run
#   -o SUBSTR   only jobs whose name contains SUBSTR
#   -f          skip CUDA preflight
#   -F          stay in foreground
#
# Output:
#   results/circuits_k128/
#     patch/<model>/<stream>/<cond>/patch_k<K>.csv       + patch_nodes_k<K>.csv
#     ablation/<model>/<stream>/<cond>/ablation_k<K>.csv + ablation_nodes_k<K>.csv
#     logs/<timestamp>/{master.log,master.pid,jobs.tsv,failed.txt,<job>.log}
#     logs/latest -> <timestamp>
#
set -uo pipefail

if [[ "${_RUN_CK_DAEMON:-0}" != 1 && " $* " != *" -n "* && " $* " != *" -F "* ]]; then
  export _RUN_CK_DAEMON=1
  _dir="$HOME/wug-test-interp/results/circuits_k128"
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
  if ! kill -0 "$_pid" 2>/dev/null; then
    echo "--- daemon exited immediately; $_out says: ---"
    cat "$_out"
    exit 1
  fi
  exit 0
fi

NPROC=8
GPUS="0,1"
TOP_K=128
DRY_RUN=0
ONLY=""
FORCE=0
while getopts ":j:g:k:no:fF" opt; do
  case "$opt" in
    j) NPROC="$OPTARG" ;;
    g) GPUS="$OPTARG" ;;
    k) TOP_K="$OPTARG" ;;
    n) DRY_RUN=1 ;;
    o) ONLY="$OPTARG" ;;
    f) FORCE=1 ;;
    F) : ;;
    *) echo "usage: $0 [-j N] [-g GPUS] [-k K] [-n] [-o SUBSTR] [-f] [-F]" >&2; exit 1 ;;
  esac
done

ROOT="$HOME/wug-test-interp"
OUT="results/circuits_k128"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$ROOT/$OUT/logs/$STAMP"
JOBS="$LOGDIR/jobs.tsv"
mkdir -p "$LOGDIR"
ln -sfn "$STAMP" "$ROOT/$OUT/logs/latest"

set -m
echo "$$" > "$LOGDIR/master.pid"
echo "master PID $$   (kill -TERM -$$)"
trap 'echo "[$(date +%T)] signal; killing children"; kill -TERM -$$ 2>/dev/null; exit 130' INT TERM

# --- preflight -------------------------------------------------------------
if [[ "$FORCE" == 0 && "$DRY_RUN" == 0 ]]; then
  ndev=$(python3 -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)' 2>/dev/null || echo 0)
  if [[ "$ndev" == "0" ]]; then
    echo "ERROR: torch.cuda.is_available() is False." >&2
    echo "  which python3 : $(command -v python3)" >&2
    echo "  CONDA_DEFAULT_ENV = ${CONDA_DEFAULT_ENV:-<unset>}" >&2
    exit 1
  fi
  IFS=',' read -ra _g <<< "$GPUS"
  for id in "${_g[@]}"; do
    (( id >= ndev )) && { echo "ERROR: -g wants gpu $id, torch sees $ndev." >&2; exit 1; }
  done
  echo "[$(date +%T)] CUDA ok: $ndev device(s), using [$GPUS]"
fi

# --- grid ------------------------------------------------------------------
MODELS=(Qwen3-VL-2B Qwen3-VL-4B)          # 2B first
STREAMS=(syntax vision)
CONDS=(target_verb_att0_opp target_verb_att1_opp
       target_verb_att2_opp target_verb_att3_opp)
HOOK=mlp_act
N_SAMPLE="${N_SAMPLE:-200}"

LAYERS_Qwen3_VL_2B="2 7 12 17 22 27"
LAYERS_Qwen3_VL_4B="2 7 12 17 22 27"

layers_for() { local v="LAYERS_${1//-/_}"; echo "${!v}"; }

emit() {  # emit <name> <outdir> <command...>
  local name="$1" dir="$2"; shift 2
  [[ -n "$ONLY" && "$name" != *"$ONLY"* ]] && return 0
  printf '%s\t%s\t%s\n' "$name" "$dir" "$*" >> "$JOBS"
}

args_for() {  # <model-short> <stream> <cond>
  local m="$1" s="$2" c="$3" full="${1}-Instruct"
  local ml; ml="$(echo "$m" | tr 'A-Z-' 'a-z_')"
  echo "--model_path Qwen/${full}" \
       "--embeddings_path embeddings/${m}/${s}/${ml}_${s}.pt" \
       "--train_csv results/eval/interp/${full}/agreement_target_natural_scored.csv" \
       "--test_csv results/eval/interp/${full}/${s}/agreement_target_wug_scored.csv" \
       "--source_input_col base_sentence_sg --base_input_col base_sentence_pl" \
       "--source_completion_A good_singular --source_completion_B bad_singular" \
       "--filter condition=${c} is_correct_all=TRUE" \
       "--layers $(layers_for "$m")" \
       "--n_sample ${N_SAMPLE} --add_inverse" \
       "--hook_points ${HOOK} --top_k ${TOP_K}"
}

: > "$JOBS"

# Method-major within each model, so the gpu-by-parity assignment splits patch
# and ablation evenly across cards (ablation runs 4 forwards/row, patch 3).
for m in "${MODELS[@]}"; do
  for s in "${STREAMS[@]}"; do
    for c in "${CONDS[@]}"; do
      A="$(args_for "$m" "$s" "$c")"
      D="$OUT/patch/${m}/${s}/${c}"
      emit "patch_${m}_${s}_${c}_k${TOP_K}" "$D" \
        "python3 core/interp/circuit.py $A \
         --out_csv ${D}/patch_k${TOP_K}.csv \
         --circuit_out ${D}/patch_nodes_k${TOP_K}.csv"
    done
  done
  for s in "${STREAMS[@]}"; do
    for c in "${CONDS[@]}"; do
      A="$(args_for "$m" "$s" "$c")"
      D="$OUT/ablation/${m}/${s}/${c}"
      emit "ablation_${m}_${s}_${c}_k${TOP_K}" "$D" \
        "python3 core/interp/ablation.py $A --ablation mean \
         --out_csv ${D}/ablation_k${TOP_K}.csv \
         --circuit_out ${D}/ablation_nodes_k${TOP_K}.csv"
    done
  done
done

N=$(wc -l < "$JOBS")
echo "[$(date +%T)] $N jobs (k=$TOP_K, hook=$HOOK), $NPROC at a time, gpus [$GPUS]" \
  | tee -a "$LOGDIR/master.log"

if [[ "$DRY_RUN" == 1 ]]; then
  cut -f1 "$JOBS" | sed 's/^/  /'
  echo "dry run — nothing executed."
  exit 0
fi

cut -f2 "$JOBS" | sort -u | while read -r d; do mkdir -p "$ROOT/$d"; done

# --- worker in its own file: no three-deep quote nesting --------------------
WORKER="$LOGDIR/worker.sh"
cat > "$WORKER" <<'WEOF'
#!/usr/bin/env bash
set -uo pipefail
line="$1"
slot=$(printf '%s' "$line" | cut -f1)
name=$(printf '%s' "$line" | cut -f2)
cmd=$(printf  '%s' "$line" | cut -f4-)

IFS=',' read -ra g <<< "$GPUS"
gpu="${g[$(( slot % ${#g[@]} ))]}"

cd "$ROOT" || exit 1
echo "[$(date +%T)] START $name (gpu $gpu)"
if CUDA_VISIBLE_DEVICES="$gpu" eval "$cmd" > "$LOGDIR/${name}.log" 2>&1; then
  echo "[$(date +%T)] OK    $name"
else
  echo "[$(date +%T)] FAIL  $name -> $LOGDIR/${name}.log"
  echo "$name" >> "$LOGDIR/failed.txt"
fi
WEOF
chmod +x "$WORKER"

export ROOT LOGDIR GPUS

# Real tab via printf. NOT $"\t" -- that is locale translation and yields a
# literal backslash-t, which silently corrupts every field offset.
nl -ba -s"$(printf '\t')" "$JOBS" \
  | xargs -d '\n' -P "$NPROC" -I{} "$WORKER" "{}" \
  2>&1 | tee -a "$LOGDIR/master.log"

nfail=$(wc -l < "$LOGDIR/failed.txt" 2>/dev/null || echo 0)
echo "[$(date +%T)] done. $nfail failures." | tee -a "$LOGDIR/master.log"
[[ -s "$LOGDIR/failed.txt" ]] && sed 's/^/  FAILED /' "$LOGDIR/failed.txt"
exit 0