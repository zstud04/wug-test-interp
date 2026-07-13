#!/usr/bin/env bash
#
# run_hooks.sh — hook-point comparison.
#
#   Qwen3-VL-2B, syntax stream, 0 attractors.
#   patch (interchange) and ablation, each on FOUR hook points, run SEPARATELY:
#
#       mlp_act    input to down_proj: act_fn(gate_proj(x)) * up_proj(x)  [d_ffn]
#       mlp_out    output of the MLP block                               [d_model]
#       attn_out   output of the attention block                         [d_model]
#       resid      output of the decoder layer (residual stream post)    [d_model]
#
#   k = 8, 16, 32, 64, 128.  4 hooks x 2 methods x 5 ks = 40 jobs.
#
# NO --toks anywhere. Both are circuit methods: the layer x token grid is the
# SEARCH SPACE, not a loop. Where the k nodes land on the token axis is a
# finding (check the `tok` column of --circuit_out), not something to assume.
# Restricting positions would also make the ablation's complement trivial and
# faithfulness ~1.0 for free.
#
# Separate runs, never `--hook_points a b`: (1) attribution is in raw activation
# units, and mlp_act (post-SwiGLU, unbounded) is not commensurable with resid
# (norms grow ~10x with depth), so a mixed top-k goes to whichever site has the
# bigger numbers; (2) resid at layer L already CONTAINS mlp_out and attn_out at
# layer L -- they are its summands, so a mixed circuit double-counts and the
# patches overwrite each other. Compare by overlaying the curves, not by pooling.
#
# Self-daemonizing. Two GPUs, 4 jobs each:
#   conda activate wug_test_env
#   bash run_hooks.sh -j 8 -g 0,1
#
#   tail -f results/hook_comparison/logs/latest/master.log
#   kill -TERM -$(cat results/hook_comparison/logs/latest/master.pid)
#
# Options:
#   -j N        TOTAL concurrent jobs across all GPUs (default 8)
#   -g LIST     comma-separated GPU ids (default 0,1); job n -> gpu n % |LIST|
#   -n          dry run: list jobs, execute nothing (foreground)
#   -o SUBSTR   only jobs whose name contains SUBSTR
#   -f          skip the CUDA preflight
#   -F          stay in the foreground
#
# Output:
#   results/hook_comparison/
#     patch/<hook>/patch_k<K>.csv        + patch_nodes_k<K>.csv
#     ablation/<hook>/ablation_k<K>.csv  + ablation_nodes_k<K>.csv
#     logs/<timestamp>/...
#
set -uo pipefail

if [[ "${_RUN_HOOKS_DAEMON:-0}" != 1 && " $* " != *" -n "* && " $* " != *" -F "* ]]; then
  export _RUN_HOOKS_DAEMON=1
  _dir="$HOME/wug-test-interp/results/hook_comparison"
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

NPROC=8              # TOTAL across all GPUs
GPUS="0,1"
DRY_RUN=0
ONLY=""
FORCE=0
while getopts ":j:g:no:fF" opt; do
  case "$opt" in
    j) NPROC="$OPTARG" ;;
    g) GPUS="$OPTARG" ;;
    n) DRY_RUN=1 ;;
    o) ONLY="$OPTARG" ;;
    f) FORCE=1 ;;
    F) : ;;
    *) echo "usage: $0 [-j N] [-g GPUS] [-n] [-o SUBSTR] [-f] [-F]" >&2; exit 1 ;;
  esac
done

ROOT="$HOME/wug-test-interp"
OUT="results/hook_comparison"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$ROOT/$OUT/logs/$STAMP"
JOBS="$LOGDIR/jobs.tsv"
mkdir -p "$LOGDIR"
ln -sfn "$STAMP" "$ROOT/$OUT/logs/latest"

set -m
echo "$$" > "$LOGDIR/master.pid"
echo "master PID $$   (kill -TERM -$$)"
trap 'echo "[$(date +%T)] signal; killing children"; kill -TERM -$$ 2>/dev/null; exit 130' INT TERM

# --- preflight: torch sees CUDA, and every requested GPU actually exists ----
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
    if (( id >= ndev )); then
      echo "ERROR: -g requests gpu $id but torch sees only $ndev device(s)." >&2
      exit 1
    fi
  done
  echo "[$(date +%T)] CUDA ok: $ndev device(s); using [$GPUS], $NPROC jobs total"
  python3 -c "import torch
for i in range(torch.cuda.device_count()): print('   gpu', i, torch.cuda.get_device_name(i))"
fi

# --- fixed cell ------------------------------------------------------------
MODEL=Qwen3-VL-2B
FULL="${MODEL}-Instruct"
STREAM=syntax
COND=target_verb_att0_opp
LAYERS="2 7 12 17 22 27"
N_SAMPLE="${N_SAMPLE:-200}"

HOOKS=(mlp_act mlp_out attn_out resid)
KS=(8 16 32 64 128)

# Identical for both methods. Differences are the script and --ablation, which
# is what makes patch-vs-ablation a controlled comparison.
ARGS="--model_path Qwen/${FULL} \
  --embeddings_path embeddings/${MODEL}/${STREAM}/qwen3_vl_2b_${STREAM}.pt \
  --train_csv results/eval/interp/${FULL}/agreement_target_natural_scored.csv \
  --test_csv results/eval/interp/${FULL}/${STREAM}/agreement_target_wug_scored.csv \
  --source_input_col base_sentence_sg --base_input_col base_sentence_pl \
  --source_completion_A good_singular --source_completion_B bad_singular \
  --filter condition=${COND} is_correct_all=TRUE \
  --layers ${LAYERS} \
  --n_sample ${N_SAMPLE} --add_inverse"

emit() {  # emit <name> <outdir> <command...>
  local name="$1" dir="$2"; shift 2
  [[ -n "$ONLY" && "$name" != *"$ONLY"* ]] && return 0
  printf '%s\t%s\t%s\n' "$name" "$dir" "$*" >> "$JOBS"
}

: > "$JOBS"

# Emit method-major, not interleaved. GPUs are assigned by line parity, so
# alternating patch/ablation would put every patch on gpu 0 and every ablation
# on gpu 1 -- and ablation runs 4 forwards per row where patch runs 3.
for h in "${HOOKS[@]}"; do
  for k in "${KS[@]}"; do
    D="$OUT/patch/${h}"
    emit "patch_${h}_k${k}" "$D" \
      "python3 core/interp/circuit.py $ARGS \
       --hook_points $h --top_k $k \
       --out_csv ${D}/patch_k${k}.csv --circuit_out ${D}/patch_nodes_k${k}.csv"
  done
done

for h in "${HOOKS[@]}"; do
  for k in "${KS[@]}"; do
    D="$OUT/ablation/${h}"
    emit "ablation_${h}_k${k}" "$D" \
      "python3 core/interp/ablation.py $ARGS \
       --hook_points $h --top_k $k --ablation mean \
       --out_csv ${D}/ablation_k${k}.csv --circuit_out ${D}/ablation_nodes_k${k}.csv"
  done
done

N=$(wc -l < "$JOBS")
echo "[$(date +%T)] $N jobs, $NPROC at a time, gpus [$GPUS]" | tee -a "$LOGDIR/master.log"

if [[ "$DRY_RUN" == 1 ]]; then
  cut -f1 "$JOBS" | sed 's/^/  /'
  echo "dry run — nothing executed."
  exit 0
fi

cut -f2 "$JOBS" | sort -u | while read -r d; do mkdir -p "$ROOT/$d"; done

export ROOT LOGDIR GPUS
run_one() {
  local slot="$1" name="$2" cmd="$3" gpu
  IFS=',' read -ra g <<< "$GPUS"
  gpu="${g[$(( slot % ${#g[@]} ))]}"
  cd "$ROOT" || return 1
  echo "[$(date +%T)] START $name (gpu $gpu)"
  if CUDA_VISIBLE_DEVICES="$gpu" eval "$cmd" > "$LOGDIR/${name}.log" 2>&1; then
    echo "[$(date +%T)] OK    $name"
  else
    echo "[$(date +%T)] FAIL  $name -> $LOGDIR/${name}.log"
    echo "$name" >> "$LOGDIR/failed.txt"
  fi
}
export -f run_one

# nl prepends the line number, so fields shift: 1=slot 2=name 3=dir 4-=cmd
nl -ba -s$'\t' "$JOBS" | xargs -d '\n' -P "$NPROC" -I{} bash -c '
    line="{}"
    slot=$(printf "%s" "$line" | cut -f1 | tr -d " ")
    name=$(printf "%s" "$line" | cut -f2)
    cmd=$(printf  "%s" "$line" | cut -f4-)
    run_one "$slot" "$name" "$cmd"
  ' 2>&1 | tee -a "$LOGDIR/master.log"

nfail=$(wc -l < "$LOGDIR/failed.txt" 2>/dev/null || echo 0)
echo "[$(date +%T)] done. $nfail failures." | tee -a "$LOGDIR/master.log"
[[ -s "$LOGDIR/failed.txt" ]] && sed 's/^/  FAILED /' "$LOGDIR/failed.txt"
exit 0