#!/usr/bin/env bash
#
# rerun_failed.sh — rerun a named subset of hook_comparison jobs.
#
#   conda activate wug_test_env
#   bash rerun_failed.sh -j 4 -g 0,1
#
# Commands are pulled VERBATIM from the original jobs.tsv, so flags, paths and
# --top_k cannot drift from the first run. Edit RERUN below to change the set.
#
# Options:
#   -j N        concurrent jobs, total across GPUs (default 4)
#   -g LIST     comma-separated GPU ids (default 0,1)
#   -J PATH     source jobfile (default logs/latest/jobs.tsv)
#   -n          dry run
#   -F          stay in foreground
#
set -uo pipefail

if [[ "${_RERUN_DAEMON:-0}" != 1 && " $* " != *" -n "* && " $* " != *" -F "* ]]; then
  export _RERUN_DAEMON=1
  _dir="$HOME/wug-test-interp/results/hook_comparison"
  mkdir -p "$_dir"
  _out="$_dir/rerun_nohup.out"
  setsid nohup bash "$0" "$@" > "$_out" 2>&1 &
  _pid=$!
  sleep 2
  echo "detached. master PID $_pid   (kill -TERM -$_pid)"
  echo "stdout: $_out"
  if ! kill -0 "$_pid" 2>/dev/null; then
    echo "--- daemon exited immediately ---"; cat "$_out"; exit 1
  fi
  exit 0
fi

ROOT="$HOME/wug-test-interp"
NPROC=4
GPUS="0,1"
SRC_JOBS="$ROOT/results/hook_comparison/logs/latest/jobs.tsv"
DRY_RUN=0

while getopts ":j:g:J:nF" opt; do
  case "$opt" in
    j) NPROC="$OPTARG" ;;
    g) GPUS="$OPTARG" ;;
    J) SRC_JOBS="$OPTARG" ;;
    n) DRY_RUN=1 ;;
    F) : ;;
    *) echo "usage: $0 [-j N] [-g GPUS] [-J jobs.tsv] [-n] [-F]" >&2; exit 1 ;;
  esac
done

# --- the jobs to rerun -----------------------------------------------------
RERUN=(
  patch_mlp_out_k16
  patch_mlp_act_k32
  patch_mlp_act_k16
  patch_mlp_out_k8
  patch_mlp_out_k32
  patch_mlp_act_k64
  patch_mlp_act_k8
  patch_attn_out_k128
  patch_attn_out_k32
  patch_attn_out_k8
  patch_mlp_out_k64
  patch_resid_k8
  patch_resid_k32
  ablation_mlp_act_k16
  ablation_mlp_out_k64
  ablation_attn_out_k16
  ablation_resid_k64
)

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$ROOT/results/hook_comparison/logs/rerun_$STAMP"
JOBS="$LOGDIR/jobs.tsv"
mkdir -p "$LOGDIR"

set -m
echo "$$" > "$LOGDIR/master.pid"
echo "master PID $$   (kill -TERM -$$)"
trap 'kill -TERM -$$ 2>/dev/null; exit 130' INT TERM

[[ -f "$SRC_JOBS" ]] || { echo "no jobfile at $SRC_JOBS" >&2; exit 1; }

# Exact, whole-field match on column 1. -F so nothing is treated as a regex,
# -x on the field so patch_resid_k8 never matches patch_resid_k80.
: > "$JOBS"
missing=0
for name in "${RERUN[@]}"; do
  line=$(awk -F'\t' -v n="$name" '$1==n' "$SRC_JOBS")
  if [[ -z "$line" ]]; then
    echo "MISSING from jobfile: $name" >&2
    missing=$((missing + 1))
  else
    printf '%s\n' "$line" >> "$JOBS"
  fi
done

N=$(wc -l < "$JOBS")
echo "matched $N / ${#RERUN[@]} jobs  (missing: $missing)"
(( missing > 0 )) && { echo "refusing to run with unmatched names" >&2; exit 1; }

if [[ "$DRY_RUN" == 1 ]]; then
  cut -f1 "$JOBS" | sed 's/^/  /'
  echo "dry run — nothing executed."
  exit 0
fi

# Recreate output dirs (column 2) in case any were never made.
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

# nl -ba -s <TAB> : the separator is a real tab via ANSI-C quoting ($'\t'),
# NOT $"\t" (which is locale translation and yields a literal backslash-t).
nl -ba -s"$(printf '\t')" "$JOBS" \
  | xargs -d '\n' -P "$NPROC" -I{} "$WORKER" "{}" \
  2>&1 | tee -a "$LOGDIR/master.log"

nfail=$(wc -l < "$LOGDIR/failed.txt" 2>/dev/null || echo 0)
echo "[$(date +%T)] done. $nfail failures." | tee -a "$LOGDIR/master.log"
[[ -s "$LOGDIR/failed.txt" ]] && sed 's/^/  FAILED /' "$LOGDIR/failed.txt"
exit 0