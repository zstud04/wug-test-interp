#!/usr/bin/env bash
#
# Reproduce results/interp/main_runs, but evaluated on the NATURAL (non-wug)
# stimuli test split instead of the wug stimuli.
#
# Difference from run_all_exps.sh: instead of --train_csv (natural) + --test_csv
# (wug), we pass a single --src_csv (natural). intervention.py then fits on the
# natural split=="train" rows and evaluates on the natural split=="test" rows.
# Everything else (methods, filters, layers, toks, sampling, --add_inverse*)
# matches main_runs. Only the das/diffmean/probe methods and the k=128 circuit
# (patch) / ablation results are produced, exactly as in main_runs.
#
# Output: results/interp/natural_runs/<Model-Instruct>/<stream>/<cond>/{das,
# diffmean,probe,patch_k128,patch_nodes_k128,ablation_k128,ablation_nodes_k128}.csv
#
# Progress is tracked in results/JOBTRACKER.md.

set -uo pipefail

# --- re-exec detached so the run survives the shell ------------------------
if [[ "${_NAT_DAEMON:-0}" != 1 && " $* " != *" -n "* && " $* " != *" -F "* ]]; then
  export _NAT_DAEMON=1
  _dir="$HOME/wug-test-interp/results/interp/natural_runs"
  mkdir -p "$_dir"
  _out="$_dir/nohup.out"
  setsid nohup bash "$0" "$@" > "$_out" 2>&1 &
  _pid=$!
  sleep 2
  echo "=============================================================="
  echo " detached. master PID : $_pid"
  echo " kill with            : kill -TERM -$_pid"
  echo " stdout               : $_out"
  echo " tracker              : $HOME/wug-test-interp/results/JOBTRACKER.md"
  echo " follow               : tail -f $_dir/logs/latest/master.log"
  echo "=============================================================="
  if ! kill -0 "$_pid" 2>/dev/null; then
    echo "--- daemon exited immediately; $_out says: ---"
    cat "$_out"
    exit 1
  fi
  exit 0
fi

NPROC=2          # jobs PER GPU (doubled per user request; watch for 4B OOM on 24GB)
GPUS_STR="0 1"   # GPUs to spread jobs across (round-robin, one pool each)
DRY_RUN=0
ONLY=""
FORCE=0
while getopts ":j:no:fF" opt; do
  case "$opt" in
    j) NPROC="$OPTARG" ;;
    n) DRY_RUN=1 ;;
    o) ONLY="$OPTARG" ;;
    f) FORCE=1 ;;
    F) : ;;
    *) echo "usage: $0 [-j N] [-n] [-o SUBSTR] [-f] [-F]" >&2; exit 1 ;;
  esac
done

# Use the project's conda env python without requiring `conda activate`.
ENV_BIN="/mnt/dv/wid/projects3/Rogers-muri-human-ai/zstuddiford/tmp/miniconda3/envs/wug_test_env/bin"
[[ -x "$ENV_BIN/python3" ]] && export PATH="$ENV_BIN:$PATH"

ROOT="$HOME/wug-test-interp"
OUT="results/interp/natural_runs"
TRACKER="$ROOT/results/JOBTRACKER.md"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$ROOT/$OUT/logs/$STAMP"
JOBS="$LOGDIR/jobs.tsv"
STATUS="$LOGDIR/status"
mkdir -p "$LOGDIR" "$STATUS"
ln -sfn "$STAMP" "$ROOT/$OUT/logs/latest"

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
    exit 1
  fi
  echo "[$(date +%T)] CUDA ok: $(python3 -c 'import torch;print(torch.cuda.get_device_name(0))')"
fi

CACHE_DIR="/mnt/dv/wid/projects3/Rogers-muri-human-ai/zstuddiford"
N_SAMPLE="${N_SAMPLE:-200}"            # circuit methods (patch, ablation)
N_SAMPLE_CELL="${N_SAMPLE_CELL:-400}"  # per-cell methods (das, diffmean, probe)
K=128                                  # main_runs uses k=128 circuits

MODELS=(Qwen3-VL-2B Qwen3-VL-4B)
# Vision omitted: no custom vision embeddings are used, so only syntax is run.
STREAMS=(syntax)
CONDS=(target_verb_att0_opp target_verb_att1_opp
       target_verb_att2_opp target_verb_att3_opp)

# Pyvene methods (das/diffmean/probe) use these (1-based; tolerate the top index).
LAYERS_Qwen3_VL_2B="1 5 10 15 20 25 28"
LAYERS_Qwen3_VL_4B="1 5 10 15 20 25 30 35 36"

# Circuit methods (patch/ablation) index raw 0-based module list -> different set,
# matching run_circuits_k128.sh (no --toks for these either).
CLAYERS_Qwen3_VL_2B="2 7 12 17 22 27"
CLAYERS_Qwen3_VL_4B="2 7 12 17 22 27"

TOKS_target_verb_att0_opp="4 5"
TOKS_target_verb_att1_opp="7 8"
TOKS_target_verb_att2_opp="10 11"
TOKS_target_verb_att3_opp="13 14"

emit() {  # emit <name> <outdir> <command...>
  local name="$1" dir="$2"; shift 2
  [[ -n "$ONLY" && "$name" != *"$ONLY"* ]] && return 0
  printf '%s\t%s\t%s\n' "$name" "$dir" "$*" >> "$JOBS"
  echo PENDING > "$STATUS/$name"
}

# emit a method only if its output CSV does not already exist (never overwrite).
emit_if_missing() {  # <name> <dir> <out_rel_csv> <command...>
  local name="$1" dir="$2" out="$3"; shift 3
  if [[ "$FORCE" == 0 && -s "$ROOT/$out" ]]; then return 0; fi
  emit "$name" "$dir" "$@"
}

layers_for()  { local v="LAYERS_${1//-/_}";  echo "${!v}"; }
clayers_for() { local v="CLAYERS_${1//-/_}"; echo "${!v}"; }
toks_for()    { local v="TOKS_$1";           echo "${!v}"; }

# Shared args WITHOUT --toks/--layers; callers add them.
# NATURAL: single --src_csv (natural scored, has train/test split column).
common_args() {  # <model-short> <stream> <cond> <embedding>
  local m="$1" s="$2" c="$3" emb="$4" full="${1}-Instruct"
  echo "--model_path Qwen/${full}" \
       "--cache_dir ${CACHE_DIR}" \
       "--embeddings_path ${emb}" \
       "--src_csv results/eval/attractors/${full}/agreement_target_natural_scored.csv" \
       "--source_input_col base_sentence_sg --base_input_col base_sentence_pl" \
       "--source_completion_A good_singular --source_completion_B bad_singular" \
       "--filter condition=${c} is_correct_all=TRUE"
}

: > "$JOBS"

for m in "${MODELS[@]}"; do
  ml="$(echo "$m" | tr 'A-Z-' 'a-z_')"
  full="${m}-Instruct"
  for s in "${STREAMS[@]}"; do
    for c in "${CONDS[@]}"; do
      A="$(common_args "$m" "$s" "$c" "embeddings/${m}/${s}/${ml}_${s}.pt")"
      T="--toks $(toks_for "$c")"
      PL="--layers $(layers_for "$m")"    # pyvene methods
      CL="--layers $(clayers_for "$m")"   # circuit/ablation methods
      D="$OUT/${full}/${s}/${c}"

      emit_if_missing "das_${m}_${s}_${c}" "$D" "$D/das.csv" \
        "python3 core/interp/das.py $A $PL $T --n_sample $N_SAMPLE_CELL --add_inverse \
         --out_csv ${D}/das.csv"
      emit_if_missing "diffmean_${m}_${s}_${c}" "$D" "$D/diffmean.csv" \
        "python3 core/interp/diffmean.py $A $PL $T --n_sample $N_SAMPLE_CELL --add_inverse_test \
         --out_csv ${D}/diffmean.csv"
      emit_if_missing "probe_${m}_${s}_${c}" "$D" "$D/probe.csv" \
        "python3 core/interp/linear_probe.py $A $PL $T --n_sample $N_SAMPLE_CELL --add_inverse_test \
         --out_csv ${D}/probe.csv"
      emit_if_missing "patch_${m}_${s}_${c}_k${K}" "$D" "$D/patch_k${K}.csv" \
        "python3 core/interp/circuit.py $A $CL --n_sample $N_SAMPLE --add_inverse \
         --hook_points mlp_act --top_k $K \
         --out_csv ${D}/patch_k${K}.csv --circuit_out ${D}/patch_nodes_k${K}.csv"
      emit_if_missing "ablation_${m}_${s}_${c}_k${K}" "$D" "$D/ablation_k${K}.csv" \
        "python3 core/interp/ablation.py $A $CL --n_sample $N_SAMPLE --add_inverse \
         --hook_points mlp_act --top_k $K --ablation mean \
         --out_csv ${D}/ablation_k${K}.csv --circuit_out ${D}/ablation_nodes_k${K}.csv"
    done
  done
done

N=$(wc -l < "$JOBS")

# --- JOBTRACKER.md regeneration (called under flock) -----------------------
regen_tracker() {
  python3 - "$JOBS" "$STATUS" "$TRACKER" "$LOGDIR" "$N" <<'PY'
import sys, os, collections, datetime
jobs_f, status_dir, tracker_f, logdir, N = sys.argv[1:6]
rows = []
for line in open(jobs_f):
    name, d, cmd = line.rstrip("\n").split("\t", 2)
    st = "PENDING"
    p = os.path.join(status_dir, name)
    if os.path.exists(p):
        st = open(p).read().strip()
    rows.append((name, d, st))
counts = collections.Counter(r[2] for r in rows)
sym = {"PENDING":"⬜","RUNNING":"🟡","OK":"✅","FAIL":"❌"}
done = counts.get("OK",0)+counts.get("FAIL",0)
with open(tracker_f, "w") as f:
    f.write("# Natural-stimuli interp runs — job tracker\n\n")
    f.write(f"_Updated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}_\n\n")
    f.write("Reproduces `results/interp/main_runs` on the **natural** (non-wug) "
            "stimuli test split (`--src_csv`, natural train/test split). "
            "Output → `results/interp/natural_runs/`.\n\n")
    f.write(f"**Progress: {done}/{N} finished** "
            f"({counts.get('OK',0)} ✅ ok, {counts.get('FAIL',0)} ❌ fail, "
            f"{counts.get('RUNNING',0)} 🟡 running, "
            f"{counts.get('PENDING',0)} ⬜ pending)\n\n")
    f.write(f"Logs: `{logdir}`\n\n")
    # group by cell dir
    cells = collections.OrderedDict()
    for name, d, st in rows:
        cells.setdefault(d, []).append((name, st))
    f.write("| Cell | das | diffmean | probe | patch_k128 | ablation_k128 |\n")
    f.write("|---|---|---|---|---|---|\n")
    order = ["das_","diffmean_","probe_","patch_","ablation_"]
    for d, items in cells.items():
        cell = d.replace("results/interp/natural_runs/","")
        m = dict(items)
        def cellsym(pref):
            for nm, st in items:
                if nm.startswith(pref):
                    return sym.get(st, st)
            return "-"
        f.write(f"| `{cell}` | " + " | ".join(cellsym(p) for p in order) + " |\n")
PY
}
export -f regen_tracker
export JOBS STATUS TRACKER LOGDIR N

{
  echo "[$(date +%T)] $N jobs, $NPROC at a time, 1 gpu"
  echo "[$(date +%T)] master pid $$"
} | tee -a "$LOGDIR/master.log"

regen_tracker

if [[ "$DRY_RUN" == 1 ]]; then
  cut -f1 "$JOBS" | sed 's/^/  /'
  echo "dry run — nothing executed. tracker written to $TRACKER"
  exit 0
fi

cut -f2 "$JOBS" | sort -u | while read -r d; do mkdir -p "$ROOT/$d"; done

export ROOT LOGDIR STATUS TRACKER
run_one() {
  local name="$1" cmd="$2"
  cd "$ROOT" || return 1
  echo RUNNING > "$STATUS/$name"
  ( flock 9; regen_tracker ) 9>"$LOGDIR/.tracker.lock"
  echo "[$(date +%T)] START $name"
  if eval "$cmd" > "$LOGDIR/${name}.log" 2>&1; then
    echo OK > "$STATUS/$name"
    echo "[$(date +%T)] OK    $name"
  else
    echo FAIL > "$STATUS/$name"
    echo "[$(date +%T)] FAIL  $name -> $LOGDIR/${name}.log"
    echo "$name" >> "$LOGDIR/failed.txt"
  fi
  ( flock 9; regen_tracker ) 9>"$LOGDIR/.tracker.lock"
}
export -f run_one

# Reduce fragmentation OOMs (per the CUDA OOM hint on 4B models).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# One xargs pool per GPU, jobs split round-robin. Each pool pins its GPU via
# CUDA_VISIBLE_DEVICES and runs $NPROC jobs at a time on that GPU.
read -r -a GPUS <<< "$GPUS_STR"
NG=${#GPUS[@]}
for gi in "${!GPUS[@]}"; do
  g="${GPUS[$gi]}"
  awk -v n="$NG" -v k="$gi" 'NR % n == k' "$JOBS" > "$JOBS.gpu${g}"
  CUDA_VISIBLE_DEVICES="$g" xargs -d '\n' -a "$JOBS.gpu${g}" -P "$NPROC" -I{} bash -c '
    line="{}"
    name=$(printf "%s" "$line" | cut -f1)
    cmd=$(printf  "%s" "$line" | cut -f3-)
    run_one "$name" "$cmd"
  ' 2>&1 | tee -a "$LOGDIR/master.log" &
done
wait

regen_tracker
nfail=$(wc -l < "$LOGDIR/failed.txt" 2>/dev/null || echo 0)
echo "[$(date +%T)] done. $nfail failures." | tee -a "$LOGDIR/master.log"
[[ -s "$LOGDIR/failed.txt" ]] && sed 's/^/  FAILED /' "$LOGDIR/failed.txt"
exit 0
