#!/usr/bin/env bash
#
# Expand results/interp/seed_replicate to full main_runs coverage across the top-5
# seeds, for ALL four combos: {2B,4B} x {syntax,vision}.
#
# Methodology matches main_runs: fit on the natural train split, evaluate on the
# wug test split (--train_csv natural + --test_csv wug/<stream>).
#
# Target per combo: {das,diffmean,probe,patch_k128,ablation_k128} x att0..att3 x 5 seeds.
# Cells whose output CSV already exists are skipped (so the pre-existing 2B
# syntax/vision att0/att1 das/diffmean/probe are not re-run).
#
# Seeds are the TOP-5 by final_overall from the 50-seed CI runs
# (results/train/CI_seed_runs/...); the matching per-seed trained embeddings live
# in embeddings/<model>/<stream>_seeds/seed_<seed>.pt.
#
# Output: results/interp/seed_replicate/<model>/<stream>/<cond>/seed_<seed>/*.csv
# Progress: results/JOBTRACKER_seed_replicate.md

set -uo pipefail

if [[ "${_SEED_DAEMON:-0}" != 1 && " $* " != *" -n "* && " $* " != *" -F "* ]]; then
  export _SEED_DAEMON=1
  _dir="$HOME/wug-test-interp/results/interp/seed_replicate"
  mkdir -p "$_dir"
  _out="$_dir/nohup.out"
  setsid nohup bash "$0" "$@" > "$_out" 2>&1 &
  _pid=$!
  sleep 2
  echo "=============================================================="
  echo " detached. master PID : $_pid"
  echo " kill with            : kill -TERM -$_pid"
  echo " stdout               : $_out"
  echo " tracker              : $HOME/wug-test-interp/results/JOBTRACKER_seed_replicate.md"
  echo " follow               : tail -f $_dir/logs/latest/master.log"
  echo "=============================================================="
  if ! kill -0 "$_pid" 2>/dev/null; then
    echo "--- daemon exited immediately; $_out says: ---"; cat "$_out"; exit 1
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

ENV_BIN="/mnt/dv/wid/projects3/Rogers-muri-human-ai/zstuddiford/tmp/miniconda3/envs/wug_test_env/bin"
[[ -x "$ENV_BIN/python3" ]] && export PATH="$ENV_BIN:$PATH"

ROOT="$HOME/wug-test-interp"
OUT="results/interp/seed_replicate"
TRACKER="$ROOT/results/JOBTRACKER_seed_replicate.md"
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

if [[ "$FORCE" == 0 && "$DRY_RUN" == 0 ]]; then
  gpu_ok=$(python3 -c 'import torch; print(int(torch.cuda.is_available()))' 2>/dev/null || echo 0)
  [[ "$gpu_ok" != "1" ]] && { echo "ERROR: cuda not available." >&2; exit 1; }
  echo "[$(date +%T)] CUDA ok: $(python3 -c 'import torch;print(torch.cuda.get_device_name(0))')"
fi

CACHE_DIR="/mnt/dv/wid/projects3/Rogers-muri-human-ai/zstuddiford"
N_SAMPLE="${N_SAMPLE:-200}"             # circuits
N_SAMPLE_CELL="${N_SAMPLE_CELL:-400}"   # das/diffmean/probe
K=128

MODELS=(Qwen3-VL-2B Qwen3-VL-4B)
STREAMS=(syntax vision)
CONDS=(target_verb_att0_opp target_verb_att1_opp
       target_verb_att2_opp target_verb_att3_opp)

# Top-5 seeds per model/stream (== existing embeddings == top by final_overall).
SEEDS_Qwen3_VL_2B_syntax="31085 37722 48126 54321 96684"
SEEDS_Qwen3_VL_2B_vision="27616 47410 68200 79404 92980"
SEEDS_Qwen3_VL_4B_syntax="2440 19075 64892 75470 80320"
SEEDS_Qwen3_VL_4B_vision="16672 33130 40760 43304 88660"
seeds_for() { local v="SEEDS_${1//-/_}_$2"; echo "${!v}"; }

LAYERS_PY_Qwen3_VL_2B="1 5 10 15 20 25 28"
LAYERS_PY_Qwen3_VL_4B="1 5 10 15 20 25 30 35 36"
LAYERS_CIRC="2 7 12 17 22 27"
pylayers_for() { local v="LAYERS_PY_${1//-/_}"; echo "${!v}"; }

TOKS_target_verb_att0_opp="4 5"
TOKS_target_verb_att1_opp="7 8"
TOKS_target_verb_att2_opp="10 11"
TOKS_target_verb_att3_opp="13 14"
toks_for() { local v="TOKS_$1"; echo "${!v}"; }

emit() {  # emit <name> <outdir> <command...>
  local name="$1" dir="$2"; shift 2
  [[ -n "$ONLY" && "$name" != *"$ONLY"* ]] && return 0
  printf '%s\t%s\t%s\n' "$name" "$dir" "$*" >> "$JOBS"
  echo PENDING > "$STATUS/$name"
}

# emit a method only if its output CSV does not already exist.
emit_if_missing() {  # <name> <dir> <out_rel_csv> <command...>
  local name="$1" dir="$2" out="$3"; shift 3
  if [[ "$FORCE" == 0 && -s "$ROOT/$out" ]]; then return 0; fi
  emit "$name" "$dir" "$@"
}

: > "$JOBS"
for m in "${MODELS[@]}"; do
  ml="$(echo "$m" | tr 'A-Z-' 'a-z_')"
  full="${m}-Instruct"
  PL="--layers $(pylayers_for "$m")"
  CL="--layers $LAYERS_CIRC"
  for s in "${STREAMS[@]}"; do
    train_csv="results/eval/attractors/${full}/agreement_target_natural_scored.csv"
    test_csv="results/eval/attractors/${full}/${s}/agreement_target_wug_scored.csv"
    for c in "${CONDS[@]}"; do
      T="--toks $(toks_for "$c")"
      for sd in $(seeds_for "$m" "$s"); do
        emb="embeddings/${m}/${s}_seeds/seed_${sd}.pt"
        A="--model_path Qwen/${full} --cache_dir ${CACHE_DIR} --embeddings_path ${emb} \
--train_csv ${train_csv} --test_csv ${test_csv} \
--source_input_col base_sentence_sg --base_input_col base_sentence_pl \
--source_completion_A good_singular --source_completion_B bad_singular \
--filter condition=${c} is_correct_all=TRUE"
        D="$OUT/${m}/${s}/${c}/seed_${sd}"
        tag="${m}_${s}_${c}_s${sd}"

        emit_if_missing "das_${tag}" "$D" "$D/das.csv" \
          "python3 core/interp/das.py $A $PL $T --n_sample $N_SAMPLE_CELL --add_inverse \
           --out_csv ${D}/das.csv"
        emit_if_missing "diffmean_${tag}" "$D" "$D/diffmean.csv" \
          "python3 core/interp/diffmean.py $A $PL $T --n_sample $N_SAMPLE_CELL --add_inverse_test \
           --out_csv ${D}/diffmean.csv"
        emit_if_missing "probe_${tag}" "$D" "$D/probe.csv" \
          "python3 core/interp/linear_probe.py $A $PL $T --n_sample $N_SAMPLE_CELL --add_inverse_test \
           --out_csv ${D}/probe.csv"
        emit_if_missing "patch_${tag}_k${K}" "$D" "$D/patch_k${K}.csv" \
          "python3 core/interp/circuit.py $A $CL --n_sample $N_SAMPLE --add_inverse \
           --hook_points mlp_act --top_k $K \
           --out_csv ${D}/patch_k${K}.csv --circuit_out ${D}/patch_nodes_k${K}.csv"
        emit_if_missing "ablation_${tag}_k${K}" "$D" "$D/ablation_k${K}.csv" \
          "python3 core/interp/ablation.py $A $CL --n_sample $N_SAMPLE --add_inverse \
           --hook_points mlp_act --top_k $K --ablation mean \
           --out_csv ${D}/ablation_k${K}.csv --circuit_out ${D}/ablation_nodes_k${K}.csv"
      done
    done
  done
done

N=$(wc -l < "$JOBS")

regen_tracker() {
  python3 - "$JOBS" "$STATUS" "$TRACKER" "$LOGDIR" "$N" <<'PY'
import sys, os, collections, datetime
jobs_f, status_dir, tracker_f, logdir, N = sys.argv[1:6]
rows=[]
for line in open(jobs_f):
    name,d,cmd=line.rstrip("\n").split("\t",2)
    st="PENDING"; p=os.path.join(status_dir,name)
    if os.path.exists(p): st=open(p).read().strip()
    rows.append((name,d,st))
c=collections.Counter(r[2] for r in rows)
sym={"PENDING":"⬜","RUNNING":"🟡","OK":"✅","FAIL":"❌"}
done=c.get("OK",0)+c.get("FAIL",0)
with open(tracker_f,"w") as f:
    f.write("# seed_replicate expansion — job tracker\n\n")
    f.write(f"_Updated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}_\n\n")
    f.write("Full main_runs method coverage across top-5 seeds for {2B,4B}x{syntax,vision}. "
            "Fit natural-train, eval wug-test. Cells already present are skipped.\n\n")
    f.write(f"**Progress: {done}/{N} scheduled jobs finished** ({c.get('OK',0)} ✅, "
            f"{c.get('FAIL',0)} ❌, {c.get('RUNNING',0)} 🟡, {c.get('PENDING',0)} ⬜)\n\n")
    f.write(f"Logs: `{logdir}`\n\n")
    # per model/stream/cond summary
    per=collections.Counter()
    tot=collections.Counter()
    for name,d,st in rows:
        parts=d.split("/")  # results interp seed_replicate <model> <stream> <cond> seed_x
        key="/".join(parts[3:6])
        tot[key]+=1
        if st=="OK": per[key]+=1
    f.write("| model / stream / cond | done / scheduled |\n|---|---|\n")
    for k in sorted(tot):
        f.write(f"| `{k}` | {per[k]}/{tot[k]} |\n")
PY
}
export -f regen_tracker
export JOBS STATUS TRACKER LOGDIR N

echo "[$(date +%T)] $N jobs, $NPROC at a time" | tee -a "$LOGDIR/master.log"
regen_tracker

if [[ "$DRY_RUN" == 1 ]]; then
  echo "scheduled $N jobs (missing cells only). tracker: $TRACKER"
  cut -f1 "$JOBS" | sed -E 's/_s[0-9]+.*//' | sort | uniq -c
  exit 0
fi

cut -f2 "$JOBS" | sort -u | while read -r d; do mkdir -p "$ROOT/$d"; done
export ROOT LOGDIR STATUS TRACKER
run_one() {
  local name="$1" cmd="$2"
  cd "$ROOT" || return 1
  echo RUNNING > "$STATUS/$name"; ( flock 9; regen_tracker ) 9>"$LOGDIR/.tracker.lock"
  echo "[$(date +%T)] START $name"
  if eval "$cmd" > "$LOGDIR/${name}.log" 2>&1; then
    echo OK > "$STATUS/$name"; echo "[$(date +%T)] OK    $name"
  else
    echo FAIL > "$STATUS/$name"; echo "[$(date +%T)] FAIL  $name -> $LOGDIR/${name}.log"
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
      line="{}"; name=$(printf "%s" "$line" | cut -f1); cmd=$(printf "%s" "$line" | cut -f3-)
      run_one "$name" "$cmd"
    ' 2>&1 | tee -a "$LOGDIR/master.log" &
done
wait

regen_tracker
nfail=$(wc -l < "$LOGDIR/failed.txt" 2>/dev/null || echo 0)
echo "[$(date +%T)] done. $nfail failures." | tee -a "$LOGDIR/master.log"
[[ -s "$LOGDIR/failed.txt" ]] && sed 's/^/  FAILED /' "$LOGDIR/failed.txt"
exit 0
