#!/usr/bin/env bash
# Sequentially run the two rerun pipelines so they never share a GPU.
# Each sub-script uses BOTH GPUs at 1 job/GPU (2x4B OOMs a 24GB A5000), so they
# must run one after another, not together.
#   1) natural_runs   (14 missing jobs, all 4B)
#   2) seed_replicate (272 missing jobs)
# Both are resume-guarded (emit_if_missing) — existing CSVs are never overwritten.
set -uo pipefail
cd "$HOME/wug-test-interp" || exit 1
LOG="results/interp/chain_reruns.log"
mkdir -p results/interp
echo "==================================================================" >> "$LOG"
echo "[$(date '+%F %T')] CHAIN START (pid $$)" >> "$LOG"

echo "[$(date '+%F %T')] >>> natural_runs" >> "$LOG"
bash pipelines/interp/run_natural_runs.sh -F >> "$LOG" 2>&1
echo "[$(date '+%F %T')] <<< natural_runs done (exit $?)" >> "$LOG"

echo "[$(date '+%F %T')] >>> seed_replicate" >> "$LOG"
bash pipelines/interp/run_seed_replicate.sh -F >> "$LOG" 2>&1
echo "[$(date '+%F %T')] <<< seed_replicate done (exit $?)" >> "$LOG"

echo "[$(date '+%F %T')] CHAIN ALL DONE" >> "$LOG"
