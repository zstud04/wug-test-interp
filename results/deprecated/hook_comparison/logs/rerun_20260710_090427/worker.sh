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
