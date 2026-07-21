#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/data

export PYTHONPATH="${PYTHONPATH:-.}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

OUT_ROOT="${OUT_ROOT:-results/trainable_baselines_strict_v1}"
LOG_DIR="${LOG_DIR:-results/logs}"
MAX_PARALLEL_DATASETS="${MAX_PARALLEL_DATASETS:-2}"
BASELINES="${BASELINES:-distmult,complex,temporal_distmult}"
EXPECTED_ROWS="${EXPECTED_ROWS:-15}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

run_one() {
  local label="$1"
  local config="$2"
  local out_dir="$OUT_ROOT/$label"
  local log="$LOG_DIR/trainable_baselines_${label}.log"
  mkdir -p "$out_dir"
  echo "$(date +%F_%T) ${label}_trainable_baselines_start config=${config} baselines=${BASELINES}" | tee -a "$LOG_DIR/trainable_baselines_queue.log"
  python -u scripts/run_trainable_baselines.py \
    --config "$config" \
    --output "$out_dir" \
    --baselines "$BASELINES" \
    --resume > "$log" 2>&1
  local rows
  rows=$(python - "$out_dir/partial_results.jsonl" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1])
print(sum(1 for _ in p.open(errors="ignore")) if p.exists() else 0)
PY
)
  if [ "$rows" -lt "$EXPECTED_ROWS" ]; then
    echo "$(date +%F_%T) ${label}_trainable_baselines_incomplete rows=${rows}/${EXPECTED_ROWS}" | tee -a "$LOG_DIR/trainable_baselines_queue.log"
    return 1
  fi
  python scripts/summarize_results.py \
    --results "$out_dir/all_results.json" \
    --out "$out_dir/summary.csv" >> "$log" 2>&1
  echo "$(date +%F_%T) ${label}_trainable_baselines_done rows=${rows}" | tee -a "$LOG_DIR/trainable_baselines_queue.log"
}

labels=(
  yago15k_temporal
  tgb_smallpedia
  icews14
  icews05_15
  tgb_yago
)

configs=(
  configs/ssptgfm_yago15k_temporal_best.yaml
  configs/ssptgfm_tgb_smallpedia_best.yaml
  configs/ssptgfm_icews14_best.yaml
  configs/ssptgfm_icews05_15_best.yaml
  configs/ssptgfm_tgb_yago_best.yaml
)

echo "$(date +%F_%T) trainable_baselines_queue_start max_parallel=${MAX_PARALLEL_DATASETS}" | tee -a "$LOG_DIR/trainable_baselines_queue.log"

pids=()
running=0
for idx in "${!labels[@]}"; do
  run_one "${labels[$idx]}" "${configs[$idx]}" &
  pids+=("$!")
  running=$((running + 1))
  if [ "$running" -ge "$MAX_PARALLEL_DATASETS" ]; then
    wait -n
    running=$((running - 1))
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "$(date +%F_%T) trainable_baselines_queue_done" | tee -a "$LOG_DIR/trainable_baselines_queue.log"
