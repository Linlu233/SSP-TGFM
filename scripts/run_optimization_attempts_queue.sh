#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/data

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PYTHONPATH:-.}"

mkdir -p results/logs results/optimization_attempts

GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-60}"
GPU_MONITOR_LOG="${GPU_MONITOR_LOG:-results/logs/optimization_attempts_gpu.log}"

monitor_gpu() {
  while true; do
    if command -v nvidia-smi >/dev/null 2>&1; then
      {
        date +%F_%T
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
      } >> "$GPU_MONITOR_LOG" 2>&1 || true
    fi
    sleep "$GPU_MONITOR_INTERVAL"
  done
}

monitor_gpu &
GPU_MONITOR_PID="$!"
cleanup() {
  kill "$GPU_MONITOR_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

run_pipeline() {
  local label="$1"
  local max_parallel="$2"
  local search_config="$3"
  local search_out="$4"
  local best_config="$5"
  local best_dir="$6"
  local expected_rows="$7"

  echo "$(date +%F_%T) queue_pipeline_start label=${label} max_parallel_seeds=${max_parallel}"
  MAX_PARALLEL_SEEDS="$max_parallel" \
    scripts/run_strict_dataset_pipeline.sh \
      "$label" \
      "$search_config" \
      "$search_out" \
      "$best_config" \
      "$best_dir" \
      "$expected_rows"
  echo "$(date +%F_%T) queue_pipeline_done label=${label}"
}

echo "$(date +%F_%T) optimization_attempt_queue_start cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"

run_pipeline \
  "icews14_formula_rank" \
  "${MAX_PARALLEL_SEEDS_ICEWS14:-2}" \
  "configs/optimization_attempts/ssptgfm_search_icews14_formula_rank.yaml" \
  "results/optimization_attempts/search_icews14_formula_rank" \
  "configs/optimization_attempts/ssptgfm_icews14_formula_rank_best.yaml" \
  "results/optimization_attempts/ssptgfm_icews14_formula_rank_best" \
  "5"

python scripts/compare_optimization_attempt.py \
  --output results/optimization_attempts/compare_icews14_formula_rank.csv \
  --input optimized results/optimization_attempts/ssptgfm_icews14_formula_rank_best/all_results.json \
  --input strict_v2 results/ssptgfm_icews14_strict_v2_best/all_results.json \
  --input fast_strong results/fast_strong_baselines_strict_v1/icews14/all_results.json

run_pipeline \
  "tgb_yago_formula_rank" \
  "${MAX_PARALLEL_SEEDS_TKGL_YAGO:-1}" \
  "configs/optimization_attempts/ssptgfm_search_tgb_yago_formula_rank.yaml" \
  "results/optimization_attempts/search_tgb_yago_formula_rank" \
  "configs/optimization_attempts/ssptgfm_tgb_yago_formula_rank_best.yaml" \
  "results/optimization_attempts/ssptgfm_tgb_yago_formula_rank_best" \
  "5"

python scripts/compare_optimization_attempt.py \
  --output results/optimization_attempts/compare_tgb_yago_formula_rank.csv \
  --input optimized results/optimization_attempts/ssptgfm_tgb_yago_formula_rank_best/all_results.json \
  --input strict_v2 results/ssptgfm_tgb_yago_strict_v2_best/all_results.json \
  --input fast_strong results/fast_strong_baselines_strict_v1/tgb_yago/all_results.json

echo "$(date +%F_%T) optimization_attempt_queue_done"
