#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "usage: $0 LABEL SEARCH_CONFIG SEARCH_OUT BEST_CONFIG BEST_DIR [EXPECTED_ROWS]" >&2
  exit 2
fi

LABEL="$1"
SEARCH_CONFIG="$2"
SEARCH_OUT="$3"
BEST_CONFIG="$4"
BEST_DIR="$5"
EXPECTED_ROWS="${6:-15}"

cd /root/autodl-tmp/data

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PYTHONPATH:-.}"
MAX_PARALLEL_SEEDS="${MAX_PARALLEL_SEEDS:-3}"

mkdir -p results/logs

echo "$(date +%F_%T) ${LABEL}_search_start"
python -u scripts/search_hparams.py --config "$SEARCH_CONFIG" --output "$SEARCH_OUT" --resume
echo "$(date +%F_%T) ${LABEL}_search_done"

python scripts/apply_best_candidate.py --search-results "$SEARCH_OUT/search_results.json" --targets "$BEST_CONFIG" --require-pass
echo "$(date +%F_%T) ${LABEL}_best_candidate_applied"

pids=()
running=0
for seed in 1 2 3 4 5; do
  out="results/ssptgfm_${LABEL}_seed${seed}"
  log="results/logs/${LABEL}_seed${seed}.log"
  echo "$(date +%F_%T) ${LABEL}_formal_seed_start seed=${seed} out=${out} log=${log}"
  bash -lc "cd /root/autodl-tmp/data && env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} PYTHONPATH=. python -u scripts/run_ssptgfm.py --config ${BEST_CONFIG} --seeds ${seed} --output ${out} --resume" > "$log" 2>&1 &
  pids+=("$!")
  running=$((running + 1))
  if [ "$running" -ge "$MAX_PARALLEL_SEEDS" ]; then
    wait -n
    running=$((running - 1))
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

inputs=()
for seed in 1 2 3 4 5; do
  inputs+=("results/ssptgfm_${LABEL}_seed${seed}")
done

python scripts/merge_parallel_results.py --inputs "${inputs[@]}" --out-dir "$BEST_DIR" --expected-rows "$EXPECTED_ROWS"
python scripts/summarize_results.py --results "$BEST_DIR/all_results.json" --out "$BEST_DIR/summary.csv"
echo "$(date +%F_%T) ${LABEL}_pipeline_done"
