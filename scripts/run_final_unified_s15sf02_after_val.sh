#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/data
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

wait_or_run_search() {
  local label="$1"
  local config="$2"
  local out_dir="$3"
  local log="$4"
  local result="$out_dir/search_results.json"
  if [[ -s "$result" ]]; then
    echo "$(date +%F_%T) ${label} search already complete"
    return
  fi
  while ps -eo comm=,args= | awk -v cfg="$config" '
    $1 ~ /^python/ && index($0, "scripts/search_hparams.py") && index($0, cfg) { found = 1 }
    END { exit found ? 0 : 1 }
  '; do
    echo "$(date +%F_%T) ${label} search running; waiting"
    sleep 120
  done
  if [[ -s "$result" ]]; then
    echo "$(date +%F_%T) ${label} search complete after wait"
    return
  fi
  echo "$(date +%F_%T) ${label} search start"
  python -u scripts/search_hparams.py --config "$config" --resume > "$log" 2>&1
  echo "$(date +%F_%T) ${label} search done"
}

wait_or_run_search \
  "icews14" \
  "configs/auto_target/ssptgfm_search_icews14_s15_fe647b370ebb_val.yaml" \
  "results/unified_s15_fe647b370ebb_val/search_icews14" \
  "results/logs/search_icews14_s15_fe647b370ebb_val.log"

wait_or_run_search \
  "icews05_15" \
  "configs/auto_target/ssptgfm_search_icews05_15_s15_fe647b370ebb_val.yaml" \
  "results/unified_s15_fe647b370ebb_val/search_icews05_15" \
  "results/logs/search_icews05_15_s15_fe647b370ebb_val.log"

wait_or_run_search \
  "yago15k_temporal" \
  "configs/auto_target/ssptgfm_search_yago15k_temporal_s15_fe647b370ebb_val.yaml" \
  "results/unified_s15_fe647b370ebb_val/search_yago15k_temporal" \
  "results/logs/search_yago15k_temporal_s15_fe647b370ebb_val.log"

python scripts/apply_best_candidate.py \
  --search-results results/unified_s15_fe647b370ebb_val/search_icews14/search_results.json \
  --targets configs/auto_target/ssptgfm_icews14_unified_s15sf02_official.yaml \
  --require-pass

python scripts/apply_best_candidate.py \
  --search-results results/unified_s15_fe647b370ebb_val/search_icews05_15/search_results.json \
  --targets configs/auto_target/ssptgfm_icews05_15_unified_s15sf02_official.yaml \
  --require-pass

python scripts/apply_best_candidate.py \
  --search-results results/unified_s15_fe647b370ebb_val/search_yago15k_temporal/search_results.json \
  --targets configs/auto_target/ssptgfm_yago15k_temporal_unified_s15sf02_official.yaml \
  --require-pass

echo "$(date +%F_%T) final formal queue start"
python -u scripts/run_final_unified_s15sf02_queue.py
echo "$(date +%F_%T) final formal queue done"
