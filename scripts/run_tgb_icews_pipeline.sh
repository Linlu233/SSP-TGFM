#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/data

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH=.

PROBE_RESULTS=results/ssptgfm_search_tgb_icews_probe/search_results.json
SEARCH_CONFIG=configs/ssptgfm_search_tgb_icews_all_losses.yaml
SEARCH_RESULTS=results/ssptgfm_search_tgb_icews_all_losses/search_results.json
BEST_CONFIG=configs/ssptgfm_tgb_icews_best.yaml
BEST_DIR=results/ssptgfm_tgb_icews_best

echo "$(date +%F_%T) tgb_icews_pipeline_wait_probe"
while [ ! -f "$PROBE_RESULTS" ]; do
  if ! screen -ls 2>/dev/null | grep -q "tgb_icews_probe"; then
    echo "$(date +%F_%T) tgb_icews_probe_missing_without_results"
    exit 1
  fi
  sleep 300
done

python - <<'PY'
import json
import math
from pathlib import Path

payload = json.loads(Path("results/ssptgfm_search_tgb_icews_probe/search_results.json").read_text(encoding="utf-8"))
score = float(payload.get("best_validation_mean", float("nan")))
if not math.isfinite(score):
    raise SystemExit(f"probe best_validation_mean is not finite: {score}")
print(f"probe_validated best_validation_mean={score}")
PY

echo "$(date +%F_%T) tgb_icews_full_search_start"
python -u scripts/search_hparams.py --config "$SEARCH_CONFIG" --resume
echo "$(date +%F_%T) tgb_icews_full_search_done"

python scripts/apply_best_candidate.py --search-results "$SEARCH_RESULTS" --targets "$BEST_CONFIG" --require-pass
echo "$(date +%F_%T) tgb_icews_best_candidate_applied"

mkdir -p results/logs
for seed in 1 2 3 4 5; do
  out="results/ssptgfm_tgb_icews_best_seed${seed}"
  log="results/logs/tgb_icews_seed${seed}.log"
  echo "$(date +%F_%T) tgb_icews_formal_seed_start seed=${seed} out=${out} log=${log}"
  python -u scripts/run_ssptgfm.py --config "$BEST_CONFIG" --seeds "$seed" --output "$out" --resume > "$log" 2>&1
  echo "$(date +%F_%T) tgb_icews_formal_seed_done seed=${seed}"
done

python scripts/merge_parallel_results.py \
  --inputs \
  results/ssptgfm_tgb_icews_best_seed1 \
  results/ssptgfm_tgb_icews_best_seed2 \
  results/ssptgfm_tgb_icews_best_seed3 \
  results/ssptgfm_tgb_icews_best_seed4 \
  results/ssptgfm_tgb_icews_best_seed5 \
  --out-dir "$BEST_DIR" \
  --expected-rows 15
python scripts/summarize_results.py --results "$BEST_DIR/all_results.json" --out "$BEST_DIR/summary.csv"
echo "$(date +%F_%T) tgb_icews_pipeline_done"
