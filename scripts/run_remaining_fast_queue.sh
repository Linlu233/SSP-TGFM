#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/data
mkdir -p results/logs

echo "$(date +%F_%T) remaining_fast_queue_start"

while screen -ls 2>/dev/null | grep -Eq "yago15k_temporal_strict_v2|tgb_smallpedia_strict_v2"; do
  echo "$(date +%F_%T) waiting_for_active_fast_screens"
  sleep 300
done

echo "$(date +%F_%T) icews14_strict_v2_queue_start"
MAX_PARALLEL_SEEDS=3 scripts/run_strict_dataset_pipeline.sh \
  icews14_strict_v2 \
  configs/ssptgfm_search_icews14_all_losses.yaml \
  results/ssptgfm_search_icews14_strict_v2 \
  configs/ssptgfm_icews14_best.yaml \
  results/ssptgfm_icews14_strict_v2_best \
  15

echo "$(date +%F_%T) icews05_15_strict_v2_queue_start"
MAX_PARALLEL_SEEDS=2 scripts/run_strict_dataset_pipeline.sh \
  icews05_15_strict_v2 \
  configs/ssptgfm_search_icews05_15_all_losses.yaml \
  results/ssptgfm_search_icews05_15_strict_v2 \
  configs/ssptgfm_icews05_15_best.yaml \
  results/ssptgfm_icews05_15_strict_v2_best \
  15

echo "$(date +%F_%T) tgb_yago_strict_v2_queue_start"
MAX_PARALLEL_SEEDS=1 scripts/run_strict_dataset_pipeline.sh \
  tgb_yago_strict_v2 \
  configs/ssptgfm_search_tgb_yago_all_losses.yaml \
  results/ssptgfm_search_tgb_yago_strict_v2 \
  configs/ssptgfm_tgb_yago_best.yaml \
  results/ssptgfm_tgb_yago_strict_v2_best \
  15

echo "$(date +%F_%T) remaining_fast_queue_done"
