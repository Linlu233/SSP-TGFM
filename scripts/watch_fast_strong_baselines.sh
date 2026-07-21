#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/data

OUT_ROOT="${OUT_ROOT:-results/fast_strong_baselines_strict_v1}"
REPORT="${REPORT:-results/fast_strong_baselines_progress.txt}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"

labels=(
  yago15k_temporal
  tgb_smallpedia
  icews14
  icews05_15
  tgb_yago
)

while true; do
  {
    echo "timestamp=$(date +%F_%T)"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
    echo "[processes]"
    ps -eo pid,ppid,stat,etime,%cpu,%mem,rss,cmd | grep -E 'run_fast_strong_baselines.py|run_fast_strong_baselines_queue.sh' | grep -v grep || true
    echo "[rows]"
    for label in "${labels[@]}"; do
      partial="$OUT_ROOT/$label/partial_results.jsonl"
      summary="$OUT_ROOT/$label/summary.csv"
      rows=0
      if [ -f "$partial" ]; then
        rows=$(wc -l < "$partial")
      fi
      if [ -f "$summary" ]; then
        status=done
      else
        status=pending
      fi
      echo "${label} ${status} rows=${rows}/20"
    done
    echo "[queue_tail]"
    tail -n 30 results/logs/fast_strong_queue.log 2>/dev/null || true
  } > "$REPORT"
  sleep "$INTERVAL_SECONDS"
done
