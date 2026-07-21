#!/usr/bin/env bash
set -u

cd /root/autodl-tmp/data || exit 1

export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.

SEARCH_CONFIG=configs/ssptgfm_search_yago15k_temporal_all_losses.yaml
BEST_CONFIG=configs/ssptgfm_yago15k_temporal_best.yaml
SEARCH_RESULTS=results/ssptgfm_search_yago15k_temporal_all_losses/search_results.json
BEST_DIR=results/ssptgfm_yago15k_temporal_best

python -u scripts/search_hparams.py --config "$SEARCH_CONFIG" --resume
search_rc=$?
echo "$(date +%F_%T) yago15k_temporal_search_exit rc=$search_rc"
if [ "$search_rc" -ne 0 ]; then
  exit "$search_rc"
fi

python scripts/apply_best_candidate.py --search-results "$SEARCH_RESULTS" --targets "$BEST_CONFIG" --require-pass
apply_rc=$?
echo "$(date +%F_%T) yago15k_temporal_apply_exit rc=$apply_rc"
if [ "$apply_rc" -ne 0 ]; then
  exit "$apply_rc"
fi

mkdir -p results/logs
for seed in 1 2 3 4 5; do
  out="results/ssptgfm_yago15k_temporal_best_seed${seed}"
  log="results/logs/yago15k_temporal_seed${seed}.log"
  setsid bash -lc "cd /root/autodl-tmp/data && env OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. python -u scripts/run_ssptgfm.py --config $BEST_CONFIG --seeds $seed --output $out --resume" > "$log" 2>&1 < /dev/null &
  echo "$(date +%F_%T) launched_yago15k_temporal_seed seed=$seed pid=$! out=$out log=$log"
done

setsid bash -lc 'cd /root/autodl-tmp/data; while true; do rows=0; for s in 1 2 3 4 5; do f="results/ssptgfm_yago15k_temporal_best_seed${s}/partial_results.jsonl"; [ -f "$f" ] && rows=$((rows + $(wc -l < "$f"))); done; ts=$(date +%F_%T); echo "$ts yago15k_rows_pending=$rows/15" >> results/yago15k_temporal_merge_watchdog.log; if [ "$rows" -ge 15 ]; then PYTHONPATH=. python scripts/merge_parallel_results.py --inputs results/ssptgfm_yago15k_temporal_best_seed1 results/ssptgfm_yago15k_temporal_best_seed2 results/ssptgfm_yago15k_temporal_best_seed3 results/ssptgfm_yago15k_temporal_best_seed4 results/ssptgfm_yago15k_temporal_best_seed5 --out-dir results/ssptgfm_yago15k_temporal_best --expected-rows 15 >> results/yago15k_temporal_merge_watchdog.log 2>&1 && PYTHONPATH=. python scripts/summarize_results.py --results results/ssptgfm_yago15k_temporal_best/all_results.json --out results/ssptgfm_yago15k_temporal_best/summary.csv >> results/yago15k_temporal_merge_watchdog.log 2>&1; exit 0; fi; sleep 120; done' >/dev/null 2>&1 < /dev/null &
echo "$(date +%F_%T) launched_yago15k_temporal_merge_watchdog pid=$!"
