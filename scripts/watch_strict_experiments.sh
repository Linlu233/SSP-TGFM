#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/data

INTERVAL_SECONDS="${INTERVAL_SECONDS:-14400}"
LOG="${LOG:-results/strict_experiment_watchdog.log}"
REPORT="${REPORT:-results/latest_progress_report.txt}"
mkdir -p results

while true; do
  {
    echo "# Latest SSP-TGFM Progress"
    echo
    echo "Time: $(date -u '+%F %T UTC')"
    echo
    echo "GPU:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
    echo
    echo "Screens:"
    screen -ls || true
    echo
    python - <<'PY'
from pathlib import Path
import json

items = [
    ("synthetic_stronggate_v4_formal", "results/ssptgfm_synthetic_stronggate_v4_formal_summary.csv", "summary"),
    ("yago15k_temporal_strict_v2_search", "results/ssptgfm_search_yago15k_temporal_strict_v2/search_results.json", "search"),
    ("yago15k_temporal_strict_v2_formal", "results/ssptgfm_yago15k_temporal_strict_v2_best/summary.csv", "summary"),
    ("tgb_smallpedia_strict_v2_search", "results/ssptgfm_search_tgb_smallpedia_strict_v2/search_results.json", "search"),
    ("tgb_smallpedia_strict_v2_formal", "results/ssptgfm_tgb_smallpedia_strict_v2_best/summary.csv", "summary"),
    ("icews14_strict_v2_search", "results/ssptgfm_search_icews14_strict_v2/search_results.json", "search"),
    ("icews14_strict_v2_formal", "results/ssptgfm_icews14_strict_v2_best/summary.csv", "summary"),
    ("icews05_15_strict_v2_search", "results/ssptgfm_search_icews05_15_strict_v2/search_results.json", "search"),
    ("icews05_15_strict_v2_formal", "results/ssptgfm_icews05_15_strict_v2_best/summary.csv", "summary"),
    ("tgb_yago_strict_v2_search", "results/ssptgfm_search_tgb_yago_strict_v2/search_results.json", "search"),
    ("tgb_yago_strict_v2_formal", "results/ssptgfm_tgb_yago_strict_v2_best/summary.csv", "summary"),
]

print("Strict target status:")
for label, path, kind in items:
    p = Path(path)
    if not p.exists():
        partial = p.with_name("partial_search_results.jsonl") if kind == "search" else p.with_name("partial_results.jsonl")
        rows = sum(1 for _ in partial.open(errors="ignore")) if partial.exists() else 0
        print(f"- {label}: pending rows={rows} path={path}")
        continue
    if kind == "search":
        data = json.loads(p.read_text(errors="ignore"))
        best = (data.get("best_candidate") or {}).get("name")
        print(f"- {label}: done formal_allowed={data.get('formal_allowed')} best={best}")
    else:
        print(f"- {label}: done path={path}")

print()
print("Excluded:")
print("- tgb_icews: excluded by user instruction on 2026-06-04; probe/pipeline screens stopped.")
print()
print("Blocked:")
print("- synthetic_main_ablation: blocked unless mamba_ssm is installable for mamba_encoder.")
PY
  } > "$REPORT"
  cat "$REPORT" >> "$LOG"
  echo >> "$LOG"
  sleep "$INTERVAL_SECONDS"
done
