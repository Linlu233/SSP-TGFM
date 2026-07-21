#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/data

echo "# Optimization Attempt Status"
echo
echo "Time: $(date -u '+%F %T UTC')"
echo
echo "GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
echo
echo "Processes:"
ps -eo pid,ppid,stat,etime,pcpu,pmem,args | grep -E "run_ssptgfm|search_hparams|run_optimization_attempts_queue|run_strict_dataset_pipeline" | grep -v grep || true
echo

python - <<'PY'
from pathlib import Path
import csv
import json

attempts = [
    (
        "icews14_formula_rank",
        Path("results/optimization_attempts/search_icews14_formula_rank"),
        Path("results/optimization_attempts/ssptgfm_icews14_formula_rank_best"),
    ),
    (
        "tgb_yago_formula_rank",
        Path("results/optimization_attempts/search_tgb_yago_formula_rank"),
        Path("results/optimization_attempts/ssptgfm_tgb_yago_formula_rank_best"),
    ),
]

for label, search_dir, formal_dir in attempts:
    print(f"{label}:")
    search_results = search_dir / "search_results.json"
    partial_search = search_dir / "partial_search_results.jsonl"
    if search_results.exists():
        data = json.loads(search_results.read_text(encoding="utf-8"))
        best = (data.get("best_candidate") or {}).get("name")
        print(f"  search: done formal_allowed={data.get('formal_allowed')} best={best}")
    else:
        rows = sum(1 for _ in partial_search.open(encoding="utf-8")) if partial_search.exists() else 0
        print(f"  search: running_or_pending rows={rows}")

    summary = formal_dir / "summary.csv"
    all_results = formal_dir / "all_results.json"
    if summary.exists():
        print("  formal: done summary=" + str(summary))
        with summary.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("method") == "ssptgfm" and row.get("ablation") == "full":
                    metrics = ", ".join(
                        f"{key}={float(row[key]):.4f}"
                        for key in ("auc", "ap", "mrr", "hits@10")
                        if row.get(key) not in (None, "")
                    )
                    print(f"  ssptgfm/full: {metrics}")
    else:
        total = 0
        for p in sorted(Path("results").glob(f"ssptgfm_{label}_seed*/partial_results.jsonl")):
            total += sum(1 for _ in p.open(encoding="utf-8"))
        if all_results.exists():
            try:
                total = len(json.loads(all_results.read_text(encoding="utf-8")))
            except Exception:
                pass
        print(f"  formal: running_or_pending rows={total}/5")
    print()
PY
