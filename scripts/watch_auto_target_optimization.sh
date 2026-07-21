#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/data

echo "# Auto Target Optimization Status"
echo
echo "Time: $(date +%F_%T) UTC"
echo
echo "GPU:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
else
  echo "nvidia-smi not found"
fi
echo
echo "Processes:"
ps -eo pid,ppid,stat,etime,pcpu,pmem,args \
  | grep -E "auto_optimize_until_target|run_ssptgfm.py|search_hparams.py" \
  | grep -v grep || true
echo
echo "State:"
python - <<'PY'
import json
from pathlib import Path

path = Path("results/auto_target_optimization/state.json")
if not path.exists():
    print("state missing")
    raise SystemExit
state = json.loads(path.read_text(encoding="utf-8"))
for dataset, info in sorted(state.get("datasets", {}).items()):
    status = info.get("status", "running_or_pending")
    existing = info.get("existing", {})
    formal = info.get("formal", {})
    print(f"{dataset}: status={status} existing_wins={existing.get('wins')} formal_wins={formal.get('wins')} bank={formal.get('bank')}")
PY
echo
echo "Recent queue log:"
tail -n 80 results/logs/auto_target_optimization.log 2>/dev/null || true
