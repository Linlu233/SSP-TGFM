#!/usr/bin/env bash
set -u

MAX_PARALLEL="${MAX_PARALLEL:-2}"
VRAM_LIMIT_MB="${VRAM_LIMIT_MB:-28000}"
LOG_DIR="results/logs"
QUEUE_LOG="${LOG_DIR}/ablation_unified_s15sf02_queue.log"

mkdir -p "${LOG_DIR}"

CONFIGS=(
  "configs/ablation_unified_s15sf02_official/ssptgfm_yago15k_temporal_unified_s15sf02_official_ablation.yaml"
  "configs/ablation_unified_s15sf02_official/ssptgfm_tgb_smallpedia_unified_s15sf02_official_ablation.yaml"
  "configs/ablation_unified_s15sf02_official/ssptgfm_tgb_yago_unified_s15sf02_official_ablation.yaml"
  "configs/ablation_unified_s15sf02_official/ssptgfm_icews14_unified_s15sf02_official_ablation.yaml"
  "configs/ablation_unified_s15sf02_official/ssptgfm_icews05_15_unified_s15sf02_official_ablation.yaml"
)

timestamp() {
  date +"%Y-%m-%d_%H:%M:%S"
}

gpu_used_mb() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' '
}

wait_for_slot() {
  while [ "$(jobs -pr | wc -l)" -ge "${MAX_PARALLEL}" ]; do
    sleep 60
  done
}

wait_for_vram() {
  while true; do
    used="$(gpu_used_mb)"
    if [ -z "${used}" ] || [ "${used}" -lt "${VRAM_LIMIT_MB}" ]; then
      break
    fi
    echo "$(timestamp) wait_vram used=${used}MB limit=${VRAM_LIMIT_MB}MB" >> "${QUEUE_LOG}"
    sleep 60
  done
}

config_output_dir() {
  conda run -n base python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['output_dir'])" "$1"
}

run_one() {
  cfg="$1"
  name="$(basename "${cfg}" .yaml)"
  out_dir="$(config_output_dir "${cfg}")"
  log="${LOG_DIR}/ablation_${name}.log"
  echo "$(timestamp) start ${name} output=${out_dir}" >> "${QUEUE_LOG}"
  PYTHONUNBUFFERED=1 conda run --no-capture-output -n base python scripts/run_ssptgfm.py --config "${cfg}" --resume > "${log}" 2>&1
  status="$?"
  echo "$(timestamp) finish ${name} status=${status}" >> "${QUEUE_LOG}"
  if [ "${status}" -eq 0 ]; then
    PYTHONUNBUFFERED=1 conda run --no-capture-output -n base python scripts/summarize_results.py --results "${out_dir}/all_results.json" --out "${out_dir}/summary.csv" >> "${log}" 2>&1
    echo "$(timestamp) summary ${name} status=$?" >> "${QUEUE_LOG}"
  fi
  return "${status}"
}

echo "$(timestamp) queue_start max_parallel=${MAX_PARALLEL} vram_limit=${VRAM_LIMIT_MB}" >> "${QUEUE_LOG}"

for cfg in "${CONFIGS[@]}"; do
  wait_for_slot
  wait_for_vram
  run_one "${cfg}" &
  sleep 10
done

wait
echo "$(timestamp) queue_done" >> "${QUEUE_LOG}"
