#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.auto_optimize_until_target import DATASETS, run_formal


ORDER = ("icews14", "icews05_15", "yago15k_temporal", "tgb_yago", "tgb_smallpedia")
MAX_PARALLEL = {
    "icews14": 2,
    "icews05_15": 2,
    "yago15k_temporal": 2,
    "tgb_yago": 1,
    "tgb_smallpedia": 1,
}


def main() -> None:
    run_root = Path("results/final_unified_s15sf02_official")
    run_root.mkdir(parents=True, exist_ok=True)
    specs = {spec.label: spec for spec in DATASETS}
    base_env = os.environ.copy()
    base_env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    base_env.setdefault("OMP_NUM_THREADS", "2")
    base_env.setdefault("MKL_NUM_THREADS", "2")
    base_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    base_env["PYTHONPATH"] = "."
    for label in ORDER:
        spec = specs[label]
        config = Path("configs/auto_target") / f"ssptgfm_{label}_unified_s15sf02_official.yaml"
        formal_dir = run_root / f"ssptgfm_{label}_unified_s15sf02_official"
        env = base_env.copy()
        env["MAX_PARALLEL_FORMAL_SEEDS"] = str(MAX_PARALLEL[label])
        print(
            {
                "event": "final_unified_dataset_start",
                "dataset": label,
                "config": str(config),
                "output": str(formal_dir),
                "max_parallel_seeds": MAX_PARALLEL[label],
            },
            flush=True,
        )
        run_formal(spec, "unified_s15sf02_official", config, formal_dir, run_root, env)
        print({"event": "final_unified_dataset_done", "dataset": label, "output": str(formal_dir)}, flush=True)


if __name__ == "__main__":
    main()
