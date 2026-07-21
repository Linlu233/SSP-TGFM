#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from ssptgfm.data import generate_synthetic_dataset, load_csv_dataset, load_tgb_dataset, split_by_labels, split_by_time
from ssptgfm.explain import explain_edge
from ssptgfm.utils import load_yaml, save_json


def load_dataset_from_config(cfg: dict):
    data_cfg = cfg.get("data", {})
    if data_cfg.get("name", "synthetic") == "synthetic":
        return generate_synthetic_dataset(
            num_nodes=int(data_cfg.get("num_nodes", 256)),
            num_relations=int(data_cfg.get("num_relations", 4)),
            num_edges=int(data_cfg.get("num_edges", 1600)),
            num_topics=int(data_cfg.get("num_topics", 8)),
            seed=int(data_cfg.get("seed", 1)),
        )
    if data_cfg.get("format") == "tgb":
        return load_tgb_dataset(name=data_cfg["name"], root=data_cfg.get("root", "data/raw/tgb"), download=bool(data_cfg.get("download", True)))
    return load_csv_dataset(data_cfg["path"], name=data_cfg.get("name"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export structure/semantic evidence for test edges.")
    parser.add_argument("--config", default="configs/ssptgfm_smoke.yaml")
    parser.add_argument("--out", default="results/explanations.json")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    dataset = load_dataset_from_config(cfg)
    split_cfg = cfg.get("split", {})
    if split_cfg.get("mode", "time") == "labels":
        splits = split_by_labels(dataset)
    else:
        splits = split_by_time(dataset.edges, float(split_cfg.get("val_ratio", 0.15)), float(split_cfg.get("test_ratio", 0.15)))
    history = splits.train.concat(splits.val)
    rows = [explain_edge(dataset, history.before(float(splits.test.time[i]), strict=True), splits.test.slice([i])) for i in range(min(args.limit, len(splits.test)))]
    save_json(rows, Path(args.out))
    print(args.out)


if __name__ == "__main__":
    main()
