#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.report_global_candidate_search import canonical_json, deep_update, short_hash


METRICS = ("val_auc", "val_ap", "val_mrr", "val_hits@10", "val_ndcg")


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def score_row(row: dict[str, Any], weights: dict[str, float]) -> float:
    total = 0.0
    used = False
    for key, weight in weights.items():
        value = row.get(key, float("nan"))
        if finite(value):
            total += float(weight) * float(value)
            used = True
    return total if used else -float("inf")


def load_candidate_configs(search_dir: Path) -> tuple[str, dict[str, float], dict[str, dict[str, Any]]]:
    config_path = Path("configs/auto_target") / f"ssptgfm_{search_dir.name}.yaml"
    if not config_path.exists():
        return "", {}, {}
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_model = cfg.get("model", {})
    base_train = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    dataset = str(data_cfg.get("name", ""))
    weights = {str(k): float(v) for k, v in cfg.get("search", {}).get("metric_weights", {}).items()}
    out: dict[str, dict[str, Any]] = {}
    for candidate in cfg.get("search", {}).get("candidates", []):
        name = str(candidate.get("name", ""))
        out[name] = {
            "model": deep_update(base_model, candidate.get("model", {})),
            "train": deep_update(base_train, candidate.get("train", {})),
        }
    return dataset, weights, out


def rows_from_partial(path: Path) -> list[dict[str, Any]]:
    dataset, weights, configs = load_candidate_configs(path.parent)
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        candidate = str(raw.get("candidate", ""))
        config = configs.get(candidate)
        if not config:
            continue
        model = config["model"]
        row = {
            "dataset": dataset,
            "search": path.parent.name,
            "candidate": candidate,
            "model_signature": short_hash(model),
            "model_config": canonical_json(model),
            "train_config": canonical_json(config["train"]),
            "score": score_row(raw, weights),
        }
        for metric in METRICS:
            row[metric] = float(raw.get(metric, float("nan")))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate partial validation rows by shared model signature.")
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for root in args.run_root:
        for path in sorted(Path(root).glob("search_*/partial_search_results.jsonl")):
            rows.extend(rows_from_partial(path))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["model_signature"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    datasets_seen = sorted({str(row["dataset"]) for row in rows if row.get("dataset")})
    for signature, sig_rows in grouped.items():
        best_by_dataset: dict[str, dict[str, Any]] = {}
        for row in sig_rows:
            dataset = str(row["dataset"])
            if dataset not in best_by_dataset or float(row["score"]) > float(best_by_dataset[dataset]["score"]):
                best_by_dataset[dataset] = row
        selected = list(best_by_dataset.values())
        means: dict[str, float] = {}
        for key in ("score", *METRICS):
            values = [float(row[key]) for row in selected if finite(row.get(key))]
            means[key] = sum(values) / len(values) if values else float("nan")
        first = selected[0]
        summary_rows.append(
            {
                "model_signature": signature,
                "datasets_completed": len(selected),
                "datasets_expected": len(datasets_seen),
                "datasets": ",".join(sorted(best_by_dataset)),
                "selected_candidates": json.dumps(
                    {str(row["dataset"]): str(row["candidate"]) for row in selected},
                    sort_keys=True,
                ),
                **{f"mean_{key}": value for key, value in means.items()},
                "model_config": first["model_config"],
            }
        )
    summary_rows.sort(
        key=lambda row: (
            int(row["datasets_completed"]),
            float(row["mean_score"]) if finite(row["mean_score"]) else -float("inf"),
        ),
        reverse=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_signature",
        "datasets_completed",
        "datasets_expected",
        "datasets",
        "selected_candidates",
        "mean_score",
        *[f"mean_{metric}" for metric in METRICS],
        "model_config",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    for row in summary_rows[:10]:
        print(
            f"{row['model_signature']} datasets={row['datasets_completed']}/{row['datasets_expected']} "
            f"score={row['mean_score']:.4g} mrr={row['mean_val_mrr']:.4g} "
            f"hits@10={row['mean_val_hits@10']:.4g} ap={row['mean_val_ap']:.4g}"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
