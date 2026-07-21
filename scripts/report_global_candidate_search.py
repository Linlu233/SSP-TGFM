#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


METRICS = ("val_auc", "val_ap", "val_mrr", "val_hits@10", "val_ndcg")


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def short_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:12]


def load_candidate_configs(search_results_path: Path) -> dict[str, dict[str, Any]]:
    config_path = Path("configs/auto_target") / f"ssptgfm_{search_results_path.parent.name}.yaml"
    if not config_path.exists():
        return {}
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_model = cfg.get("model", {})
    base_train = cfg.get("train", {})
    out: dict[str, dict[str, Any]] = {}
    for candidate in cfg.get("search", {}).get("candidates", []):
        name = str(candidate.get("name", ""))
        out[name] = {
            "model": deep_update(base_model, candidate.get("model", {})),
            "train": deep_update(base_train, candidate.get("train", {})),
        }
    return out


def rows_from_search(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate_configs = load_candidate_configs(path)
    rows: list[dict[str, Any]] = []
    for summary in payload.get("candidate_summaries", []):
        means = summary.get("means", {})
        candidate_name = str(summary.get("candidate", ""))
        config = candidate_configs.get(candidate_name, {})
        model_config = config.get("model", {})
        row = {
            "dataset": str(payload.get("dataset", "")),
            "search": path.parent.name,
            "candidate": candidate_name,
            "model_signature": short_hash(model_config) if model_config else "",
            "model_config": canonical_json(model_config) if model_config else "",
            "train_config": canonical_json(config.get("train", {})) if config else "",
            "score": float(summary.get("score", float("nan"))),
        }
        for metric in METRICS:
            row[metric] = float(means.get(metric, float("nan")))
        rows.append(row)
    return rows


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate validation-only searches by one shared candidate or one shared model across datasets.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--group-by",
        choices=("candidate", "model"),
        default="candidate",
        help="candidate fixes model and training hyperparameters; model fixes only the model config and lets each dataset use validation-selected train hyperparameters.",
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    rows: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("search_*/*search_results.json")):
        rows.extend(rows_from_search(path))
    datasets = sorted({row["dataset"] for row in rows if row["dataset"]})
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row["candidate"]) if args.group_by == "candidate" else str(row["model_signature"])
        if key:
            grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for group_key, cand_rows in sorted(grouped.items()):
        best_by_dataset: dict[str, dict[str, Any]] = {}
        for row in cand_rows:
            dataset = str(row["dataset"])
            old = best_by_dataset.get(dataset)
            if old is None or float(row["score"]) > float(old["score"]):
                best_by_dataset[dataset] = row
        selected_rows = list(best_by_dataset.values()) if args.group_by == "model" else cand_rows
        present = sorted({row["dataset"] for row in selected_rows if row["dataset"]})
        metric_means: dict[str, float] = {}
        for metric in ("score", *METRICS):
            values = [float(row[metric]) for row in selected_rows if finite(row.get(metric))]
            metric_means[metric] = sum(values) / len(values) if values else float("nan")
        first = selected_rows[0] if selected_rows else {}
        summary_rows.append(
            {
                "group": group_key,
                "model_signature": first.get("model_signature", "") if args.group_by == "model" else "",
                "model_config": first.get("model_config", "") if args.group_by == "model" else "",
                "datasets_completed": len(present),
                "datasets_expected": len(datasets),
                "datasets": ",".join(present),
                "selected_candidates": json.dumps(
                    {str(row["dataset"]): str(row["candidate"]) for row in selected_rows},
                    sort_keys=True,
                    ensure_ascii=True,
                ),
                **{f"mean_{key}": value for key, value in metric_means.items()},
            }
        )
    summary_rows.sort(
        key=lambda row: (
            int(row["datasets_completed"]),
            float(row["mean_score"]) if finite(row["mean_score"]) else -float("inf"),
        ),
        reverse=True,
    )

    out_path = Path(args.out) if args.out else run_root / "global_candidate_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
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
            f"{row['group']} datasets={row['datasets_completed']}/{row['datasets_expected']} "
            f"score={row['mean_score']:.4g} mrr={row['mean_val_mrr']:.4g} "
            f"hits@10={row['mean_val_hits@10']:.4g} ap={row['mean_val_ap']:.4g}"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
