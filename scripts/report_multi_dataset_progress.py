#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


METRICS = ("auc", "ap", "mrr", "hits@10", "ndcg")


def rows_from_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    return [row for row in rows if isinstance(row, dict)]


def finite_mean(rows: list[dict[str, Any]], key: str) -> float:
    vals: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            vals.append(value)
    return mean(vals) if vals else float("nan")


def best_external(paths: list[Path]) -> dict[str, float]:
    best = {metric: -float("inf") for metric in METRICS}
    for path in paths:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in rows_from_json(path):
            if str(row.get("method", "")).lower() == "ssptgfm":
                continue
            key = (str(row.get("dataset", "")), str(row.get("method", "")), str(row.get("ablation", "")))
            grouped.setdefault(key, []).append(row)
        for group_rows in grouped.values():
            for metric in METRICS:
                value = finite_mean(group_rows, f"test_{metric}")
                if math.isfinite(value):
                    best[metric] = max(best[metric], value)
    return best


def best_ssptgfm(paths: list[Path]) -> tuple[Path | None, dict[str, float]]:
    best_path: Path | None = None
    best_values = {metric: -float("inf") for metric in METRICS}
    best_score = -float("inf")
    for path in paths:
        rows = [row for row in rows_from_json(path) if str(row.get("method", "")).lower() == "ssptgfm"]
        if not rows:
            continue
        values = {metric: finite_mean(rows, f"test_{metric}") for metric in METRICS}
        score = sum(value for value in values.values() if math.isfinite(value))
        if score > best_score:
            best_score = score
            best_values = values
            best_path = path
    return best_path, best_values


def main() -> None:
    parser = argparse.ArgumentParser(description="Report SSP-TGFM progress on all active datasets.")
    parser.add_argument("--run-root", default="results/auto_target_multi_sota")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    specs = {
        "icews14": {
            "baselines": [
                Path("results/ssptgfm_icews14_strict_v2_best/all_results.json"),
                Path("results/fast_strong_baselines_strict_v1/icews14/all_results.json"),
            ],
            "ours": [
                Path("results/ssptgfm_icews14_strict_v2_best/all_results.json"),
                *run_root.glob("ssptgfm_icews14_*_best/all_results.json"),
            ],
        },
        "icews05_15": {
            "baselines": [
                Path("results/ssptgfm_icews05_15_strict_v2_best/all_results.json"),
                Path("results/fast_strong_baselines_strict_v1/icews05_15/all_results.json"),
            ],
            "ours": [
                Path("results/ssptgfm_icews05_15_strict_v2_best/all_results.json"),
                *run_root.glob("ssptgfm_icews05_15_*_best/all_results.json"),
            ],
        },
        "tgb_smallpedia": {
            "baselines": [
                Path("results/ssptgfm_tgb_smallpedia_strict_v2_best/all_results.json"),
                Path("results/fast_strong_baselines_strict_v1/tgb_smallpedia/all_results.json"),
            ],
            "ours": [
                Path("results/ssptgfm_tgb_smallpedia_strict_v2_best/all_results.json"),
                *run_root.glob("ssptgfm_tgb_smallpedia_*_best/all_results.json"),
            ],
        },
        "tgb_yago": {
            "baselines": [
                Path("results/ssptgfm_tgb_yago_strict_v2_best/all_results.json"),
                Path("results/fast_strong_baselines_strict_v1/tgb_yago/all_results.json"),
            ],
            "ours": [
                Path("results/ssptgfm_tgb_yago_strict_v2_best/all_results.json"),
                *run_root.glob("ssptgfm_tgb_yago_*_best/all_results.json"),
            ],
        },
        "yago15k_temporal": {
            "baselines": [
                Path("results/ssptgfm_yago15k_temporal_strict_v2_best/all_results.json"),
                Path("results/fast_strong_baselines_strict_v1/yago15k_temporal/all_results.json"),
            ],
            "ours": [
                Path("results/ssptgfm_yago15k_temporal_strict_v2_best/all_results.json"),
                *run_root.glob("ssptgfm_yago15k_temporal_*_best/all_results.json"),
            ],
        },
    }
    for name, spec in specs.items():
        base = best_external(spec["baselines"])
        ours_path, ours = best_ssptgfm(spec["ours"])
        wins = 0
        cells = []
        for metric in METRICS:
            win = math.isfinite(ours[metric]) and math.isfinite(base[metric]) and ours[metric] > base[metric]
            wins += int(win)
            cells.append(f"{metric}:{ours[metric]:.4g}/{base[metric]:.4g}{'*' if win else ''}")
        print(f"{name}: wins={wins}/5 source={ours_path} " + " ".join(cells))


if __name__ == "__main__":
    main()
