#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = ("test_auc", "test_ap", "test_mrr", "test_hits@1", "test_hits@10", "test_hits@50", "test_hits@100", "test_ndcg")
GROUP_KEYS = ("dataset", "method", "ablation", "scenario", "few_shot_ratio")


def load_rows(label: str, path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["source"] = label
        out.append(item)
    return out


def finite(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare optimized SSP-TGFM runs against existing strict baselines.")
    parser.add_argument("--input", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for label, raw_path in args.input:
        rows.extend(load_rows(label, Path(raw_path)))
    if not rows:
        raise SystemExit("no rows found in comparison inputs")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["source"],) + tuple(row.get(k, "") for k in GROUP_KEYS)
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        source, dataset, method, ablation, scenario, few_shot_ratio = key
        out: dict[str, Any] = {
            "source": source,
            "dataset": dataset,
            "method": method,
            "ablation": ablation,
            "scenario": scenario,
            "few_shot_ratio": few_shot_ratio,
        }
        for metric in METRICS:
            values = finite([row.get(metric) for row in group_rows])
            out[f"{metric}_mean"] = mean(values) if values else ""
            out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else ""
            out[f"{metric}_count"] = len(values)
        summary_rows.append(out)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary_rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"wrote {out_path}")
    for metric in ("test_auc", "test_ap", "test_mrr", "test_hits@10"):
        col = f"{metric}_mean"
        ranked = [row for row in summary_rows if row.get(col) != ""]
        ranked.sort(key=lambda row: float(row[col]), reverse=True)
        if not ranked:
            continue
        best = ranked[0]
        optimized = next((row for row in ranked if row["source"] == "optimized" and row["method"] == "ssptgfm" and row["ablation"] == "full"), None)
        if optimized is None:
            print(f"{metric}: best={best['source']}/{best['method']}/{best['ablation']} {float(best[col]):.6f}")
            continue
        delta = float(optimized[col]) - float(best[col])
        print(
            f"{metric}: optimized={float(optimized[col]):.6f} "
            f"best={best['source']}/{best['method']}/{best['ablation']} {float(best[col]):.6f} "
            f"delta_to_best={delta:.6f}"
        )


if __name__ == "__main__":
    main()
