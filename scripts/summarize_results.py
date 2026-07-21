#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssptgfm.metrics import paired_significance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/ssptgfm_synthetic/all_results.json")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    with open(args.results, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("rows", [])
    if not rows:
        raise SystemExit("no rows found")
    df = pd.DataFrame(rows)
    group_cols = [c for c in ["dataset", "method", "ablation", "scenario", "few_shot_ratio"] if c in df.columns]
    metric_cols = [c for c in df.columns if c.startswith("test_") or c.startswith("val_")]
    summary = df.groupby(group_cols)[metric_cols].agg(["mean", "std", "count"])
    ci = df.groupby(group_cols)[metric_cols].std() * 1.96 / (df.groupby(group_cols)[metric_cols].count() ** 0.5)
    ci.columns = pd.MultiIndex.from_product([ci.columns, ["ci95"]])
    summary = pd.concat([summary, ci], axis=1).sort_index(axis=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out)
    print(summary.to_string())
    if {"ablation", "seed"}.issubset(df.columns):
        for metric in metric_cols:
            df = df.copy()
            df["comparison"] = df["method"].astype(str) + "/" + df["ablation"].astype(str)
            pivot = df.pivot_table(index=["dataset", "scenario", "few_shot_ratio", "seed"], columns="comparison", values=metric, aggfunc="mean")
            target = "ssptgfm/full"
            if target not in pivot.columns:
                continue
            for comparison in pivot.columns:
                if comparison == target:
                    continue
                stats = paired_significance(pivot[comparison].to_numpy(), pivot[target].to_numpy())
                print(f"paired {metric}: {target} vs {comparison}: {stats}")


if __name__ == "__main__":
    main()
