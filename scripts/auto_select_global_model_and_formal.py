#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.auto_optimize_until_target import DATASETS, DatasetSpec  # noqa: E402
from scripts.report_global_candidate_search import short_hash  # noqa: E402
from scripts.run_global_model_formal import bank_name_from_search_dir, candidate_configs  # noqa: E402


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def selected_specs(labels: set[str] | None) -> list[DatasetSpec]:
    specs = [spec for spec in DATASETS if labels is None or spec.label in labels]
    if labels is not None:
        missing = labels.difference({spec.label for spec in specs})
        if missing:
            raise SystemExit(f"unknown dataset labels: {','.join(sorted(missing))}")
    return specs


def root_complete_for_specs(root: Path, specs: list[DatasetSpec]) -> bool:
    for spec in specs:
        if not any(root.glob(f"search_{spec.label}_*/search_results.json")):
            return False
    return True


def best_by_signature(
    run_roots: list[Path],
    specs: list[DatasetSpec],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in specs:
        for run_root in run_roots:
            for path in sorted(run_root.glob(f"search_{spec.label}_*/search_results.json")):
                bank = bank_name_from_search_dir(spec, path.parent)
                if bank is None:
                    continue
                configs = candidate_configs(path)
                payload = json.loads(path.read_text(encoding="utf-8"))
                for summary in payload.get("candidate_summaries", []):
                    candidate = str(summary.get("candidate", ""))
                    config = configs.get(candidate)
                    if not config:
                        continue
                    score = float(summary.get("score", float("nan")))
                    if not finite(score):
                        continue
                    model = config["model"]
                    signature = short_hash(model)
                    metrics = summary.get("means", {})
                    row = {
                        "dataset": spec.label,
                        "run_root": str(run_root),
                        "search_results": str(path),
                        "bank": bank,
                        "candidate": candidate,
                        "score": score,
                        "metrics": metrics,
                    }
                    by_dataset = grouped.setdefault(signature, {})
                    if spec.label not in by_dataset or score > float(by_dataset[spec.label]["score"]):
                        by_dataset[spec.label] = row
    return grouped


def choose_signature(
    run_roots: list[Path],
    specs: list[DatasetSpec],
) -> tuple[str | None, dict[str, Any]]:
    grouped = best_by_signature(run_roots, specs)
    expected = {spec.label for spec in specs}
    summaries: list[dict[str, Any]] = []
    for signature, by_dataset in grouped.items():
        present = set(by_dataset)
        if not expected.issubset(present):
            continue
        rows = [by_dataset[spec.label] for spec in specs]
        score_values = [float(row["score"]) for row in rows if finite(row.get("score"))]
        mean_score = sum(score_values) / len(score_values) if score_values else -float("inf")
        metric_means: dict[str, float] = {}
        for metric in ("val_auc", "val_ap", "val_mrr", "val_hits@10", "val_ndcg"):
            values = [float(row["metrics"].get(metric, float("nan"))) for row in rows if finite(row["metrics"].get(metric))]
            metric_means[metric] = sum(values) / len(values) if values else float("nan")
        summaries.append(
            {
                "model_signature": signature,
                "mean_score": mean_score,
                "metric_means": metric_means,
                "datasets": {row["dataset"]: row for row in rows},
            }
        )
    summaries.sort(
        key=lambda row: (
            float(row["mean_score"]),
            float(row["metric_means"].get("val_mrr", -float("inf"))),
            float(row["metric_means"].get("val_hits@10", -float("inf"))),
            float(row["metric_means"].get("val_ap", -float("inf"))),
        ),
        reverse=True,
    )
    if not summaries:
        return None, {"complete_signatures": [], "expected_datasets": sorted(expected)}
    return str(summaries[0]["model_signature"]), {"complete_signatures": summaries, "expected_datasets": sorted(expected)}


def write_selection(out_root: Path, payload: dict[str, Any]) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "validation_global_model_auto_selection.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Select one shared SSP-TGFM model by validation only, then run formal tests.")
    parser.add_argument("--search-run-root", action="append", required=True)
    parser.add_argument("--require-complete-root", action="append", default=[])
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--datasets", required=True, help="Comma-separated DatasetSpec labels.")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    labels = {part.strip() for part in args.datasets.split(",") if part.strip()}
    specs = selected_specs(labels)
    run_roots = [Path(path) for path in args.search_run_root]
    required_roots = [Path(path) for path in args.require_complete_root]
    out_root = Path(args.out_root)

    while True:
        incomplete_roots = [str(root) for root in required_roots if not root_complete_for_specs(root, specs)]
        signature, payload = choose_signature(run_roots, specs)
        payload.update(
            {
                "selection_uses": "validation metrics only",
                "test_usage": "not used before this selection",
                "search_run_roots": [str(path) for path in run_roots],
                "required_complete_roots": [str(path) for path in required_roots],
                "incomplete_required_roots": incomplete_roots,
                "selected_model_signature": signature,
            }
        )
        selection_path = write_selection(out_root, payload)
        if signature is not None and not incomplete_roots:
            break
        if not args.wait:
            print(
                {
                    "event": "global_model_selection_waiting",
                    "selection": str(selection_path),
                    "signature_ready": signature is not None,
                    "incomplete_required_roots": incomplete_roots,
                },
                flush=True,
            )
            return
        print(
            {
                "event": "global_model_selection_poll",
                "selection": str(selection_path),
                "signature_ready": signature is not None,
                "incomplete_required_roots": incomplete_roots,
                "sleep_sec": args.poll_seconds,
            },
            flush=True,
        )
        time.sleep(max(1, int(args.poll_seconds)))

    print({"event": "global_model_auto_selected", "signature": signature, "selection": str(selection_path)}, flush=True)
    if args.dry_run:
        return
    cmd = [
        sys.executable,
        "-u",
        "scripts/run_global_model_formal.py",
        *sum((["--search-run-root", str(root)] for root in run_roots), []),
        "--out-root",
        str(out_root),
        "--model-signature",
        str(signature),
        "--datasets",
        ",".join(spec.label for spec in specs),
    ]
    if args.resume:
        cmd.append("--resume")
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONPATH"] = "."
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
