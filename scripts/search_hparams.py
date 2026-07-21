#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_ssptgfm import build_dataset, load_tgb_negative_sets, make_model, make_train_config, validate_full_formula_config
from ssptgfm.data import limit_train_edges, split_by_labels, split_by_time
from ssptgfm.experiment_splits import build_scenario
from ssptgfm.text import encode_texts
from ssptgfm.training import train_one_seed
from ssptgfm.utils import ensure_dir, env_report, load_yaml, save_json, set_seed


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _finite_mean(rows: list[dict[str, Any]], metric: str) -> float:
    values = [float(row.get(metric, float("nan"))) for row in rows]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _candidate_score(
    means: dict[str, float],
    metric: str,
    weights: dict[str, float],
    minima: dict[str, float],
) -> float:
    if metric == "composite":
        total = 0.0
        used = False
        for key, weight in weights.items():
            value = means.get(key, float("nan"))
            if np.isfinite(value):
                total += float(weight) * float(value)
                used = True
        return total if used else -float("inf")
    if metric == "all_minima_margin":
        if not minima:
            raise ValueError("search.metric=all_minima_margin requires search.metric_minima")
        margins: list[float] = []
        for key, threshold in minima.items():
            value = means.get(key, float("nan"))
            if not np.isfinite(value):
                return -float("inf")
            denom = max(abs(float(threshold)), 1e-8)
            margins.append((float(value) - float(threshold)) / denom)
        return float(min(margins))
    return float(means.get(metric, -float("inf")))


def _passes_minima(means: dict[str, float], minima: dict[str, float]) -> bool:
    return all(np.isfinite(means.get(key, float("nan"))) and means[key] >= threshold for key, threshold in minima.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only hyperparameter search for SSP-TGFM.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true", help="Preserve and skip rows already present in partial_search_results.jsonl.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    search = cfg.get("search", {})
    candidates = search.get("candidates", [])
    if not candidates:
        raise SystemExit("search.candidates is empty")
    metric = str(search.get("metric", "val_auc"))
    metric_names = [str(x) for x in search.get("metrics", [])]
    if metric not in {"composite", "all_minima_margin"} and metric not in metric_names:
        metric_names.append(metric)
    metric_weights = {str(k): float(v) for k, v in search.get("metric_weights", {}).items()}
    if metric == "composite" and not metric_weights:
        raise SystemExit("search.metric=composite requires search.metric_weights")
    metric_minima = {str(k): float(v) for k, v in search.get("metric_minima", {}).items()}
    for key in metric_weights:
        if key not in metric_names:
            metric_names.append(key)
    for key in metric_minima:
        if key not in metric_names:
            metric_names.append(key)
    out_dir = ensure_dir(args.output or cfg.get("output_dir", "results/ssptgfm_search"))
    strict_search = bool(search.get("strict_full_formula", cfg.get("strict_full_formula", True)))

    dataset = build_dataset(cfg)
    dataset.validate()
    split_cfg = cfg.get("split", {})
    if split_cfg.get("mode", "time") == "labels":
        base_splits = split_by_labels(dataset)
        scenario_name = "label_split"
        scenario_meta = {}
    else:
        dataset, scenario = build_scenario(
            dataset,
            cfg.get("scenario", {"name": "standard"}),
            val_ratio=float(split_cfg.get("val_ratio", 0.15)),
            test_ratio=float(split_cfg.get("test_ratio", 0.15)),
            seed=int(cfg.get("data", {}).get("seed", 1)),
        )
        base_splits = scenario.splits
        scenario_name = scenario.name
        scenario_meta = scenario.metadata
    base_splits.assert_no_temporal_leakage()
    train_edge_limit = search.get("train_edge_limit", None)
    if train_edge_limit is not None:
        limit = int(train_edge_limit)
        if limit > 0 and len(base_splits.train) > limit:
            base_splits.train = limit_train_edges(
                base_splits.train,
                limit,
                mode=str(search.get("train_edge_sample", "prefix")),
            )
            base_splits.assert_no_temporal_leakage()

    requested_cuda = cfg.get("device", "cuda") == "cuda"
    if bool(cfg.get("require_cuda", requested_cuda)) and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device("cuda" if requested_cuda and torch.cuda.is_available() else "cpu")
    print(f"runtime_device={device}")
    if device.type == "cuda":
        print(f"cuda_device={torch.cuda.get_device_name(device)}")
    text_cfg = cfg.get("text", {})
    node_text = encode_texts(
        dataset.node_texts,
        backend=str(text_cfg.get("backend", "hashing")),
        model_name=str(text_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
        cache_dir=str(text_cfg.get("cache_dir", "data/processed/text_cache")),
        dim=int(text_cfg.get("dim", 384)),
        device=str(device),
    )
    rel_text = encode_texts(
        dataset.relation_texts,
        backend=str(text_cfg.get("backend", "hashing")),
        model_name=str(text_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
        cache_dir=str(text_cfg.get("cache_dir", "data/processed/text_cache")),
        dim=int(text_cfg.get("dim", 384)),
        device=str(device),
    )
    node_text_t = torch.tensor(node_text, dtype=torch.float32, device=device)
    rel_text_t = torch.tensor(rel_text, dtype=torch.float32, device=device)
    eval_cfg = cfg.get("eval", {})
    tgb_val_rank_edges = eval_cfg.get("tgb_val_edges", cfg.get("train", {}).get("val_rank_edges", None))
    if tgb_val_rank_edges is not None:
        tgb_val_rank_edges = int(tgb_val_rank_edges)
    tgb_val_negative_sets = load_tgb_negative_sets(cfg, dataset, "val")

    partial_path = Path(out_dir) / "partial_search_results.jsonl"
    rows: list[dict[str, Any]] = []
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    if partial_path.exists():
        if not args.resume:
            raise SystemExit(f"{partial_path} already exists; pass --resume to preserve and continue it")
        for line in partial_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            completed[(str(row.get("candidate")), int(row.get("seed")))] = row
    best_score = -float("inf")
    best_candidate: dict[str, Any] | None = None
    best_passes_minima = False
    candidate_summaries: list[dict[str, Any]] = []
    seeds = [int(s) for s in search.get("seeds", cfg.get("seeds", [1, 2, 3]))]
    for candidate in candidates:
        cand_name = str(candidate.get("name", f"candidate_{len(rows)}"))
        cand_cfg = deep_update(cfg, {"model": candidate.get("model", {}), "train": candidate.get("train", {})})
        if strict_search:
            validate_full_formula_config(cand_cfg, context=f"search candidate {cand_name}")
        candidate_rows: list[dict[str, Any]] = []
        for seed in seeds:
            completed_row = completed.get((cand_name, seed))
            if completed_row is not None:
                value = float(completed_row.get(metric, float("nan")))
                print({"event": "resume_skip", "candidate": cand_name, "seed": seed, metric: value})
                candidate_rows.append(completed_row)
                continue
            set_seed(seed)
            model = make_model(cand_cfg, dataset, text_dim=node_text_t.size(1))
            train_cfg = make_train_config(cand_cfg)
            print(
                {
                    "event": "candidate_seed_start",
                    "candidate": cand_name,
                    "seed": seed,
                    "device": str(device),
                    "node_text_device": str(node_text_t.device),
                }
            )
            try:
                model, val_metrics = train_one_seed(
                    dataset,
                    copy.deepcopy(base_splits),
                    model,
                    node_text_t,
                    rel_text_t,
                    train_cfg,
                    device,
                    seed,
                    tgb_val_negative_sets=tgb_val_negative_sets,
                    tgb_val_rank_edges=tgb_val_rank_edges,
                )
                value = float(val_metrics.get(metric, float("nan")))
                row = {"candidate": cand_name, "seed": seed, metric: value, **val_metrics}
            except Exception as exc:
                value = float("nan")
                row = {
                    "candidate": cand_name,
                    "seed": seed,
                    metric: value,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            finally:
                try:
                    del model
                except UnboundLocalError:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            rows.append(row)
            candidate_rows.append(row)
            with open(partial_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rows[-1], ensure_ascii=False, sort_keys=True) + "\n")
            print(rows[-1])
        means = {key: _finite_mean(candidate_rows, key) for key in metric_names}
        passes = _passes_minima(means, metric_minima) if metric_minima else True
        mean_score = _candidate_score(means, metric, metric_weights, metric_minima)
        summary = {
            "candidate": cand_name,
            "score": mean_score,
            "passes_metric_minima": passes,
            "means": means,
            "seeds_completed": len(candidate_rows),
        }
        candidate_summaries.append(summary)
        print({"event": "candidate_summary", **summary}, flush=True)
        if mean_score > best_score:
            best_score = mean_score
            best_candidate = copy.deepcopy(candidate)
            best_passes_minima = passes

    if best_candidate is None:
        raise SystemExit("no finite validation score found")
    selected_cfg = deep_update(cfg, {"model": best_candidate.get("model", {}), "train": best_candidate.get("train", {})})
    selected_cfg.pop("search", None)
    save_json(
        {
            "environment": env_report(),
            "dataset": dataset.name,
            "scenario": {"name": scenario_name, **scenario_meta},
            "selection_metric": metric,
            "selection_metrics": metric_names,
            "metric_weights": metric_weights,
            "metric_minima": metric_minima,
            "formal_allowed": bool(best_passes_minima),
            "best_candidate": best_candidate,
            "best_candidate_passes_minima": bool(best_passes_minima),
            "best_validation_mean": best_score,
            "candidate_summaries": candidate_summaries,
            "rows": rows,
            "leakage_control": {
                "selection_uses": "validation metrics only",
                "test_usage": "not evaluated by this script",
                "temporal_split": "strict train < validation < test",
            },
        },
        Path(out_dir) / "search_results.json",
    )
    save_json(selected_cfg, Path(out_dir) / "selected_config.json")


if __name__ == "__main__":
    main()
