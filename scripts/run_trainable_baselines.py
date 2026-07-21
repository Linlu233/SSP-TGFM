#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_ssptgfm import build_dataset, make_train_config
from ssptgfm.baseline_training import (
    build_baseline,
    evaluate_baseline_binary,
    evaluate_baseline_ranking,
    train_baseline_one_seed,
)
from ssptgfm.data import split_by_labels, subsample_train_edges
from ssptgfm.experiment_splits import build_scenario, exact_k_shot_train
from ssptgfm.negative_sampling import KnownFacts, NegativeSampler
from ssptgfm.text import encode_texts
from ssptgfm.utils import Timer, count_parameters, cuda_memory_mb, env_report, ensure_dir, load_yaml, save_json, set_seed


DEFAULT_BASELINES = ["distmult", "complex", "temporal_distmult"]


def completed_rows(partial_path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str, float, int]]]:
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, str, float, int]] = set()
    if not partial_path.exists():
        return rows, keys
    for line in partial_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        keys.add((str(row.get("method")), str(row.get("ablation")), float(row.get("few_shot_ratio", 1.0)), int(row.get("seed"))))
    return rows, keys


def save_payload(
    out_dir: Path,
    cfg: dict,
    rows: list[dict[str, Any]],
    scenario_name: str,
    scenario_meta: dict[str, Any],
    val_start_time: float,
    test_start_time: float,
) -> None:
    save_json(
        {
            "config": cfg,
            "environment": env_report(),
            "rows": rows,
            "temporal_split": {
                "val_start_time": val_start_time,
                "test_start_time": test_start_time,
            },
            "scenario": {"name": scenario_name, **scenario_meta},
            "leakage_control": {
                "hyperparameter_selection": "training config only; early stopping uses validation binary AUC",
                "history_for_train_batch": "edges with time strictly before batch min time",
                "history_for_validation": "train edges only, causally restricted by query time",
                "history_for_test": "train+validation edges only, causally restricted by query time",
                "filtered_eval": True,
                "test_labels_not_used_for_selection": True,
            },
        },
        out_dir / "all_results.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainable KG/TKG baselines under the SSP-TGFM protocol.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baselines", default=",".join(DEFAULT_BASELINES))
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    if args.seeds:
        cfg["seeds"] = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    out_dir = ensure_dir(args.output)

    dataset = build_dataset(cfg)
    dataset.validate()
    split_cfg = cfg.get("split", {})
    if split_cfg.get("mode", "time") == "labels":
        base_splits = split_by_labels(dataset)
        scenario_name = "label_split"
        scenario_meta: dict[str, Any] = {}
    else:
        dataset, scenario = build_scenario(
            dataset,
            cfg.get("scenario", {"name": "standard"}),
            val_ratio=float(split_cfg.get("val_ratio", 0.15)),
            test_ratio=float(split_cfg.get("test_ratio", 0.15)),
            seed=int(cfg.get("scenario", {}).get("seed", cfg.get("data", {}).get("seed", 1))),
        )
        base_splits = scenario.splits
        scenario_name = scenario.name
        scenario_meta = scenario.metadata
    base_splits.assert_no_temporal_leakage()

    requested_cuda = cfg.get("device", "cuda") == "cuda"
    if bool(cfg.get("require_cuda", requested_cuda)) and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device("cuda" if requested_cuda and torch.cuda.is_available() else "cpu")
    print({"event": "trainable_baselines_runtime", "device": str(device)}, flush=True)
    if device.type == "cuda":
        print({"event": "cuda_device", "name": torch.cuda.get_device_name(device)}, flush=True)

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

    config = make_train_config(cfg, cfg.get("baseline_train", {}))
    filtered_rank_edges = cfg.get("eval", {}).get("filtered_rank_edges", 200)
    if filtered_rank_edges is not None:
        filtered_rank_edges = int(filtered_rank_edges)
    baselines = [name.strip() for name in args.baselines.split(",") if name.strip()]
    seeds = [int(s) for s in cfg.get("seeds", [1, 2, 3, 4, 5])]
    ratios = [float(x) for x in cfg.get("few_shot_ratios", [1.0])]
    hidden_dim = int(cfg.get("baseline", {}).get("hidden_dim", cfg.get("model", {}).get("hidden_dim", 128)))
    partial_path = out_dir / "partial_results.jsonl"
    if partial_path.exists() and not args.resume:
        partial_path.unlink()
    rows, done = completed_rows(partial_path) if args.resume else ([], set())

    for ratio in ratios:
        for seed in seeds:
            set_seed(seed)
            splits = copy.deepcopy(base_splits)
            splits.train = subsample_train_edges(splits.train, ratio)
            if "k_shot" in cfg:
                k_cfg = cfg["k_shot"]
                splits.train = exact_k_shot_train(splits.train, int(k_cfg.get("k", 5)), by=str(k_cfg.get("by", "relation")))
            splits.assert_no_temporal_leakage()
            history_pool = splits.train.concat(splits.val, sort=False)
            known_all = KnownFacts.from_edges(dataset.edges)
            test_sampler = NegativeSampler(
                dataset.num_nodes,
                known_all,
                splits.train,
                mode=config.negative_mode_eval,
                filter_scope=config.filter_scope,
                seed=seed + 909,
            )
            for method in baselines:
                key = (method, "trainable_baseline", ratio, seed)
                if key in done:
                    print({"event": "resume_skip", "dataset": dataset.name, "method": method, "seed": seed}, flush=True)
                    continue
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                print({"event": "trainable_baseline_start", "dataset": dataset.name, "method": method, "seed": seed}, flush=True)
                model = build_baseline(method, dataset, text_dim=node_text_t.size(1), hidden_dim=hidden_dim)
                start = perf_counter()
                with Timer() as timer:
                    model, val_metrics = train_baseline_one_seed(
                        dataset,
                        splits,
                        model,
                        node_text_t,
                        rel_text_t,
                        config,
                        device,
                        seed,
                    )
                    test_binary = evaluate_baseline_binary(
                        dataset,
                        splits.test,
                        history_pool,
                        model,
                        node_text_t,
                        rel_text_t,
                        test_sampler,
                        config,
                        device,
                    )
                    test_rank = evaluate_baseline_ranking(
                        dataset,
                        splits.test,
                        history_pool,
                        known_all,
                        model,
                        node_text_t,
                        rel_text_t,
                        config,
                        filter_scope=config.filter_scope,
                        max_eval_edges=filtered_rank_edges,
                    )
                row = {
                    "dataset": dataset.name,
                    "method": method,
                    "scenario": scenario_name,
                    "ablation": "trainable_baseline",
                    "few_shot_ratio": ratio,
                    "seed": seed,
                    "train_edges": len(splits.train),
                    "val_edges": len(splits.val),
                    "test_edges": len(splits.test),
                    "params_trainable": count_parameters(model, trainable_only=True),
                    "params_total": count_parameters(model, trainable_only=False),
                    "selected_params": {
                        "hidden_dim": hidden_dim,
                        "lr": config.lr,
                        "weight_decay": config.weight_decay,
                        "epochs": config.epochs,
                        "patience": config.patience,
                    },
                    "selection_metric": "val_auc",
                    "train_eval_time_sec": timer.elapsed,
                    "wall_time_sec": perf_counter() - start,
                    **cuda_memory_mb(),
                    **val_metrics,
                    **{f"test_{k}": v for k, v in test_binary.items()},
                    **{f"test_{k}": v for k, v in test_rank.items()},
                }
                rows.append(row)
                done.add(key)
                with partial_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                save_payload(out_dir, cfg, rows, scenario_name, scenario_meta, base_splits.val_start_time, base_splits.test_start_time)
                print(row, flush=True)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    save_payload(out_dir, cfg, rows, scenario_name, scenario_meta, base_splits.val_start_time, base_splits.test_start_time)


if __name__ == "__main__":
    main()
