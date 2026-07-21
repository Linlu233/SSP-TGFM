#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_ssptgfm import build_dataset, make_train_config
from ssptgfm.data import EdgeTensor, split_by_labels, subsample_train_edges
from ssptgfm.experiment_splits import build_scenario, exact_k_shot_train
from ssptgfm.metrics import binary_metrics, ranking_metrics
from ssptgfm.negative_sampling import KnownFacts, NegativeSampler, make_labels
from ssptgfm.training import _time_grouped_batches
from ssptgfm.utils import env_report, ensure_dir, load_yaml, save_json, set_seed


DEFAULT_BASELINES = [
    "edgebank_exact",
    "edgebank_time_decay",
    "historical_frequency",
    "relational_popularity",
]


class TemporalHistoryState:
    """Incremental causal history used by fast non-parametric temporal baselines."""

    def __init__(self, num_nodes: int, num_relations: int, history_pool: EdgeTensor) -> None:
        self.num_nodes = int(num_nodes)
        self.num_relations = int(num_relations)
        self.history_pool = history_pool.sort_by_time()
        self.cursor = 0
        self.triple_counts: dict[tuple[int, int, int], int] = defaultdict(int)
        self.triple_last: dict[tuple[int, int, int], float] = {}
        self.pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        self.pair_last: dict[tuple[int, int], float] = {}
        self.sr_tail: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.rd_head: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.out_pair: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.in_pair: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.rel_tail_counts = np.zeros((self.num_relations, self.num_nodes), dtype=np.float32)
        self.rel_head_counts = np.zeros((self.num_relations, self.num_nodes), dtype=np.float32)
        self.out_degree = np.zeros(self.num_nodes, dtype=np.float32)
        self.in_degree = np.zeros(self.num_nodes, dtype=np.float32)
        self.rel_counts = np.zeros(self.num_relations, dtype=np.float32)

    def advance(self, time_value: float) -> None:
        src = self.history_pool.src
        dst = self.history_pool.dst
        rel = self.history_pool.rel
        tim = self.history_pool.time
        n = len(self.history_pool)
        while self.cursor < n and float(tim[self.cursor]) < time_value:
            s = int(src[self.cursor])
            d = int(dst[self.cursor])
            r = int(rel[self.cursor])
            t = float(tim[self.cursor])
            self._add(s, r, d, t)
            self.cursor += 1

    def _add(self, s: int, r: int, d: int, t: float) -> None:
        triple = (s, r, d)
        pair = (s, d)
        self.triple_counts[triple] += 1
        self.triple_last[triple] = max(t, self.triple_last.get(triple, -float("inf")))
        self.pair_counts[pair] += 1
        self.pair_last[pair] = max(t, self.pair_last.get(pair, -float("inf")))
        self.sr_tail[(s, r)][d] += 1
        self.rd_head[(r, d)][s] += 1
        self.out_pair[s][d] += 1
        self.in_pair[d][s] += 1
        self.rel_tail_counts[r, d] += 1.0
        self.rel_head_counts[r, s] += 1.0
        self.out_degree[s] += 1.0
        self.in_degree[d] += 1.0
        self.rel_counts[r] += 1.0

    def score_edges(self, method: str, edges: EdgeTensor, params: dict[str, float]) -> np.ndarray:
        scores = np.zeros(len(edges), dtype=np.float32)
        tau = max(float(params.get("tau", 1.0)), 1e-6)
        for idx, (s_t, d_t, r_t, time_t) in enumerate(zip(edges.src, edges.dst, edges.rel, edges.time)):
            s = int(s_t)
            d = int(d_t)
            r = int(r_t)
            t = float(time_t)
            scores[idx] = self._score_one(method, s, r, d, t, tau)
        return scores

    def _score_one(self, method: str, s: int, r: int, d: int, t: float, tau: float) -> float:
        triple = (s, r, d)
        if method == "edgebank_exact":
            return 1.0 if self.triple_counts.get(triple, 0) > 0 else 0.0
        if method == "edgebank_time_decay":
            last = self.triple_last.get(triple)
            if last is None:
                return 0.0
            return math.exp(-max(0.0, t - last) / tau)
        if method == "historical_frequency":
            return (
                2.0 * math.log1p(self.triple_counts.get(triple, 0))
                + 0.5 * math.log1p(self.pair_counts.get((s, d), 0))
                + 0.05 * math.log1p(float(self.rel_tail_counts[r, d]))
                + 0.05 * math.log1p(float(self.rel_head_counts[r, s]))
                + 0.01 * math.log1p(float(self.in_degree[d] + self.out_degree[s]))
            )
        if method == "relational_popularity":
            sr_count = self.sr_tail.get((s, r), {}).get(d, 0)
            rd_count = self.rd_head.get((r, d), {}).get(s, 0)
            return (
                math.log1p(sr_count)
                + math.log1p(rd_count)
                + 0.1 * math.log1p(float(self.rel_tail_counts[r, d]))
                + 0.1 * math.log1p(float(self.rel_head_counts[r, s]))
            )
        raise ValueError(f"unknown fast baseline: {method}")

    def score_candidates(
        self,
        method: str,
        src: int,
        rel: int,
        dst: int,
        time_value: float,
        corrupt_head: bool,
        params: dict[str, float],
    ) -> np.ndarray:
        tau = max(float(params.get("tau", 1.0)), 1e-6)
        scores = np.zeros(self.num_nodes, dtype=np.float32)
        if method == "edgebank_exact":
            if corrupt_head:
                for cand in self.rd_head.get((rel, dst), {}):
                    scores[cand] = 1.0
            else:
                for cand in self.sr_tail.get((src, rel), {}):
                    scores[cand] = 1.0
            return scores
        if method == "edgebank_time_decay":
            if corrupt_head:
                for cand in self.rd_head.get((rel, dst), {}):
                    last = self.triple_last.get((cand, rel, dst))
                    if last is not None:
                        scores[cand] = math.exp(-max(0.0, time_value - last) / tau)
            else:
                for cand in self.sr_tail.get((src, rel), {}):
                    last = self.triple_last.get((src, rel, cand))
                    if last is not None:
                        scores[cand] = math.exp(-max(0.0, time_value - last) / tau)
            return scores
        if method == "historical_frequency":
            if corrupt_head:
                scores += 0.05 * np.log1p(self.rel_head_counts[rel])
                scores += 0.01 * np.log1p(self.out_degree)
                for cand, count in self.rd_head.get((rel, dst), {}).items():
                    scores[cand] += 2.0 * math.log1p(count)
                for cand, count in self.in_pair.get(dst, {}).items():
                    scores[cand] += 0.5 * math.log1p(count)
            else:
                scores += 0.05 * np.log1p(self.rel_tail_counts[rel])
                scores += 0.01 * np.log1p(self.in_degree)
                for cand, count in self.sr_tail.get((src, rel), {}).items():
                    scores[cand] += 2.0 * math.log1p(count)
                for cand, count in self.out_pair.get(src, {}).items():
                    scores[cand] += 0.5 * math.log1p(count)
            return scores
        if method == "relational_popularity":
            if corrupt_head:
                scores += 0.1 * np.log1p(self.rel_head_counts[rel])
                for cand, count in self.rd_head.get((rel, dst), {}).items():
                    scores[cand] += math.log1p(count)
            else:
                scores += 0.1 * np.log1p(self.rel_tail_counts[rel])
                for cand, count in self.sr_tail.get((src, rel), {}).items():
                    scores[cand] += math.log1p(count)
            return scores
        raise ValueError(f"unknown fast baseline: {method}")


def tau_grid_from_splits(train: EdgeTensor, val: EdgeTensor) -> list[float]:
    times = np.concatenate([train.time.numpy(), val.time.numpy()])
    if times.size == 0:
        return [1.0]
    span = max(float(times.max() - times.min()), 1.0)
    raw = [1.0, span / 100.0, span / 20.0, span / 5.0, span, span * 5.0]
    return sorted({float(max(x, 1e-6)) for x in raw})


def evaluate_binary(
    dataset,
    positives: EdgeTensor,
    history_pool: EdgeTensor,
    method: str,
    params: dict[str, float],
    sampler: NegativeSampler,
    config,
) -> dict[str, float]:
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    positives = positives.sort_by_time()
    state = TemporalHistoryState(dataset.num_nodes, dataset.num_relations, history_pool)
    for time_value, batches in _time_grouped_batches(positives, batch_size=config.batch_size, shuffle_within_time=False, seed=0):
        state.advance(float(time_value))
        hist = history_pool.before(float(time_value), strict=True)
        for idx in batches:
            pos = positives.slice(idx.cpu())
            neg = sampler.sample(pos, config.num_neg_eval, hist)
            batch = pos.concat(neg, sort=False)
            labels = make_labels(len(pos), len(neg), "cpu").numpy()
            scores = state.score_edges(method, batch, params)
            labels_all.append(labels)
            scores_all.append(scores)
    return binary_metrics(np.concatenate(labels_all), np.concatenate(scores_all))


def evaluate_ranking(
    dataset,
    positives: EdgeTensor,
    history_pool: EdgeTensor,
    known_facts: KnownFacts,
    method: str,
    params: dict[str, float],
    filter_scope: str,
    max_eval_edges: int | None,
) -> dict[str, float]:
    positives = positives.sort_by_time()
    history_pool = history_pool.sort_by_time()
    state = TemporalHistoryState(dataset.num_nodes, dataset.num_relations, history_pool)
    ranks: list[float] = []
    eval_count = len(positives) if max_eval_edges is None else min(len(positives), max_eval_edges)
    for idx in range(eval_count):
        pos = positives.slice([idx])
        s = int(pos.src[0])
        d = int(pos.dst[0])
        r = int(pos.rel[0])
        tm = float(pos.time[0])
        state.advance(tm)
        for corrupt_head in (False, True):
            keep, true_index = known_facts.filtered_nodes_for_query(
                s,
                r,
                d,
                tm,
                corrupt_head=corrupt_head,
                num_nodes=dataset.num_nodes,
                scope=filter_scope,
            )
            if true_index is None:
                continue
            keep_np = keep.numpy().astype(bool)
            scores = state.score_candidates(method, s, r, d, tm, corrupt_head, params)
            cand_scores = scores[keep_np]
            true_score = scores[true_index]
            ranks.append(1.0 + float(np.sum(cand_scores > true_score)) + 0.5 * float(np.sum(cand_scores == true_score) - 1))
    return ranking_metrics(ranks)


def select_params(
    dataset,
    splits,
    method: str,
    seed: int,
    config,
) -> tuple[dict[str, float], dict[str, float]]:
    known_val = KnownFacts.from_edges(splits.train.concat(splits.val, sort=False))
    sampler = NegativeSampler(
        dataset.num_nodes,
        known_val,
        splits.train,
        mode=config.negative_mode_eval,
        filter_scope=config.filter_scope,
        seed=seed + 1717,
    )
    if method != "edgebank_time_decay":
        params: dict[str, float] = {}
        val = evaluate_binary(dataset, splits.val, splits.train, method, params, sampler, config)
        return params, {f"val_{k}": v for k, v in val.items()}
    best_params: dict[str, float] = {}
    best_metrics: dict[str, float] = {}
    best_score = -float("inf")
    for tau in tau_grid_from_splits(splits.train, splits.val):
        params = {"tau": float(tau)}
        val = evaluate_binary(dataset, splits.val, splits.train, method, params, sampler, config)
        score = float(val.get("auc", float("nan")))
        tie = float(val.get("ap", float("nan")))
        composite = score if np.isfinite(score) else -float("inf")
        composite += 1e-6 * (tie if np.isfinite(tie) else 0.0)
        print({"event": "fast_baseline_candidate", "method": method, "seed": seed, "tau": tau, **{f"val_{k}": v for k, v in val.items()}}, flush=True)
        if composite > best_score:
            best_score = composite
            best_params = params
            best_metrics = val
    return best_params, {f"val_{k}": v for k, v in best_metrics.items()}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast strong historical baselines with validation-only selection.")
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
        scenario_meta = {}
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
    config = make_train_config(cfg)
    filtered_rank_edges = cfg.get("eval", {}).get("filtered_rank_edges", 200)
    if filtered_rank_edges is not None:
        filtered_rank_edges = int(filtered_rank_edges)
    baselines = [name.strip() for name in args.baselines.split(",") if name.strip()]
    seeds = [int(s) for s in cfg.get("seeds", [1, 2, 3, 4, 5])]
    ratios = [float(x) for x in cfg.get("few_shot_ratios", [1.0])]
    partial_path = Path(out_dir) / "partial_results.jsonl"
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
                key = (method, "strong_baseline", ratio, seed)
                if key in done:
                    print({"event": "resume_skip", "dataset": dataset.name, "method": method, "seed": seed}, flush=True)
                    continue
                start = perf_counter()
                print({"event": "fast_baseline_start", "dataset": dataset.name, "method": method, "seed": seed}, flush=True)
                params, val_metrics = select_params(dataset, splits, method, seed, config)
                test_binary = evaluate_binary(dataset, splits.test, history_pool, method, params, test_sampler, config)
                test_rank = evaluate_ranking(
                    dataset,
                    splits.test,
                    history_pool,
                    known_all,
                    method,
                    params,
                    filter_scope=config.filter_scope,
                    max_eval_edges=filtered_rank_edges,
                )
                row = {
                    "dataset": dataset.name,
                    "method": method,
                    "scenario": scenario_name,
                    "ablation": "strong_baseline",
                    "few_shot_ratio": ratio,
                    "seed": seed,
                    "train_edges": len(splits.train),
                    "val_edges": len(splits.val),
                    "test_edges": len(splits.test),
                    "params_trainable": 0,
                    "params_total": 0,
                    "selected_params": params,
                    "selection_metric": "val_auc",
                    "train_eval_time_sec": perf_counter() - start,
                    **val_metrics,
                    **{f"test_{k}": v for k, v in test_binary.items()},
                    **{f"test_{k}": v for k, v in test_rank.items()},
                }
                rows.append(row)
                done.add(key)
                with partial_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                print(row, flush=True)

    save_json(
        {
            "config": cfg,
            "environment": env_report(),
            "rows": rows,
            "scenario": {"name": scenario_name, **scenario_meta},
            "leakage_control": {
                "hyperparameter_selection": "validation binary AUC only",
                "history_for_validation": "train edges only, causally restricted by query time",
                "history_for_test": "train+validation edges only, causally restricted by query time",
                "filtered_eval": True,
                "test_labels_not_used_for_selection": True,
            },
        },
        Path(out_dir) / "all_results.json",
    )


if __name__ == "__main__":
    main()
