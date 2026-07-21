#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_ssptgfm import build_dataset
from ssptgfm.data import EdgeTensor, split_by_labels
from ssptgfm.experiment_splits import build_scenario
from ssptgfm.features import GraphHistoryIndex
from ssptgfm.metrics import binary_metrics, ranking_metrics
from ssptgfm.negative_sampling import KnownFacts, NegativeSampler, make_labels
from ssptgfm.training import TrainConfig, _time_grouped_batches
from ssptgfm.utils import ensure_dir, load_yaml, save_json


def _score_features(feats: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return feats.float() @ weights.to(feats.device).float()


def _history_eval_binary(
    num_nodes: int,
    positives: EdgeTensor,
    history_pool: EdgeTensor,
    sampler: NegativeSampler,
    num_neg: int,
    batch_size: int,
    weights: torch.Tensor,
) -> dict[str, float]:
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    positives = positives.sort_by_time()
    history_pool = history_pool.sort_by_time()
    hist_index = GraphHistoryIndex(num_nodes, history_pool.slice([]))
    cursor = 0
    for time_value, batches in _time_grouped_batches(positives, batch_size=batch_size, shuffle_within_time=False, seed=0):
        while cursor < len(history_pool) and float(history_pool.time[cursor]) < time_value:
            start = cursor
            cursor += 1
            while cursor < len(history_pool) and float(history_pool.time[cursor]) < time_value:
                cursor += 1
            hist_index.add_edges(history_pool.slice(slice(start, cursor)))
        hist = history_pool.slice(slice(0, cursor))
        for idx in batches:
            pos = positives.slice(idx.cpu())
            neg = sampler.sample(pos, num_neg_per_pos=num_neg, history=hist)
            batch = pos.concat(neg, sort=False)
            feats = hist_index.history_prior_features_for_edges(batch, "cpu", feature_dim=int(weights.numel()))
            scores = _score_features(feats, weights)
            labels = make_labels(len(pos), len(neg), device=torch.device("cpu"))
            labels_all.append(labels.numpy())
            scores_all.append(scores.numpy())
    return binary_metrics(np.concatenate(labels_all), np.concatenate(scores_all))


def _history_eval_ranking(
    num_nodes: int,
    positives: EdgeTensor,
    history_pool: EdgeTensor,
    known_facts: KnownFacts,
    batch_size: int,
    weights: torch.Tensor,
    filter_scope: str,
    max_eval_edges: int | None,
) -> dict[str, float]:
    ranks: list[float] = []
    positives = positives.sort_by_time()
    history_pool = history_pool.sort_by_time()
    hist_index = GraphHistoryIndex(num_nodes, history_pool.slice([]))
    cursor = 0
    eval_count = len(positives) if max_eval_edges is None else min(len(positives), max_eval_edges)
    processed = 0
    for time_value, groups in _time_grouped_batches(positives, batch_size=1, shuffle_within_time=False, seed=0):
        while cursor < len(history_pool) and float(history_pool.time[cursor]) < time_value:
            start = cursor
            cursor += 1
            while cursor < len(history_pool) and float(history_pool.time[cursor]) < time_value:
                cursor += 1
            hist_index.add_edges(history_pool.slice(slice(start, cursor)))
        for group in groups:
            if processed >= eval_count:
                break
            pos = positives.slice(group.cpu())
            s = int(pos.src[0])
            d = int(pos.dst[0])
            r = int(pos.rel[0])
            tm = float(pos.time[0])
            for corrupt_head in (False, True):
                keep, true_index = known_facts.filtered_nodes_for_query(
                    s,
                    r,
                    d,
                    tm,
                    corrupt_head=corrupt_head,
                    num_nodes=num_nodes,
                    scope=filter_scope,
                )
                if true_index is None:
                    continue
                cand_nodes = torch.nonzero(keep, as_tuple=False).view(-1).tolist()
                chunks: list[torch.Tensor] = []
                for start in range(0, len(cand_nodes), batch_size):
                    chunk_nodes = cand_nodes[start : start + batch_size]
                    if corrupt_head:
                        chunk = [(node, d, r, tm) for node in chunk_nodes]
                    else:
                        chunk = [(s, node, r, tm) for node in chunk_nodes]
                    feats = hist_index.history_prior_features_for_candidates(chunk, "cpu", feature_dim=int(weights.numel()))
                    chunks.append(_score_features(feats, weights).detach().cpu())
                all_scores = torch.cat(chunks).numpy()
                true_score = all_scores[true_index]
                rank = 1.0 + float(np.sum(all_scores > true_score)) + 0.5 * float(np.sum(all_scores == true_score) - 1)
                ranks.append(rank)
            processed += 1
        if processed >= eval_count:
            break
    return ranking_metrics(ranks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast causal history-prior validation probe.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--rank-edges", type=int, default=100)
    parser.add_argument("--num-neg", type=int, default=25)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    dataset = build_dataset(cfg)
    dataset.validate()
    split_cfg = cfg.get("split", {})
    if split_cfg.get("mode", "time") == "labels":
        splits = split_by_labels(dataset)
    else:
        dataset, scenario = build_scenario(
            dataset,
            cfg.get("scenario", {"name": "standard"}),
            val_ratio=float(split_cfg.get("val_ratio", 0.15)),
            test_ratio=float(split_cfg.get("test_ratio", 0.15)),
            seed=int(cfg.get("data", {}).get("seed", 1)),
        )
        splits = scenario.splits
    splits.assert_no_temporal_leakage()
    train_cfg = TrainConfig(
        batch_size=int(cfg.get("train", {}).get("batch_size", 512)),
        negative_mode_eval=str(cfg.get("train", {}).get("negative_mode_eval", "filtered")),
        filter_scope=str(cfg.get("train", {}).get("filter_scope", "exact")),
    )
    if args.split == "val":
        positives = splits.val
        history_pool = splits.train
        known = KnownFacts.from_edges(splits.train.concat(splits.val, sort=False))
    else:
        positives = splits.test
        history_pool = splits.train.concat(splits.val, sort=False)
        known = KnownFacts.from_edges(dataset.edges)
    sampler = NegativeSampler(
        dataset.num_nodes,
        known,
        splits.train,
        mode=train_cfg.negative_mode_eval,
        filter_scope=train_cfg.filter_scope,
        seed=20260615,
    )
    weight_bank = {
        "exact": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "pair": [0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 0.0],
        "rel_cond": [0.5, 0.2, 0.1, 0.4, 0.4, 0.1, 1.0, 1.0, 0.5, 0.4, 0.2, 0.2],
        "rank_heavy": [1.5, 0.7, 0.3, 0.2, 0.2, 0.0, 1.5, 1.5, 1.0, 0.8, 0.3, 0.4],
        "copy_heavy": [2.0, 0.5, 0.2, 0.1, 0.1, 0.0, 2.0, 2.0, 1.0, 1.0, 0.3, 0.2],
    }
    rows = []
    for name, weights in weight_bank.items():
        w = torch.tensor(weights, dtype=torch.float32)
        binary = _history_eval_binary(
            dataset.num_nodes,
            positives,
            history_pool,
            sampler,
            num_neg=args.num_neg,
            batch_size=train_cfg.batch_size,
            weights=w,
        )
        ranking = _history_eval_ranking(
            dataset.num_nodes,
            positives,
            history_pool,
            known,
            batch_size=train_cfg.batch_size,
            weights=w,
            filter_scope=train_cfg.filter_scope,
            max_eval_edges=args.rank_edges,
        )
        row = {"name": name, **{f"{args.split}_{k}": v for k, v in binary.items()}, **{f"{args.split}_{k}": v for k, v in ranking.items()}}
        rows.append(row)
        print(row, flush=True)
    out_dir = ensure_dir(args.output)
    save_json(
        {
            "config": args.config,
            "split": args.split,
            "rank_edges": args.rank_edges,
            "num_neg": args.num_neg,
            "rows": rows,
            "leakage_control": "history_pool is train for validation and train+validation for test, with strict per-query causal cutoff",
        },
        Path(out_dir) / f"history_prior_{args.split}.json",
    )


if __name__ == "__main__":
    main()
