#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_ssptgfm import validate_full_formula_config


METRICS = ("auc", "ap", "mrr", "hits@10", "ndcg")
TARGET_WINS = 3
SEEDS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    base_search_config: Path
    base_formal_config: Path
    baseline_paths: tuple[Path, ...]
    existing_result_paths: tuple[Path, ...]
    max_parallel_seeds: int
    search_epochs: int
    formal_epochs: int
    search_val_rank_edges: int
    formal_eval_rank_edges: int
    neg_values: tuple[int, ...]
    batch_size: int | None = None


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        label="icews14",
        base_search_config=Path("configs/optimization_attempts/ssptgfm_search_icews14_formula_rank.yaml"),
        base_formal_config=Path("configs/optimization_attempts/ssptgfm_icews14_formula_rank_best.yaml"),
        baseline_paths=(
            Path("results/ssptgfm_icews14_strict_v2_best/all_results.json"),
            Path("results/fast_strong_baselines_strict_v1/icews14/all_results.json"),
        ),
        existing_result_paths=(Path("results/optimization_attempts/ssptgfm_icews14_formula_rank_best/all_results.json"),),
        max_parallel_seeds=2,
        search_epochs=14,
        formal_epochs=24,
        search_val_rank_edges=100,
        formal_eval_rank_edges=250,
        neg_values=(5, 10, 15),
        batch_size=512,
    ),
    DatasetSpec(
        label="icews05_15",
        base_search_config=Path("configs/ssptgfm_search_icews05_15_all_losses.yaml"),
        base_formal_config=Path("configs/ssptgfm_icews05_15_best.yaml"),
        baseline_paths=(
            Path("results/ssptgfm_icews05_15_strict_v2_best/all_results.json"),
            Path("results/fast_strong_baselines_strict_v1/icews05_15/all_results.json"),
        ),
        existing_result_paths=(Path("results/ssptgfm_icews05_15_best/all_results.json"),),
        max_parallel_seeds=1,
        search_epochs=10,
        formal_epochs=20,
        search_val_rank_edges=75,
        formal_eval_rank_edges=150,
        neg_values=(1, 2, 5),
        batch_size=4096,
    ),
    DatasetSpec(
        label="tgb_smallpedia",
        base_search_config=Path("configs/ssptgfm_search_tgb_smallpedia_all_losses.yaml"),
        base_formal_config=Path("configs/ssptgfm_tgb_smallpedia_best.yaml"),
        baseline_paths=(
            Path("results/ssptgfm_tgb_smallpedia_strict_v2_best/all_results.json"),
            Path("results/fast_strong_baselines_strict_v1/tgb_smallpedia/all_results.json"),
        ),
        existing_result_paths=(Path("results/ssptgfm_tgb_smallpedia_best/all_results.json"),),
        max_parallel_seeds=2,
        search_epochs=14,
        formal_epochs=24,
        search_val_rank_edges=100,
        formal_eval_rank_edges=250,
        neg_values=(2, 5, 10),
        batch_size=512,
    ),
    DatasetSpec(
        label="tgb_yago",
        base_search_config=Path("configs/optimization_attempts/ssptgfm_search_tgb_yago_formula_rank.yaml"),
        base_formal_config=Path("configs/optimization_attempts/ssptgfm_tgb_yago_formula_rank_best.yaml"),
        baseline_paths=(
            Path("results/ssptgfm_tgb_yago_strict_v2_best/all_results.json"),
            Path("results/fast_strong_baselines_strict_v1/tgb_yago/all_results.json"),
        ),
        existing_result_paths=(Path("results/optimization_attempts/ssptgfm_tgb_yago_formula_rank_best/all_results.json"),),
        max_parallel_seeds=1,
        search_epochs=14,
        formal_epochs=20,
        search_val_rank_edges=75,
        formal_eval_rank_edges=150,
        neg_values=(5, 8, 10),
        batch_size=4096,
    ),
    DatasetSpec(
        label="yago15k_temporal",
        base_search_config=Path("configs/ssptgfm_search_yago15k_temporal_all_losses.yaml"),
        base_formal_config=Path("configs/ssptgfm_yago15k_temporal_best.yaml"),
        baseline_paths=(
            Path("results/ssptgfm_yago15k_temporal_strict_v2_best/all_results.json"),
            Path("results/fast_strong_baselines_strict_v1/yago15k_temporal/all_results.json"),
        ),
        existing_result_paths=(Path("results/ssptgfm_yago15k_temporal_best/all_results.json"),),
        max_parallel_seeds=1,
        search_epochs=14,
        formal_epochs=24,
        search_val_rank_edges=150,
        formal_eval_rank_edges=300,
        neg_values=(2, 5, 10),
        batch_size=512,
    ),
)


BANKS: tuple[dict[str, Any], ...] = (
    {
        "name": "balanced_rank_v1",
        "weights": {"val_mrr": 0.4, "val_hits@10": 0.25, "val_ap": 0.2, "val_auc": 0.1, "val_ndcg": 0.05},
        "presets": (
            (128, 16, 16, 4, 4, 1, 7e-4, 0.005, 1e-4, 0.005, 0.001, 0.2),
            (160, 24, 24, 6, 4, 1, 5e-4, 0.005, 1e-4, 0.005, 0.001, 0.2),
            (192, 24, 24, 6, 8, 1, 5e-4, 0.002, 1e-4, 0.010, 0.001, 0.2),
            (192, 32, 32, 8, 8, 1, 3e-4, 0.002, 5e-5, 0.010, 0.001, 0.15),
        ),
    },
    {
        "name": "ranking_focus_v1",
        "weights": {"val_mrr": 0.55, "val_hits@10": 0.25, "val_ndcg": 0.1, "val_ap": 0.05, "val_auc": 0.05},
        "presets": (
            (128, 24, 24, 4, 4, 1, 7e-4, 0.001, 5e-5, 0.010, 0.001, 0.15),
            (160, 24, 24, 6, 4, 1, 7e-4, 0.001, 5e-5, 0.010, 0.001, 0.15),
            (192, 32, 32, 8, 8, 1, 5e-4, 0.001, 5e-5, 0.010, 0.002, 0.10),
            (192, 32, 32, 8, 8, 2, 3e-4, 0.001, 5e-5, 0.010, 0.002, 0.10),
        ),
    },
    {
        "name": "ap_auc_focus_v1",
        "weights": {"val_ap": 0.4, "val_auc": 0.25, "val_mrr": 0.2, "val_hits@10": 0.1, "val_ndcg": 0.05},
        "presets": (
            (128, 16, 16, 4, 4, 1, 1e-3, 0.010, 1e-4, 0.005, 0.001, 0.2),
            (160, 24, 24, 6, 4, 1, 7e-4, 0.010, 1e-4, 0.005, 0.001, 0.2),
            (192, 24, 24, 6, 8, 1, 5e-4, 0.005, 1e-4, 0.005, 0.001, 0.2),
            (192, 32, 32, 8, 8, 1, 3e-4, 0.005, 5e-5, 0.010, 0.001, 0.15),
        ),
    },
    {
        "name": "history_prior_v1",
        "weights": {"val_mrr": 0.35, "val_hits@10": 0.25, "val_ap": 0.25, "val_auc": 0.1, "val_ndcg": 0.05},
        "history_prior": True,
        "history_scales": (0.1, 0.3, 0.7),
        "presets": (
            (128, 16, 16, 4, 4, 1, 7e-4, 0.005, 1e-4, 0.005, 0.001, 0.2),
            (160, 24, 24, 6, 4, 1, 7e-4, 0.005, 1e-4, 0.005, 0.001, 0.2),
            (192, 24, 24, 6, 8, 1, 5e-4, 0.002, 5e-5, 0.010, 0.001, 0.15),
        ),
    },
    {
        "name": "history_rank_v1",
        "weights": {"val_mrr": 0.55, "val_hits@10": 0.25, "val_ndcg": 0.1, "val_ap": 0.05, "val_auc": 0.05},
        "history_prior": True,
        "history_scales": (0.3, 0.7, 1.2),
        "presets": (
            (128, 24, 24, 4, 4, 1, 7e-4, 0.001, 5e-5, 0.010, 0.001, 0.15),
            (160, 24, 24, 6, 4, 1, 7e-4, 0.001, 5e-5, 0.010, 0.001, 0.15),
            (192, 32, 32, 8, 8, 1, 5e-4, 0.001, 5e-5, 0.010, 0.002, 0.10),
        ),
    },
    {
        "name": "history_ap_auc_v1",
        "weights": {"val_ap": 0.4, "val_auc": 0.25, "val_mrr": 0.2, "val_hits@10": 0.1, "val_ndcg": 0.05},
        "history_prior": True,
        "history_scales": (0.1, 0.3, 0.7),
        "presets": (
            (128, 16, 16, 4, 4, 1, 1e-3, 0.010, 1e-4, 0.005, 0.001, 0.2),
            (160, 24, 24, 6, 4, 1, 7e-4, 0.010, 1e-4, 0.005, 0.001, 0.2),
            (192, 24, 24, 6, 8, 1, 5e-4, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "history_prior_probe",
        "weights": {"val_mrr": 0.35, "val_hits@10": 0.25, "val_ap": 0.25, "val_auc": 0.1, "val_ndcg": 0.05},
        "history_prior": True,
        "history_scales": (0.3, 0.7, 1.2),
        "search_epochs": 3,
        "patience": 2,
        "val_rank_edges": 10,
        "presets": (
            (128, 16, 16, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
            (160, 24, 24, 6, 4, 1, 7e-4, 0.002, 5e-5, 0.010, 0.001, 0.15),
        ),
    },
    {
        "name": "history_rank_probe",
        "weights": {"val_mrr": 0.55, "val_hits@10": 0.25, "val_ndcg": 0.1, "val_ap": 0.05, "val_auc": 0.05},
        "history_prior": True,
        "history_scales": (0.7, 1.2, 2.0),
        "search_epochs": 3,
        "patience": 2,
        "val_rank_edges": 10,
        "presets": (
            (128, 24, 24, 4, 4, 1, 1e-3, 0.001, 5e-5, 0.010, 0.001, 0.15),
            (160, 24, 24, 6, 4, 1, 7e-4, 0.001, 5e-5, 0.010, 0.002, 0.10),
        ),
    },
    {
        "name": "history_ap_auc_probe",
        "weights": {"val_ap": 0.45, "val_auc": 0.25, "val_mrr": 0.15, "val_hits@10": 0.1, "val_ndcg": 0.05},
        "history_prior": True,
        "history_scales": (0.3, 0.7, 1.2),
        "search_epochs": 3,
        "patience": 2,
        "val_rank_edges": 10,
        "presets": (
            (128, 16, 16, 4, 4, 1, 1e-3, 0.010, 1e-4, 0.005, 0.001, 0.2),
            (160, 24, 24, 6, 4, 1, 7e-4, 0.010, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "history_fixed_probe",
        "weights": {"val_mrr": 0.35, "val_hits@10": 0.25, "val_ap": 0.25, "val_auc": 0.1, "val_ndcg": 0.05},
        "history_prior": True,
        "history_modes": {
            "rank": [1.5, 0.7, 0.3, 0.2, 0.2, 0.0, 1.5, 1.5, 1.0, 0.8, 0.3, 0.4],
            "copy": [2.0, 0.5, 0.2, 0.1, 0.1, 0.0, 2.0, 2.0, 1.0, 1.0, 0.3, 0.2],
        },
        "history_scales": (0.5, 1.0, 2.0),
        "search_epochs": 2,
        "patience": 1,
        "val_rank_edges": 25,
        "presets": (
            (128, 16, 16, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "history_sota_probe",
        "weights": {"val_mrr": 0.6, "val_hits@10": 0.15, "val_ap": 0.1, "val_auc": 0.05, "val_ndcg": 0.1},
        "history_prior": True,
        "history_dim": 20,
        "history_modes": {
            "rank20": [1.8, 0.7, 0.25, 0.15, 0.15, 0.0, 1.8, 1.8, 0.8, 0.8, 0.25, 0.3, 1.0, 0.4, 0.2, 0.2, 0.6, 0.6, 0.3, 0.3],
            "h1_20": [2.5, 0.5, 0.1, 0.05, 0.05, 0.0, 2.5, 2.5, 1.0, 1.0, 0.1, 0.2, 1.5, 0.2, 0.1, 0.1, 0.8, 0.8, 0.2, 0.2],
        },
        "history_scales": (1.0, 2.0, 4.0),
        "search_epochs": 1,
        "patience": 1,
        "val_rank_edges": 50,
        "presets": (
            (128, 16, 16, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "history_sota_rankloss",
        "weights": {"val_mrr": 0.65, "val_hits@10": 0.1, "val_ap": 0.05, "val_auc": 0.05, "val_ndcg": 0.15},
        "history_prior": True,
        "history_dim": 20,
        "history_modes": {
            "rank20": [1.8, 0.7, 0.25, 0.15, 0.15, 0.0, 1.8, 1.8, 0.8, 0.8, 0.25, 0.3, 1.0, 0.4, 0.2, 0.2, 0.6, 0.6, 0.3, 0.3],
            "h1_20": [2.5, 0.5, 0.1, 0.05, 0.05, 0.0, 2.5, 2.5, 1.0, 1.0, 0.1, 0.2, 1.5, 0.2, 0.1, 0.1, 0.8, 0.8, 0.2, 0.2],
        },
        "history_scales": (1.0, 2.0),
        "rank_losses": ((0.1, 1.0), (0.3, 1.0), (0.5, 0.5)),
        "search_epochs": 1,
        "patience": 1,
        "val_rank_edges": 50,
        "presets": (
            (128, 16, 16, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "history_decay_rankloss",
        "weights": {"val_mrr": 0.7, "val_hits@10": 0.1, "val_ap": 0.05, "val_auc": 0.05, "val_ndcg": 0.1},
        "history_prior": True,
        "history_dim": 28,
        "history_modes": {
            "decay_rank28": [
                1.4, 0.55, 0.2, 0.1, 0.1, 0.0, 1.3, 1.3, 0.7, 0.55,
                0.2, 0.2, 0.9, 0.3, 0.15, 0.15, 0.45, 0.45, 0.25, 0.25,
                0.9, 0.8, 0.35, 0.15, 0.55, 0.2, 0.1, 0.1,
            ],
            "decay_h1_28": [
                2.0, 0.45, 0.1, 0.05, 0.05, 0.0, 2.0, 2.0, 0.9, 0.9,
                0.1, 0.15, 1.3, 0.2, 0.08, 0.08, 0.65, 0.65, 0.15, 0.15,
                1.2, 1.0, 0.25, 0.1, 0.7, 0.25, 0.08, 0.08,
            ],
        },
        "history_scales": (0.35, 0.6, 0.9),
        "rank_losses": ((0.0, 1.0), (0.05, 1.0), (0.1, 0.75)),
        "search_epochs": 2,
        "patience": 1,
        "val_rank_edges": 75,
        "presets": (
            (128, 16, 16, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "history_decay_fastscan",
        "weights": {"val_mrr": 0.7, "val_hits@10": 0.1, "val_ap": 0.05, "val_auc": 0.05, "val_ndcg": 0.1},
        "history_prior": True,
        "history_dim": 28,
        "history_modes": {
            "decay_rank28": [
                1.4, 0.55, 0.2, 0.1, 0.1, 0.0, 1.3, 1.3, 0.7, 0.55,
                0.2, 0.2, 0.9, 0.3, 0.15, 0.15, 0.45, 0.45, 0.25, 0.25,
                0.9, 0.8, 0.35, 0.15, 0.55, 0.2, 0.1, 0.1,
            ],
            "decay_h1_28": [
                2.0, 0.45, 0.1, 0.05, 0.05, 0.0, 2.0, 2.0, 0.9, 0.9,
                0.1, 0.15, 1.3, 0.2, 0.08, 0.08, 0.65, 0.65, 0.15, 0.15,
                1.2, 1.0, 0.25, 0.1, 0.7, 0.25, 0.08, 0.08,
            ],
        },
        "history_scales": (0.25, 0.5),
        "rank_losses": ((0.0, 1.0), (0.05, 1.0)),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 50000,
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 5,
        "num_neg_eval": 5,
        "skip_formal": True,
        "presets": (
            (128, 16, 16, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "rankloss_fastscan",
        "weights": {"val_mrr": 0.7, "val_hits@10": 0.1, "val_ap": 0.05, "val_auc": 0.05, "val_ndcg": 0.1},
        "rank_losses": ((0.05, 1.0), (0.1, 0.75), (0.2, 0.5)),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 50000,
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 5,
        "num_neg_eval": 5,
        "skip_formal": True,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
            (128, 16, 16, 4, 4, 1, 7e-4, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "rankloss_uniform_probe",
        "weights": {"val_mrr": 0.62, "val_hits@10": 0.18, "val_ndcg": 0.1, "val_ap": 0.06, "val_auc": 0.04},
        "rank_losses": ((0.05, 1.0), (0.1, 0.75), (0.2, 0.5)),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 50000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1, 2),
        "val_binary_edges": 256,
        "val_rank_edges": 5,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "rankloss_micro_fastscan",
        "weights": {"val_mrr": 0.7, "val_hits@10": 0.1, "val_ap": 0.05, "val_auc": 0.05, "val_ndcg": 0.1},
        "rank_losses": ((0.1, 0.75), (0.2, 0.5)),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 10000,
        "override_neg_values": (1,),
        "val_binary_edges": 128,
        "val_rank_edges": 3,
        "num_neg_eval": 3,
        "skip_formal": True,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "rank_bpr_micro_fastscan",
        "weights": {"val_mrr": 0.72, "val_hits@10": 0.1, "val_ap": 0.04, "val_auc": 0.04, "val_ndcg": 0.1},
        "rank_losses": ((0.05, 0.0, "bpr"), (0.1, 0.0, "bpr"), (0.05, 0.5, "softplus")),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 10000,
        "override_neg_values": (1,),
        "val_binary_edges": 128,
        "val_rank_edges": 3,
        "num_neg_eval": 3,
        "skip_formal": True,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "history_light_fastscan",
        "weights": {"val_mrr": 0.7, "val_hits@10": 0.1, "val_ap": 0.05, "val_auc": 0.05, "val_ndcg": 0.1},
        "history_prior": True,
        "history_dim": 12,
        "history_modes": {
            "light12": [0.7, 0.25, 0.1, 0.08, 0.08, 0.0, 0.6, 0.6, 0.25, 0.25, 0.08, 0.08],
            "rank12": [1.0, 0.35, 0.15, 0.1, 0.1, 0.0, 0.9, 0.9, 0.4, 0.35, 0.12, 0.12],
        },
        "history_scales": (0.1, 0.25),
        "rank_losses": ((0.0, 1.0), (0.05, 1.0)),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 50000,
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 5,
        "num_neg_eval": 5,
        "skip_formal": True,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "history_popularity_midscan",
        "weights": {"val_mrr": 0.48, "val_hits@10": 0.18, "val_ap": 0.18, "val_auc": 0.08, "val_ndcg": 0.08},
        "history_prior": True,
        "history_dim": 40,
        "history_modes": {
            "freq40": [
                2.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01,
                0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            "relpop40": [
                0.8, 0.2, 0.0, 0.7, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.02, 0.02,
                0.02, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            "hybrid40": [
                1.5, 0.4, 0.1, 0.35, 0.35, 0.0, 0.6, 0.6, 0.2, 0.15,
                0.1, 0.0, 0.8, 0.2, 0.0, 0.0, 0.1, 0.1, 0.08, 0.08,
                0.25, 0.15, 0.1, 0.0, 0.35, 0.05, 0.03, 0.03, 0.02, 0.02,
                0.02, 0.0, 0.0, 0.08, 0.08, 0.02, 0.04, 0.06, 0.04, 0.0,
            ],
        },
        "history_scales": (0.25, 0.5, 0.8),
        "rank_losses": ((0.0, 1.0), (0.05, 0.0, "bpr"), (0.1, 0.5, "softplus")),
        "search_epochs": 3,
        "patience": 2,
        "search_seeds": (1, 2),
        "train_edge_limit": 50000,
        "override_neg_values": (1, 2),
        "val_binary_edges": 512,
        "val_rank_edges": 20,
        "num_neg_eval": 10,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
            (128, 16, 16, 4, 4, 1, 7e-4, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "history_popularity_quickscan",
        "weights": {"val_mrr": 0.5, "val_hits@10": 0.2, "val_ap": 0.15, "val_auc": 0.05, "val_ndcg": 0.1},
        "history_prior": True,
        "history_dim": 40,
        "history_modes": {
            "freq40": [
                2.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01,
                0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            "hybrid40": [
                1.5, 0.4, 0.1, 0.35, 0.35, 0.0, 0.6, 0.6, 0.2, 0.15,
                0.1, 0.0, 0.8, 0.2, 0.0, 0.0, 0.1, 0.1, 0.08, 0.08,
                0.25, 0.15, 0.1, 0.0, 0.35, 0.05, 0.03, 0.03, 0.02, 0.02,
                0.02, 0.0, 0.0, 0.08, 0.08, 0.02, 0.04, 0.06, 0.04, 0.0,
            ],
        },
        "history_scales": (0.5,),
        "rank_losses": ((0.0, 1.0), (0.05, 0.0, "bpr")),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 30000,
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 10,
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 14,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "history_popularity_tinysearch",
        "weights": {"val_ap": 0.45, "val_auc": 0.25, "val_ndcg": 0.2, "val_mrr": 0.05, "val_hits@10": 0.05},
        "history_prior": True,
        "history_dim": 40,
        "history_modes": {
            "freq40": [
                2.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01,
                0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            "hybrid40": [
                1.5, 0.4, 0.1, 0.35, 0.35, 0.0, 0.6, 0.6, 0.2, 0.15,
                0.1, 0.0, 0.8, 0.2, 0.0, 0.0, 0.1, 0.1, 0.08, 0.08,
                0.25, 0.15, 0.1, 0.0, 0.35, 0.05, 0.03, 0.03, 0.02, 0.02,
                0.02, 0.0, 0.0, 0.08, 0.08, 0.02, 0.04, 0.06, 0.04, 0.0,
            ],
        },
        "history_scales": (0.5, 0.8),
        "rank_losses": ((0.0, 1.0),),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 20000,
        "override_neg_values": (1,),
        "val_binary_edges": 512,
        "val_rank_edges": 0,
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 14,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "struct_residual_fastscan",
        "weights": {"val_mrr": 0.62, "val_hits@10": 0.18, "val_ap": 0.08, "val_auc": 0.04, "val_ndcg": 0.08},
        "struct_residual": True,
        "struct_scales": (0.35,),
        "struct_aux": (0.1,),
        "rank_losses": ((0.05, 0.0, "bpr"),),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 12000,
        "override_neg_values": (1,),
        "val_binary_edges": 128,
        "val_rank_edges": 3,
        "num_neg_eval": 3,
        "skip_formal": True,
        "formal_epochs": 18,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
            (128, 16, 16, 4, 4, 1, 7e-4, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "struct_history_fastscan",
        "weights": {"val_mrr": 0.55, "val_hits@10": 0.15, "val_ap": 0.15, "val_auc": 0.05, "val_ndcg": 0.1},
        "struct_residual": True,
        "struct_scales": (0.25, 0.5),
        "struct_aux": (0.1,),
        "history_prior": True,
        "history_dim": 40,
        "history_modes": {
            "freq40": [
                2.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01,
                0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        },
        "history_scales": (0.35,),
        "rank_losses": ((0.05, 0.0, "bpr"),),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 12000,
        "override_neg_values": (1,),
        "val_binary_edges": 128,
        "val_rank_edges": 3,
        "num_neg_eval": 3,
        "skip_formal": True,
        "formal_epochs": 18,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "struct_history_mlp_uniform_fastscan",
        "weights": {"val_mrr": 0.5, "val_hits@10": 0.18, "val_ap": 0.17, "val_auc": 0.05, "val_ndcg": 0.1},
        "struct_residual": True,
        "struct_scales": (0.25, 0.5),
        "struct_aux": (0.05, 0.1),
        "history_prior": True,
        "history_dim": 42,
        "history_prior_mode": "mlp",
        "history_prior_layer_norm": True,
        "history_scales": (0.25, 0.5),
        "rank_losses": ((0.05, 0.0, "bpr"), (0.1, 0.0, "bpr")),
        "search_epochs": 2,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 20000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1, 2),
        "val_binary_edges": 256,
        "val_rank_edges": 8,
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 20,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
            (128, 16, 16, 4, 4, 1, 7e-4, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "struct_history_mlp_uniform_probe",
        "weights": {"val_mrr": 0.5, "val_hits@10": 0.18, "val_ap": 0.17, "val_auc": 0.05, "val_ndcg": 0.1},
        "struct_residual": True,
        "struct_scales": (0.25, 0.5),
        "struct_aux": (0.05,),
        "history_prior": True,
        "history_dim": 42,
        "history_prior_mode": "mlp",
        "history_prior_layer_norm": True,
        "history_scales": (0.25, 0.5),
        "rank_losses": ((0.05, 0.0, "bpr"),),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 15000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1,),
        "val_binary_edges": 192,
        "val_rank_edges": 5,
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 18,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "struct_history_hardneg_probe",
        "weights": {"val_mrr": 0.55, "val_hits@10": 0.2, "val_ap": 0.12, "val_auc": 0.03, "val_ndcg": 0.1},
        "struct_residual": True,
        "struct_scales": (0.25, 0.5),
        "struct_aux": (0.05,),
        "history_prior": True,
        "history_dim": 42,
        "history_prior_mode": "mlp",
        "history_prior_layer_norm": True,
        "history_scales": (0.25, 0.5),
        "rank_losses": ((0.1, 0.0, "bpr"),),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 15000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (2,),
        "negative_mode_train": "relation_hard",
        "val_binary_edges": 192,
        "val_rank_edges": 5,
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 18,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "struct_history_popularity_probe",
        "weights": {"val_mrr": 0.45, "val_hits@10": 0.2, "val_ap": 0.2, "val_auc": 0.05, "val_ndcg": 0.1},
        "struct_residual": True,
        "struct_scales": (0.1, 0.25),
        "struct_aux": (0.05,),
        "history_prior": True,
        "history_dim": 42,
        "history_modes": {
            "histpop42": [
                2.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.01,
                0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0,
            ],
            "relpop42": [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.05, 0.05,
                0.05, 0.0, 0.0, 0.05, 0.05, 0.0, 0.02, 0.02, 0.0, 0.0,
                0.0, 0.0,
            ],
            "hybridpop42": [
                1.5, 0.4, 0.1, 0.2, 0.2, 0.0, 0.5, 0.5, 0.15, 0.0,
                0.0, 0.0, 0.5, 0.15, 0.0, 0.0, 0.3, 0.3, 0.05, 0.05,
                0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.03, 0.03,
                0.03, 0.0, 0.0, 0.04, 0.04, 0.0, 0.02, 0.02, 0.0, 0.0,
                0.0, 0.0,
            ],
        },
        "history_scales": (0.5, 1.0),
        "rank_losses": ((0.05, 0.0, "bpr"),),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 20000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 5,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 18,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "struct_history_relpop_fixed_probe",
        "weights": {"val_mrr": 0.45, "val_hits@10": 0.2, "val_ap": 0.2, "val_auc": 0.05, "val_ndcg": 0.1},
        "struct_residual": True,
        "struct_scales": (0.0, 0.1),
        "struct_aux": (0.0,),
        "history_prior": True,
        "history_dim": 48,
        "history_modes": {
            "relpopfix48": [
                2.0, 0.5, 0.1, 0.1, 0.1, 0.0, 0.6, 0.6, 0.1, 0.0,
                0.0, 0.0, 0.5, 0.15, 0.0, 0.0, 0.4, 0.4, 0.1, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.03, 0.03,
                0.03, 0.0, 0.0, 0.04, 0.04, 0.0, 0.02, 0.02, 0.0, 0.0,
                0.0, 0.0, 2.0, 2.0, 0.1, 0.1, 0.5, 0.5,
            ],
        },
        "history_scales": (1.0, 2.0, 4.0),
        "rank_losses": ((0.0, 0.0, "bpr"),),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 20000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 5,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 18,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "struct_history_gate_relpop_probe",
        "weights": {"val_mrr": 0.45, "val_hits@10": 0.2, "val_ap": 0.2, "val_auc": 0.05, "val_ndcg": 0.1},
        "struct_residual": True,
        "struct_scales": (0.0,),
        "struct_aux": (0.0,),
        "history_prior": True,
        "history_prior_gate": True,
        "history_gate_init_biases": (-1.0, 0.0, 1.0),
        "history_dim": 48,
        "history_modes": {
            "relpopfix48": [
                2.0, 0.5, 0.1, 0.1, 0.1, 0.0, 0.6, 0.6, 0.1, 0.0,
                0.0, 0.0, 0.5, 0.15, 0.0, 0.0, 0.4, 0.4, 0.1, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.03, 0.03,
                0.03, 0.0, 0.0, 0.04, 0.04, 0.0, 0.02, 0.02, 0.0, 0.0,
                0.0, 0.0, 2.0, 2.0, 0.1, 0.1, 0.5, 0.5,
            ],
        },
        "history_scales": (1.0, 2.0, 4.0),
        "rank_losses": ((0.0, 0.0, "bpr"),),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 20000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 5,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 18,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
            (128, 16, 16, 4, 4, 1, 7e-4, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "struct_history_role64_rankloss_probe",
        "weights": {"val_mrr": 0.55, "val_hits@10": 0.2, "val_ap": 0.12, "val_auc": 0.03, "val_ndcg": 0.1},
        "struct_residual": True,
        "struct_scales": (0.0,),
        "struct_aux": (0.0,),
        "history_prior": True,
        "history_dim": 64,
        "history_modes": {
            "role64": [
                2.0, 0.5, 0.1, 0.1, 0.1, 0.0, 0.6, 0.6, 0.1, 0.0,
                0.0, 0.0, 0.5, 0.15, 0.0, 0.0, 0.4, 0.4, 0.1, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.03, 0.03,
                0.03, 0.0, 0.0, 0.04, 0.04, 0.0, 0.02, 0.02, 0.0, 0.0,
                0.0, 0.0, 2.0, 2.0, 0.1, 0.1, 0.5, 0.5, 0.8, 0.8,
                0.25, 0.25, 0.15, 0.0, 0.35, 0.0, 0.6, 0.4, 0.2, 0.2,
                0.05, 0.05, 0.1, 0.1,
            ],
        },
        "history_scales": (0.5, 1.0, 2.0),
        "rank_losses": ((0.05, 1.0), (0.1, 0.75), (0.2, 0.5)),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 50000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 5,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 18,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
            (128, 16, 16, 4, 4, 1, 7e-4, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "struct_history_role64_mlp_rankloss_probe",
        "weights": {"val_mrr": 0.55, "val_hits@10": 0.2, "val_ap": 0.12, "val_auc": 0.03, "val_ndcg": 0.1},
        "struct_residual": True,
        "struct_scales": (0.0,),
        "struct_aux": (0.0,),
        "history_prior": True,
        "history_dim": 64,
        "history_prior_mode": "mlp",
        "history_prior_layer_norm": True,
        "history_scales": (0.25, 0.5),
        "rank_losses": ((0.05, 1.0), (0.1, 0.75)),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 50000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 5,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 18,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "struct_history_rankloss_unified_probe",
        "weights": {"val_mrr": 0.62, "val_hits@10": 0.18, "val_ndcg": 0.1, "val_ap": 0.06, "val_auc": 0.04},
        "struct_residual": True,
        "struct_scales": (0.0,),
        "struct_aux": (0.0,),
        "history_prior": True,
        "history_dim": 48,
        "history_prior_mode": "mlp",
        "history_prior_layer_norm": False,
        "history_scales": (0.0, 0.02),
        "rank_losses": ((0.1, 0.75), (0.2, 0.5)),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 50000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1, 2),
        "val_binary_edges": 256,
        "val_rank_edges": 5,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 18,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "relation_entity_prior_unified_probe",
        "weights": {"val_mrr": 0.62, "val_hits@10": 0.18, "val_ndcg": 0.1, "val_ap": 0.06, "val_auc": 0.04},
        "struct_residual": True,
        "struct_scales": (0.0,),
        "struct_aux": (0.0,),
        "history_prior": True,
        "history_dim": 48,
        "history_prior_mode": "mlp",
        "history_prior_layer_norm": False,
        "history_scales": (0.0,),
        "relation_entity_prior": True,
        "relation_entity_prior_ranks": (16, 32),
        "relation_entity_prior_scales": (0.1, 0.3, 0.7),
        "rank_losses": ((0.05, 1.0), (0.1, 0.75)),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 50000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 5,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 18,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "candidate_listwise_unified_probe",
        "weights": {"val_mrr": 0.62, "val_hits@10": 0.18, "val_ndcg": 0.1, "val_ap": 0.06, "val_auc": 0.04},
        "struct_residual": True,
        "struct_scales": (0.0,),
        "struct_aux": (0.0,),
        "history_prior": True,
        "history_dim": 64,
        "history_prior_mode": "mlp",
        "history_prior_layer_norm": True,
        "history_scales": (0.25, 0.5),
        "candidate_rank_losses": ((0.05, 16, "both"), (0.1, 16, "both"), (0.05, 32, "both")),
        "rank_losses": ((0.05, 0.0, "bpr"),),
        "negative_mode_train": "relation_hard",
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 50000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 5,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 18,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
            (128, 16, 16, 4, 4, 1, 7e-4, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "candidate_listwise_balanced_probe",
        "weights": {"val_mrr": 0.42, "val_hits@10": 0.18, "val_ap": 0.2, "val_auc": 0.1, "val_ndcg": 0.1},
        "struct_residual": True,
        "struct_scales": (0.0,),
        "struct_aux": (0.0,),
        "history_prior": True,
        "history_dim": 64,
        "history_prior_mode": "mlp",
        "history_prior_layer_norm": True,
        "history_scales": (0.1, 0.25),
        "candidate_rank_losses": ((0.005, 8, "both", 16), (0.01, 8, "both", 16), (0.02, 8, "both", 32)),
        "rank_losses": ((0.02, 0.0, "bpr"), (0.05, 0.0, "bpr")),
        "negative_mode_train": "relation_hard",
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 50000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 5,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 18,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
            (128, 16, 16, 4, 4, 1, 7e-4, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "candidate_listwise_yago_diag_probe",
        "weights": {"val_mrr": 0.55, "val_hits@10": 0.2, "val_ndcg": 0.1, "val_ap": 0.1, "val_auc": 0.05},
        "struct_residual": True,
        "struct_scales": (0.0,),
        "struct_aux": (0.0,),
        "history_prior": True,
        "history_dim": 64,
        "history_prior_mode": "mlp",
        "history_prior_layer_norm": True,
        "history_scales": (0.1,),
        "candidate_rank_losses": ((0.005, 8, "both", 16), (0.01, 8, "both", 16)),
        "rank_losses": ((0.02, 0.0, "bpr"), (0.05, 0.0, "bpr")),
        "negative_mode_train": "relation_hard",
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 50000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 5,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 18,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "struct_history_relpop_tune_probe",
        "weights": {"val_mrr": 0.45, "val_hits@10": 0.2, "val_ap": 0.2, "val_auc": 0.05, "val_ndcg": 0.1},
        "struct_residual": True,
        "struct_scales": (0.0,),
        "struct_aux": (0.0,),
        "history_prior": True,
        "freeze_history_prior": False,
        "history_dim": 48,
        "history_modes": {
            "relpopfix48": [
                2.0, 0.5, 0.1, 0.1, 0.1, 0.0, 0.6, 0.6, 0.1, 0.0,
                0.0, 0.0, 0.5, 0.15, 0.0, 0.0, 0.4, 0.4, 0.1, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.03, 0.03,
                0.03, 0.0, 0.0, 0.04, 0.04, 0.0, 0.02, 0.02, 0.0, 0.0,
                0.0, 0.0, 2.0, 2.0, 0.1, 0.1, 0.5, 0.5,
            ],
        },
        "history_scales": (0.5, 1.0, 2.0),
        "rank_losses": ((0.0, 0.0, "bpr"),),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 20000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 5,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 18,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "struct_history_listwise_relpop_probe",
        "weights": {"val_mrr": 0.5, "val_hits@10": 0.22, "val_ap": 0.13, "val_auc": 0.05, "val_ndcg": 0.1},
        "struct_residual": True,
        "struct_scales": (0.0,),
        "struct_aux": (0.0,),
        "history_prior": True,
        "history_dim": 48,
        "history_modes": {
            "relpopfix48": [
                2.0, 0.5, 0.1, 0.1, 0.1, 0.0, 0.6, 0.6, 0.1, 0.0,
                0.0, 0.0, 0.5, 0.15, 0.0, 0.0, 0.4, 0.4, 0.1, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.03, 0.03,
                0.03, 0.0, 0.0, 0.04, 0.04, 0.0, 0.02, 0.02, 0.0, 0.0,
                0.0, 0.0, 2.0, 2.0, 0.1, 0.1, 0.5, 0.5,
            ],
        },
        "history_scales": (1.0, 2.0),
        "rank_losses": ((0.05, 0.0, "sampled_softmax"), (0.1, 0.0, "sampled_softmax"), (0.2, 0.0, "sampled_softmax")),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 20000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (2, 5),
        "val_binary_edges": 256,
        "val_rank_edges": 5,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 18,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "struct_history_advhard_relpop_probe",
        "weights": {"val_mrr": 0.52, "val_hits@10": 0.22, "val_ap": 0.11, "val_auc": 0.05, "val_ndcg": 0.1},
        "struct_residual": True,
        "struct_scales": (0.0,),
        "struct_aux": (0.0,),
        "history_prior": True,
        "history_dim": 48,
        "history_modes": {
            "relpopfix48": [
                2.0, 0.5, 0.1, 0.1, 0.1, 0.0, 0.6, 0.6, 0.1, 0.0,
                0.0, 0.0, 0.5, 0.15, 0.0, 0.0, 0.4, 0.4, 0.1, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.03, 0.03,
                0.03, 0.0, 0.0, 0.04, 0.04, 0.0, 0.02, 0.02, 0.0, 0.0,
                0.0, 0.0, 2.0, 2.0, 0.1, 0.1, 0.5, 0.5,
            ],
        },
        "history_scales": (1.0, 2.0),
        "rank_losses": ((0.05, 0.5, "adv_bpr"), (0.1, 0.5, "adv_bpr"), (0.2, 0.5, "adv_bpr")),
        "negative_mode_train": "relation_hard",
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 20000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (2, 5),
        "val_binary_edges": 256,
        "val_rank_edges": 5,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 18,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "global_unified_sota_probe",
        "weights": {"val_mrr": 0.5, "val_hits@10": 0.2, "val_ap": 0.15, "val_ndcg": 0.1, "val_auc": 0.05},
        "struct_residual": True,
        "struct_scales": (0.25, 0.35),
        "struct_aux": (0.1,),
        "history_prior": True,
        "history_dim": 64,
        "history_prior_mode": "mlp",
        "history_prior_layer_norm": True,
        "history_scales": (0.1, 0.25),
        "rank_losses": ((0.05, 0.0, "bpr"),),
        "candidate_rank_losses": ((0.005, 8, "both", 16),),
        "negative_mode_train": "relation_hard",
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 20000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 10,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 20,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
            (128, 16, 16, 4, 4, 1, 7e-4, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "global_relpop_residual_rank_probe",
        "weights": {"val_mrr": 0.5, "val_hits@10": 0.2, "val_ap": 0.15, "val_ndcg": 0.1, "val_auc": 0.05},
        "struct_residual": True,
        "struct_scales": (0.05, 0.1),
        "struct_aux": (0.01, 0.05),
        "history_prior": True,
        "freeze_history_prior": False,
        "history_dim": 48,
        "history_modes": {
            "relpopfix48": [
                2.0, 0.5, 0.1, 0.1, 0.1, 0.0, 0.6, 0.6, 0.1, 0.0,
                0.0, 0.0, 0.5, 0.15, 0.0, 0.0, 0.4, 0.4, 0.1, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.03, 0.03,
                0.03, 0.0, 0.0, 0.04, 0.04, 0.0, 0.02, 0.02, 0.0, 0.0,
                0.0, 0.0, 2.0, 2.0, 0.1, 0.1, 0.5, 0.5,
            ],
        },
        "history_scales": (1.0, 2.0),
        "rank_losses": ((0.02, 0.0, "bpr"), (0.05, 0.0, "bpr")),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 20000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 10,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 20,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "global_relpop_frozen_epsilon_probe",
        "weights": {"val_mrr": 0.5, "val_hits@10": 0.2, "val_ap": 0.15, "val_ndcg": 0.1, "val_auc": 0.05},
        "struct_residual": True,
        "struct_scales": (0.001, 0.01, 0.05),
        "struct_aux": (0.001,),
        "history_prior": True,
        "freeze_history_prior": True,
        "history_dim": 48,
        "history_modes": {
            "relpopfix48": [
                2.0, 0.5, 0.1, 0.1, 0.1, 0.0, 0.6, 0.6, 0.1, 0.0,
                0.0, 0.0, 0.5, 0.15, 0.0, 0.0, 0.4, 0.4, 0.1, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.03, 0.03,
                0.03, 0.0, 0.0, 0.04, 0.04, 0.0, 0.02, 0.02, 0.0, 0.0,
                0.0, 0.0, 2.0, 2.0, 0.1, 0.1, 0.5, 0.5,
            ],
        },
        "history_scales": (1.0, 2.0, 4.0),
        "rank_losses": ((0.001, 0.0, "bpr"), (0.005, 0.0, "bpr")),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 20000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 10,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 20,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "global_relpop_frozen_enhanced_probe",
        "weights": {"val_mrr": 0.5, "val_hits@10": 0.2, "val_ap": 0.15, "val_ndcg": 0.1, "val_auc": 0.05},
        "struct_residual": True,
        "struct_scales": (0.01, 0.05),
        "struct_aux": (0.001,),
        "history_prior": True,
        "freeze_history_prior": True,
        "history_dim": 48,
        "history_modes": {
            "relpopfix48": [
                2.0, 0.5, 0.1, 0.1, 0.1, 0.0, 0.6, 0.6, 0.1, 0.0,
                0.0, 0.0, 0.5, 0.15, 0.0, 0.0, 0.4, 0.4, 0.1, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.03, 0.03,
                0.03, 0.0, 0.0, 0.04, 0.04, 0.0, 0.02, 0.02, 0.0, 0.0,
                0.0, 0.0, 2.0, 2.0, 0.1, 0.1, 0.5, 0.5,
            ],
        },
        "history_scales": (1.0, 2.0),
        "rank_losses": ((0.001, 0.0, "bpr"), (0.005, 0.0, "bpr")),
        "search_epochs": 3,
        "patience": 2,
        "search_seeds": (1,),
        "train_edge_limit": 50000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1,),
        "val_binary_edges": 512,
        "val_rank_edges": 30,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 10,
        "formal_epochs": 24,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "global_relpop_rep_enhanced_probe",
        "weights": {"val_mrr": 0.5, "val_hits@10": 0.2, "val_ap": 0.15, "val_ndcg": 0.1, "val_auc": 0.05},
        "struct_residual": True,
        "struct_scales": (0.01,),
        "struct_aux": (0.001,),
        "history_prior": True,
        "freeze_history_prior": True,
        "history_dim": 48,
        "history_modes": {
            "relpopfix48": [
                2.0, 0.5, 0.1, 0.1, 0.1, 0.0, 0.6, 0.6, 0.1, 0.0,
                0.0, 0.0, 0.5, 0.15, 0.0, 0.0, 0.4, 0.4, 0.1, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.03, 0.03,
                0.03, 0.0, 0.0, 0.04, 0.04, 0.0, 0.02, 0.02, 0.0, 0.0,
                0.0, 0.0, 2.0, 2.0, 0.1, 0.1, 0.5, 0.5,
            ],
        },
        "history_scales": (1.0,),
        "relation_entity_prior": True,
        "relation_entity_prior_ranks": (8, 16),
        "relation_entity_prior_scales": (0.02, 0.05),
        "rank_losses": ((0.001, 0.0, "bpr"),),
        "search_epochs": 3,
        "patience": 2,
        "search_seeds": (1,),
        "train_edge_limit": 50000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1,),
        "val_binary_edges": 512,
        "val_rank_edges": 30,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 10,
        "formal_epochs": 24,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
    {
        "name": "global_relpop_gate_probe",
        "weights": {"val_mrr": 0.5, "val_hits@10": 0.2, "val_ap": 0.15, "val_ndcg": 0.1, "val_auc": 0.05},
        "struct_residual": True,
        "struct_scales": (0.01, 0.05),
        "struct_aux": (0.001,),
        "history_prior": True,
        "history_prior_gate": True,
        "history_gate_init_biases": (-2.0, -1.0, 0.0),
        "freeze_history_prior": True,
        "history_dim": 48,
        "history_modes": {
            "relpopfix48": [
                2.0, 0.5, 0.1, 0.1, 0.1, 0.0, 0.6, 0.6, 0.1, 0.0,
                0.0, 0.0, 0.5, 0.15, 0.0, 0.0, 0.4, 0.4, 0.1, 0.1,
                0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.03, 0.03,
                0.03, 0.0, 0.0, 0.04, 0.04, 0.0, 0.02, 0.02, 0.0, 0.0,
                0.0, 0.0, 2.0, 2.0, 0.1, 0.1, 0.5, 0.5,
            ],
        },
        "history_scales": (1.0, 2.0, 4.0),
        "rank_losses": ((0.001, 0.0, "bpr"),),
        "search_epochs": 1,
        "patience": 1,
        "search_seeds": (1,),
        "train_edge_limit": 20000,
        "train_edge_sample": "temporal_uniform",
        "override_neg_values": (1,),
        "val_binary_edges": 256,
        "val_rank_edges": 10,
        "val_eval_sample": "temporal_uniform",
        "num_neg_eval": 5,
        "skip_formal": True,
        "formal_epochs": 20,
        "presets": (
            (96, 12, 12, 4, 4, 1, 1e-3, 0.005, 1e-4, 0.005, 0.001, 0.2),
        ),
    },
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def rows_from_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload if isinstance(payload, list) else [])
    return [row for row in rows if isinstance(row, dict)]


def finite_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return mean(values) if values else None


def is_external_baseline(row: dict[str, Any]) -> bool:
    return str(row.get("method", "")).lower() != "ssptgfm"


def thresholds(paths: tuple[Path, ...], prefix: str) -> dict[str, float]:
    best = {metric: -float("inf") for metric in METRICS}
    for path in paths:
        rows = [row for row in rows_from_json(path) if is_external_baseline(row)]
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in rows:
            key = (
                str(row.get("dataset", "")),
                str(row.get("method", "")),
                str(row.get("ablation", "")),
            )
            grouped.setdefault(key, []).append(row)
        for metric in METRICS:
            for group_rows in grouped.values():
                value = finite_mean(group_rows, f"{prefix}_{metric}")
                if value is not None:
                    best[metric] = max(best[metric], value)
    return best


def metric_means(path: Path, prefix: str) -> dict[str, float]:
    rows = rows_from_json(path)
    return {
        metric: finite_mean(rows, f"{prefix}_{metric}") or -float("inf")
        for metric in METRICS
    }


def count_wins(values: dict[str, float], target: dict[str, float]) -> tuple[int, dict[str, dict[str, float | bool]]]:
    detail: dict[str, dict[str, float | bool]] = {}
    wins = 0
    for metric in METRICS:
        value = values.get(metric, -float("inf"))
        baseline = target.get(metric, -float("inf"))
        win = math.isfinite(value) and math.isfinite(baseline) and value > baseline
        wins += int(win)
        detail[metric] = {"value": value, "baseline": baseline, "win": win}
    return wins, detail


def candidate_patches(spec: DatasetSpec, bank: dict[str, Any]) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    neg_values = tuple(bank.get("override_neg_values", spec.neg_values))
    history_scales = tuple(bank.get("history_scales", (None,)))
    history_modes = bank.get("history_modes") or {None: None}
    history_gate_biases = tuple(bank.get("history_gate_init_biases", (None,)))
    rank_losses = tuple(bank.get("rank_losses", ((None, None),)))
    candidate_rank_losses = tuple(bank.get("candidate_rank_losses", ((None, None, None, None),)))
    struct_scales = tuple(bank.get("struct_scales", (None,)))
    struct_aux_values = tuple(bank.get("struct_aux", (None,)))
    relation_entity_prior_ranks = tuple(bank.get("relation_entity_prior_ranks", (None,)))
    relation_entity_prior_scales = tuple(bank.get("relation_entity_prior_scales", (None,)))
    for neg in neg_values:
        for preset in bank["presets"]:
            for history_scale in history_scales:
                for history_mode_name, history_weights in history_modes.items():
                    for history_gate_bias in history_gate_biases:
                        for rank_loss in rank_losses:
                            for candidate_rank_loss in candidate_rank_losses:
                                for struct_scale in struct_scales:
                                    for struct_aux in struct_aux_values:
                                        for relation_prior_rank in relation_entity_prior_ranks:
                                            for relation_prior_scale in relation_entity_prior_scales:
                                                if len(candidate_rank_loss) == 3:
                                                    lambda_candidate_rank, candidate_rank_size, candidate_rank_sides = candidate_rank_loss
                                                    candidate_rank_queries = None
                                                else:
                                                    (
                                                        lambda_candidate_rank,
                                                        candidate_rank_size,
                                                        candidate_rank_sides,
                                                        candidate_rank_queries,
                                                    ) = candidate_rank_loss
                                                if len(rank_loss) == 2:
                                                    lambda_rank, rank_margin = rank_loss
                                                    rank_loss_type = "hinge"
                                                else:
                                                    lambda_rank, rank_margin, rank_loss_type = rank_loss
                                                hidden, rel_rank, adapter_rank, prompt_tokens, prompt_heads, layers, lr, align, kl, meta, ood, temp = preset
                                                if hidden % prompt_heads != 0:
                                                    continue
                                                name = (
                                                    f"{bank['name']}_h{hidden}_r{rel_rank}_a{adapter_rank}"
                                                    f"_p{prompt_tokens}x{prompt_heads}_l{layers}_neg{neg}_lr{lr:g}"
                                                )
                                                if history_scale is not None:
                                                    name += f"_hp{float(history_scale):g}"
                                                if history_mode_name is not None:
                                                    name += f"_{history_mode_name}"
                                                if bool(bank.get("history_prior_gate", False)):
                                                    name += f"_hpg{float(history_gate_bias):g}"
                                                if struct_scale is not None:
                                                    name += f"_sr{float(struct_scale):g}"
                                                if struct_aux is not None:
                                                    name += f"_sa{float(struct_aux):g}"
                                                if bool(bank.get("relation_entity_prior", False)):
                                                    name += f"_rep{int(relation_prior_rank)}s{float(relation_prior_scale):g}"
                                                if lambda_rank is not None:
                                                    name += f"_rl{float(lambda_rank):g}_m{float(rank_margin):g}_{rank_loss_type}"
                                                if lambda_candidate_rank is not None:
                                                    name += (
                                                        f"_crl{float(lambda_candidate_rank):g}"
                                                        f"_cs{int(candidate_rank_size)}_{candidate_rank_sides}"
                                                    )
                                                    if candidate_rank_queries is not None:
                                                        name += f"_cq{int(candidate_rank_queries)}"
                                                name = name.replace(".", "p")
                                                model_patch = {
                                                    "hidden_dim": hidden,
                                                    "relation_rank": rel_rank,
                                                    "adapter_rank": adapter_rank,
                                                    "prompt_tokens": prompt_tokens,
                                                    "prompt_heads": prompt_heads,
                                                    "temporal_layers": layers,
                                                    "temporal_encoder": "mlp",
                                                    "time_encoder": "fourier",
                                                    "use_struct": True,
                                                    "use_sem": True,
                                                    "use_cross": True,
                                                    "use_gate": True,
                                                    "use_variational": True,
                                                }
                                                if bool(bank.get("history_prior", False)):
                                                    model_patch.update(
                                                        {
                                                            "use_history_prior": True,
                                                            "history_prior_dim": int(bank.get("history_dim", 12)),
                                                            "history_prior_hidden_dim": max(16, hidden // 4),
                                                            "history_prior_init_scale": float(history_scale),
                                                            "history_prior_mode": str(bank.get("history_prior_mode", "mlp")),
                                                            "history_prior_layer_norm": bool(bank.get("history_prior_layer_norm", False)),
                                                        }
                                                    )
                                                if history_weights is not None:
                                                    model_patch.update(
                                                        {
                                                            "history_prior_mode": "linear",
                                                            "history_prior_weights": history_weights,
                                                            "freeze_history_prior": bool(bank.get("freeze_history_prior", True)),
                                                        }
                                                    )
                                                    if bool(bank.get("history_prior_gate", False)):
                                                        model_patch.update(
                                                            {
                                                                "use_history_prior_gate": True,
                                                                "history_prior_gate_hidden_dim": max(8, hidden // 8),
                                                                "history_prior_gate_init_bias": float(history_gate_bias),
                                                            }
                                                        )
                                                if bool(bank.get("struct_residual", False)):
                                                    model_patch.update(
                                                        {
                                                            "use_struct_feature_residual": True,
                                                            "struct_feature_hidden_dim": max(16, hidden),
                                                            "struct_feature_init_scale": float(struct_scale),
                                                        }
                                                    )
                                                if bool(bank.get("relation_entity_prior", False)):
                                                    model_patch.update(
                                                        {
                                                            "use_relation_entity_prior": True,
                                                            "relation_entity_prior_rank": int(relation_prior_rank),
                                                            "relation_entity_prior_init_scale": float(relation_prior_scale),
                                                        }
                                                    )
                                                train_patch = {
                                                    "lr": lr,
                                                    "num_neg_train": neg,
                                                    "lambda_align": align,
                                                    "lambda_kl": kl,
                                                    "lambda_meta": meta,
                                                    "lambda_ood": ood,
                                                    "align_temperature": temp,
                                                }
                                                if bank.get("negative_mode_train", None) is not None:
                                                    train_patch["negative_mode_train"] = str(bank["negative_mode_train"])
                                                if struct_aux is not None:
                                                    train_patch["lambda_struct_aux"] = float(struct_aux)
                                                if lambda_rank is not None:
                                                    train_patch.update(
                                                        {
                                                            "lambda_rank": float(lambda_rank),
                                                            "rank_margin": float(rank_margin),
                                                            "rank_loss_type": str(rank_loss_type),
                                                        }
                                                    )
                                                if lambda_candidate_rank is not None:
                                                    train_patch.update(
                                                        {
                                                            "lambda_candidate_rank": float(lambda_candidate_rank),
                                                            "candidate_rank_size": int(candidate_rank_size),
                                                            "candidate_rank_sides": str(candidate_rank_sides),
                                                        }
                                                    )
                                                    if candidate_rank_queries is not None:
                                                        train_patch["candidate_rank_queries"] = int(candidate_rank_queries)
                                                patches.append({"name": name, "model": model_patch, "train": train_patch})
    return patches


def render_search_config(spec: DatasetSpec, bank: dict[str, Any], run_root: Path) -> tuple[Path, Path]:
    cfg = load_yaml(spec.base_search_config)
    search_out = run_root / f"search_{spec.label}_{bank['name']}"
    cfg["output_dir"] = str(search_out)
    cfg["seeds"] = [1, 2]
    cfg["baselines"] = []
    cfg["strict_full_formula"] = True
    cfg.setdefault("model", {})
    cfg["model"].update(
        {
            "use_struct": True,
            "use_sem": True,
            "use_cross": True,
            "use_gate": True,
            "use_variational": True,
        }
    )
    if bool(bank.get("history_prior", False)):
        cfg["model"]["use_history_prior"] = True
    cfg.setdefault("train", {})
    search_epochs = int(bank.get("search_epochs", spec.search_epochs))
    patience = int(bank.get("patience", 4))
    val_rank_edges = int(bank.get("val_rank_edges", spec.search_val_rank_edges))
    cfg["train"].update(
        {
            "epochs": search_epochs,
            "patience": patience,
            "early_stop_metric": "val_composite",
            "early_stop_weights": bank["weights"],
            "val_rank_edges": val_rank_edges,
            "progress_every_batches": 100,
            "progress_every_seconds": 120.0,
        }
    )
    if bank.get("val_binary_edges", None) is not None:
        cfg["train"]["val_binary_edges"] = int(bank["val_binary_edges"])
    if bank.get("val_eval_sample", None) is not None:
        cfg["train"]["val_eval_sample"] = str(bank["val_eval_sample"])
    if bank.get("num_neg_eval", None) is not None:
        cfg["train"]["num_neg_eval"] = int(bank["num_neg_eval"])
    if spec.batch_size is not None:
        cfg["train"]["batch_size"] = spec.batch_size
    cfg.setdefault("eval", {})
    cfg["eval"]["filtered_rank_edges"] = spec.formal_eval_rank_edges
    cfg["search"] = {
        "metric": "composite",
        "metrics": [f"val_{metric}" for metric in METRICS],
        "metric_weights": bank["weights"],
        "strict_full_formula": True,
        "candidates": candidate_patches(spec, bank),
    }
    if bank.get("search_seeds", None) is not None:
        cfg["search"]["seeds"] = [int(seed) for seed in bank["search_seeds"]]
    if bank.get("train_edge_limit", None) is not None:
        cfg["search"]["train_edge_limit"] = int(bank["train_edge_limit"])
    if bank.get("train_edge_sample", None) is not None:
        cfg["search"]["train_edge_sample"] = str(bank["train_edge_sample"])
    path = Path("configs/auto_target") / f"ssptgfm_search_{spec.label}_{bank['name']}.yaml"
    write_yaml(path, cfg)
    return path, search_out


def render_formal_config(spec: DatasetSpec, bank: dict[str, Any], search_results: Path, run_root: Path) -> tuple[Path, Path]:
    cfg = load_yaml(spec.base_formal_config)
    payload = json.loads(search_results.read_text(encoding="utf-8"))
    candidate = payload["best_candidate"]
    formal_dir = run_root / f"ssptgfm_{spec.label}_{bank['name']}_best"
    cfg["output_dir"] = str(formal_dir)
    cfg["seeds"] = list(SEEDS)
    cfg["baselines"] = []
    cfg["strict_full_formula"] = True
    cfg["model"] = {**cfg.get("model", {}), **candidate.get("model", {})}
    cfg["train"] = {**cfg.get("train", {}), **candidate.get("train", {})}
    cfg["train"].update(
        {
            "epochs": int(bank.get("formal_epochs", spec.formal_epochs)),
            "patience": 5,
            "early_stop_metric": "val_composite",
            "early_stop_weights": bank["weights"],
            "val_rank_edges": spec.search_val_rank_edges,
            "progress_every_batches": 100,
            "progress_every_seconds": 120.0,
        }
    )
    if spec.batch_size is not None:
        cfg["train"]["batch_size"] = spec.batch_size
    cfg.setdefault("eval", {})
    cfg["eval"]["filtered_rank_edges"] = spec.formal_eval_rank_edges
    validate_full_formula_config(cfg, context=f"auto_target {spec.label} {bank['name']}")
    path = Path("configs/auto_target") / f"ssptgfm_{spec.label}_{bank['name']}_best.yaml"
    write_yaml(path, cfg)
    return path, formal_dir


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)


def wait_search_proc(item: tuple[subprocess.Popen[bytes], Any, str]) -> None:
    proc, handle, bank_name = item
    code = proc.wait()
    handle.close()
    if code != 0:
        raise subprocess.CalledProcessError(code, proc.args)
    print({"event": "search_done", "bank": bank_name}, flush=True)


def run_search_banks_parallel(
    spec: DatasetSpec,
    run_root: Path,
    env: dict[str, str],
    max_parallel: int,
    banks: tuple[dict[str, Any], ...] = BANKS,
) -> dict[str, dict[str, Any]]:
    pending: list[tuple[subprocess.Popen[bytes], Any, str]] = []
    search_outputs: dict[str, tuple[Path, Path]] = {}
    max_parallel = max(1, int(max_parallel))
    for bank in banks:
        search_config, search_out = render_search_config(spec, bank, run_root)
        search_outputs[bank["name"]] = (search_config, search_out)
        search_results = search_out / "search_results.json"
        if search_results.exists():
            print({"event": "search_skip_existing", "dataset": spec.label, "bank": bank["name"]}, flush=True)
            continue
        log_path = Path("results/logs") / f"auto_target_{spec.label}_{bank['name']}_search.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("ab")
        cmd = [
            sys.executable,
            "-u",
            "scripts/search_hparams.py",
            "--config",
            str(search_config),
            "--output",
            str(search_out),
            "--resume",
        ]
        log.write(f"\n$ {' '.join(cmd)}\n".encode("utf-8"))
        log.flush()
        print(
            {
                "event": "search_start",
                "dataset": spec.label,
                "bank": bank["name"],
                "config": str(search_config),
                "max_parallel_searches": max_parallel,
            },
            flush=True,
        )
        pending.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log, bank["name"]))
        if len(pending) >= max_parallel:
            wait_search_proc(pending.pop(0))
    for item in pending:
        wait_search_proc(item)

    payloads: dict[str, dict[str, Any]] = {}
    for bank in banks:
        _, search_out = search_outputs[bank["name"]]
        search_results = search_out / "search_results.json"
        if not search_results.exists():
            raise FileNotFoundError(f"missing search result after search: {search_results}")
        payloads[bank["name"]] = json.loads(search_results.read_text(encoding="utf-8"))
    return payloads


def run_formal(
    spec: DatasetSpec,
    bank_name: str,
    formal_config: Path,
    formal_dir: Path,
    run_root: Path,
    env: dict[str, str],
) -> None:
    seed_dirs = [run_root / f"ssptgfm_{spec.label}_{bank_name}_seed{seed}" for seed in SEEDS]
    pids: list[tuple[subprocess.Popen[bytes], Any]] = []
    max_parallel_formal = max(1, int(env.get("MAX_PARALLEL_FORMAL_SEEDS", spec.max_parallel_seeds)))
    for seed, out_dir in zip(SEEDS, seed_dirs):
        partial = out_dir / "partial_results.jsonl"
        if partial.exists() and sum(1 for _ in partial.open("r", encoding="utf-8")) >= 1:
            continue
        log_path = Path("results/logs") / f"auto_target_{spec.label}_{bank_name}_seed{seed}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("ab")
        cmd = [
            sys.executable,
            "-u",
            "scripts/run_ssptgfm.py",
            "--config",
            str(formal_config),
            "--seeds",
            str(seed),
            "--output",
            str(out_dir),
            "--resume",
        ]
        pids.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log))
        if len(pids) >= max_parallel_formal:
            proc, handle = pids.pop(0)
            code = proc.wait()
            handle.close()
            if code != 0:
                raise subprocess.CalledProcessError(code, proc.args)
    for proc, handle in pids:
        code = proc.wait()
        handle.close()
        if code != 0:
            raise subprocess.CalledProcessError(code, proc.args)

    run_command(
        [
            sys.executable,
            "scripts/merge_parallel_results.py",
            "--inputs",
            *[str(path) for path in seed_dirs],
            "--out-dir",
            str(formal_dir),
            "--expected-rows",
            str(len(SEEDS)),
        ],
        Path("results/logs") / f"auto_target_{spec.label}_{bank_name}_merge.log",
        env,
    )
    run_command(
        [
            sys.executable,
            "scripts/summarize_results.py",
            "--results",
            str(formal_dir / "all_results.json"),
            "--out",
            str(formal_dir / "summary.csv"),
        ],
        Path("results/logs") / f"auto_target_{spec.label}_{bank_name}_summary.log",
        env,
    )


def write_compare_csv(spec: DatasetSpec, formal_dir: Path, out_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    ours = metric_means(formal_dir / "all_results.json", "test")
    base = thresholds(spec.baseline_paths, "test")
    wins, detail = count_wins(ours, base)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "optimized", "best_external_baseline", "delta", "win"])
        writer.writeheader()
        for metric in METRICS:
            writer.writerow(
                {
                    "metric": metric,
                    "optimized": detail[metric]["value"],
                    "best_external_baseline": detail[metric]["baseline"],
                    "delta": float(detail[metric]["value"]) - float(detail[metric]["baseline"]),
                    "win": detail[metric]["win"],
                }
            )
    status_path = out_path.with_suffix(".status.json")
    status_path.write_text(
        json.dumps({"dataset": spec.label, "wins": wins, "target_wins": TARGET_WINS, "passed": wins >= TARGET_WINS, "detail": detail}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"datasets": {}}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def best_existing(spec: DatasetSpec) -> tuple[Path | None, int, dict[str, dict[str, float | bool]]]:
    base = thresholds(spec.baseline_paths, "test")
    best_path: Path | None = None
    best_wins = -1
    best_detail: dict[str, dict[str, float | bool]] = {}
    for path in spec.existing_result_paths:
        if not path.exists():
            continue
        wins, detail = count_wins(metric_means(path, "test"), base)
        if wins > best_wins:
            best_path = path
            best_wins = wins
            best_detail = detail
    return best_path, best_wins, best_detail


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only SSP-TGFM auto optimization queue.")
    parser.add_argument("--run-root", default="results/auto_target_optimization")
    parser.add_argument("--state", default="results/auto_target_optimization/state.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset labels. TGB-ICEWS is intentionally absent.")
    parser.add_argument("--banks", default=None, help="Comma-separated search bank names.")
    parser.add_argument(
        "--allow-dataset-local-formal",
        action="store_true",
        help=(
            "Allow the legacy per-dataset best-candidate formal stage. "
            "Leave disabled when the paper requires one shared model candidate across datasets."
        ),
    )
    args = parser.parse_args()

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = Path(args.state)
    state = load_state(state_path) if args.resume else {"datasets": {}}
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONPATH"] = "."
    selected = set(args.datasets.split(",")) if args.datasets else None
    selected_banks = set(args.banks.split(",")) if args.banks else None
    active_banks = tuple(bank for bank in BANKS if selected_banks is None or bank["name"] in selected_banks)
    if not active_banks:
        raise SystemExit("no matching search banks selected")

    print({"event": "auto_target_start", "run_root": str(run_root), "target_wins": TARGET_WINS, "metrics": METRICS}, flush=True)
    for spec in DATASETS:
        if selected is not None and spec.label not in selected:
            continue
        ds_state = state["datasets"].setdefault(spec.label, {})
        existing_path, existing_wins, existing_detail = best_existing(spec)
        ds_state["existing"] = {
            "path": str(existing_path) if existing_path else None,
            "wins": existing_wins,
            "detail": existing_detail,
        }
        save_state(state_path, state)
        print({"event": "dataset_start", "dataset": spec.label, "existing_wins": existing_wins}, flush=True)
        if existing_wins >= TARGET_WINS:
            ds_state["status"] = "already_passed"
            save_state(state_path, state)
            print({"event": "dataset_skip_already_passed", "dataset": spec.label}, flush=True)
            continue

        max_parallel_searches = int(env.get("MAX_PARALLEL_SEARCHES", "3"))
        payloads = run_search_banks_parallel(spec, run_root, env, max_parallel_searches, banks=active_banks)
        best_bank: dict[str, Any] | None = None
        best_search_score = -float("inf")
        best_search_only_bank: dict[str, Any] | None = None
        best_search_only_score = -float("inf")
        for bank in active_banks:
            payload = payloads[bank["name"]]
            score = float(payload.get("best_validation_mean", -float("inf")))
            ds_state.setdefault("searches", {})[bank["name"]] = {
                "score": score,
                "search_results": str(run_root / f"search_{spec.label}_{bank['name']}" / "search_results.json"),
                "best_candidate": payload.get("best_candidate", {}).get("name"),
            }
            save_state(state_path, state)
            if bool(bank.get("skip_formal", False)):
                if score > best_search_only_score:
                    best_search_only_score = score
                    best_search_only_bank = bank
                continue
            if score > best_search_score:
                best_search_score = score
                best_bank = bank

        if best_bank is None:
            ds_state["status"] = "no_formal_candidate"
            if best_search_only_bank is not None:
                ds_state["best_search"] = {
                    "bank": best_search_only_bank["name"],
                    "score": best_search_only_score,
                    "search_results": str(run_root / f"search_{spec.label}_{best_search_only_bank['name']}" / "search_results.json"),
                    "note": "best candidate came from a search-only bank; run a formal-capable bank before formal test",
                }
            save_state(state_path, state)
            continue

        if not args.allow_dataset_local_formal:
            ds_state["status"] = "formal_blocked_global_model_required"
            ds_state["best_dataset_local_formal_candidate"] = {
                "bank": best_bank["name"],
                "score": best_search_score,
                "search_results": str(run_root / f"search_{spec.label}_{best_bank['name']}" / "search_results.json"),
                "note": (
                    "Per-dataset formal selection is disabled. Select one shared candidate "
                    "across datasets with report_global_candidate_search.py before formal tests."
                ),
            }
            save_state(state_path, state)
            print(
                {
                    "event": "formal_blocked_global_model_required",
                    "dataset": spec.label,
                    "bank": best_bank["name"],
                },
                flush=True,
            )
            continue

        best_search_results = run_root / f"search_{spec.label}_{best_bank['name']}" / "search_results.json"
        formal_config, formal_dir = render_formal_config(spec, best_bank, best_search_results, run_root)
        if not (formal_dir / "all_results.json").exists():
            print({"event": "formal_start", "dataset": spec.label, "bank": best_bank["name"], "formal_config": str(formal_config)}, flush=True)
            run_formal(spec, best_bank["name"], formal_config, formal_dir, run_root, env)
        compare_path = run_root / f"compare_{spec.label}_{best_bank['name']}.csv"
        write_compare_csv(spec, formal_dir, compare_path)
        status = json.loads(compare_path.with_suffix(".status.json").read_text(encoding="utf-8"))
        ds_state["formal"] = {
            "bank": best_bank["name"],
            "dir": str(formal_dir),
            "compare": str(compare_path),
            "wins": status["wins"],
            "passed": status["passed"],
        }
        ds_state["status"] = "passed" if status["passed"] else "formal_done_target_not_met"
        save_state(state_path, state)
        print({"event": "dataset_done", "dataset": spec.label, **ds_state["formal"]}, flush=True)

    print({"event": "auto_target_done", "state": str(state_path)}, flush=True)


if __name__ == "__main__":
    main()
