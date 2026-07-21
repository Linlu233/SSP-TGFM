from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ssptgfm.data import EdgeTensor, TemporalDataset, TemporalSplits, split_by_time


@dataclass
class SplitScenario:
    name: str
    splits: TemporalSplits
    metadata: dict[str, object]


def exact_k_shot_train(train: EdgeTensor, k: int, by: str = "relation") -> EdgeTensor:
    if k <= 0:
        raise ValueError("k must be positive")
    train = train.sort_by_time()
    groups = train.rel if by == "relation" else train.src
    keep: list[int] = []
    counts: dict[int, int] = {}
    for idx, group in enumerate(groups.tolist()):
        g = int(group)
        if counts.get(g, 0) < k:
            keep.append(idx)
            counts[g] = counts.get(g, 0) + 1
    if not keep:
        raise ValueError("k-shot selection produced no train edges")
    return train.slice(keep).sort_by_time()


def _mask_by_nodes(edges: EdgeTensor, nodes: set[int], include: bool) -> torch.Tensor:
    src = edges.src.cpu().numpy()
    dst = edges.dst.cpu().numpy()
    mask_np = np.asarray([(int(s) in nodes or int(d) in nodes) for s, d in zip(src, dst)], dtype=bool)
    if not include:
        mask_np = ~mask_np
    return torch.as_tensor(mask_np)


def new_node_scenario(dataset: TemporalDataset, val_ratio: float, test_ratio: float, holdout_ratio: float, seed: int) -> SplitScenario:
    base = split_by_time(dataset.edges, val_ratio=val_ratio, test_ratio=test_ratio)
    rng = np.random.default_rng(seed)
    nodes = np.arange(dataset.num_nodes)
    holdout_count = max(1, int(round(dataset.num_nodes * holdout_ratio)))
    holdout = set(rng.choice(nodes, size=holdout_count, replace=False).astype(int).tolist())
    train = base.train.slice(_mask_by_nodes(base.train, holdout, include=False))
    val_hold = base.val.slice(_mask_by_nodes(base.val, holdout, include=True))
    test_hold = base.test.slice(_mask_by_nodes(base.test, holdout, include=True))
    val = val_hold if len(val_hold) else base.val
    test = test_hold if len(test_hold) else base.test
    scenario = SplitScenario(
        name="new_node_inductive",
        splits=TemporalSplits(train.sort_by_time(), val.sort_by_time(), test.sort_by_time(), base.val_start_time, base.test_start_time),
        metadata={"holdout_nodes": sorted(holdout), "holdout_ratio": holdout_ratio},
    )
    scenario.splits.assert_no_temporal_leakage()
    return scenario


def new_relation_scenario(dataset: TemporalDataset, val_ratio: float, test_ratio: float, holdout_ratio: float, seed: int) -> SplitScenario:
    base = split_by_time(dataset.edges, val_ratio=val_ratio, test_ratio=test_ratio)
    rng = np.random.default_rng(seed)
    rels = np.arange(dataset.num_relations)
    holdout_count = max(1, int(round(dataset.num_relations * holdout_ratio)))
    holdout = set(rng.choice(rels, size=holdout_count, replace=False).astype(int).tolist())
    train_mask = torch.as_tensor([int(r) not in holdout for r in base.train.rel.tolist()])
    val_mask = torch.as_tensor([int(r) in holdout for r in base.val.rel.tolist()])
    test_mask = torch.as_tensor([int(r) in holdout for r in base.test.rel.tolist()])
    train = base.train.slice(train_mask)
    val = base.val.slice(val_mask) if bool(val_mask.any()) else base.val
    test = base.test.slice(test_mask) if bool(test_mask.any()) else base.test
    scenario = SplitScenario(
        name="new_relation_inductive",
        splits=TemporalSplits(train.sort_by_time(), val.sort_by_time(), test.sort_by_time(), base.val_start_time, base.test_start_time),
        metadata={"holdout_relations": sorted(holdout), "holdout_ratio": holdout_ratio},
    )
    scenario.splits.assert_no_temporal_leakage()
    return scenario


def ood_time_scenario(dataset: TemporalDataset, val_ratio: float, test_ratio: float) -> SplitScenario:
    base = split_by_time(dataset.edges, val_ratio=val_ratio, test_ratio=test_ratio)
    return SplitScenario(
        name="ood_time",
        splits=base,
        metadata={"ood_definition": "test timestamps are later than all train and validation timestamps"},
    )


def hallucination_stress_dataset(dataset: TemporalDataset) -> TemporalDataset:
    stressed_texts = []
    n = dataset.num_nodes
    for idx, text in enumerate(dataset.node_texts):
        stressed_texts.append(f"{text} semantic_alias_group_{n - 1 - idx}")
    return TemporalDataset(
        name=f"{dataset.name}_hallucination_stress",
        num_nodes=dataset.num_nodes,
        num_relations=dataset.num_relations,
        edges=dataset.edges,
        node_texts=stressed_texts,
        relation_texts=dataset.relation_texts,
        node_features=dataset.node_features,
        edge_split=dataset.edge_split,
    )


def build_scenario(dataset: TemporalDataset, scenario_cfg: dict, val_ratio: float, test_ratio: float, seed: int) -> tuple[TemporalDataset, SplitScenario]:
    scenario = str(scenario_cfg.get("name", "standard"))
    if scenario == "standard":
        splits = split_by_time(dataset.edges, val_ratio=val_ratio, test_ratio=test_ratio)
        return dataset, SplitScenario("standard", splits, {})
    if scenario == "ood_time":
        return dataset, ood_time_scenario(dataset, val_ratio, test_ratio)
    if scenario == "new_node":
        return dataset, new_node_scenario(dataset, val_ratio, test_ratio, float(scenario_cfg.get("holdout_ratio", 0.2)), seed)
    if scenario == "new_relation":
        return dataset, new_relation_scenario(dataset, val_ratio, test_ratio, float(scenario_cfg.get("holdout_ratio", 0.25)), seed)
    if scenario == "hallucination_stress":
        stressed = hallucination_stress_dataset(dataset)
        splits = split_by_time(stressed.edges, val_ratio=val_ratio, test_ratio=test_ratio)
        return stressed, SplitScenario("hallucination_stress", splits, {"text_conflict": True})
    raise ValueError(f"unknown scenario: {scenario}")
