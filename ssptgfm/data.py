from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
import os
from typing import Iterable

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class EdgeTensor:
    src: torch.Tensor
    dst: torch.Tensor
    rel: torch.Tensor
    time: torch.Tensor

    def __post_init__(self) -> None:
        n = int(self.src.numel())
        if not (self.dst.numel() == self.rel.numel() == self.time.numel() == n):
            raise ValueError("src, dst, rel, and time must have equal length")

    @classmethod
    def from_arrays(
        cls,
        src: Iterable[int] | np.ndarray | torch.Tensor,
        dst: Iterable[int] | np.ndarray | torch.Tensor,
        rel: Iterable[int] | np.ndarray | torch.Tensor,
        time: Iterable[float] | np.ndarray | torch.Tensor,
    ) -> "EdgeTensor":
        return cls(
            torch.as_tensor(src, dtype=torch.long).cpu(),
            torch.as_tensor(dst, dtype=torch.long).cpu(),
            torch.as_tensor(rel, dtype=torch.long).cpu(),
            torch.as_tensor(time, dtype=torch.float32).cpu(),
        )

    def __len__(self) -> int:
        return int(self.src.numel())

    @property
    def device(self) -> torch.device:
        return self.src.device

    def to(self, device: torch.device | str) -> "EdgeTensor":
        return EdgeTensor(
            self.src.to(device),
            self.dst.to(device),
            self.rel.to(device),
            self.time.to(device),
        )

    def cpu(self) -> "EdgeTensor":
        return self.to("cpu")

    def slice(self, index: slice | torch.Tensor | np.ndarray | list[int]) -> "EdgeTensor":
        return EdgeTensor(self.src[index], self.dst[index], self.rel[index], self.time[index])

    def before(self, cutoff: float, strict: bool = True) -> "EdgeTensor":
        mask = self.time < cutoff if strict else self.time <= cutoff
        return self.slice(mask)

    def sort_by_time(self) -> "EdgeTensor":
        order = torch.argsort(self.time, stable=True)
        return self.slice(order)

    def concat(self, other: "EdgeTensor", sort: bool = True) -> "EdgeTensor":
        merged = EdgeTensor(
            torch.cat([self.src, other.src]),
            torch.cat([self.dst, other.dst]),
            torch.cat([self.rel, other.rel]),
            torch.cat([self.time, other.time]),
        )
        return merged.sort_by_time() if sort else merged

    def unique_nodes(self) -> torch.Tensor:
        return torch.unique(torch.cat([self.src, self.dst]))

    def tuples(self, include_time: bool = True) -> list[tuple[int, ...]]:
        if include_time:
            return [
                (int(s), int(r), int(d), float(t))
                for s, r, d, t in zip(self.src, self.rel, self.dst, self.time)
            ]
        return [(int(s), int(r), int(d)) for s, r, d in zip(self.src, self.rel, self.dst)]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "src": self.src.numpy(),
                "dst": self.dst.numpy(),
                "rel": self.rel.numpy(),
                "time": self.time.numpy(),
            }
        )


@dataclass
class TemporalDataset:
    name: str
    num_nodes: int
    num_relations: int
    edges: EdgeTensor
    node_texts: list[str]
    relation_texts: list[str]
    node_features: np.ndarray | None = None
    edge_split: np.ndarray | None = None
    raw_node_ids: list[int] | None = None
    raw_relation_ids: list[int] | None = None
    source_format: str | None = None
    source_root: str | None = None
    tgb_first_dst_id: int | None = None
    tgb_last_dst_id: int | None = None

    def validate(self) -> None:
        if len(self.node_texts) != self.num_nodes:
            raise ValueError("node_texts length must equal num_nodes")
        if len(self.relation_texts) != self.num_relations:
            raise ValueError("relation_texts length must equal num_relations")
        if len(self.edges) == 0:
            raise ValueError("dataset has no edges")
        if int(self.edges.src.max()) >= self.num_nodes or int(self.edges.dst.max()) >= self.num_nodes:
            raise ValueError("edge endpoint id exceeds num_nodes")
        if int(self.edges.rel.max()) >= self.num_relations:
            raise ValueError("edge relation id exceeds num_relations")
        if self.edge_split is not None and len(self.edge_split) != len(self.edges):
            raise ValueError("edge_split length must equal number of edges")


@dataclass
class TemporalSplits:
    train: EdgeTensor
    val: EdgeTensor
    test: EdgeTensor
    val_start_time: float
    test_start_time: float

    def assert_no_temporal_leakage(self) -> None:
        if len(self.train) and not bool(torch.all(self.train.time < self.val_start_time)):
            raise AssertionError("train contains events at or after validation start time")
        if len(self.val) and not bool(torch.all((self.val.time >= self.val_start_time) & (self.val.time < self.test_start_time))):
            raise AssertionError("validation contains events outside [val_start, test_start)")
        if len(self.test) and not bool(torch.all(self.test.time >= self.test_start_time)):
            raise AssertionError("test contains events before test start time")
        if len(self.train) and len(self.val):
            if float(self.train.time.max()) >= float(self.val.time.min()):
                raise AssertionError("train/validation time order is not strict")
        if len(self.val) and len(self.test):
            if float(self.val.time.max()) >= float(self.test.time.min()):
                raise AssertionError("validation/test time order is not strict")


def split_by_time(
    edges: EdgeTensor,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> TemporalSplits:
    if not (0.0 < val_ratio < 1.0 and 0.0 < test_ratio < 1.0 and val_ratio + test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be positive and sum to < 1")
    edges = edges.sort_by_time()
    unique_times = torch.unique(edges.time).cpu().numpy()
    if unique_times.size < 3:
        raise ValueError("strict temporal split needs at least three unique timestamps")
    train_end_idx = max(1, int(np.floor(unique_times.size * (1.0 - val_ratio - test_ratio))))
    test_start_idx = max(train_end_idx + 1, int(np.floor(unique_times.size * (1.0 - test_ratio))))
    test_start_idx = min(test_start_idx, unique_times.size - 1)
    val_start = float(unique_times[train_end_idx])
    test_start = float(unique_times[test_start_idx])
    train = edges.slice(edges.time < val_start)
    val = edges.slice((edges.time >= val_start) & (edges.time < test_start))
    test = edges.slice(edges.time >= test_start)
    splits = TemporalSplits(train=train, val=val, test=test, val_start_time=val_start, test_start_time=test_start)
    splits.assert_no_temporal_leakage()
    return splits


def split_by_labels(dataset: TemporalDataset) -> TemporalSplits:
    if dataset.edge_split is None:
        raise ValueError("dataset has no edge_split labels")
    labels = np.asarray(dataset.edge_split)
    if np.issubdtype(labels.dtype, np.integer):
        train_mask = labels == 0
        val_mask = labels == 1
        test_mask = labels == 2
    else:
        train_mask = labels == "train"
        val_mask = np.isin(labels, ["val", "valid", "validation"])
        test_mask = labels == "test"
    train = dataset.edges.slice(torch.as_tensor(train_mask))
    val = dataset.edges.slice(torch.as_tensor(val_mask))
    test = dataset.edges.slice(torch.as_tensor(test_mask))
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError("label split requires non-empty train/val/test edges")
    val_start = float(val.time.min())
    test_start = float(test.time.min())
    splits = TemporalSplits(
        train=train.sort_by_time(),
        val=val.sort_by_time(),
        test=test.sort_by_time(),
        val_start_time=val_start,
        test_start_time=test_start,
    )
    splits.assert_no_temporal_leakage()
    return splits


def subsample_train_edges(train: EdgeTensor, ratio: float) -> EdgeTensor:
    if ratio >= 1.0:
        return train
    if not (0.0 < ratio <= 1.0):
        raise ValueError("train ratio must be in (0, 1]")
    train = train.sort_by_time()
    cutoff_count = max(1, int(np.ceil(len(train) * ratio)))
    return train.slice(slice(0, cutoff_count))


def limit_train_edges(train: EdgeTensor, limit: int, mode: str = "prefix") -> EdgeTensor:
    if limit <= 0 or len(train) <= limit:
        return train
    train = train.sort_by_time()
    if mode == "prefix":
        return train.slice(slice(0, int(limit)))
    if mode == "temporal_uniform":
        indices = torch.linspace(0, len(train) - 1, steps=int(limit)).round().long()
        return train.slice(indices)
    raise ValueError(f"unknown train edge limit mode: {mode}")


def load_csv_dataset(data_dir: str | Path, name: str | None = None) -> TemporalDataset:
    data_dir = Path(data_dir)
    edges_path = data_dir / "edges.csv"
    nodes_path = data_dir / "nodes.csv"
    rels_path = data_dir / "relations.csv"
    if not edges_path.exists():
        raise FileNotFoundError(f"missing {edges_path}")
    edges_df = pd.read_csv(edges_path)
    required = {"src", "dst", "rel", "time"}
    if not required.issubset(edges_df.columns):
        raise ValueError(f"edges.csv must contain columns {sorted(required)}")

    node_ids = sorted(set(edges_df["src"]).union(set(edges_df["dst"])))
    rel_ids = sorted(set(edges_df["rel"]))
    node_map = {raw: idx for idx, raw in enumerate(node_ids)}
    rel_map = {raw: idx for idx, raw in enumerate(rel_ids)}

    if nodes_path.exists():
        nodes_df = pd.read_csv(nodes_path)
        if not {"id", "text"}.issubset(nodes_df.columns):
            raise ValueError("nodes.csv must contain columns id,text")
        text_by_id = {row["id"]: str(row["text"]) for _, row in nodes_df.iterrows()}
    else:
        text_by_id = {}
    if rels_path.exists():
        rels_df = pd.read_csv(rels_path)
        if not {"id", "text"}.issubset(rels_df.columns):
            raise ValueError("relations.csv must contain columns id,text")
        rel_text_by_id = {row["id"]: str(row["text"]) for _, row in rels_df.iterrows()}
    else:
        rel_text_by_id = {}

    raw_edge_tensor = EdgeTensor.from_arrays(
        [node_map[x] for x in edges_df["src"]],
        [node_map[x] for x in edges_df["dst"]],
        [rel_map[x] for x in edges_df["rel"]],
        edges_df["time"].astype(float).to_numpy(),
    )
    order = torch.argsort(raw_edge_tensor.time, stable=True)
    edge_tensor = raw_edge_tensor.slice(order)
    edge_split = None
    if "split" in edges_df.columns:
        split_raw = edges_df["split"].astype(str).str.lower().to_numpy()
        edge_split = split_raw[order.numpy()]
    node_texts = [text_by_id.get(raw, f"node {raw}") for raw in node_ids]
    relation_texts = [rel_text_by_id.get(raw, f"relation {raw}") for raw in rel_ids]
    dataset = TemporalDataset(
        name=name or data_dir.name,
        num_nodes=len(node_ids),
        num_relations=len(rel_ids),
        edges=edge_tensor,
        node_texts=node_texts,
        relation_texts=relation_texts,
        edge_split=edge_split,
        raw_node_ids=[int(x) for x in node_ids],
        raw_relation_ids=[int(x) for x in rel_ids],
        source_format="csv",
        source_root=str(data_dir),
    )
    dataset.validate()
    return dataset


def _load_tkgl_icews_csv_fast(name: str, root: str | Path) -> TemporalDataset | None:
    if name != "tkgl-icews":
        return None
    data_dir = Path(root) / "tkgl_icews"
    edge_path = data_dir / "tkgl-icews_edgelist.csv"
    if not edge_path.exists():
        return None

    df = pd.read_csv(
        edge_path,
        usecols=["date", "head", "tail", "relation_type"],
        dtype={"date": np.int64, "head": np.int64, "tail": np.int64, "relation_type": np.int64},
    )
    src0 = df["head"].to_numpy(copy=False)
    dst0 = df["tail"].to_numpy(copy=False)
    rel0 = df["relation_type"].to_numpy(copy=False)
    time0_i64 = df["date"].to_numpy(copy=False)

    num_nodes = int(max(src0.max(), dst0.max())) + 1
    num_base_rels = int(rel0.max()) + 1
    val_time, test_time = np.quantile(time0_i64, [0.70, 0.85])
    split0 = np.empty(time0_i64.shape[0], dtype=np.int8)
    split0[time0_i64 <= val_time] = 0
    split0[(time0_i64 > val_time) & (time0_i64 <= test_time)] = 1
    split0[time0_i64 > test_time] = 2

    src = np.concatenate([src0, dst0]).astype(np.int64, copy=False)
    dst = np.concatenate([dst0, src0]).astype(np.int64, copy=False)
    rel = np.concatenate([rel0, rel0 + num_base_rels]).astype(np.int64, copy=False)
    time = np.concatenate([time0_i64, time0_i64]).astype(np.float32, copy=False)
    edge_split = np.concatenate([split0, split0])
    del df, src0, dst0, rel0, time0_i64, split0
    gc.collect()

    dataset = TemporalDataset(
        name=name,
        num_nodes=num_nodes,
        num_relations=num_base_rels * 2,
        edges=EdgeTensor.from_arrays(src, dst, rel, time),
        node_texts=[f"node {idx}" for idx in range(num_nodes)],
        relation_texts=[f"relation {idx}" for idx in range(num_base_rels * 2)],
        edge_split=edge_split,
        raw_node_ids=list(range(num_nodes)),
        raw_relation_ids=list(range(num_base_rels * 2)),
        source_format="tgb",
        source_root=str(Path(root)),
        tgb_first_dst_id=0,
        tgb_last_dst_id=num_nodes - 1,
    )
    dataset.validate()
    return dataset


def load_tgb_dataset(name: str, root: str | Path = "data/raw/tgb", download: bool = True) -> TemporalDataset:
    fast_dataset = _load_tkgl_icews_csv_fast(name, root)
    if fast_dataset is not None:
        return fast_dataset

    try:
        from tgb.linkproppred.dataset import LinkPropPredDataset
        from tgb.utils.info import PROJ_DIR
    except ImportError as exc:
        raise ImportError("Install py-tgb to use data.format=tgb") from exc

    root_path = Path(root)
    if root_path.is_absolute():
        tgb_root = os.path.relpath(root_path, PROJ_DIR)
    else:
        tgb_root = str(root_path)
    ds = LinkPropPredDataset(name=name, root=tgb_root, download=download, preprocess=True)
    full = ds.full_data
    src = np.asarray(full["sources"], dtype=np.int64)
    dst = np.asarray(full["destinations"], dtype=np.int64)
    time = np.asarray(full["timestamps"], dtype=np.float32)
    edge_type = full.get("edge_type", ds.edge_type)
    if edge_type is None:
        rel = np.zeros_like(src, dtype=np.int64)
    else:
        rel = np.asarray(edge_type).reshape(-1).astype(np.int64)
    split = np.full(src.shape[0], "unassigned", dtype=object)
    split[np.asarray(ds.train_mask, dtype=bool)] = "train"
    split[np.asarray(ds.val_mask, dtype=bool)] = "val"
    split[np.asarray(ds.test_mask, dtype=bool)] = "test"
    if np.any(split == "unassigned"):
        raise ValueError("TGB official split masks do not cover every edge")
    unique_nodes = sorted(set(src.tolist()).union(set(dst.tolist())))
    unique_rels = sorted(set(rel.tolist()))
    node_map = {raw: idx for idx, raw in enumerate(unique_nodes)}
    rel_map = {raw: idx for idx, raw in enumerate(unique_rels)}
    raw_edges = EdgeTensor.from_arrays(
        [node_map[int(x)] for x in src],
        [node_map[int(x)] for x in dst],
        [rel_map[int(x)] for x in rel],
        time,
    )
    order = torch.argsort(raw_edges.time, stable=True)
    edges = raw_edges.slice(order)
    node_texts = [f"node {raw}" for raw in unique_nodes]
    relation_texts = [f"relation {raw}" for raw in unique_rels]
    dataset = TemporalDataset(
        name=name,
        num_nodes=len(unique_nodes),
        num_relations=len(unique_rels),
        edges=edges,
        node_texts=node_texts,
        relation_texts=relation_texts,
        edge_split=split[order.numpy()].astype(str),
        raw_node_ids=[int(x) for x in unique_nodes],
        raw_relation_ids=[int(x) for x in unique_rels],
        source_format="tgb",
        source_root=str(root_path),
        tgb_first_dst_id=int(ds.min_dst_idx),
        tgb_last_dst_id=int(ds.max_dst_idx),
    )
    dataset.validate()
    return dataset


def generate_synthetic_dataset(
    num_nodes: int = 256,
    num_relations: int = 4,
    num_edges: int = 1600,
    num_topics: int = 8,
    seed: int = 1,
) -> TemporalDataset:
    rng = np.random.default_rng(seed)
    node_topic = rng.integers(0, num_topics, size=num_nodes)
    relation_topic = rng.integers(0, num_topics, size=num_relations)
    topic_words = [
        "graph",
        "temporal",
        "semantic",
        "neural",
        "citation",
        "product",
        "event",
        "relation",
    ]
    node_texts = [
        f"node {i} topic {node_topic[i]} {topic_words[node_topic[i] % len(topic_words)]}"
        for i in range(num_nodes)
    ]
    relation_texts = [
        f"relation {r} prefers topic {relation_topic[r]} {topic_words[relation_topic[r] % len(topic_words)]}"
        for r in range(num_relations)
    ]
    src: list[int] = []
    dst: list[int] = []
    rel: list[int] = []
    times: list[float] = []
    topic_to_nodes = {k: np.where(node_topic == k)[0] for k in range(num_topics)}
    for t in range(num_edges):
        r = int(rng.integers(0, num_relations))
        if rng.random() < 0.72:
            topic = int(relation_topic[r])
            pool = topic_to_nodes.get(topic)
            if pool is None or pool.size < 2:
                pool = np.arange(num_nodes)
        else:
            pool = np.arange(num_nodes)
        u, v = rng.choice(pool, size=2, replace=False)
        if rng.random() < 0.12 and src:
            idx = int(rng.integers(0, len(src)))
            u, v, r = src[idx], dst[idx], rel[idx]
        src.append(int(u))
        dst.append(int(v))
        rel.append(int(r))
        times.append(float(t // max(1, num_edges // 80)))
    dataset = TemporalDataset(
        name="synthetic",
        num_nodes=num_nodes,
        num_relations=num_relations,
        edges=EdgeTensor.from_arrays(src, dst, rel, times).sort_by_time(),
        node_texts=node_texts,
        relation_texts=relation_texts,
        raw_node_ids=list(range(num_nodes)),
        raw_relation_ids=list(range(num_relations)),
        source_format="synthetic",
    )
    dataset.validate()
    return dataset
