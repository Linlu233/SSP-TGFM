from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from ssptgfm.data import EdgeTensor
from ssptgfm.features import GraphHistoryIndex


def _bit_width(max_value: int) -> int:
    return max(1, int(max_value).bit_length())


class _FactStore(Protocol):
    def contains(self, s: int, r: int, d: int, t: float | None = None) -> bool:
        ...

    def contains_many(
        self,
        src: np.ndarray,
        rel: int,
        dst: np.ndarray,
        t: float | None = None,
    ) -> np.ndarray:
        ...


@dataclass
class _PackedFactStore:
    keys: np.ndarray
    node_bits: int
    rel_bits: int
    max_node: int
    max_rel: int
    time_values: np.ndarray | None = None
    time_bits: int = 0

    @classmethod
    def from_edges(cls, edges: EdgeTensor, temporal: bool) -> "_FactStore":
        if len(edges) == 0:
            return cls(
                keys=np.asarray([], dtype=np.uint64),
                node_bits=1,
                rel_bits=1,
                max_node=-1,
                max_rel=-1,
                time_values=np.asarray([], dtype=np.float32) if temporal else None,
                time_bits=1 if temporal else 0,
            )
        src_np = edges.src.numpy()
        dst_np = edges.dst.numpy()
        rel_np = edges.rel.numpy()
        if int(src_np.min()) < 0 or int(dst_np.min()) < 0 or int(rel_np.min()) < 0:
            return _TupleFactStore.from_edges(edges, temporal)
        max_node = max(int(src_np.max()), int(dst_np.max()))
        max_rel = int(rel_np.max())
        node_bits = _bit_width(max_node)
        rel_bits = _bit_width(max_rel)
        time_values = None
        time_bits = 0
        if temporal:
            times = edges.time.numpy().astype(np.float32, copy=False)
            time_values, time_ids = np.unique(times, return_inverse=True)
            time_bits = _bit_width(int(time_values.size - 1))
        else:
            time_ids = None
        total_bits = (time_bits if temporal else 0) + rel_bits + 2 * node_bits
        if total_bits > 64:
            return _TupleFactStore.from_edges(edges, temporal)

        rel_u = rel_np.view(np.uint64)
        src_u = src_np.view(np.uint64)
        dst_u = dst_np.view(np.uint64)
        if temporal:
            keys = time_ids.astype(np.uint64, copy=True)
            keys <<= np.uint64(rel_bits)
            np.bitwise_or(keys, rel_u, out=keys)
        else:
            keys = rel_u.copy()
        keys <<= np.uint64(node_bits)
        np.bitwise_or(keys, src_u, out=keys)
        keys <<= np.uint64(node_bits)
        np.bitwise_or(keys, dst_u, out=keys)
        return cls(
            keys=np.unique(keys),
            node_bits=node_bits,
            rel_bits=rel_bits,
            max_node=max_node,
            max_rel=max_rel,
            time_values=time_values,
            time_bits=time_bits,
        )

    def _time_id(self, t: float | None) -> int | None:
        if self.time_values is None:
            return None
        if t is None or self.time_values.size == 0:
            return None
        time_value = np.float32(t)
        idx = int(np.searchsorted(self.time_values, time_value))
        if idx >= self.time_values.size or self.time_values[idx] != time_value:
            return None
        return idx

    def _pack_scalar(self, s: int, r: int, d: int, t: float | None) -> np.uint64 | None:
        if s < 0 or d < 0 or r < 0 or s > self.max_node or d > self.max_node or r > self.max_rel:
            return None
        if self.time_values is not None:
            time_id = self._time_id(t)
            if time_id is None:
                return None
            key = np.uint64(time_id)
            key <<= np.uint64(self.rel_bits)
            key |= np.uint64(r)
        else:
            key = np.uint64(r)
        key <<= np.uint64(self.node_bits)
        key |= np.uint64(s)
        key <<= np.uint64(self.node_bits)
        key |= np.uint64(d)
        return key

    def _pack_many(
        self,
        src: np.ndarray,
        rel: int,
        dst: np.ndarray,
        t: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        src_i = np.asarray(src, dtype=np.int64)
        dst_i = np.asarray(dst, dtype=np.int64)
        valid = (
            (src_i >= 0)
            & (dst_i >= 0)
            & (rel >= 0)
            & (src_i <= self.max_node)
            & (dst_i <= self.max_node)
            & (rel <= self.max_rel)
        )
        keys = np.zeros(src_i.shape, dtype=np.uint64)
        if not np.any(valid):
            return keys, valid
        if self.time_values is not None:
            time_id = self._time_id(t)
            if time_id is None:
                return keys, np.zeros(src_i.shape, dtype=bool)
            keys[valid] = np.uint64(time_id)
            keys[valid] <<= np.uint64(self.rel_bits)
            keys[valid] |= np.uint64(rel)
        else:
            keys[valid] = np.uint64(rel)
        keys[valid] <<= np.uint64(self.node_bits)
        keys[valid] |= src_i[valid].view(np.uint64)
        keys[valid] <<= np.uint64(self.node_bits)
        keys[valid] |= dst_i[valid].view(np.uint64)
        return keys, valid

    def contains(self, s: int, r: int, d: int, t: float | None = None) -> bool:
        key = self._pack_scalar(s, r, d, t)
        if key is None or self.keys.size == 0:
            return False
        idx = int(np.searchsorted(self.keys, key))
        return idx < self.keys.size and bool(self.keys[idx] == key)

    def contains_many(
        self,
        src: np.ndarray,
        rel: int,
        dst: np.ndarray,
        t: float | None = None,
    ) -> np.ndarray:
        keys, valid = self._pack_many(src, rel, dst, t)
        out = np.zeros(keys.shape, dtype=bool)
        if self.keys.size == 0 or not np.any(valid):
            return out
        idx = np.searchsorted(self.keys, keys[valid])
        in_range = idx < self.keys.size
        valid_positions = np.nonzero(valid)[0]
        matched_positions = valid_positions[in_range]
        out[matched_positions] = self.keys[idx[in_range]] == keys[matched_positions]
        return out


@dataclass
class _TupleFactStore:
    values: set[tuple[int, int, int] | tuple[int, int, int, float]]
    temporal: bool

    @classmethod
    def from_edges(cls, edges: EdgeTensor, temporal: bool) -> "_TupleFactStore":
        if temporal:
            return cls(
                {
                    (int(s), int(r), int(d), float(t))
                    for s, r, d, t in zip(edges.src, edges.rel, edges.dst, edges.time)
                },
                temporal=True,
            )
        return cls({(int(s), int(r), int(d)) for s, r, d in zip(edges.src, edges.rel, edges.dst)}, temporal=False)

    def contains(self, s: int, r: int, d: int, t: float | None = None) -> bool:
        if self.temporal:
            return (s, r, d, float(t)) in self.values
        return (s, r, d) in self.values

    def contains_many(
        self,
        src: np.ndarray,
        rel: int,
        dst: np.ndarray,
        t: float | None = None,
    ) -> np.ndarray:
        out = np.zeros(np.asarray(src).shape, dtype=bool)
        for idx, (s, d) in enumerate(zip(src, dst)):
            out[idx] = self.contains(int(s), rel, int(d), t)
        return out


@dataclass
class KnownFacts:
    exact: _FactStore
    timeless: _FactStore
    by_time: dict[float, set[tuple[int, int, int]]]

    @classmethod
    def from_edges(cls, edges: EdgeTensor) -> "KnownFacts":
        return cls(
            exact=_PackedFactStore.from_edges(edges, temporal=True),
            timeless=_PackedFactStore.from_edges(edges, temporal=False),
            by_time={},
        )

    def contains(self, s: int, r: int, d: int, t: float, scope: str = "exact") -> bool:
        if scope == "exact":
            return self.exact.contains(s, r, d, float(t))
        if scope == "timeless":
            return self.timeless.contains(s, r, d)
        if scope == "both":
            return self.exact.contains(s, r, d, float(t)) or self.timeless.contains(s, r, d)
        raise ValueError(f"unknown filter scope: {scope}")

    def contains_many(
        self,
        src: np.ndarray,
        rel: int,
        dst: np.ndarray,
        time: float,
        scope: str = "exact",
    ) -> np.ndarray:
        if scope == "exact":
            return self.exact.contains_many(src, rel, dst, float(time))
        if scope == "timeless":
            return self.timeless.contains_many(src, rel, dst)
        if scope == "both":
            return self.exact.contains_many(src, rel, dst, float(time)) | self.timeless.contains_many(src, rel, dst)
        raise ValueError(f"unknown filter scope: {scope}")

    def filtered_nodes_for_query(
        self,
        src: int,
        rel: int,
        dst: int,
        time: float,
        corrupt_head: bool,
        num_nodes: int,
        scope: str = "exact",
    ) -> tuple[torch.Tensor, int | None]:
        true_index = src if corrupt_head else dst
        nodes = np.arange(num_nodes, dtype=np.int64)
        if corrupt_head:
            cand_src = nodes
            cand_dst = np.full(num_nodes, dst, dtype=np.int64)
        else:
            cand_src = np.full(num_nodes, src, dtype=np.int64)
            cand_dst = nodes
        keep_np = np.ones(num_nodes, dtype=bool)
        if corrupt_head and 0 <= dst < num_nodes:
            keep_np[dst] = False
        elif not corrupt_head and 0 <= src < num_nodes:
            keep_np[src] = False
        not_true = nodes != true_index
        known = self.contains_many(cand_src, rel, cand_dst, time, scope)
        keep_np[not_true & known] = False
        true_pos = None
        if 0 <= true_index < num_nodes and keep_np[true_index]:
            true_pos = int(np.count_nonzero(keep_np[: true_index + 1]) - 1)
        return torch.from_numpy(keep_np), true_pos


class NegativeSampler:
    def __init__(
        self,
        num_nodes: int,
        known_facts: KnownFacts,
        train_edges: EdgeTensor,
        mode: str = "random",
        filter_scope: str = "exact",
        seed: int = 1,
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.known_facts = known_facts
        self.train_seen_nodes = set(train_edges.unique_nodes().cpu().tolist())
        self.mode = mode
        self.filter_scope = filter_scope
        self.rng = np.random.default_rng(seed)
        self.unseen_nodes = np.asarray([n for n in range(self.num_nodes) if n not in self.train_seen_nodes], dtype=np.int64)
        self.rel_heads: dict[int, np.ndarray] = {}
        self.rel_tails: dict[int, np.ndarray] = {}
        for rel in torch.unique(train_edges.rel).cpu().tolist():
            mask = train_edges.rel == int(rel)
            self.rel_heads[int(rel)] = np.unique(train_edges.src[mask].cpu().numpy().astype(np.int64, copy=False))
            self.rel_tails[int(rel)] = np.unique(train_edges.dst[mask].cpu().numpy().astype(np.int64, copy=False))

    def sample(self, positives: EdgeTensor, num_neg_per_pos: int = 1, history: EdgeTensor | None = None) -> EdgeTensor:
        if num_neg_per_pos < 1:
            raise ValueError("num_neg_per_pos must be >= 1")
        hist_index = (
            GraphHistoryIndex(self.num_nodes, history)
            if self.mode == "historical" and history is not None and len(history)
            else None
        )
        src: list[int] = []
        dst: list[int] = []
        rel: list[int] = []
        time: list[float] = []
        for s_t, d_t, r_t, tm_t in zip(positives.src.cpu(), positives.dst.cpu(), positives.rel.cpu(), positives.time.cpu()):
            s = int(s_t)
            d = int(d_t)
            r = int(r_t)
            tm = float(tm_t)
            for _ in range(num_neg_per_pos):
                ns, nd = self._sample_one(s, d, r, tm, hist_index)
                src.append(ns)
                dst.append(nd)
                rel.append(r)
                time.append(tm)
        return EdgeTensor.from_arrays(src, dst, rel, time)

    def sample_candidate_nodes(
        self,
        s: int,
        d: int,
        r: int,
        tm: float,
        *,
        corrupt_head: bool,
        num_negatives: int,
        history_index: GraphHistoryIndex | None = None,
    ) -> list[int]:
        if num_negatives < 1:
            return []
        true_node = s if corrupt_head else d
        nodes: list[int] = []
        seen = {int(true_node)}
        for _ in range(max(200, num_negatives * 50)):
            node = self._sample_candidate_node(s, d, r, tm, corrupt_head, history_index)
            if node in seen:
                continue
            ns, nd = (node, d) if corrupt_head else (s, node)
            if ns == nd:
                continue
            if self.known_facts.contains(ns, r, nd, tm, self.filter_scope):
                continue
            seen.add(node)
            nodes.append(node)
            if len(nodes) >= num_negatives:
                return nodes
        for node in range(self.num_nodes):
            if node in seen:
                continue
            ns, nd = (node, d) if corrupt_head else (s, node)
            if ns == nd:
                continue
            if self.known_facts.contains(ns, r, nd, tm, self.filter_scope):
                continue
            seen.add(node)
            nodes.append(int(node))
            if len(nodes) >= num_negatives:
                break
        return nodes

    def _sample_one(
        self,
        s: int,
        d: int,
        r: int,
        tm: float,
        hist_index: GraphHistoryIndex | None,
    ) -> tuple[int, int]:
        for _ in range(200):
            corrupt_head = bool(self.rng.random() < 0.5)
            if self.mode == "historical" and hist_index is not None and hist_index.observed_pairs:
                hs, hd = hist_index.observed_pairs[int(self.rng.integers(0, len(hist_index.observed_pairs)))]
                cand = (hs, d) if corrupt_head else (s, hd)
            elif self.mode == "inductive":
                node = int(self.rng.choice(self.unseen_nodes)) if self.unseen_nodes.size else int(self.rng.integers(0, self.num_nodes))
                cand = (node, d) if corrupt_head else (s, node)
            elif self.mode == "random":
                node = int(self.rng.integers(0, self.num_nodes))
                cand = (node, d) if corrupt_head else (s, node)
                ns, nd = cand
                if ns != nd:
                    return ns, nd
            elif self.mode == "filtered":
                node = self._sample_filtered_candidate_node(s, d, r, tm, corrupt_head)
                cand = (node, d) if corrupt_head else (s, node)
            elif self.mode == "relation_hard":
                node = self._sample_relation_hard_candidate_node(s, d, r, tm, corrupt_head)
                cand = (node, d) if corrupt_head else (s, node)
            else:
                raise ValueError(f"unknown negative mode: {self.mode}")
            ns, nd = cand
            if ns == nd:
                continue
            if not self.known_facts.contains(ns, r, nd, tm, self.filter_scope):
                return ns, nd
        for node in range(self.num_nodes):
            cand = (s, node)
            if node != s and not self.known_facts.contains(cand[0], r, cand[1], tm, self.filter_scope):
                return cand
        raise RuntimeError("failed to sample a filtered negative")

    def _sample_candidate_node(
        self,
        s: int,
        d: int,
        r: int,
        tm: float,
        corrupt_head: bool,
        hist_index: GraphHistoryIndex | None,
    ) -> int:
        if self.mode == "historical" and hist_index is not None and hist_index.observed_pairs:
            hs, hd = hist_index.observed_pairs[int(self.rng.integers(0, len(hist_index.observed_pairs)))]
            return int(hs if corrupt_head else hd)
        if self.mode == "inductive":
            return int(self.rng.choice(self.unseen_nodes)) if self.unseen_nodes.size else int(self.rng.integers(0, self.num_nodes))
        if self.mode == "relation_hard":
            pool = self.rel_heads.get(r) if corrupt_head else self.rel_tails.get(r)
            if pool is not None and pool.size:
                return int(pool[int(self.rng.integers(0, pool.size))])
        return int(self.rng.integers(0, self.num_nodes))

    def _sample_filtered_candidate_node(self, s: int, d: int, r: int, tm: float, corrupt_head: bool) -> int:
        for _ in range(200):
            node = int(self.rng.integers(0, self.num_nodes))
            ns, nd = (node, d) if corrupt_head else (s, node)
            if ns == nd:
                continue
            if not self.known_facts.contains(ns, r, nd, tm, self.filter_scope):
                return node
        for node in range(self.num_nodes):
            ns, nd = (node, d) if corrupt_head else (s, node)
            if ns != nd and not self.known_facts.contains(ns, r, nd, tm, self.filter_scope):
                return int(node)
        return int(self.rng.integers(0, self.num_nodes))

    def _sample_relation_hard_candidate_node(self, s: int, d: int, r: int, tm: float, corrupt_head: bool) -> int:
        pool = self.rel_heads.get(r) if corrupt_head else self.rel_tails.get(r)
        if pool is not None and pool.size:
            for _ in range(200):
                node = int(pool[int(self.rng.integers(0, pool.size))])
                ns, nd = (node, d) if corrupt_head else (s, node)
                if ns == nd:
                    continue
                if not self.known_facts.contains(ns, r, nd, tm, self.filter_scope):
                    return node
        return self._sample_filtered_candidate_node(s, d, r, tm, corrupt_head)


def make_labels(num_pos: int, num_neg: int, device: torch.device | str) -> torch.Tensor:
    return torch.cat(
        [
            torch.ones(num_pos, dtype=torch.float32, device=device),
            torch.zeros(num_neg, dtype=torch.float32, device=device),
        ]
    )
