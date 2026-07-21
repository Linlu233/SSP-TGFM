from __future__ import annotations

from collections import defaultdict
import math

import torch

from ssptgfm.data import EdgeTensor

HISTORY_PRIOR_FEATURE_DIM = 64


class GraphHistoryIndex:
    """Causal structural feature index built only from history edges."""

    def __init__(self, num_nodes: int, history: EdgeTensor, ppr_alpha: float = 0.15, ppr_steps: int = 20):
        self.num_nodes = int(num_nodes)
        self.ppr_alpha = float(ppr_alpha)
        self.ppr_steps = int(ppr_steps)
        self.adj: list[set[int]] = [set() for _ in range(num_nodes)]
        self.deg = torch.zeros(num_nodes, dtype=torch.float32)
        self.last_pair_time: dict[tuple[int, int], float] = {}
        self.last_direct_pair_time: dict[tuple[int, int], float] = {}
        self.last_triple_time: dict[tuple[int, int, int], float] = {}
        self.direct_pair_count: dict[tuple[int, int], int] = defaultdict(int)
        self.undirected_pair_count: dict[tuple[int, int], int] = defaultdict(int)
        self.triple_count: dict[tuple[int, int, int], int] = defaultdict(int)
        self.sr_count: dict[tuple[int, int], int] = defaultdict(int)
        self.rd_count: dict[tuple[int, int], int] = defaultdict(int)
        self.rel_count: dict[int, int] = defaultdict(int)
        self.out_count: dict[int, int] = defaultdict(int)
        self.in_count: dict[int, int] = defaultdict(int)
        self.last_sr_time: dict[tuple[int, int], float] = {}
        self.last_rd_time: dict[tuple[int, int], float] = {}
        self.last_rel_time: dict[int, float] = {}
        self.last_out_time: dict[int, float] = {}
        self.last_in_time: dict[int, float] = {}
        self.time_min: float | None = None
        self.time_max: float | None = None
        self.total_events = 0
        self.observed_pairs: list[tuple[int, int]] = []
        self._transition_cache: dict[str, torch.Tensor] = {}
        self._dense_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._ppr_row_cache: dict[str, dict[int, torch.Tensor]] = {}
        self._ppr_col_cache: dict[str, dict[int, torch.Tensor]] = {}
        self.add_edges(history)

    def add_edges(self, edges: EdgeTensor) -> None:
        edges = edges.cpu()
        if len(edges) == 0:
            return
        self._transition_cache.clear()
        self._dense_cache.clear()
        self._ppr_row_cache.clear()
        self._ppr_col_cache.clear()
        for s_t, d_t, r_t, t_t in zip(edges.src, edges.dst, edges.rel, edges.time):
            s = int(s_t)
            d = int(d_t)
            r = int(r_t)
            t = float(t_t)
            self.time_min = t if self.time_min is None else min(self.time_min, t)
            self.time_max = t if self.time_max is None else max(self.time_max, t)
            if d not in self.adj[s]:
                self.adj[s].add(d)
                self.adj[d].add(s)
                self.deg[s] += 1.0
                self.deg[d] += 1.0
                self.observed_pairs.append((s, d))
                self.observed_pairs.append((d, s))
            key = (min(s, d), max(s, d))
            self.last_pair_time[key] = max(t, self.last_pair_time.get(key, -1.0))
            self.last_direct_pair_time[(s, d)] = max(t, self.last_direct_pair_time.get((s, d), -1.0))
            self.last_triple_time[(s, r, d)] = max(t, self.last_triple_time.get((s, r, d), -1.0))
            self.direct_pair_count[(s, d)] += 1
            self.undirected_pair_count[key] += 1
            self.triple_count[(s, r, d)] += 1
            self.sr_count[(s, r)] += 1
            self.rd_count[(r, d)] += 1
            self.rel_count[r] += 1
            self.out_count[s] += 1
            self.in_count[d] += 1
            self.last_sr_time[(s, r)] = max(t, self.last_sr_time.get((s, r), -1.0))
            self.last_rd_time[(r, d)] = max(t, self.last_rd_time.get((r, d), -1.0))
            self.last_rel_time[r] = max(t, self.last_rel_time.get(r, -1.0))
            self.last_out_time[s] = max(t, self.last_out_time.get(s, -1.0))
            self.last_in_time[d] = max(t, self.last_in_time.get(d, -1.0))
            self.total_events += 1

    @staticmethod
    def _recency(last_time: float | None, query_time: float) -> float:
        if last_time is None:
            return 0.0
        delta = max(0.0, float(query_time) - float(last_time))
        return 1.0 / (1.0 + math.log1p(delta))

    @staticmethod
    def _time_decay(last_time: float | None, query_time: float, tau: float) -> float:
        if last_time is None:
            return 0.0
        delta = max(0.0, float(query_time) - float(last_time))
        return math.exp(-delta / max(float(tau), 1.0))

    def _history_prior_row(self, s: int, d: int, r: int, t: float) -> list[float]:
        exact = float(self.triple_count.get((s, r, d), 0))
        reverse_exact = float(self.triple_count.get((d, r, s), 0))
        direct_pair = float(self.direct_pair_count.get((s, d), 0))
        reverse_pair = float(self.direct_pair_count.get((d, s), 0))
        undirected_pair = float(self.undirected_pair_count.get((min(s, d), max(s, d)), 0))
        sr = float(self.sr_count.get((s, r), 0))
        rd = float(self.rd_count.get((r, d), 0))
        rel = float(self.rel_count.get(r, 0))
        out_count = float(self.out_count.get(s, 0))
        in_count = float(self.in_count.get(d, 0))
        total = float(max(1, self.total_events))
        sr_tail = exact
        rd_head = exact
        rel_tail = rd
        rel_head = sr
        exact_given_sr = exact / max(1.0, sr)
        exact_given_rd = exact / max(1.0, rd)
        exact_given_pair = exact / max(1.0, direct_pair)
        sr_rd_overlap = math.sqrt(max(0.0, sr * rd)) / max(1.0, rel)
        sr_given_rel = sr / max(1.0, rel)
        rd_given_rel = rd / max(1.0, rel)
        pair_given_sr = direct_pair / max(1.0, sr)
        pair_given_rd = direct_pair / max(1.0, rd)
        exact_recency = self._recency(self.last_triple_time.get((s, r, d)), t)
        pair_recency = self._recency(self.last_direct_pair_time.get((s, d)), t)
        reverse_pair_recency = self._recency(self.last_direct_pair_time.get((d, s)), t)
        exact_fast_decay = self._time_decay(self.last_triple_time.get((s, r, d)), t, tau=86400.0 * 7.0)
        exact_slow_decay = self._time_decay(self.last_triple_time.get((s, r, d)), t, tau=86400.0 * 30.0)
        pair_slow_decay = self._time_decay(self.last_direct_pair_time.get((s, d)), t, tau=86400.0 * 30.0)
        reverse_pair_slow_decay = self._time_decay(self.last_direct_pair_time.get((d, s)), t, tau=86400.0 * 30.0)
        span = max(1.0, float((self.time_max or t) - (self.time_min or t)))
        exact_span_fast_decay = self._time_decay(self.last_triple_time.get((s, r, d)), t, tau=span / 100.0)
        exact_span_mid_decay = self._time_decay(self.last_triple_time.get((s, r, d)), t, tau=span / 20.0)
        exact_span_slow_decay = self._time_decay(self.last_triple_time.get((s, r, d)), t, tau=span / 5.0)
        pair_span_mid_decay = self._time_decay(self.last_direct_pair_time.get((s, d)), t, tau=span / 20.0)
        sr_recency = self._recency(self.last_sr_time.get((s, r)), t)
        rd_recency = self._recency(self.last_rd_time.get((r, d)), t)
        rel_recency = self._recency(self.last_rel_time.get(r), t)
        node_out_recency = self._recency(self.last_out_time.get(s), t)
        node_in_recency = self._recency(self.last_in_time.get(d), t)
        log_rel = math.log1p(rel)
        log_sr = math.log1p(sr)
        log_rd = math.log1p(rd)
        log_out = math.log1p(out_count)
        log_in = math.log1p(in_count)
        source_rel_share = log_sr / max(1.0, log_rel)
        dest_rel_share = log_rd / max(1.0, log_rel)
        source_rel_pmi = math.log((sr + 1.0) * (total + 1.0) / max(1.0, (out_count + 1.0) * (rel + 1.0)))
        dest_rel_pmi = math.log((rd + 1.0) * (total + 1.0) / max(1.0, (in_count + 1.0) * (rel + 1.0)))
        pair_pmi = math.log((direct_pair + 1.0) * (total + 1.0) / max(1.0, (out_count + 1.0) * (in_count + 1.0)))
        reverse_pair_pmi = math.log((reverse_pair + 1.0) * (total + 1.0) / max(1.0, (in_count + 1.0) * (out_count + 1.0)))
        exact_given_undirected = exact / max(1.0, undirected_pair)
        reverse_exact_given_reverse_pair = reverse_exact / max(1.0, reverse_pair)
        role_match_strength = math.sqrt(max(0.0, source_rel_share * dest_rel_share))
        role_count_product = log_sr * log_rd / max(1.0, log_rel)
        source_role_span_decay = self._time_decay(self.last_sr_time.get((s, r)), t, tau=span / 20.0)
        dest_role_span_decay = self._time_decay(self.last_rd_time.get((r, d)), t, tau=span / 20.0)
        source_activity_inverse = 1.0 / math.sqrt(1.0 + out_count)
        dest_activity_inverse = 1.0 / math.sqrt(1.0 + in_count)
        source_tail_balance = log_out / max(1.0, log_in + log_out)
        dest_head_balance = log_in / max(1.0, log_in + log_out)
        return [
            math.log1p(exact),
            math.log1p(direct_pair),
            math.log1p(undirected_pair),
            math.log1p(sr),
            math.log1p(rd),
            math.log1p(rel),
            exact_given_sr,
            exact_given_rd,
            exact_given_pair,
            exact_recency,
            pair_recency,
            math.log1p(reverse_exact),
            1.0 if exact > 0.0 else 0.0,
            1.0 if direct_pair > 0.0 else 0.0,
            math.log1p(reverse_pair),
            reverse_pair_recency,
            sr_given_rel,
            rd_given_rel,
            pair_given_sr,
            pair_given_rd,
            exact_fast_decay,
            exact_slow_decay,
            pair_slow_decay,
            reverse_pair_slow_decay,
            sr_rd_overlap,
            exact / total,
            sr / total,
            rd / total,
            math.log1p(out_count),
            math.log1p(in_count),
            math.log1p(out_count + in_count),
            out_count / total,
            in_count / total,
            sr_recency,
            rd_recency,
            rel_recency,
            node_out_recency,
            node_in_recency,
            exact_span_fast_decay,
            exact_span_mid_decay,
            exact_span_slow_decay,
            pair_span_mid_decay,
            math.log1p(sr_tail),
            math.log1p(rd_head),
            math.log1p(rel_tail),
            math.log1p(rel_head),
            sr_tail / max(1.0, sr),
            rd_head / max(1.0, rd),
            source_rel_share,
            dest_rel_share,
            source_rel_pmi,
            dest_rel_pmi,
            pair_pmi,
            reverse_pair_pmi,
            exact_given_undirected,
            reverse_exact_given_reverse_pair,
            role_match_strength,
            role_count_product,
            source_role_span_decay,
            dest_role_span_decay,
            source_activity_inverse,
            dest_activity_inverse,
            source_tail_balance,
            dest_head_balance,
        ]

    @staticmethod
    def _fit_history_prior_dim(rows: list[list[float]], feature_dim: int) -> list[list[float]]:
        if feature_dim == HISTORY_PRIOR_FEATURE_DIM:
            return rows
        if feature_dim < HISTORY_PRIOR_FEATURE_DIM:
            return [row[:feature_dim] for row in rows]
        return [row + [0.0] * (feature_dim - HISTORY_PRIOR_FEATURE_DIM) for row in rows]

    def history_prior_features_for_edges(
        self,
        edges: EdgeTensor,
        device: torch.device | str,
        feature_dim: int = HISTORY_PRIOR_FEATURE_DIM,
    ) -> torch.Tensor:
        device = torch.device(device)
        if len(edges) == 0:
            return torch.empty(0, feature_dim, dtype=torch.float32, device=device)
        rows = [
            self._history_prior_row(int(s), int(d), int(r), float(t))
            for s, d, r, t in zip(edges.src.cpu(), edges.dst.cpu(), edges.rel.cpu(), edges.time.cpu())
        ]
        rows = self._fit_history_prior_dim(rows, feature_dim)
        return torch.tensor(rows, dtype=torch.float32, device=device)

    def history_prior_features_for_candidates(
        self,
        candidates: list[tuple[int, int, int, float]],
        device: torch.device | str,
        feature_dim: int = HISTORY_PRIOR_FEATURE_DIM,
    ) -> torch.Tensor:
        device = torch.device(device)
        if not candidates:
            return torch.empty(0, feature_dim, dtype=torch.float32, device=device)
        rows = [self._history_prior_row(int(s), int(d), int(r), float(t)) for s, d, r, t in candidates]
        rows = self._fit_history_prior_dim(rows, feature_dim)
        return torch.tensor(rows, dtype=torch.float32, device=device)

    def _transition_sparse(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        cached = self._transition_cache.get(key)
        if cached is not None:
            return cached
        if not self.observed_pairs:
            idx = torch.empty(2, 0, dtype=torch.long, device=device)
            values = torch.empty(0, dtype=torch.float32, device=device)
        else:
            src = torch.tensor([x[0] for x in self.observed_pairs], dtype=torch.long, device=device)
            dst = torch.tensor([x[1] for x in self.observed_pairs], dtype=torch.long, device=device)
            deg = self.deg.to(device).clamp_min(1.0)
            idx = torch.stack([src, dst], dim=0)
            values = 1.0 / deg[src]
        transition = torch.sparse_coo_tensor(
            idx,
            values,
            (self.num_nodes, self.num_nodes),
            device=device,
        ).coalesce()
        self._transition_cache[key] = transition
        return transition

    def _dense_adj_transition(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = str(device)
        cached = self._dense_cache.get(key)
        if cached is not None:
            return cached
        adj = torch.zeros(self.num_nodes, self.num_nodes, dtype=torch.float32, device=device)
        for node, neigh in enumerate(self.adj):
            if neigh:
                idx = torch.tensor(list(neigh), dtype=torch.long, device=device)
                adj[node, idx] = 1.0
        deg = self.deg.to(device)
        transition = torch.zeros_like(adj)
        transition[deg > 0] = adj[deg > 0] / deg[deg > 0].unsqueeze(-1)
        isolated = deg == 0
        if bool(isolated.any()):
            idx = torch.nonzero(isolated, as_tuple=False).view(-1)
            transition[idx, idx] = 1.0
        self._dense_cache[key] = (adj, transition, deg)
        return adj, transition, deg

    def _sparse_ppr_for_pairs(
        self,
        sources: torch.Tensor,
        dests: torch.Tensor,
        device: torch.device | str,
        source_chunk_size: int = 512,
    ) -> torch.Tensor:
        device = torch.device(device)
        if sources.numel() == 0:
            return torch.empty(0, dtype=torch.float32, device=device)
        sources = sources.to(device)
        dests = dests.to(device)
        unique_sources = torch.unique(sources, sorted=True)
        unique_dests = torch.unique(dests, sorted=True)
        if unique_dests.numel() <= 8 and unique_sources.numel() > unique_dests.numel() * 4:
            return self._sparse_ppr_to_dests(sources, dests, device)
        out = torch.zeros(sources.numel(), dtype=torch.float32, device=device)
        unique_sources, inverse = torch.unique(sources, sorted=True, return_inverse=True)
        transition_t = self._transition_sparse(device).transpose(0, 1).coalesce()
        cache = self._ppr_row_cache.setdefault(str(device), {})
        if device.type == "cuda":
            total_memory = torch.cuda.get_device_properties(device).total_memory
            cache_bytes = min(256 * 1024**2, max(64 * 1024**2, total_memory // 256))
        else:
            cache_bytes = 256 * 1024**2
        max_cache_rows = max(1, min(self.num_nodes, cache_bytes // max(1, self.num_nodes * 4)))
        missing_sources: list[int] = []
        missing_positions: list[int] = []
        for pos, src_t in enumerate(unique_sources.detach().cpu().tolist()):
            cached = cache.get(int(src_t))
            if cached is None:
                missing_sources.append(int(src_t))
                missing_positions.append(pos)
                continue
            mask = inverse == pos
            if bool(mask.any()):
                out[mask] = cached[dests[mask]]

        for start in range(0, len(missing_sources), source_chunk_size):
            end = min(start + source_chunk_size, len(missing_sources))
            src_chunk = torch.tensor(missing_sources[start:end], dtype=torch.long, device=device)
            rows = torch.arange(src_chunk.numel(), device=device)
            prob = torch.zeros(src_chunk.numel(), self.num_nodes, dtype=torch.float32, device=device)
            prob[rows, src_chunk] = 1.0
            restart = prob.clone()
            for _ in range(self.ppr_steps):
                prob = (1.0 - self.ppr_alpha) * torch.sparse.mm(transition_t, prob.T).T
                prob = prob + self.ppr_alpha * restart
            for local_row, unique_pos in enumerate(missing_positions[start:end]):
                mask = inverse == unique_pos
                if bool(mask.any()):
                    out[mask] = prob[local_row, dests[mask]]
                if len(cache) < max_cache_rows:
                    cache[int(src_chunk[local_row])] = prob[local_row].detach().clone()
        out = torch.where(sources == dests, torch.ones_like(out), out)
        return out

    def _sparse_ppr_to_dests(
        self,
        sources: torch.Tensor,
        dests: torch.Tensor,
        device: torch.device,
        dest_chunk_size: int = 8,
    ) -> torch.Tensor:
        sources = sources.to(device)
        dests = dests.to(device)
        out = torch.zeros(sources.numel(), dtype=torch.float32, device=device)
        unique_dests, inverse = torch.unique(dests, sorted=True, return_inverse=True)
        transition = self._transition_sparse(device)
        cache = self._ppr_col_cache.setdefault(str(device), {})
        if device.type == "cuda":
            total_memory = torch.cuda.get_device_properties(device).total_memory
            cache_bytes = min(256 * 1024**2, max(64 * 1024**2, total_memory // 256))
        else:
            cache_bytes = 256 * 1024**2
        max_cache_cols = max(1, min(self.num_nodes, cache_bytes // max(1, self.num_nodes * 4)))
        missing_dests: list[int] = []
        missing_positions: list[int] = []
        for pos, dst_t in enumerate(unique_dests.detach().cpu().tolist()):
            cached = cache.get(int(dst_t))
            if cached is None:
                missing_dests.append(int(dst_t))
                missing_positions.append(pos)
                continue
            mask = inverse == pos
            if bool(mask.any()):
                out[mask] = cached[sources[mask]]

        for start in range(0, len(missing_dests), dest_chunk_size):
            end = min(start + dest_chunk_size, len(missing_dests))
            dst_chunk = torch.tensor(missing_dests[start:end], dtype=torch.long, device=device)
            cols = torch.arange(dst_chunk.numel(), device=device)
            prob = torch.zeros(self.num_nodes, dst_chunk.numel(), dtype=torch.float32, device=device)
            prob[dst_chunk, cols] = 1.0
            score = torch.zeros_like(prob)
            decay = 1.0
            for _ in range(self.ppr_steps):
                score = score + float(self.ppr_alpha) * decay * prob
                prob = torch.sparse.mm(transition, prob)
                decay *= 1.0 - float(self.ppr_alpha)
            score = score + decay * prob
            for local_col, unique_pos in enumerate(missing_positions[start:end]):
                mask = inverse == unique_pos
                if bool(mask.any()):
                    out[mask] = score[sources[mask], local_col]
                if len(cache) < max_cache_cols:
                    cache[int(dst_chunk[local_col])] = score[:, local_col].detach().clone()
        out = torch.where(sources == dests, torch.ones_like(out), out)
        return out

    def ppr(self, src: int, dst: int) -> float:
        if src == dst:
            return 1.0
        if not self.adj[src] or not self.adj[dst]:
            return 0.0
        prob = {src: 1.0}
        restart = {src: 1.0}
        for _ in range(self.ppr_steps):
            nxt: dict[int, float] = {}
            for node, mass in prob.items():
                neigh = self.adj[node]
                if not neigh:
                    nxt[node] = nxt.get(node, 0.0) + (1.0 - self.ppr_alpha) * mass
                    continue
                share = (1.0 - self.ppr_alpha) * mass / len(neigh)
                for other in neigh:
                    nxt[other] = nxt.get(other, 0.0) + share
            for node, mass in restart.items():
                nxt[node] = nxt.get(node, 0.0) + self.ppr_alpha * mass
            prob = nxt
        return float(prob.get(dst, 0.0))

    def ppr_vector(self, src: int) -> dict[int, float]:
        if not self.adj[src]:
            return {}
        prob = {src: 1.0}
        restart = {src: 1.0}
        for _ in range(self.ppr_steps):
            nxt: dict[int, float] = {}
            for node, mass in prob.items():
                neigh = self.adj[node]
                if not neigh:
                    nxt[node] = nxt.get(node, 0.0) + (1.0 - self.ppr_alpha) * mass
                    continue
                share = (1.0 - self.ppr_alpha) * mass / len(neigh)
                for other in neigh:
                    nxt[other] = nxt.get(other, 0.0) + share
            for node, mass in restart.items():
                nxt[node] = nxt.get(node, 0.0) + self.ppr_alpha * mass
            prob = nxt
        return prob

    def features_for_edges(self, edges: EdgeTensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats: list[list[float]] = []
        ood: list[float] = []
        ppr_cache: dict[int, dict[int, float]] = {}
        for s_t, d_t, t_t in zip(edges.src.cpu(), edges.dst.cpu(), edges.time.cpu()):
            s = int(s_t)
            d = int(d_t)
            t = float(t_t)
            deg_s = float(self.deg[s])
            deg_d = float(self.deg[d])
            neigh_s = self.adj[s]
            neigh_d = self.adj[d]
            cn = float(len(neigh_s.intersection(neigh_d)))
            direct = 1.0 if d in neigh_s else 0.0
            if s == d:
                ppr_uv = 1.0
            else:
                if s not in ppr_cache:
                    ppr_cache[s] = self.ppr_vector(s)
                ppr_uv = float(ppr_cache[s].get(d, 0.0))
            last_t = self.last_pair_time.get((min(s, d), max(s, d)))
            if last_t is None:
                delta_t = t + 1.0
            else:
                delta_t = max(0.0, t - last_t)
            feats.append(
                [
                    float(ppr_uv),
                    float(cn),
                    float(deg_s),
                    float(deg_d),
                    float(delta_t),
                ]
            )
            ood.append(1.0 if (deg_s == 0.0 or deg_d == 0.0 or (cn == 0.0 and direct == 0.0)) else 0.0)
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(ood, dtype=torch.float32).unsqueeze(-1)

    def features_for_edges_sparse_ppr(
        self,
        edges: EdgeTensor,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device)
        if len(edges) == 0:
            return torch.empty(0, 5, dtype=torch.float32, device=device), torch.empty(0, 1, dtype=torch.float32, device=device)
        edges_cpu = edges.cpu()
        sources = edges_cpu.src
        dests = edges_cpu.dst
        ppr_uv = self._sparse_ppr_for_pairs(sources, dests, device)
        deg = self.deg.to(device)
        deg_s = deg[sources.to(device)]
        deg_d = deg[dests.to(device)]
        cn: list[float] = []
        direct: list[float] = []
        delta_t: list[float] = []
        for s_t, d_t, t_t in zip(edges_cpu.src, edges_cpu.dst, edges_cpu.time):
            s = int(s_t)
            d = int(d_t)
            t = float(t_t)
            neigh_s = self.adj[s]
            neigh_d = self.adj[d]
            cn.append(float(len(neigh_s.intersection(neigh_d))))
            direct.append(1.0 if d in neigh_s else 0.0)
            last_t = self.last_pair_time.get((min(s, d), max(s, d)))
            delta_t.append(t + 1.0 if last_t is None else max(0.0, t - last_t))
        cn_t = torch.tensor(cn, dtype=torch.float32, device=device)
        direct_t = torch.tensor(direct, dtype=torch.float32, device=device)
        delta = torch.tensor(delta_t, dtype=torch.float32, device=device)
        feats = torch.stack([ppr_uv, cn_t, deg_s, deg_d, delta], dim=-1)
        ood = ((deg_s == 0.0) | (deg_d == 0.0) | ((cn_t == 0.0) & (direct_t == 0.0))).float().unsqueeze(-1)
        return feats, ood

    def features_for_candidates(
        self,
        src: int,
        dst: int,
        candidates: list[tuple[int, int, int, float]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ppr_cache: dict[int, dict[int, float]] = {}
        feats: list[list[float]] = []
        ood: list[float] = []
        for cand_s, cand_d, _, cand_t in candidates:
            s = int(cand_s)
            d = int(cand_d)
            t = float(cand_t)
            deg_s = float(self.deg[s])
            deg_d = float(self.deg[d])
            neigh_s = self.adj[s]
            neigh_d = self.adj[d]
            cn = float(len(neigh_s.intersection(neigh_d)))
            direct = 1.0 if d in neigh_s else 0.0
            if s == d:
                ppr_uv = 1.0
            else:
                if s not in ppr_cache:
                    ppr_cache[s] = self.ppr_vector(s)
                ppr_uv = float(ppr_cache[s].get(d, 0.0))
            last_t = self.last_pair_time.get((min(s, d), max(s, d)))
            if last_t is None:
                delta_t = t + 1.0
            else:
                delta_t = max(0.0, t - last_t)
            feats.append([ppr_uv, cn, deg_s, deg_d, float(delta_t)])
            ood.append(1.0 if (deg_s == 0.0 or deg_d == 0.0 or (cn == 0.0 and direct == 0.0)) else 0.0)
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(ood, dtype=torch.float32).unsqueeze(-1)

    def features_for_candidates_sparse_ppr(
        self,
        candidates: list[tuple[int, int, int, float]],
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not candidates:
            device = torch.device(device)
            return torch.empty(0, 5, dtype=torch.float32, device=device), torch.empty(0, 1, dtype=torch.float32, device=device)
        batch = EdgeTensor.from_arrays(
            [x[0] for x in candidates],
            [x[1] for x in candidates],
            [x[2] for x in candidates],
            [x[3] for x in candidates],
        )
        return self.features_for_edges_sparse_ppr(batch, device)

    def features_for_candidates_tensor(
        self,
        candidates: list[tuple[int, int, int, float]],
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = torch.device(device)
        if not candidates:
            return torch.empty(0, 5, device=device), torch.empty(0, 1, device=device)
        sources = torch.tensor([x[0] for x in candidates], dtype=torch.long, device=device)
        dests = torch.tensor([x[1] for x in candidates], dtype=torch.long, device=device)
        times = [float(x[3]) for x in candidates]
        adj, transition, deg = self._dense_adj_transition(device)
        deg_s = deg[sources]
        deg_d = deg[dests]
        cn = (adj[sources] * adj[dests]).sum(dim=-1)
        direct = adj[sources, dests]

        unique_sources, inverse = torch.unique(sources, sorted=True, return_inverse=True)
        prob = torch.zeros(unique_sources.numel(), self.num_nodes, dtype=torch.float32, device=device)
        prob[torch.arange(unique_sources.numel(), device=device), unique_sources] = 1.0
        restart = prob.clone()
        for _ in range(self.ppr_steps):
            prob = (1.0 - self.ppr_alpha) * (prob @ transition)
            prob = prob + self.ppr_alpha * restart
        ppr_uv = prob[inverse, dests]
        ppr_uv = torch.where(sources == dests, torch.ones_like(ppr_uv), ppr_uv)

        delta_t = []
        for s_t, d_t, tm in zip(sources.detach().cpu().tolist(), dests.detach().cpu().tolist(), times):
            last_t = self.last_pair_time.get((min(s_t, d_t), max(s_t, d_t)))
            delta_t.append(tm + 1.0 if last_t is None else max(0.0, tm - last_t))
        delta = torch.tensor(delta_t, dtype=torch.float32, device=device)
        feats = torch.stack([ppr_uv, cn, deg_s, deg_d, delta], dim=-1)
        ood = ((deg_s == 0.0) | (deg_d == 0.0) | ((cn == 0.0) & (direct == 0.0))).float().unsqueeze(-1)
        return feats, ood


def relation_frequency_ood(train: EdgeTensor, query: EdgeTensor, num_relations: int) -> torch.Tensor:
    counts = torch.bincount(train.rel.cpu(), minlength=num_relations).float()
    return relation_frequency_ood_from_counts(counts, query, num_relations)


def relation_frequency_ood_from_counts(counts: torch.Tensor, query: EdgeTensor, num_relations: int) -> torch.Tensor:
    counts = counts.detach().cpu().float()
    if counts.numel() < num_relations:
        counts = torch.nn.functional.pad(counts, (0, num_relations - counts.numel()))
    seen = counts > 0
    return (~seen[query.rel.cpu()]).float().unsqueeze(-1)


def build_relation_histories(edges: EdgeTensor) -> dict[int, EdgeTensor]:
    buckets: dict[int, list[int]] = defaultdict(list)
    for idx, rel in enumerate(edges.rel.cpu().tolist()):
        buckets[int(rel)].append(idx)
    return {rel: edges.slice(indices) for rel, indices in buckets.items()}
