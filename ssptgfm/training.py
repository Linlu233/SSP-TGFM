from __future__ import annotations

from dataclasses import dataclass, field
import sys
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ssptgfm.data import EdgeTensor, TemporalDataset, TemporalSplits
from ssptgfm.features import GraphHistoryIndex, relation_frequency_ood, relation_frequency_ood_from_counts
from ssptgfm.losses import total_loss
from ssptgfm.meta import inner_update_prompts, query_with_adapted_prompts
from ssptgfm.metrics import binary_metrics, ranking_metrics
from ssptgfm.model import SSPTGFM
from ssptgfm.negative_sampling import KnownFacts, NegativeSampler, make_labels


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 256
    lr: float = 0.001
    weight_decay: float = 0.0001
    num_neg_train: int = 1
    num_neg_eval: int = 50
    negative_mode_train: str = "filtered"
    negative_mode_eval: str = "filtered"
    filter_scope: str = "exact"
    lambda_align: float = 0.1
    lambda_kl: float = 0.001
    lambda_meta: float = 0.0
    meta_lr: float = 0.01
    meta_support_size: int = 16
    meta_query_size: int = 16
    lambda_ood: float = 0.0
    lambda_rank: float = 0.0
    rank_margin: float = 1.0
    rank_loss_type: str = "hinge"
    lambda_candidate_rank: float = 0.0
    candidate_rank_size: int = 32
    candidate_rank_sides: str = "both"
    candidate_rank_queries: int = 0
    candidate_rank_tail_pool: str = "sampler"
    lambda_struct_aux: float = 0.0
    align_temperature: float = 0.2
    patience: int = 10
    early_stop_metric: str = "val_auc"
    early_stop_minima: dict[str, float] = field(default_factory=dict)
    early_stop_weights: dict[str, float] = field(default_factory=dict)
    val_binary_edges: int | None = None
    val_rank_edges: int | None = None
    val_eval_sample: str = "prefix"
    eval_batch_size_candidates: int | None = None
    grad_clip: float = 1.0
    amp: bool = False
    progress_every_batches: int = 100
    progress_every_seconds: float = 60.0


def _cuda_memory_stats(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    torch.cuda.synchronize(device)
    return {
        "cuda_allocated_mb": int(torch.cuda.memory_allocated(device) / 1024 / 1024),
        "cuda_reserved_mb": int(torch.cuda.memory_reserved(device) / 1024 / 1024),
        "cuda_max_allocated_mb": int(torch.cuda.max_memory_allocated(device) / 1024 / 1024),
    }


def _metric_lookup(metrics: dict[str, float], key: str) -> float:
    normalized = key[4:] if key.startswith("val_") else key
    return float(metrics.get(normalized, float("nan")))


def _validation_score(metrics: dict[str, float], config: TrainConfig) -> float:
    metric = config.early_stop_metric
    if metric in {"all_minima_margin", "val_all_minima_margin"}:
        if not config.early_stop_minima:
            raise ValueError("early_stop_metric=all_minima_margin requires train.early_stop_minima")
        margins: list[float] = []
        for key, threshold in config.early_stop_minima.items():
            value = _metric_lookup(metrics, key)
            if not np.isfinite(value):
                return -float("inf")
            denom = max(abs(float(threshold)), 1e-8)
            margins.append((value - float(threshold)) / denom)
        return float(min(margins))
    if metric in {"composite", "val_composite"}:
        if not config.early_stop_weights:
            raise ValueError("early_stop_metric=composite requires train.early_stop_weights")
        score = 0.0
        used = False
        for key, weight in config.early_stop_weights.items():
            value = _metric_lookup(metrics, key)
            if np.isfinite(value):
                score += float(weight) * value
                used = True
        return score if used else -float("inf")
    return _metric_lookup(metrics, metric)


def _candidate_batch_size(config: TrainConfig) -> int:
    return int(config.eval_batch_size_candidates or config.batch_size)


def _batch_edges(edges: EdgeTensor, indices: torch.Tensor) -> EdgeTensor:
    return edges.slice(indices.cpu())


def _select_eval_edges(positives: EdgeTensor, max_eval_edges: int | None, eval_sample: str) -> EdgeTensor:
    positives = positives.sort_by_time()
    if max_eval_edges is None or max_eval_edges <= 0 or len(positives) <= max_eval_edges:
        return positives
    sample_size = int(max_eval_edges)
    if eval_sample == "temporal_uniform":
        if sample_size == 1:
            idx = torch.tensor([len(positives) // 2], dtype=torch.long)
        else:
            idx = torch.linspace(0, len(positives) - 1, steps=sample_size).round().long()
        return positives.slice(idx)
    if eval_sample == "prefix":
        return positives.slice(slice(0, sample_size))
    raise ValueError(f"unknown eval_sample: {eval_sample}")


def _time_coherent_batches(edges: EdgeTensor, batch_size: int, shuffle: bool, seed: int) -> list[torch.Tensor]:
    batches: list[torch.Tensor] = []
    generator = torch.Generator().manual_seed(seed)
    if len(edges) == 0:
        return batches
    if len(edges) <= 1 or bool(torch.all(edges.time[:-1] <= edges.time[1:])):
        order = torch.arange(len(edges), dtype=torch.long)
        sorted_time = edges.time
    else:
        order = torch.argsort(edges.time)
        sorted_time = edges.time[order]
    _, counts = torch.unique_consecutive(sorted_time, return_counts=True)
    cursor = 0
    for count_t in counts.tolist():
        next_cursor = cursor + int(count_t)
        indices = order[cursor:next_cursor].clone()
        if shuffle and indices.numel() > 1:
            indices = indices[torch.randperm(indices.numel(), generator=generator)]
        for start in range(0, indices.numel(), batch_size):
            batches.append(indices[start : start + batch_size])
        cursor = next_cursor
    if shuffle and len(batches) > 1:
        order = torch.randperm(len(batches), generator=generator).tolist()
        batches = [batches[i] for i in order]
    return batches


def _time_grouped_batches(
    edges: EdgeTensor,
    batch_size: int,
    shuffle_within_time: bool,
    seed: int,
) -> list[tuple[float, list[torch.Tensor]]]:
    generator = torch.Generator().manual_seed(seed)
    groups: list[tuple[float, list[torch.Tensor]]] = []
    if len(edges) == 0:
        return groups
    if len(edges) <= 1 or bool(torch.all(edges.time[:-1] <= edges.time[1:])):
        order = torch.arange(len(edges), dtype=torch.long)
        sorted_time = edges.time
    else:
        order = torch.argsort(edges.time)
        sorted_time = edges.time[order]
    unique_times, counts = torch.unique_consecutive(sorted_time, return_counts=True)
    cursor = 0
    for time_value, count_t in zip(unique_times.tolist(), counts.tolist()):
        next_cursor = cursor + int(count_t)
        indices = order[cursor:next_cursor].clone()
        if shuffle_within_time and indices.numel() > 1:
            indices = indices[torch.randperm(indices.numel(), generator=generator)]
        batches = [indices[start : start + batch_size] for start in range(0, indices.numel(), batch_size)]
        groups.append((float(time_value), batches))
        cursor = next_cursor
    return groups


def _edge_nodes(edges: EdgeTensor, device: torch.device) -> torch.Tensor:
    return torch.cat([edges.src, edges.dst]).to(device)


def _add_degree_counts(degree: torch.Tensor, edges: EdgeTensor) -> None:
    if len(edges) == 0:
        return
    src = edges.src.to(degree.device)
    dst = edges.dst.to(degree.device)
    one = torch.ones(src.numel(), 1, dtype=degree.dtype, device=degree.device)
    degree.index_add_(0, src, one)
    degree.index_add_(0, dst, one)


def _struct_features(
    hist_index: GraphHistoryIndex,
    edges: EdgeTensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.autocast(device_type="cuda", enabled=False):
        if device.type == "cuda" and hist_index.num_nodes <= 4096:
            candidates = [
                (int(s), int(d), int(r), float(t))
                for s, d, r, t in zip(edges.src.cpu(), edges.dst.cpu(), edges.rel.cpu(), edges.time.cpu())
            ]
            return hist_index.features_for_candidates_tensor(candidates, device)
        if device.type == "cuda":
            return hist_index.features_for_edges_sparse_ppr(edges, device)
        return hist_index.features_for_edges(edges)


def _history_prior_features(
    model: SSPTGFM,
    hist_index: GraphHistoryIndex,
    edges: EdgeTensor,
    device: torch.device,
) -> torch.Tensor | None:
    if not getattr(model, "use_history_prior", False):
        return None
    return hist_index.history_prior_features_for_edges(edges, device, feature_dim=model.history_prior_dim)


def _history_prior_features_for_candidates(
    model: SSPTGFM,
    hist_index: GraphHistoryIndex,
    candidates: list[tuple[int, int, int, float]],
    device: torch.device,
) -> torch.Tensor | None:
    if not getattr(model, "use_history_prior", False):
        return None
    return hist_index.history_prior_features_for_candidates(candidates, device, feature_dim=model.history_prior_dim)


def _batch_rank_loss(
    scores: torch.Tensor,
    pos_count: int,
    neg_count: int,
    margin: float,
    loss_type: str = "hinge",
) -> torch.Tensor:
    if pos_count <= 0 or neg_count <= 0 or neg_count % pos_count != 0:
        return scores.new_tensor(0.0)
    neg_per_pos = neg_count // pos_count
    pos_scores = scores[:pos_count].view(pos_count, 1)
    neg_scores = scores[pos_count : pos_count + neg_count].view(pos_count, neg_per_pos)
    diff = pos_scores - neg_scores
    if loss_type == "hinge":
        return torch.relu(float(margin) - diff).mean()
    if loss_type == "bpr":
        return torch.nn.functional.softplus(-diff).mean()
    if loss_type == "softplus":
        return torch.nn.functional.softplus(float(margin) - diff).mean()
    if loss_type in {"sampled_softmax", "softmax", "listwise", "ce"}:
        logits = torch.cat([pos_scores, neg_scores], dim=1)
        target = torch.zeros(pos_count, dtype=torch.long, device=scores.device)
        return torch.nn.functional.cross_entropy(logits, target)
    if loss_type in {"adv_bpr", "self_adversarial_bpr"}:
        temperature = max(float(margin), 1e-6)
        weights = torch.softmax(neg_scores.detach() / temperature, dim=1)
        return (weights * torch.nn.functional.softplus(-diff)).sum(dim=1).mean()
    if loss_type in {"adv_softplus", "self_adversarial_softplus"}:
        temperature = max(float(margin), 1e-6)
        weights = torch.softmax(neg_scores.detach() / temperature, dim=1)
        return (weights * torch.nn.functional.softplus(-diff)).sum(dim=1).mean()
    raise ValueError(f"unknown rank_loss_type: {loss_type}")


def _candidate_listwise_rank_loss(
    model: SSPTGFM,
    positives: EdgeTensor,
    hist_index: GraphHistoryIndex,
    sampler: NegativeSampler,
    context: dict[str, torch.Tensor],
    device: torch.device,
    num_relations: int,
    relation_counts: torch.Tensor,
    *,
    candidate_rank_size: int,
    candidate_rank_sides: str,
    candidate_rank_queries: int = 0,
    tail_candidate_pool: np.ndarray | None = None,
) -> torch.Tensor:
    if len(positives) == 0 or candidate_rank_size <= 0:
        return context["sem_all"].new_tensor(0.0)
    sides = candidate_rank_sides.lower()
    if sides == "both":
        corrupt_options = (False, True)
    elif sides in {"tail", "tails"}:
        corrupt_options = (False,)
    elif sides in {"head", "heads"}:
        corrupt_options = (True,)
    else:
        raise ValueError(f"unknown candidate_rank_sides: {candidate_rank_sides}")
    if candidate_rank_queries > 0 and len(positives) > candidate_rank_queries:
        indices = torch.linspace(0, len(positives) - 1, steps=int(candidate_rank_queries)).round().long()
        positives = positives.slice(indices)
    candidates: list[tuple[int, int, int, float]] = []
    group_sizes: list[int] = []
    for s_t, d_t, r_t, tm_t in zip(positives.src.cpu(), positives.dst.cpu(), positives.rel.cpu(), positives.time.cpu()):
        s = int(s_t)
        d = int(d_t)
        r = int(r_t)
        tm = float(tm_t)
        for corrupt_head in corrupt_options:
            true_node = s if corrupt_head else d
            if (not corrupt_head) and tail_candidate_pool is not None and tail_candidate_pool.size:
                sampled: list[int] = []
                seen = {int(true_node), int(s)}
                attempts = max(200, int(candidate_rank_size) * 50)
                for _ in range(attempts):
                    node = int(tail_candidate_pool[int(sampler.rng.integers(0, tail_candidate_pool.size))])
                    if node in seen:
                        continue
                    if sampler.known_facts.contains(s, r, node, tm, sampler.filter_scope):
                        continue
                    seen.add(node)
                    sampled.append(node)
                    if len(sampled) >= int(candidate_rank_size):
                        break
                if len(sampled) < int(candidate_rank_size):
                    for node_t in tail_candidate_pool:
                        node = int(node_t)
                        if node in seen:
                            continue
                        if sampler.known_facts.contains(s, r, node, tm, sampler.filter_scope):
                            continue
                        seen.add(node)
                        sampled.append(node)
                        if len(sampled) >= int(candidate_rank_size):
                            break
                neg_nodes = sampled
            else:
                neg_nodes = sampler.sample_candidate_nodes(
                    s,
                    d,
                    r,
                    tm,
                    corrupt_head=corrupt_head,
                    num_negatives=int(candidate_rank_size),
                    history_index=hist_index,
                )
            if not neg_nodes:
                continue
            nodes = [true_node, *neg_nodes]
            if corrupt_head:
                query_candidates = [(node, d, r, tm) for node in nodes]
            else:
                query_candidates = [(s, node, r, tm) for node in nodes]
            candidates.extend(query_candidates)
            group_sizes.append(len(query_candidates))
    if not candidates:
        return context["sem_all"].new_tensor(0.0)
    batch = EdgeTensor.from_arrays(
        [x[0] for x in candidates],
        [x[1] for x in candidates],
        [x[2] for x in candidates],
        [x[3] for x in candidates],
    )
    if device.type == "cuda" and model.num_nodes <= 4096:
        sf, ood_s = hist_index.features_for_candidates_tensor(candidates, device)
    elif device.type == "cuda":
        sf, ood_s = hist_index.features_for_candidates_sparse_ppr(candidates, device)
    else:
        first_s, first_d, _, _ = candidates[0]
        sf, ood_s = hist_index.features_for_candidates(first_s, first_d, candidates)
    hp = _history_prior_features_for_candidates(model, hist_index, candidates, device)
    ood_e = relation_frequency_ood_from_counts(relation_counts, batch, num_relations)
    out = model.score_edges_from_context(
        batch,
        struct_features=sf,
        ood_s=ood_s,
        ood_e=ood_e,
        context=context,
        history_prior_features=hp,
    )
    losses: list[torch.Tensor] = []
    target = torch.zeros(1, dtype=torch.long, device=device)
    offset = 0
    for group_size in group_sizes:
        scores = out.final_score[offset : offset + group_size].float().view(1, -1)
        losses.append(torch.nn.functional.cross_entropy(scores, target))
        offset += group_size
    return torch.stack(losses).mean()


def causal_history_for_batch(history_pool: EdgeTensor, batch: EdgeTensor) -> EdgeTensor:
    if len(batch) == 0:
        return history_pool.slice([])
    min_t = float(batch.time.min())
    return history_pool.before(min_t, strict=True)


def _plain_lp_loss(
    model: SSPTGFM,
    batch: EdgeTensor,
    history: EdgeTensor,
    labels: torch.Tensor,
    node_text_emb: torch.Tensor,
    rel_text_emb: torch.Tensor,
    num_relations: int,
    node_features: torch.Tensor | None = None,
    use_adapted: dict[str, torch.Tensor] | None = None,
    hist_index: GraphHistoryIndex | None = None,
    relation_counts: torch.Tensor | None = None,
    history_context: EdgeTensor | None = None,
    history_degree: torch.Tensor | None = None,
) -> torch.Tensor:
    hist_index = GraphHistoryIndex(model.num_nodes, history) if hist_index is None else hist_index
    sf, ood_s = _struct_features(hist_index, batch, node_text_emb.device)
    hp = _history_prior_features(model, hist_index, batch, node_text_emb.device)
    ood_e = (
        relation_frequency_ood(history, batch, num_relations)
        if relation_counts is None
        else relation_frequency_ood_from_counts(relation_counts, batch, num_relations)
    )
    call = query_with_adapted_prompts if use_adapted is not None else None
    if call is None:
        out = model(
            batch,
            history_context if history_context is not None else history,
            node_text_emb=node_text_emb,
            rel_text_emb=rel_text_emb,
            struct_features=sf,
            ood_s=ood_s,
            ood_e=ood_e,
            node_features=node_features,
            history_degree=history_degree,
            history_prior_features=hp,
        )
    else:
        out = call(
            model,
            use_adapted,
            batch,
            history_context if history_context is not None else history,
            node_text_emb=node_text_emb,
            rel_text_emb=rel_text_emb,
            struct_features=sf,
            ood_s=ood_s,
            ood_e=ood_e,
            node_features=node_features,
            history_degree=history_degree,
            history_prior_features=hp,
        )
    return torch.nn.functional.binary_cross_entropy_with_logits(out.final_score, labels.float())


def _meta_episode_loss(
    model: SSPTGFM,
    pos: EdgeTensor,
    neg: EdgeTensor,
    hist: EdgeTensor,
    labels: torch.Tensor,
    node_text_emb: torch.Tensor,
    rel_text_emb: torch.Tensor,
    config: TrainConfig,
    num_relations: int,
    device: torch.device,
    node_features: torch.Tensor | None = None,
    hist_index: GraphHistoryIndex | None = None,
    relation_counts: torch.Tensor | None = None,
    history_context: EdgeTensor | None = None,
    history_degree: torch.Tensor | None = None,
) -> torch.Tensor:
    if config.lambda_meta <= 0.0 or len(pos) == 0 or len(neg) == 0:
        return labels.new_tensor(0.0)
    if len(pos) >= 2:
        support_pos_n = max(1, min(config.meta_support_size, len(pos) // 2))
        query_pos_n = max(1, min(config.meta_query_size, len(pos) - support_pos_n))
        support_pos = pos.slice(slice(0, support_pos_n))
        query_pos = pos.slice(slice(support_pos_n, support_pos_n + query_pos_n))
    else:
        support_pos_n = 1
        query_pos_n = 1
        support_pos = pos
        query_pos = pos
    if len(neg) >= 2:
        support_neg_n = max(1, min(config.meta_support_size, len(neg) // 2))
        query_neg_n = max(1, min(config.meta_query_size, len(neg) - support_neg_n))
        support_neg = neg.slice(slice(0, support_neg_n))
        query_neg = neg.slice(slice(support_neg_n, support_neg_n + query_neg_n))
    else:
        support_neg_n = 1
        query_neg_n = 1
        support_neg = neg
        query_neg = neg
    support = support_pos.concat(support_neg, sort=False)
    query = query_pos.concat(query_neg, sort=False)
    support_labels = make_labels(support_pos_n, support_neg_n, device)
    query_labels = make_labels(query_pos_n, query_neg_n, device)
    support_loss = _plain_lp_loss(
        model,
        support,
        hist,
        support_labels,
        node_text_emb,
        rel_text_emb,
        num_relations,
        node_features=node_features,
        hist_index=hist_index,
        relation_counts=relation_counts,
        history_context=history_context,
        history_degree=history_degree,
    )
    adapted = inner_update_prompts(model, support_loss, config.meta_lr, create_graph=True)
    return _plain_lp_loss(
        model,
        query,
        hist,
        query_labels,
        node_text_emb,
        rel_text_emb,
        num_relations,
        node_features=node_features,
        use_adapted=adapted,
        hist_index=hist_index,
        relation_counts=relation_counts,
        history_context=history_context,
        history_degree=history_degree,
    )


def combine_scores_for_binary_eval(
    model: SSPTGFM,
    positives: EdgeTensor,
    history_pool: EdgeTensor,
    sampler: NegativeSampler,
    node_text_emb: torch.Tensor,
    rel_text_emb: torch.Tensor,
    device: torch.device,
    num_neg_per_pos: int,
    batch_size: int,
    num_relations: int,
    node_features: torch.Tensor | None = None,
    max_eval_edges: int | None = None,
    eval_sample: str = "prefix",
) -> tuple[np.ndarray, np.ndarray]:
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        positives = _select_eval_edges(positives, max_eval_edges, eval_sample)
        history_pool = history_pool.sort_by_time()
        history_context_pool = history_pool.to(device) if device.type == "cuda" else history_pool
        hist_index = GraphHistoryIndex(model.num_nodes, history_pool.slice([]))
        relation_counts = torch.zeros(num_relations, dtype=torch.float32)
        history_degree = torch.zeros(model.num_nodes, 1, dtype=torch.float32, device=device)
        cursor = 0
        for time_value, batches in _time_grouped_batches(positives, batch_size=batch_size, shuffle_within_time=False, seed=0):
            while cursor < len(history_pool) and float(history_pool.time[cursor]) < time_value:
                start = cursor
                cursor += 1
                while cursor < len(history_pool) and float(history_pool.time[cursor]) < time_value:
                    cursor += 1
                newly_available = history_pool.slice(slice(start, cursor))
                hist_index.add_edges(newly_available)
                relation_counts += torch.bincount(newly_available.rel.cpu(), minlength=num_relations).float()
                _add_degree_counts(history_degree, history_context_pool.slice(slice(start, cursor)))
            hist = history_pool.slice(slice(0, cursor))
            hist_context = history_context_pool.slice(slice(0, cursor))
            query_time = torch.tensor([time_value], dtype=torch.float32, device=device)
            context = model.encode_context(
                hist_context,
                query_time,
                node_text_emb=node_text_emb,
                rel_text_emb=rel_text_emb,
                node_features=node_features,
                history_degree=history_degree,
            )
            for idx in batches:
                pos = _batch_edges(positives, idx)
                neg = sampler.sample(pos, num_neg_per_pos=num_neg_per_pos, history=hist)
                batch = pos.concat(neg, sort=False)
                sf, ood_s = _struct_features(hist_index, batch, device)
                hp = _history_prior_features(model, hist_index, batch, device)
                ood_e = relation_frequency_ood_from_counts(relation_counts, batch, num_relations)
                out = model.score_edges_from_context(
                    batch,
                    struct_features=sf,
                    ood_s=ood_s,
                    ood_e=ood_e,
                    context=context,
                    history_prior_features=hp,
                )
                labels = make_labels(len(pos), len(neg), device=device)
                labels_all.append(labels.detach().cpu().numpy())
                scores_all.append(out.final_score.detach().cpu().numpy())
    return np.concatenate(labels_all), np.concatenate(scores_all)



def tgb_tail_ranking_eval(
    model: SSPTGFM,
    positives: EdgeTensor,
    history_pool: EdgeTensor,
    negative_sets: dict[tuple[int, int, int], np.ndarray],
    node_text_emb: torch.Tensor,
    rel_text_emb: torch.Tensor,
    device: torch.device,
    batch_size_candidates: int,
    num_relations: int,
    node_features: torch.Tensor | None = None,
    max_eval_edges: int | None = None,
    eval_sample: str = "prefix",
    raw_node_ids: list[int] | None = None,
    raw_relation_ids: list[int] | None = None,
    tgb_first_dst_id: int | None = None,
    tgb_last_dst_id: int | None = None,
) -> dict[str, float]:
    """TGB time-filtered tail prediction protocol for tkgl-* datasets.

    TGB TKG negative samples are keyed by (timestamp, raw_source, raw_relation)
    and rank the true destination against all time-filtered destination candidates.
    """
    if not negative_sets:
        return {"tgb_mrr": float("nan"), "tgb_hits@10": float("nan")}
    model.eval()
    positives = _select_eval_edges(positives, max_eval_edges, eval_sample)
    history_pool = history_pool.sort_by_time()
    history_context_pool = history_pool.to(device) if device.type == "cuda" else history_pool
    raw_node_ids = raw_node_ids or list(range(model.num_nodes))
    raw_relation_ids = raw_relation_ids or list(range(num_relations))
    raw_node_to_internal = {int(raw): idx for idx, raw in enumerate(raw_node_ids)}
    if tgb_first_dst_id is None:
        tgb_first_dst_id = min(raw_node_to_internal) if raw_node_to_internal else 0
    if tgb_last_dst_id is None:
        tgb_last_dst_id = max(raw_node_to_internal) if raw_node_to_internal else model.num_nodes - 1
    raw_all_dst = np.arange(int(tgb_first_dst_id), int(tgb_last_dst_id) + 1, dtype=np.int64)
    ranks: list[float] = []
    with torch.no_grad():
        hist_index = GraphHistoryIndex(model.num_nodes, history_pool.slice([]))
        relation_counts = torch.zeros(num_relations, dtype=torch.float32)
        history_degree = torch.zeros(model.num_nodes, 1, dtype=torch.float32, device=device)
        cursor = 0
        processed = 0
        eval_count = len(positives)
        for time_value, groups in _time_grouped_batches(positives, batch_size=1, shuffle_within_time=False, seed=0):
            while cursor < len(history_pool) and float(history_pool.time[cursor]) < time_value:
                start = cursor
                cursor += 1
                while cursor < len(history_pool) and float(history_pool.time[cursor]) < time_value:
                    cursor += 1
                newly_available = history_pool.slice(slice(start, cursor))
                hist_index.add_edges(newly_available)
                relation_counts += torch.bincount(newly_available.rel.cpu(), minlength=num_relations).float()
                _add_degree_counts(history_degree, history_context_pool.slice(slice(start, cursor)))
            hist_context = history_context_pool.slice(slice(0, cursor))
            context = model.encode_context(
                hist_context,
                torch.tensor([time_value], dtype=torch.float32, device=device),
                node_text_emb=node_text_emb,
                rel_text_emb=rel_text_emb,
                node_features=node_features,
                history_degree=history_degree,
            )
            for group in groups:
                if processed >= eval_count:
                    break
                pos = positives.slice(group.cpu())
                s = int(pos.src[0])
                d = int(pos.dst[0])
                r = int(pos.rel[0])
                tm = float(pos.time[0])
                raw_s = int(raw_node_ids[s])
                raw_r = int(raw_relation_ids[r])
                key = (int(tm), raw_s, raw_r)
                conflict_raw_nodes = negative_sets.get(key)
                if conflict_raw_nodes is None:
                    processed += 1
                    continue
                conflict_idx = np.asarray(conflict_raw_nodes, dtype=np.int64)
                valid_conflict = conflict_idx[(conflict_idx >= 0) & (conflict_idx < raw_all_dst.size)]
                neg_raw_nodes = np.delete(raw_all_dst, valid_conflict, axis=0)
                raw_true_dst = int(raw_node_ids[d])
                neg_nodes = [
                    raw_node_to_internal[int(raw)]
                    for raw in neg_raw_nodes
                    if int(raw) != raw_true_dst and int(raw) in raw_node_to_internal
                ]
                if not neg_nodes:
                    processed += 1
                    continue
                candidates = [(s, d, r, tm), *[(s, int(node), r, tm) for node in neg_nodes]]
                scores: list[torch.Tensor] = []
                for start in range(0, len(candidates), batch_size_candidates):
                    chunk = candidates[start : start + batch_size_candidates]
                    batch = EdgeTensor.from_arrays(
                        [x[0] for x in chunk],
                        [x[1] for x in chunk],
                        [x[2] for x in chunk],
                        [x[3] for x in chunk],
                    )
                    if device.type == "cuda" and model.num_nodes <= 4096:
                        sf, ood_s = hist_index.features_for_candidates_tensor(chunk, device)
                    elif device.type == "cuda":
                        sf, ood_s = hist_index.features_for_candidates_sparse_ppr(chunk, device)
                    else:
                        sf, ood_s = hist_index.features_for_candidates(s, d, chunk)
                    hp = _history_prior_features_for_candidates(model, hist_index, chunk, device)
                    ood_e = relation_frequency_ood_from_counts(relation_counts, batch, num_relations)
                    out = model.score_edges_from_context(
                        batch,
                        struct_features=sf,
                        ood_s=ood_s,
                        ood_e=ood_e,
                        context=context,
                        history_prior_features=hp,
                    )
                    scores.append(out.final_score.detach().cpu())
                all_scores = torch.cat(scores).numpy()
                true_score = all_scores[0]
                neg_scores = all_scores[1:]
                optimistic_rank = float(np.sum(neg_scores > true_score))
                pessimistic_rank = float(np.sum(neg_scores >= true_score))
                ranks.append(0.5 * (optimistic_rank + pessimistic_rank) + 1.0)
                processed += 1
            if processed >= eval_count:
                break
    if not ranks:
        return {"tgb_mrr": float("nan"), "tgb_hits@10": float("nan")}
    arr = np.asarray(ranks, dtype=np.float64)
    return {"tgb_mrr": float(np.mean(1.0 / arr)), "tgb_hits@10": float(np.mean(arr <= 10))}

def filtered_ranking_eval(
    model: SSPTGFM,
    positives: EdgeTensor,
    history_pool: EdgeTensor,
    known_facts: KnownFacts,
    node_text_emb: torch.Tensor,
    rel_text_emb: torch.Tensor,
    device: torch.device,
    batch_size_candidates: int,
    num_relations: int,
    filter_scope: str = "exact",
    max_eval_edges: int | None = None,
    node_features: torch.Tensor | None = None,
    eval_sample: str = "prefix",
) -> dict[str, float]:
    ranks: list[float] = []
    model.eval()
    eval_count = len(positives) if max_eval_edges is None else min(len(positives), max_eval_edges)
    with torch.no_grad():
        positives = _select_eval_edges(positives, max_eval_edges, eval_sample)
        history_pool = history_pool.sort_by_time()
        history_context_pool = history_pool.to(device) if device.type == "cuda" else history_pool
        hist_index = GraphHistoryIndex(model.num_nodes, history_pool.slice([]))
        relation_counts = torch.zeros(num_relations, dtype=torch.float32)
        history_degree = torch.zeros(model.num_nodes, 1, dtype=torch.float32, device=device)
        cursor = 0
        processed = 0
        for time_value, groups in _time_grouped_batches(positives, batch_size=1, shuffle_within_time=False, seed=0):
            while cursor < len(history_pool) and float(history_pool.time[cursor]) < time_value:
                start = cursor
                cursor += 1
                while cursor < len(history_pool) and float(history_pool.time[cursor]) < time_value:
                    cursor += 1
                newly_available = history_pool.slice(slice(start, cursor))
                hist_index.add_edges(newly_available)
                relation_counts += torch.bincount(newly_available.rel.cpu(), minlength=num_relations).float()
                _add_degree_counts(history_degree, history_context_pool.slice(slice(start, cursor)))
            hist = history_pool.slice(slice(0, cursor))
            hist_context = history_context_pool.slice(slice(0, cursor))
            context = model.encode_context(
                hist_context,
                torch.tensor([time_value], dtype=torch.float32, device=device),
                node_text_emb=node_text_emb,
                rel_text_emb=rel_text_emb,
                node_features=node_features,
                history_degree=history_degree,
            )
            for group in groups:
                if processed >= eval_count:
                    break
                pos = positives.slice(group.cpu())
                s = int(pos.src[0])
                d = int(pos.dst[0])
                r = int(pos.rel[0])
                tm = float(pos.time[0])
                ood_e_all = relation_frequency_ood_from_counts(relation_counts, pos, num_relations)
                for corrupt_head in (False, True):
                    keep, true_index = known_facts.filtered_nodes_for_query(
                        s,
                        r,
                        d,
                        tm,
                        corrupt_head=corrupt_head,
                        num_nodes=model.num_nodes,
                        scope=filter_scope,
                    )
                    if true_index is None:
                        continue
                    cand_nodes = torch.nonzero(keep, as_tuple=False).view(-1).tolist()
                    scores = []
                    for start in range(0, len(cand_nodes), batch_size_candidates):
                        chunk_nodes = cand_nodes[start : start + batch_size_candidates]
                        if corrupt_head:
                            chunk = [(node, d, r, tm) for node in chunk_nodes]
                        else:
                            chunk = [(s, node, r, tm) for node in chunk_nodes]
                        batch = EdgeTensor.from_arrays(
                            [x[0] for x in chunk],
                            [x[1] for x in chunk],
                            [x[2] for x in chunk],
                            [x[3] for x in chunk],
                        )
                        if device.type == "cuda" and model.num_nodes <= 4096:
                            sf, ood_s = hist_index.features_for_candidates_tensor(chunk, device)
                        elif device.type == "cuda":
                            sf, ood_s = hist_index.features_for_candidates_sparse_ppr(chunk, device)
                        else:
                            sf, ood_s = hist_index.features_for_candidates(s, d, chunk)
                        hp = _history_prior_features_for_candidates(model, hist_index, chunk, device)
                        ood_e = ood_e_all.expand(len(batch), -1)
                        out = model.score_edges_from_context(
                            batch,
                            struct_features=sf,
                            ood_s=ood_s,
                            ood_e=ood_e,
                            context=context,
                            history_prior_features=hp,
                        )
                        scores.append(out.final_score.detach().cpu())
                    all_scores = torch.cat(scores).numpy()
                    true_score = all_scores[true_index]
                    rank = 1.0 + float(np.sum(all_scores > true_score)) + 0.5 * float(np.sum(all_scores == true_score) - 1)
                    ranks.append(rank)
                processed += 1
            if processed >= eval_count:
                break
    return ranking_metrics(ranks)


def train_one_seed(
    dataset: TemporalDataset,
    splits: TemporalSplits,
    model: SSPTGFM,
    node_text_emb: torch.Tensor,
    rel_text_emb: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    seed: int,
    node_features: torch.Tensor | None = None,
    tgb_val_negative_sets: dict[tuple[int, int, int], np.ndarray] | None = None,
    tgb_val_rank_edges: int | None = None,
) -> tuple[SSPTGFM, dict[str, float]]:
    train_start = perf_counter()
    print(
        {
            "event": "train_seed_start",
            "seed": seed,
            "train_edges": len(splits.train),
            "val_edges": len(splits.val),
            "test_edges": len(splits.test),
            "num_nodes": dataset.num_nodes,
            "num_relations": dataset.num_relations,
        },
        flush=True,
    )
    model.to(device)
    node_text_emb = node_text_emb.to(device)
    rel_text_emb = rel_text_emb.to(device)
    node_features = node_features.to(device) if node_features is not None else None
    print(
        {
            "event": "train_seed_tensors_ready",
            "seed": seed,
            "elapsed_sec": perf_counter() - train_start,
            **_cuda_memory_stats(device),
        },
        flush=True,
    )
    known_train = KnownFacts.from_edges(splits.train)
    print(
        {
            "event": "train_known_facts_ready",
            "seed": seed,
            "scope": "train",
            "edges": len(splits.train),
            "elapsed_sec": perf_counter() - train_start,
        },
        flush=True,
    )
    known_val = KnownFacts.from_edges(splits.train.concat(splits.val, sort=False))
    print(
        {
            "event": "train_known_facts_ready",
            "seed": seed,
            "scope": "train_val",
            "edges": len(splits.train) + len(splits.val),
            "elapsed_sec": perf_counter() - train_start,
        },
        flush=True,
    )
    train_sampler = NegativeSampler(
        dataset.num_nodes,
        known_train,
        splits.train,
        mode=config.negative_mode_train,
        filter_scope=config.filter_scope,
        seed=seed,
    )
    print({"event": "train_sampler_ready", "seed": seed, "elapsed_sec": perf_counter() - train_start}, flush=True)
    def make_val_sampler() -> NegativeSampler:
        return NegativeSampler(
            dataset.num_nodes,
            known_val,
            splits.train,
            mode=config.negative_mode_eval,
            filter_scope=config.filter_scope,
            seed=seed + 1009,
        )

    val_sampler = make_val_sampler()
    print({"event": "val_sampler_ready", "seed": seed, "elapsed_sec": perf_counter() - train_start}, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler(device="cuda", enabled=bool(config.amp and device.type == "cuda"))
    best_score = -float("inf")
    early_stop_metric = config.early_stop_metric
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_loss_parts: dict[str, float] = {}
    stale = 0
    train_edges = splits.train.sort_by_time()
    train_edges_context = train_edges.to(device) if device.type == "cuda" else train_edges
    tail_candidate_pool: np.ndarray | None = None
    if str(config.candidate_rank_tail_pool).lower() in {"tgb_dst", "official_tgb_dst", "dst_range"}:
        raw_node_ids = dataset.raw_node_ids or list(range(dataset.num_nodes))
        raw_to_internal = {int(raw): idx for idx, raw in enumerate(raw_node_ids)}
        if dataset.tgb_first_dst_id is not None and dataset.tgb_last_dst_id is not None:
            tail_candidate_pool = np.asarray(
                [
                    raw_to_internal[int(raw)]
                    for raw in range(int(dataset.tgb_first_dst_id), int(dataset.tgb_last_dst_id) + 1)
                    if int(raw) in raw_to_internal
                ],
                dtype=np.int64,
            )
    for epoch in range(1, config.epochs + 1):
        epoch_start = perf_counter()
        model.train()
        time_groups = _time_grouped_batches(train_edges, batch_size=config.batch_size, shuffle_within_time=True, seed=seed + epoch)
        losses: list[float] = []
        epoch_parts: dict[str, list[float]] = {}
        total_batches = sum(len(batches) for _, batches in time_groups)
        total_time_groups = len(time_groups)
        print(
            {
                "event": "epoch_start",
                "epoch": epoch,
                "epochs": config.epochs,
                "batches": total_batches,
                "time_groups": total_time_groups,
            },
            flush=True,
        )
        pbar = tqdm(total=total_batches, desc=f"epoch {epoch}", leave=False, disable=not sys.stderr.isatty())
        hist_index = GraphHistoryIndex(dataset.num_nodes, train_edges.slice([]))
        relation_counts = torch.zeros(dataset.num_relations, dtype=torch.float32)
        history_degree = torch.zeros(dataset.num_nodes, 1, dtype=torch.float32, device=device)
        cursor = 0
        processed_batches = 0
        last_progress = epoch_start
        progress_every_batches = max(0, int(config.progress_every_batches))
        progress_every_seconds = max(0.0, float(config.progress_every_seconds))
        for time_group_idx, (time_value, batches) in enumerate(time_groups, start=1):
            while cursor < len(train_edges) and float(train_edges.time[cursor]) < time_value:
                start = cursor
                cursor += 1
                while cursor < len(train_edges) and float(train_edges.time[cursor]) < time_value:
                    cursor += 1
                newly_available = train_edges.slice(slice(start, cursor))
                hist_index.add_edges(newly_available)
                relation_counts += torch.bincount(newly_available.rel.cpu(), minlength=dataset.num_relations).float()
                _add_degree_counts(history_degree, train_edges_context.slice(slice(start, cursor)))
            hist = train_edges.slice(slice(0, cursor))
            hist_context = train_edges_context.slice(slice(0, cursor))
            shared_context = None
            query_time = torch.tensor([time_value], dtype=torch.float32, device=device)
            now = perf_counter()
            should_log_group = (
                time_group_idx == 1
                or time_group_idx == total_time_groups
                or (progress_every_seconds > 0.0 and now - last_progress >= progress_every_seconds)
            )
            if should_log_group:
                print(
                    {
                        "event": "train_time_group",
                        "epoch": epoch,
                        "time_group": time_group_idx,
                        "time_groups": total_time_groups,
                        "batch": processed_batches,
                        "batches": total_batches,
                        "time_value": time_value,
                        "group_batches": len(batches),
                        "history_edges": len(hist),
                        "elapsed_sec": now - epoch_start,
                        **_cuda_memory_stats(device),
                    },
                    flush=True,
                )
                last_progress = perf_counter()
            if len(batches) == 1:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(config.amp and device.type == "cuda")):
                    shared_context = model.encode_context(
                        hist_context,
                        query_time,
                        node_text_emb=node_text_emb,
                        rel_text_emb=rel_text_emb,
                        node_features=node_features,
                        history_degree=history_degree,
                    )
            for idx in batches:
                pos = _batch_edges(train_edges, idx)
                neg = train_sampler.sample(pos, config.num_neg_train, history=hist)
                batch = pos.concat(neg, sort=False)
                sf, ood_s = _struct_features(hist_index, batch, device)
                hp = _history_prior_features(model, hist_index, batch, device)
                ood_e = relation_frequency_ood_from_counts(relation_counts, batch, dataset.num_relations)
                labels = make_labels(len(pos), len(neg), device=device)
                context = shared_context
                if context is None:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(config.amp and device.type == "cuda")):
                        context = model.encode_context(
                            hist_context,
                            query_time,
                            node_text_emb=node_text_emb,
                            rel_text_emb=rel_text_emb,
                            node_features=node_features,
                            history_degree=history_degree,
                        )
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(config.amp and device.type == "cuda")):
                    out = model.score_edges_from_context(
                        batch,
                        struct_features=sf,
                        ood_s=ood_s,
                        ood_e=ood_e,
                        context=context,
                        history_prior_features=hp,
                    )
                    edge_nodes = _edge_nodes(batch, device)
                    meta = _meta_episode_loss(
                        model,
                        pos,
                        neg,
                        hist,
                        labels,
                        node_text_emb,
                        rel_text_emb,
                        config,
                        dataset.num_relations,
                        device,
                        node_features=node_features,
                        hist_index=hist_index,
                        relation_counts=relation_counts,
                        history_context=hist_context,
                        history_degree=history_degree,
                    )
                    loss, parts = total_loss(
                        out,
                        labels,
                        edge_nodes,
                        lambda_align=config.lambda_align,
                        lambda_kl=config.lambda_kl,
                        lambda_meta=config.lambda_meta,
                        meta_loss=meta,
                        lambda_ood=config.lambda_ood,
                        align_temperature=config.align_temperature,
                        lambda_struct_aux=config.lambda_struct_aux,
                    )
                    if config.lambda_rank > 0.0:
                        rank_loss = _batch_rank_loss(
                            out.final_score.float(),
                            len(pos),
                            len(neg),
                            config.rank_margin,
                            config.rank_loss_type,
                        )
                        loss = loss + float(config.lambda_rank) * rank_loss
                        parts["rank"] = float(rank_loss.detach().cpu())
                        parts["weighted_rank"] = float((float(config.lambda_rank) * rank_loss).detach().cpu())
                    if config.lambda_candidate_rank > 0.0:
                        candidate_rank_loss = _candidate_listwise_rank_loss(
                            model,
                            pos,
                            hist_index,
                            train_sampler,
                            context,
                            device,
                            dataset.num_relations,
                            relation_counts,
                            candidate_rank_size=config.candidate_rank_size,
                            candidate_rank_sides=config.candidate_rank_sides,
                            candidate_rank_queries=config.candidate_rank_queries,
                            tail_candidate_pool=tail_candidate_pool,
                        )
                        loss = loss + float(config.lambda_candidate_rank) * candidate_rank_loss
                        parts["candidate_rank"] = float(candidate_rank_loss.detach().cpu())
                        parts["weighted_candidate_rank"] = float(
                            (float(config.lambda_candidate_rank) * candidate_rank_loss).detach().cpu()
                        )
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if config.grad_clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(opt)
                scaler.update()
                losses.append(parts["loss"])
                for key, value in parts.items():
                    epoch_parts.setdefault(key, []).append(value)
                pbar.update(1)
                pbar.set_postfix(loss=np.mean(losses[-20:]))
                processed_batches += 1
                now = perf_counter()
                should_log_batch = (
                    processed_batches == 1
                    or processed_batches == total_batches
                    or (progress_every_batches > 0 and processed_batches % progress_every_batches == 0)
                    or (progress_every_seconds > 0.0 and now - last_progress >= progress_every_seconds)
                )
                if should_log_batch:
                    recent_window = losses[-min(len(losses), max(1, progress_every_batches or 20)) :]
                    print(
                        {
                            "event": "train_progress",
                            "epoch": epoch,
                            "batch": processed_batches,
                            "batches": total_batches,
                            "time_group": time_group_idx,
                            "time_groups": total_time_groups,
                            "history_edges": len(hist),
                            "pos_edges": len(pos),
                            "neg_edges": len(neg),
                            "loss_recent": float(np.mean(recent_window)),
                            "elapsed_sec": now - epoch_start,
                            "avg_sec_per_batch": (now - epoch_start) / max(1, processed_batches),
                            **_cuda_memory_stats(device),
                        },
                        flush=True,
                    )
                    last_progress = perf_counter()
        pbar.close()
        print({"event": "validation_start", "epoch": epoch}, flush=True)
        val_labels, val_scores = combine_scores_for_binary_eval(
            model,
            splits.val,
            splits.train,
            make_val_sampler(),
            node_text_emb,
            rel_text_emb,
            device,
            num_neg_per_pos=config.num_neg_eval,
            batch_size=config.batch_size,
            num_relations=dataset.num_relations,
                node_features=node_features,
                max_eval_edges=config.val_binary_edges,
                eval_sample=config.val_eval_sample,
        )
        val = binary_metrics(val_labels, val_scores)
        if config.val_rank_edges is not None and config.val_rank_edges > 0:
            print({"event": "validation_ranking_start", "epoch": epoch, "max_edges": config.val_rank_edges}, flush=True)
            val_rank = filtered_ranking_eval(
                model,
                splits.val,
                splits.train,
                known_val,
                node_text_emb,
                rel_text_emb,
                device,
                batch_size_candidates=_candidate_batch_size(config),
                num_relations=dataset.num_relations,
                filter_scope=config.filter_scope,
                max_eval_edges=config.val_rank_edges,
                node_features=node_features,
                eval_sample=config.val_eval_sample,
            )
            val.update(val_rank)
        if tgb_val_negative_sets is not None:
            print({"event": "validation_tgb_ranking_start", "epoch": epoch, "max_edges": tgb_val_rank_edges}, flush=True)
            val_tgb = tgb_tail_ranking_eval(
                model,
                splits.val,
                splits.train,
                tgb_val_negative_sets,
                node_text_emb,
                rel_text_emb,
                device,
                batch_size_candidates=_candidate_batch_size(config),
                num_relations=dataset.num_relations,
                node_features=node_features,
                max_eval_edges=tgb_val_rank_edges,
                eval_sample=config.val_eval_sample,
                raw_node_ids=dataset.raw_node_ids,
                raw_relation_ids=dataset.raw_relation_ids,
            )
            val.update(val_tgb)
        val_score = _validation_score(val, config)
        if np.isfinite(val_score) and val_score > best_score:
            best_score = val_score
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_loss_parts = {f"train_{key}": float(np.mean(values)) for key, values in epoch_parts.items() if values}
        else:
            stale += 1
        print(
            {
                "event": "epoch_end",
                "epoch": epoch,
                "early_stop_metric": early_stop_metric,
                "val_score": val_score,
                **{f"val_{key}": value for key, value in val.items()},
                "elapsed_sec": perf_counter() - epoch_start,
            },
            flush=True,
        )
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    val_labels, val_scores = combine_scores_for_binary_eval(
        model,
        splits.val,
        splits.train,
        make_val_sampler(),
        node_text_emb,
        rel_text_emb,
        device,
        num_neg_per_pos=config.num_neg_eval,
        batch_size=config.batch_size,
        num_relations=dataset.num_relations,
        node_features=node_features,
        max_eval_edges=config.val_binary_edges,
        eval_sample=config.val_eval_sample,
    )
    final_val = binary_metrics(val_labels, val_scores)
    if config.val_rank_edges is not None and config.val_rank_edges > 0:
        final_val_rank = filtered_ranking_eval(
            model,
            splits.val,
            splits.train,
            known_val,
            node_text_emb,
            rel_text_emb,
            device,
            batch_size_candidates=_candidate_batch_size(config),
            num_relations=dataset.num_relations,
            filter_scope=config.filter_scope,
            max_eval_edges=config.val_rank_edges,
            node_features=node_features,
            eval_sample=config.val_eval_sample,
        )
        final_val.update(final_val_rank)
    if tgb_val_negative_sets is not None:
        final_val_tgb = tgb_tail_ranking_eval(
            model,
            splits.val,
            splits.train,
            tgb_val_negative_sets,
            node_text_emb,
            rel_text_emb,
            device,
            batch_size_candidates=_candidate_batch_size(config),
            num_relations=dataset.num_relations,
            node_features=node_features,
            max_eval_edges=tgb_val_rank_edges,
            eval_sample=config.val_eval_sample,
            raw_node_ids=dataset.raw_node_ids,
            raw_relation_ids=dataset.raw_relation_ids,
            tgb_first_dst_id=getattr(dataset, "tgb_first_dst_id", None),
            tgb_last_dst_id=getattr(dataset, "tgb_last_dst_id", None),
        )
        final_val.update(final_val_tgb)
    val_metrics = {f"val_{k}": v for k, v in final_val.items()}
    val_metrics.update(best_loss_parts)
    return model, val_metrics


def evaluate_test(
    dataset: TemporalDataset,
    splits: TemporalSplits,
    model: SSPTGFM,
    node_text_emb: torch.Tensor,
    rel_text_emb: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    seed: int,
    filtered_rank_edges: int | None = 200,
    node_features: torch.Tensor | None = None,
    tgb_test_negative_sets: dict[tuple[int, int, int], np.ndarray] | None = None,
    tgb_eval_edges: int | None = None,
) -> dict[str, float]:
    model.to(device)
    node_text_emb = node_text_emb.to(device)
    rel_text_emb = rel_text_emb.to(device)
    node_features = node_features.to(device) if node_features is not None else None
    known_all = KnownFacts.from_edges(dataset.edges)
    sampler = NegativeSampler(
        dataset.num_nodes,
        known_all,
        splits.train,
        mode=config.negative_mode_eval,
        filter_scope=config.filter_scope,
        seed=seed + 809,
    )
    history_pool = splits.train.concat(splits.val, sort=False)
    labels, scores = combine_scores_for_binary_eval(
        model,
        splits.test,
        history_pool,
        sampler,
        node_text_emb,
        rel_text_emb,
        device,
        num_neg_per_pos=config.num_neg_eval,
        batch_size=config.batch_size,
        num_relations=dataset.num_relations,
        node_features=node_features,
    )
    metrics = {f"test_{k}": v for k, v in binary_metrics(labels, scores).items()}
    if filtered_rank_edges is not None and filtered_rank_edges > 0:
        rank_metrics = filtered_ranking_eval(
            model,
            splits.test,
            history_pool,
            known_all,
            node_text_emb,
            rel_text_emb,
            device,
            batch_size_candidates=_candidate_batch_size(config),
            num_relations=dataset.num_relations,
            filter_scope=config.filter_scope,
            max_eval_edges=filtered_rank_edges,
            node_features=node_features,
        )
        metrics.update({f"test_{k}": v for k, v in rank_metrics.items()})
    if tgb_test_negative_sets is not None:
        tgb_metrics = tgb_tail_ranking_eval(
            model,
            splits.test,
            history_pool,
            tgb_test_negative_sets,
            node_text_emb,
            rel_text_emb,
            device,
            batch_size_candidates=_candidate_batch_size(config),
            num_relations=dataset.num_relations,
            node_features=node_features,
            max_eval_edges=tgb_eval_edges,
            raw_node_ids=dataset.raw_node_ids,
            raw_relation_ids=dataset.raw_relation_ids,
            tgb_first_dst_id=getattr(dataset, "tgb_first_dst_id", None),
            tgb_last_dst_id=getattr(dataset, "tgb_last_dst_id", None),
        )
        metrics.update({f"test_{k}": v for k, v in tgb_metrics.items()})
    return metrics
