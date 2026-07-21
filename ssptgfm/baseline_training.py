from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from time import perf_counter

from ssptgfm.baselines import ComplExBaseline, DistMultBaseline, LMMLPBaseline, StructureMLPBaseline, TemporalDistMultBaseline
from ssptgfm.data import EdgeTensor, TemporalDataset, TemporalSplits
from ssptgfm.features import GraphHistoryIndex
from ssptgfm.metrics import binary_metrics, ranking_metrics
from ssptgfm.negative_sampling import KnownFacts, NegativeSampler, make_labels
from ssptgfm.training import TrainConfig, _struct_features, _time_grouped_batches, causal_history_for_batch


def _requires_struct_features(model: nn.Module) -> bool:
    return bool(getattr(model, "requires_struct_features", True))


def _history_index_by_time(num_nodes: int, history_pool: EdgeTensor, query_edges: EdgeTensor) -> dict[float, tuple[EdgeTensor, GraphHistoryIndex]]:
    by_time: dict[float, tuple[EdgeTensor, GraphHistoryIndex]] = {}
    for time_value in torch.unique(query_edges.time).cpu().tolist():
        hist = history_pool.before(float(time_value), strict=True)
        by_time[float(time_value)] = (hist, GraphHistoryIndex(num_nodes, hist))
    return by_time


def _advance_history_index(
    num_nodes: int,
    history_pool: EdgeTensor,
    time_value: float,
    hist_index: GraphHistoryIndex | None,
    cursor: int,
) -> tuple[EdgeTensor, GraphHistoryIndex, int]:
    if hist_index is None:
        hist_index = GraphHistoryIndex(num_nodes, history_pool.slice([]))
    while cursor < len(history_pool) and float(history_pool.time[cursor]) < time_value:
        start = cursor
        cursor += 1
        while cursor < len(history_pool) and float(history_pool.time[cursor]) < time_value:
            cursor += 1
        hist_index.add_edges(history_pool.slice(slice(start, cursor)))
    return history_pool.slice(slice(0, cursor)), hist_index, cursor


def build_baseline(name: str, dataset: TemporalDataset, text_dim: int, hidden_dim: int) -> nn.Module:
    if name == "lm_mlp":
        return LMMLPBaseline(text_dim=text_dim, num_relations=dataset.num_relations, hidden_dim=hidden_dim)
    if name == "structure_mlp":
        return StructureMLPBaseline(num_relations=dataset.num_relations, hidden_dim=hidden_dim)
    if name == "distmult":
        return DistMultBaseline(num_nodes=dataset.num_nodes, num_relations=dataset.num_relations, hidden_dim=hidden_dim)
    if name == "complex":
        return ComplExBaseline(num_nodes=dataset.num_nodes, num_relations=dataset.num_relations, hidden_dim=hidden_dim)
    if name == "temporal_distmult":
        return TemporalDistMultBaseline(num_nodes=dataset.num_nodes, num_relations=dataset.num_relations, hidden_dim=hidden_dim)
    raise ValueError(f"unknown baseline: {name}")


def _score_batch(
    model: nn.Module,
    batch: EdgeTensor,
    hist: EdgeTensor,
    node_text_emb: torch.Tensor,
    rel_text_emb: torch.Tensor,
) -> torch.Tensor:
    sf = None
    if _requires_struct_features(model):
        hist_index = GraphHistoryIndex(node_text_emb.size(0), hist)
        sf, _ = _struct_features(hist_index, batch, node_text_emb.device)
    return model(batch, node_text_emb=node_text_emb, rel_text_emb=rel_text_emb, struct_features=sf)


def train_baseline_one_seed(
    dataset: TemporalDataset,
    splits: TemporalSplits,
    model: nn.Module,
    node_text_emb: torch.Tensor,
    rel_text_emb: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, dict[str, float]]:
    model.to(device)
    node_text_emb = node_text_emb.to(device)
    rel_text_emb = rel_text_emb.to(device)
    known_train = KnownFacts.from_edges(splits.train)
    known_val = KnownFacts.from_edges(splits.train.concat(splits.val, sort=False))
    train_sampler = NegativeSampler(dataset.num_nodes, known_train, splits.train, config.negative_mode_train, config.filter_scope, seed)
    val_sampler = NegativeSampler(dataset.num_nodes, known_val, splits.train, config.negative_mode_eval, config.filter_scope, seed + 17)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_edges = splits.train.sort_by_time()
    best_auc = -float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    stale = 0
    needs_struct = _requires_struct_features(model)
    for epoch in range(1, config.epochs + 1):
        epoch_start = perf_counter()
        print(
            {
                "event": "baseline_epoch_start",
                "model": model.__class__.__name__,
                "epoch": epoch,
                "epochs": config.epochs,
                "requires_struct_features": needs_struct,
            },
            flush=True,
        )
        model.train()
        hist_index = GraphHistoryIndex(dataset.num_nodes, train_edges.slice([])) if needs_struct else None
        cursor = 0
        time_groups = _time_grouped_batches(train_edges, batch_size=config.batch_size, shuffle_within_time=True, seed=seed + epoch)
        for time_group_idx, (time_value, batches) in enumerate(time_groups, start=1):
            if needs_struct:
                hist, hist_index, cursor = _advance_history_index(dataset.num_nodes, train_edges, float(time_value), hist_index, cursor)
            else:
                hist = train_edges.before(time_value, strict=True)
                hist_index = None
            if time_group_idx == 1 or time_group_idx == len(time_groups) or time_group_idx % 10 == 0:
                print(
                    {
                        "event": "baseline_epoch_time_group",
                        "model": model.__class__.__name__,
                        "epoch": epoch,
                        "time_group": time_group_idx,
                        "time_groups": len(time_groups),
                        "batches": len(batches),
                    },
                    flush=True,
                )
            for idx in batches:
                pos = train_edges.slice(idx.cpu())
                neg = train_sampler.sample(pos, config.num_neg_train, hist)
                batch = pos.concat(neg, sort=False)
                labels = make_labels(len(pos), len(neg), device)
                sf = None
                if hist_index is not None:
                    sf, _ = _struct_features(hist_index, batch, node_text_emb.device)
                scores = model(batch, node_text_emb=node_text_emb, rel_text_emb=rel_text_emb, struct_features=sf)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(scores, labels)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                opt.step()
        print(
            {
                "event": "baseline_validation_start",
                "model": model.__class__.__name__,
                "epoch": epoch,
            },
            flush=True,
        )
        val = evaluate_baseline_binary(dataset, splits.val, splits.train, model, node_text_emb, rel_text_emb, val_sampler, config, device)
        print(
            {
                "event": "baseline_epoch_end",
                "model": model.__class__.__name__,
                "epoch": epoch,
                "val_auc": val.get("auc"),
                "elapsed_sec": perf_counter() - epoch_start,
            },
            flush=True,
        )
        if np.isfinite(val.get("auc", np.nan)) and val["auc"] > best_auc:
            best_auc = val["auc"]
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    val = evaluate_baseline_binary(dataset, splits.val, splits.train, model, node_text_emb, rel_text_emb, val_sampler, config, device)
    return model, {f"val_{k}": v for k, v in val.items()}


def evaluate_baseline_binary(
    dataset: TemporalDataset,
    positives: EdgeTensor,
    history_pool: EdgeTensor,
    model: nn.Module,
    node_text_emb: torch.Tensor,
    rel_text_emb: torch.Tensor,
    sampler: NegativeSampler,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    labels_all: list[np.ndarray] = []
    scores_all: list[np.ndarray] = []
    with torch.no_grad():
        positives = positives.sort_by_time()
        history_pool = history_pool.sort_by_time()
        needs_struct = _requires_struct_features(model)
        hist_index = GraphHistoryIndex(dataset.num_nodes, history_pool.slice([])) if needs_struct else None
        cursor = 0
        for time_value, batches in _time_grouped_batches(positives, batch_size=config.batch_size, shuffle_within_time=False, seed=0):
            if needs_struct:
                hist, hist_index, cursor = _advance_history_index(dataset.num_nodes, history_pool, float(time_value), hist_index, cursor)
            else:
                hist = history_pool.before(time_value, strict=True)
                hist_index = None
            for idx in batches:
                pos = positives.slice(idx.cpu())
                neg = sampler.sample(pos, config.num_neg_eval, hist)
                batch = pos.concat(neg, sort=False)
                labels = make_labels(len(pos), len(neg), device)
                sf = None
                if hist_index is not None:
                    sf, _ = _struct_features(hist_index, batch, node_text_emb.device)
                scores = model(batch, node_text_emb=node_text_emb, rel_text_emb=rel_text_emb, struct_features=sf)
                labels_all.append(labels.cpu().numpy())
                scores_all.append(scores.detach().cpu().numpy())
    return binary_metrics(np.concatenate(labels_all), np.concatenate(scores_all))


def evaluate_baseline_ranking(
    dataset: TemporalDataset,
    positives: EdgeTensor,
    history_pool: EdgeTensor,
    known_facts: KnownFacts,
    model: nn.Module,
    node_text_emb: torch.Tensor,
    rel_text_emb: torch.Tensor,
    config: TrainConfig,
    filter_scope: str = "exact",
    max_eval_edges: int | None = 200,
) -> dict[str, float]:
    model.eval()
    ranks: list[float] = []
    eval_count = len(positives) if max_eval_edges is None else min(len(positives), max_eval_edges)
    with torch.no_grad():
        positives = positives.sort_by_time()
        history_pool = history_pool.sort_by_time()
        needs_struct = _requires_struct_features(model)
        hist_index = GraphHistoryIndex(dataset.num_nodes, history_pool.slice([])) if needs_struct else None
        cursor = 0
        for idx in range(eval_count):
            pos = positives.slice([idx])
            if needs_struct:
                hist, hist_index, cursor = _advance_history_index(
                    dataset.num_nodes,
                    history_pool,
                    float(pos.time[0]),
                    hist_index,
                    cursor,
                )
            else:
                hist = causal_history_for_batch(history_pool, pos)
                hist_index = None
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
                    num_nodes=dataset.num_nodes,
                    scope=filter_scope,
                )
                if true_index is None:
                    continue
                cand_nodes = torch.nonzero(keep, as_tuple=False).view(-1).tolist()
                scores = []
                for start in range(0, len(cand_nodes), config.batch_size):
                    chunk_nodes = cand_nodes[start : start + config.batch_size]
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
                    if hist_index is None:
                        sf = None
                        score = model(batch, node_text_emb=node_text_emb, rel_text_emb=rel_text_emb, struct_features=sf)
                    elif node_text_emb.device.type == "cuda" and dataset.num_nodes <= 4096:
                        sf, _ = hist_index.features_for_candidates_tensor(chunk, node_text_emb.device)
                        score = model(batch, node_text_emb=node_text_emb, rel_text_emb=rel_text_emb, struct_features=sf)
                    elif node_text_emb.device.type == "cuda":
                        sf, _ = hist_index.features_for_candidates_sparse_ppr(chunk, node_text_emb.device)
                        score = model(batch, node_text_emb=node_text_emb, rel_text_emb=rel_text_emb, struct_features=sf)
                    else:
                        sf, _ = hist_index.features_for_candidates(s, d, chunk)
                        score = model(batch, node_text_emb=node_text_emb, rel_text_emb=rel_text_emb, struct_features=sf)
                    scores.append(score.detach().cpu())
                all_scores = torch.cat(scores).numpy()
                true_score = all_scores[true_index]
                ranks.append(1.0 + float(np.sum(all_scores > true_score)) + 0.5 * float(np.sum(all_scores == true_score) - 1))
    return ranking_metrics(ranks)
