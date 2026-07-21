from __future__ import annotations

from collections import deque

import torch

from ssptgfm.data import EdgeTensor, TemporalDataset
from ssptgfm.features import GraphHistoryIndex


def shortest_evidence_path(history: EdgeTensor, src: int, dst: int, max_depth: int = 3) -> list[int]:
    index = GraphHistoryIndex(int(max(int(history.src.max()), int(history.dst.max()), src, dst) + 1), history) if len(history) else None
    if index is None:
        return []
    queue: deque[tuple[int, list[int]]] = deque([(src, [src])])
    seen = {src}
    while queue:
        node, path = queue.popleft()
        if len(path) - 1 >= max_depth:
            continue
        for nxt in index.adj[node]:
            if nxt == dst:
                return path + [dst]
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + [nxt]))
    return []


def semantic_evidence_terms(dataset: TemporalDataset, src: int, dst: int, rel: int, top_k: int = 8) -> list[str]:
    text = f"{dataset.node_texts[src]} {dataset.node_texts[dst]} {dataset.relation_texts[rel]}"
    tokens = []
    seen = set()
    for raw in text.replace("_", " ").replace("-", " ").split():
        tok = "".join(ch for ch in raw.lower() if ch.isalnum())
        if len(tok) < 3 or tok in seen:
            continue
        seen.add(tok)
        tokens.append(tok)
        if len(tokens) >= top_k:
            break
    return tokens


def explain_edge(dataset: TemporalDataset, history: EdgeTensor, edge: EdgeTensor) -> dict[str, object]:
    if len(edge) != 1:
        raise ValueError("explain_edge expects a single edge")
    src = int(edge.src[0])
    dst = int(edge.dst[0])
    rel = int(edge.rel[0])
    return {
        "edge": {
            "src": src,
            "dst": dst,
            "rel": rel,
            "time": float(edge.time[0]),
            "relation_text": dataset.relation_texts[rel],
        },
        "structural_path": shortest_evidence_path(history.cpu(), src, dst),
        "semantic_terms": semantic_evidence_terms(dataset, src, dst, rel),
    }
