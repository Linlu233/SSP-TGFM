from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from scipy.stats import wilcoxon


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = labels.astype(np.int32)
    scores = scores.astype(np.float64)
    out: dict[str, float] = {}
    if np.unique(labels).size > 1:
        out["auc"] = float(roc_auc_score(labels, scores))
        out["ap"] = float(average_precision_score(labels, scores))
    else:
        out["auc"] = float("nan")
        out["ap"] = float("nan")
    try:
        out["ndcg"] = float(ndcg_score(labels.reshape(1, -1), scores.reshape(1, -1)))
    except ValueError:
        out["ndcg"] = float("nan")
    return out


def ranking_metrics(ranks: list[float], hits: tuple[int, ...] = (1, 10, 50, 100)) -> dict[str, float]:
    if not ranks:
        base = {"mrr": float("nan")}
        base.update({f"hits@{k}": float("nan") for k in hits})
        return base
    arr = np.asarray(ranks, dtype=np.float64)
    out = {"mrr": float(np.mean(1.0 / arr))}
    for k in hits:
        out[f"hits@{k}"] = float(np.mean(arr <= k))
    return out


def summarize_seed_metrics(seed_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not seed_metrics:
        return {}
    keys = sorted({k for row in seed_metrics for k in row})
    out: dict[str, float] = {}
    for key in keys:
        vals = np.asarray([row[key] for row in seed_metrics if key in row and np.isfinite(row[key])], dtype=np.float64)
        if vals.size == 0:
            continue
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        out[f"{key}_ci95"] = float(1.96 * out[f"{key}_std"] / np.sqrt(max(vals.size, 1)))
    return out


def paired_significance(
    baseline: list[float] | np.ndarray,
    candidate: list[float] | np.ndarray,
) -> dict[str, float]:
    base = np.asarray(baseline, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    mask = np.isfinite(base) & np.isfinite(cand)
    base = base[mask]
    cand = cand[mask]
    if base.size < 2:
        return {"n": float(base.size), "wilcoxon_p": float("nan"), "cohens_dz": float("nan")}
    diff = cand - base
    try:
        p_value = float(wilcoxon(diff).pvalue)
    except ValueError:
        p_value = 1.0
    std = float(diff.std(ddof=1))
    dz = float(diff.mean() / std) if std > 0 else float("inf") if diff.mean() > 0 else 0.0
    return {"n": float(base.size), "wilcoxon_p": p_value, "cohens_dz": dz}
