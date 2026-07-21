from __future__ import annotations

import torch
from torch.nn import functional as F

from ssptgfm.model import SSPTGFMOutput


def contrastive_align_loss(
    struct_nodes: torch.Tensor,
    sem_nodes: torch.Tensor,
    node_ids: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    node_ids = torch.unique(node_ids)
    if node_ids.numel() < 2:
        return struct_nodes.new_tensor(0.0)
    z = F.normalize(struct_nodes[node_ids], dim=-1)
    e = F.normalize(sem_nodes[node_ids], dim=-1)
    logits = z @ e.T / max(temperature, 1e-6)
    target = torch.arange(node_ids.numel(), device=logits.device)
    return F.cross_entropy(logits, target, reduction="sum")


def gaussian_kl(
    q_mu: torch.Tensor,
    q_logvar: torch.Tensor,
    p_mu: torch.Tensor,
    p_logvar: torch.Tensor,
    node_ids: torch.Tensor,
) -> torch.Tensor:
    node_ids = torch.unique(node_ids)
    if node_ids.numel() == 0:
        return q_mu.new_tensor(0.0)
    qm = q_mu[node_ids]
    qlv = q_logvar[node_ids]
    pm = p_mu[node_ids]
    plv = p_logvar[node_ids]
    kl = 0.5 * (plv - qlv + (torch.exp(qlv) + (qm - pm).pow(2)) / torch.exp(plv).clamp_min(1e-8) - 1.0)
    return kl.sum(dim=-1).sum()


def total_loss(
    output: SSPTGFMOutput,
    labels: torch.Tensor,
    edge_nodes: torch.Tensor,
    lambda_align: float = 0.1,
    lambda_kl: float = 0.001,
    lambda_ood: float = 0.0,
    lambda_meta: float = 0.0,
    meta_loss: torch.Tensor | None = None,
    align_temperature: float = 0.2,
    lambda_struct_aux: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    final_score = output.final_score.float()
    gate = output.gate.float()
    lp = F.binary_cross_entropy_with_logits(final_score, labels.float())
    active_nodes = torch.unique(edge_nodes)
    align = contrastive_align_loss(output.struct_nodes.float(), output.sem_nodes.float(), active_nodes, temperature=align_temperature)
    kl = gaussian_kl(output.q_mu.float(), output.q_logvar.float(), output.p_mu.float(), output.p_logvar.float(), active_nodes)
    # Encourage confident gates away from 0.5 only when explicitly requested.
    ood_reg = (gate * (1.0 - gate)).mean()
    meta = final_score.new_tensor(0.0) if meta_loss is None else meta_loss.float()
    struct_aux = final_score.new_tensor(0.0)
    if lambda_struct_aux > 0.0 and output.struct_feature_score is not None:
        struct_aux = F.binary_cross_entropy_with_logits(output.struct_feature_score.float(), labels.float())
    loss = (
        lp
        + lambda_align * align
        + lambda_kl * kl
        + lambda_meta * meta
        + lambda_ood * ood_reg
        + lambda_struct_aux * struct_aux
    )
    parts = {
        "loss": float(loss.detach().cpu()),
        "lp": float(lp.detach().cpu()),
        "align": float(align.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "ood_reg": float(ood_reg.detach().cpu()),
        "meta": float(meta.detach().cpu()),
        "struct_aux": float(struct_aux.detach().cpu()),
        "weighted_align": float((lambda_align * align).detach().cpu()),
        "weighted_kl": float((lambda_kl * kl).detach().cpu()),
        "weighted_ood_reg": float((lambda_ood * ood_reg).detach().cpu()),
        "weighted_meta": float((lambda_meta * meta).detach().cpu()),
        "weighted_struct_aux": float((lambda_struct_aux * struct_aux).detach().cpu()),
    }
    return loss, parts
