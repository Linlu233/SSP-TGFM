from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ssptgfm.data import EdgeTensor


class LMMLPBaseline(nn.Module):
    requires_struct_features = False

    def __init__(self, text_dim: int, num_relations: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        self.node_proj = nn.Linear(text_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        edges: EdgeTensor,
        node_text_emb: torch.Tensor,
        rel_text_emb: torch.Tensor | None = None,
        struct_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edges = edges.to(node_text_emb.device)
        u = F.normalize(self.node_proj(node_text_emb[edges.src].float()), dim=-1)
        v = F.normalize(self.node_proj(node_text_emb[edges.dst].float()), dim=-1)
        r = self.rel_emb(edges.rel)
        sim = F.cosine_similarity(u, v, dim=-1, eps=1e-8).unsqueeze(-1)
        return self.scorer(torch.cat([u, v, r, u * v, sim], dim=-1)).squeeze(-1)


class StructureMLPBaseline(nn.Module):
    requires_struct_features = True

    def __init__(self, num_relations: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + 5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        edges: EdgeTensor,
        node_text_emb: torch.Tensor,
        rel_text_emb: torch.Tensor | None = None,
        struct_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if struct_features is None:
            raise ValueError("StructureMLPBaseline requires struct_features")
        edges = edges.to(struct_features.device)
        r = self.rel_emb(edges.rel)
        return self.scorer(torch.cat([r, struct_features.to(r.device).float()], dim=-1)).squeeze(-1)


class DistMultBaseline(nn.Module):
    requires_struct_features = False

    def __init__(self, num_nodes: int, num_relations: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(
        self,
        edges: EdgeTensor,
        node_text_emb: torch.Tensor,
        rel_text_emb: torch.Tensor | None = None,
        struct_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edges = edges.to(node_text_emb.device)
        src = self.node_emb(edges.src)
        rel = self.rel_emb(edges.rel)
        dst = self.node_emb(edges.dst)
        return torch.sum(src * rel * dst, dim=-1)


class ComplExBaseline(nn.Module):
    requires_struct_features = False

    def __init__(self, num_nodes: int, num_relations: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.node_re = nn.Embedding(num_nodes, hidden_dim)
        self.node_im = nn.Embedding(num_nodes, hidden_dim)
        self.rel_re = nn.Embedding(num_relations, hidden_dim)
        self.rel_im = nn.Embedding(num_relations, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in (self.node_re, self.node_im, self.rel_re, self.rel_im):
            nn.init.xavier_uniform_(emb.weight)

    def forward(
        self,
        edges: EdgeTensor,
        node_text_emb: torch.Tensor,
        rel_text_emb: torch.Tensor | None = None,
        struct_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edges = edges.to(node_text_emb.device)
        sr = self.node_re(edges.src)
        si = self.node_im(edges.src)
        rr = self.rel_re(edges.rel)
        ri = self.rel_im(edges.rel)
        dr = self.node_re(edges.dst)
        di = self.node_im(edges.dst)
        return torch.sum(sr * rr * dr + si * rr * di + sr * ri * di - si * ri * dr, dim=-1)


class TemporalDistMultBaseline(nn.Module):
    requires_struct_features = False

    def __init__(self, num_nodes: int, num_relations: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        for module in self.time_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _time_features(time: torch.Tensor) -> torch.Tensor:
        signed_log = torch.sign(time) * torch.log1p(torch.abs(time))
        return torch.stack(
            [
                signed_log / 10.0,
                torch.sin(signed_log),
                torch.cos(signed_log),
                torch.sin(signed_log / 7.0),
                torch.cos(signed_log / 7.0),
                torch.sign(time),
            ],
            dim=-1,
        )

    def forward(
        self,
        edges: EdgeTensor,
        node_text_emb: torch.Tensor,
        rel_text_emb: torch.Tensor | None = None,
        struct_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edges = edges.to(node_text_emb.device)
        src = self.node_emb(edges.src)
        rel = self.rel_emb(edges.rel) + self.time_proj(self._time_features(edges.time.float()))
        dst = self.node_emb(edges.dst)
        return torch.sum(src * rel * dst, dim=-1)
