from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ssptgfm.data import EdgeTensor
from ssptgfm.time_encoding import TimeEncoder


class LowRankLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int = 16, bias: bool = True) -> None:
        super().__init__()
        rank = min(rank, in_dim, out_dim)
        self.left = nn.Parameter(torch.empty(in_dim, rank))
        self.right = nn.Parameter(torch.empty(rank, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        nn.init.xavier_uniform_(self.left)
        nn.init.xavier_uniform_(self.right)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.left @ self.right
        if self.bias is not None:
            y = y + self.bias
        return y


class BilinearLowRank(nn.Module):
    def __init__(self, num_relations: int, dim: int, rank: int = 16) -> None:
        super().__init__()
        self.left = nn.Parameter(torch.empty(num_relations, rank, dim))
        self.right = nn.Parameter(torch.empty(num_relations, rank, dim))
        nn.init.xavier_uniform_(self.left)
        nn.init.xavier_uniform_(self.right)

    def forward(self, x: torch.Tensor, rel: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        left = self.left[rel]
        right = self.right[rel]
        lx = torch.einsum("bd,brd->br", x, left)
        ry = torch.einsum("bd,brd->br", y, right)
        return (lx * ry).sum(dim=-1)


class CrossPromptBlock(nn.Module):
    def __init__(self, num_relations: int, dim: int, prompt_tokens: int = 4, heads: int = 4) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by prompt attention heads")
        self.heads = int(heads)
        self.head_dim = int(dim // heads)
        self.scale = self.head_dim**-0.5
        self.prompts = nn.Parameter(torch.randn(num_relations, prompt_tokens, dim) * 0.02)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        p = self.prompts[rel]
        q = self.q_proj(x).view(bsz, 1, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(p).view(bsz, p.size(1), self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(p).view(bsz, p.size(1), self.heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(bsz, -1)
        out = self.out_proj(out)
        y = self.norm(x + out)
        return self.norm(y + self.ffn(y))


class TemporalGraphEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        dim: int,
        time_dim: int = 32,
        num_layers: int = 2,
        mode: str = "mlp",
        time_mode: str = "fourier",
        node_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        self.num_layers = num_layers
        self.mode = mode
        self.node_emb = nn.Embedding(num_nodes, dim)
        self.rel_emb = nn.Embedding(num_relations, dim)
        self.time_encoder = TimeEncoder(time_dim, mode=time_mode)
        self.feature_proj = nn.Linear(node_feature_dim, dim) if node_feature_dim else None
        self.history_chunk_size = 262_144
        self.gradient_checkpoint = False
        self.msg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim * 2 + time_dim, dim * 2),
                    nn.GELU(),
                    nn.Linear(dim * 2, dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.update = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim * 2, dim * 2),
                    nn.GELU(),
                    nn.Linear(dim * 2, dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        if mode == "lstm":
            self.recurrent = nn.ModuleList([nn.GRUCell(dim, dim) for _ in range(num_layers)])
        elif mode == "transformer":
            self.recurrent = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=dim,
                        nhead=4,
                        dim_feedforward=dim * 2,
                        dropout=0.1,
                        batch_first=True,
                    )
                    for _ in range(num_layers)
                ]
            )
        elif mode == "mamba":
            try:
                from mamba_ssm import Mamba
            except ImportError as exc:
                raise ImportError("temporal_encoder=mamba requires installing mamba-ssm") from exc
            self.recurrent = nn.ModuleList([Mamba(d_model=dim, d_state=16, d_conv=4, expand=2) for _ in range(num_layers)])
        elif mode != "mlp":
            raise ValueError(f"unknown temporal encoder mode: {mode}")
        else:
            self.recurrent = nn.ModuleList()

    def forward(
        self,
        history: EdgeTensor,
        query_time: torch.Tensor,
        node_features: torch.Tensor | None = None,
        history_degree: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.node_emb.weight
        if self.feature_proj is not None:
            if node_features is None:
                raise ValueError("node features are required by this encoder")
            h = h + self.feature_proj(node_features.to(h.device).float())
        if len(history) == 0:
            return h
        q_time = query_time.to(h.device).float().view(-1).min()
        chunk_size = max(1, int(self.history_chunk_size))
        if history_degree is None:
            deg = torch.zeros(h.size(0), 1, device=h.device)
            for start in range(0, len(history), chunk_size):
                end = min(start + chunk_size, len(history))
                src = history.src[start:end].to(h.device)
                dst = history.dst[start:end].to(h.device)
                one = torch.ones(src.size(0), 1, device=h.device)
                deg.index_add_(0, dst, one)
                deg.index_add_(0, src, one)
        else:
            deg = history_degree.to(h.device, dtype=torch.float32)
            if deg.dim() == 1:
                deg = deg.unsqueeze(-1)
        for layer in range(self.num_layers):
            agg = torch.zeros_like(h)
            for start in range(0, len(history), chunk_size):
                end = min(start + chunk_size, len(history))
                src = history.src[start:end].to(h.device)
                dst = history.dst[start:end].to(h.device)
                rel_idx = history.rel[start:end].to(h.device)
                tm = history.time[start:end].to(h.device)
                if self.gradient_checkpoint and self.training:
                    def chunk_messages(
                        h_chunk: torch.Tensor,
                        layer_i: int = layer,
                        src_i: torch.Tensor = src,
                        dst_i: torch.Tensor = dst,
                        rel_idx_i: torch.Tensor = rel_idx,
                        tm_i: torch.Tensor = tm,
                        q_time_i: torch.Tensor = q_time,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        return self._chunk_messages(layer_i, h_chunk, src_i, dst_i, rel_idx_i, tm_i, q_time_i)

                    src_msg, dst_msg = checkpoint(
                        chunk_messages,
                        h,
                        use_reentrant=False,
                    )
                else:
                    src_msg, dst_msg = self._chunk_messages(layer, h, src, dst, rel_idx, tm, q_time)
                if agg.dtype != src_msg.dtype:
                    agg = agg.to(src_msg.dtype)
                agg.index_add_(0, dst, src_msg)
                agg.index_add_(0, src, dst_msg)
            agg = agg / deg.to(agg.dtype).clamp_min(1.0)
            if self.mode == "mlp":
                new_h = self.update[layer](torch.cat([h, agg], dim=-1))
            elif self.mode == "lstm":
                new_h = self.recurrent[layer](agg, h)
            elif self.mode in {"transformer", "mamba"}:
                seq = torch.stack([h, agg], dim=1)
                new_h = self.recurrent[layer](seq)[:, -1, :]
            else:
                raise RuntimeError("unreachable encoder mode")
            h = self.norm[layer](h + new_h)
        return h

    def _chunk_messages(
        self,
        layer: int,
        h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        rel_idx: torch.Tensor,
        tm: torch.Tensor,
        q_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = torch.clamp(q_time.float().view(-1).min() - tm.float(), min=0.0)
        t_enc = self.time_encoder(delta)
        rel = self.rel_emb(rel_idx)
        src_msg = self.msg[layer](torch.cat([h[src], rel, t_enc], dim=-1))
        dst_msg = self.msg[layer](torch.cat([h[dst], rel, t_enc], dim=-1))
        return src_msg, dst_msg


@dataclass
class SSPTGFMOutput:
    final_score: torch.Tensor
    struct_score: torch.Tensor
    sem_score: torch.Tensor
    cross_score: torch.Tensor
    gate: torch.Tensor
    struct_nodes: torch.Tensor
    sem_nodes: torch.Tensor
    q_mu: torch.Tensor
    q_logvar: torch.Tensor
    p_mu: torch.Tensor
    p_logvar: torch.Tensor
    history_prior_score: torch.Tensor | None = None
    history_prior_gate: torch.Tensor | None = None
    struct_feature_score: torch.Tensor | None = None
    relation_entity_prior_score: torch.Tensor | None = None


class SSPTGFM(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        text_dim: int,
        hidden_dim: int = 128,
        time_dim: int = 32,
        prompt_tokens: int = 4,
        prompt_heads: int = 4,
        relation_rank: int = 16,
        adapter_rank: int = 16,
        temporal_layers: int = 2,
        temporal_encoder: str = "mlp",
        time_encoder: str = "fourier",
        node_feature_dim: int | None = None,
        use_struct: bool = True,
        use_sem: bool = True,
        use_cross: bool = True,
        use_gate: bool = True,
        use_variational: bool = True,
        use_history_prior: bool = False,
        history_prior_dim: int = 12,
        history_prior_hidden_dim: int | None = None,
        history_prior_init_scale: float = 0.05,
        history_prior_mode: str = "mlp",
        history_prior_weights: list[float] | tuple[float, ...] | None = None,
        freeze_history_prior: bool = False,
        history_prior_layer_norm: bool = False,
        use_history_prior_gate: bool = False,
        history_prior_gate_hidden_dim: int | None = None,
        history_prior_gate_init_bias: float = 0.0,
        use_struct_feature_residual: bool = False,
        struct_feature_hidden_dim: int | None = None,
        struct_feature_init_scale: float = 0.1,
        use_relation_entity_prior: bool = False,
        relation_entity_prior_rank: int = 16,
        relation_entity_prior_init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.use_struct = use_struct
        self.use_sem = use_sem
        self.use_cross = use_cross
        self.use_gate = use_gate
        self.use_variational = use_variational
        self.use_history_prior = use_history_prior
        self.use_struct_feature_residual = use_struct_feature_residual
        self.use_history_prior_gate = use_history_prior_gate
        self.use_relation_entity_prior = use_relation_entity_prior
        self.history_prior_dim = int(history_prior_dim)
        self.history_prior_mode = history_prior_mode
        self.struct_encoder = TemporalGraphEncoder(
            num_nodes=num_nodes,
            num_relations=num_relations,
            dim=hidden_dim,
            time_dim=time_dim,
            num_layers=temporal_layers,
            mode=temporal_encoder,
            time_mode=time_encoder,
            node_feature_dim=node_feature_dim,
        )
        self.text_proj = LowRankLinear(text_dim, hidden_dim, rank=adapter_rank)
        self.rel_text_proj = LowRankLinear(text_dim, hidden_dim, rank=adapter_rank)
        self.prompt_s = CrossPromptBlock(num_relations, hidden_dim, prompt_tokens, prompt_heads)
        self.prompt_e = CrossPromptBlock(num_relations, hidden_dim, prompt_tokens, prompt_heads)
        self.struct_bilinear = BilinearLowRank(num_relations, hidden_dim, relation_rank)
        self.sem_bilinear = BilinearLowRank(num_relations, hidden_dim, relation_rank)
        self.cross_bilinear = BilinearLowRank(num_relations, hidden_dim, relation_rank)
        self.time_score = nn.Sequential(nn.Linear(time_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.time_encoder = TimeEncoder(time_dim, mode=time_encoder)
        self.struct_conf = nn.Linear(5, 1)
        self.sem_conf = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate = nn.Linear(4, 1)
        with torch.no_grad():
            self.gate.weight.copy_(torch.tensor([[1.0, -1.0, 1.0, -1.0]]))
            self.gate.bias.zero_()
        self.q_mu = nn.Linear(hidden_dim, hidden_dim)
        self.q_logvar = nn.Linear(hidden_dim, hidden_dim)
        self.p_mu = nn.Linear(hidden_dim, hidden_dim)
        self.p_logvar = nn.Linear(hidden_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        if self.use_history_prior:
            if history_prior_mode == "linear":
                self.history_prior = nn.Linear(self.history_prior_dim, 1, bias=False)
            elif history_prior_mode == "mlp":
                prior_hidden = int(history_prior_hidden_dim or max(8, hidden_dim // 4))
                self.history_prior = nn.Sequential(
                    nn.Linear(self.history_prior_dim, prior_hidden),
                    nn.GELU(),
                    nn.Linear(prior_hidden, 1),
                )
            else:
                raise ValueError(f"unknown history_prior_mode: {history_prior_mode}")
            self.history_prior_scale = nn.Parameter(torch.tensor(float(history_prior_init_scale)))
            self.history_prior_norm = nn.LayerNorm(self.history_prior_dim) if history_prior_layer_norm else nn.Identity()
            with torch.no_grad():
                weights = torch.tensor(
                    history_prior_weights
                    or [1.0, 0.7, 0.5, 0.25, 0.25, 0.1, 0.8, 0.8, 0.6, 0.5, 0.3, 0.3],
                    dtype=torch.float32,
                )
                if isinstance(self.history_prior, nn.Linear):
                    self.history_prior.weight.zero_()
                    n = min(self.history_prior.weight.size(1), weights.numel())
                    self.history_prior.weight[0, :n] = weights[:n]
                else:
                    prior_hidden = self.history_prior[0].weight.size(0)
                    first = self.history_prior[0]
                    last = self.history_prior[-1]
                    if isinstance(first, nn.Linear):
                        first.weight.zero_()
                        first.bias.zero_()
                        for row in range(min(prior_hidden, self.history_prior_dim)):
                            first.weight[row, row] = 1.0
                    if isinstance(last, nn.Linear):
                        last.weight.zero_()
                        last.bias.zero_()
                        n = min(last.weight.size(1), weights.numel())
                        last.weight[0, :n] = weights[:n]
            if freeze_history_prior:
                for param in self.history_prior.parameters():
                    param.requires_grad = False
                self.history_prior_scale.requires_grad = False
            if self.use_history_prior_gate:
                gate_hidden = int(history_prior_gate_hidden_dim or max(8, hidden_dim // 8))
                self.history_prior_gate = nn.Sequential(
                    nn.Linear(self.history_prior_dim + 8, gate_hidden),
                    nn.GELU(),
                    nn.Linear(gate_hidden, 1),
                )
                with torch.no_grad():
                    last = self.history_prior_gate[-1]
                    if isinstance(last, nn.Linear):
                        last.bias.fill_(float(history_prior_gate_init_bias))
            else:
                self.history_prior_gate = None
        else:
            self.history_prior = None
            self.history_prior_scale = None
            self.history_prior_norm = None
            self.history_prior_gate = None
        if self.use_struct_feature_residual:
            struct_hidden = int(struct_feature_hidden_dim or hidden_dim)
            self.struct_feature_scorer = nn.Sequential(
                nn.Linear(hidden_dim + 5, struct_hidden),
                nn.GELU(),
                nn.Linear(struct_hidden, 1),
            )
            self.struct_feature_scale = nn.Parameter(torch.tensor(float(struct_feature_init_scale)))
        else:
            self.struct_feature_scorer = None
            self.struct_feature_scale = None
        if self.use_relation_entity_prior:
            prior_rank = max(1, int(relation_entity_prior_rank))
            self.head_entity_prior = nn.Embedding(num_nodes, prior_rank)
            self.tail_entity_prior = nn.Embedding(num_nodes, prior_rank)
            self.rel_head_prior = nn.Embedding(num_relations, prior_rank)
            self.rel_tail_prior = nn.Embedding(num_relations, prior_rank)
            self.relation_entity_prior_scale = nn.Parameter(torch.tensor(float(relation_entity_prior_init_scale)))
            for emb in (self.head_entity_prior, self.tail_entity_prior, self.rel_head_prior, self.rel_tail_prior):
                nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        else:
            self.head_entity_prior = None
            self.tail_entity_prior = None
            self.rel_head_prior = None
            self.rel_tail_prior = None
            self.relation_entity_prior_scale = None

    def encode_context(
        self,
        history: EdgeTensor,
        query_time: torch.Tensor,
        node_text_emb: torch.Tensor,
        rel_text_emb: torch.Tensor,
        node_features: torch.Tensor | None = None,
        history_degree: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        device = node_text_emb.device
        history = history.to(device)
        struct_all = self.struct_encoder(history, query_time, node_features=node_features, history_degree=history_degree)
        sem_all = F.normalize(self.text_proj(node_text_emb.float()), dim=-1)
        rel_sem = F.normalize(self.rel_text_proj(rel_text_emb.float()), dim=-1)
        q_mu_all = self.q_mu(struct_all)
        q_logvar_all = self.q_logvar(struct_all).clamp(-8.0, 8.0)
        p_mu_all = self.p_mu(sem_all)
        p_logvar_all = self.p_logvar(sem_all).clamp(-8.0, 8.0)
        if self.training and self.use_variational:
            eps = torch.randn_like(q_mu_all)
            struct_score_nodes = q_mu_all + eps * torch.exp(0.5 * q_logvar_all)
        else:
            struct_score_nodes = q_mu_all
        return {
            "struct_score_nodes": struct_score_nodes,
            "sem_all": sem_all,
            "rel_sem": rel_sem,
            "q_mu_all": q_mu_all,
            "q_logvar_all": q_logvar_all,
            "p_mu_all": p_mu_all,
            "p_logvar_all": p_logvar_all,
        }

    def score_edges_from_context(
        self,
        edges: EdgeTensor,
        struct_features: torch.Tensor,
        ood_s: torch.Tensor | None,
        ood_e: torch.Tensor | None,
        context: dict[str, torch.Tensor],
        history_prior_features: torch.Tensor | None = None,
    ) -> SSPTGFMOutput:
        device = context["sem_all"].device
        edges = edges.to(device)
        struct_features = struct_features.to(device)
        ood_s = torch.zeros(len(edges), 1, device=device) if ood_s is None else ood_s.to(device)
        ood_e = torch.zeros(len(edges), 1, device=device) if ood_e is None else ood_e.to(device)
        struct_score_nodes = context["struct_score_nodes"]
        sem_all = context["sem_all"]
        rel_sem = context["rel_sem"]
        su = struct_score_nodes[edges.src]
        sv = struct_score_nodes[edges.dst]
        eu = sem_all[edges.src]
        ev = sem_all[edges.dst]
        er = rel_sem[edges.rel]
        su_p = self.prompt_s(su, edges.rel)
        sv_p = self.prompt_s(sv, edges.rel)
        eu_p = self.prompt_e(eu, edges.rel)
        ev_p = self.prompt_e(ev, edges.rel)
        struct_bilinear = self.struct_bilinear(su_p, edges.rel, sv_p)
        sem_bilinear = self.sem_bilinear(eu_p, edges.rel, ev_p)
        cross_score = self.cross_bilinear(su_p, edges.rel, ev_p)
        t_score = self.time_score(self.time_encoder(edges.time)).squeeze(-1)
        struct_score = struct_bilinear
        if self.use_cross:
            struct_score = struct_score + self.beta * cross_score
        struct_score = struct_score + self.gamma * t_score
        sem_score = self.alpha * sem_bilinear
        full_score = torch.zeros_like(struct_score)
        if self.use_struct:
            full_score = full_score + struct_score
        if self.use_sem:
            full_score = full_score + sem_score
        c_s = torch.sigmoid(self.struct_conf(struct_features))
        sem_sim = F.cosine_similarity(eu, ev, dim=-1, eps=1e-8).unsqueeze(-1)
        c_e = torch.sigmoid(self.sem_conf(torch.cat([eu, ev, er, sem_sim], dim=-1)))
        gate = torch.sigmoid(self.gate(torch.cat([c_s, ood_s, c_e, ood_e], dim=-1)))
        if self.use_gate and self.use_struct and self.use_sem:
            final_score = gate.squeeze(-1) * struct_score + (1.0 - gate.squeeze(-1)) * sem_score
        else:
            final_score = full_score
        history_prior_score = None
        history_prior_gate = None
        if self.use_history_prior:
            if history_prior_features is None:
                history_prior_features = torch.zeros(len(edges), self.history_prior_dim, device=device)
            history_prior_features = history_prior_features.to(device=device, dtype=struct_score.dtype)
            if self.history_prior is None or self.history_prior_scale is None:
                raise RuntimeError("use_history_prior=True requires initialized history prior modules")
            if self.history_prior_norm is not None:
                history_prior_features = self.history_prior_norm(history_prior_features)
            history_prior_score = self.history_prior(history_prior_features).squeeze(-1)
            history_contribution = history_prior_score
            if self.history_prior_gate is not None:
                gate_inputs = torch.cat(
                    [
                        history_prior_features,
                        history_prior_score.unsqueeze(-1),
                        final_score.detach().unsqueeze(-1),
                        struct_score.detach().unsqueeze(-1),
                        sem_score.detach().unsqueeze(-1),
                        c_s.to(history_prior_features.dtype),
                        ood_s.to(history_prior_features.dtype),
                        c_e.to(history_prior_features.dtype),
                        ood_e.to(history_prior_features.dtype),
                    ],
                    dim=-1,
                )
                history_prior_gate = torch.sigmoid(self.history_prior_gate(gate_inputs)).squeeze(-1)
                history_contribution = history_prior_gate * history_prior_score
            final_score = final_score + self.history_prior_scale.to(final_score.dtype) * history_contribution
        struct_feature_score = None
        if self.use_struct_feature_residual:
            if self.struct_feature_scorer is None or self.struct_feature_scale is None:
                raise RuntimeError("use_struct_feature_residual=True requires initialized structure feature modules")
            rel_struct = self.struct_encoder.rel_emb(edges.rel)
            sf_residual = struct_features.to(device=device, dtype=rel_struct.dtype)
            sf_residual = torch.log1p(torch.clamp(sf_residual, min=0.0))
            struct_feature_score = self.struct_feature_scorer(
                torch.cat([rel_struct, sf_residual], dim=-1)
            ).squeeze(-1)
            final_score = final_score + self.struct_feature_scale.to(final_score.dtype) * struct_feature_score
        relation_entity_prior_score = None
        if self.use_relation_entity_prior:
            if (
                self.head_entity_prior is None
                or self.tail_entity_prior is None
                or self.rel_head_prior is None
                or self.rel_tail_prior is None
                or self.relation_entity_prior_scale is None
            ):
                raise RuntimeError("use_relation_entity_prior=True requires initialized prior modules")
            head_score = (self.head_entity_prior(edges.src) * self.rel_head_prior(edges.rel)).sum(dim=-1)
            tail_score = (self.tail_entity_prior(edges.dst) * self.rel_tail_prior(edges.rel)).sum(dim=-1)
            relation_entity_prior_score = head_score + tail_score
            final_score = final_score + self.relation_entity_prior_scale.to(final_score.dtype) * relation_entity_prior_score
        return SSPTGFMOutput(
            final_score=final_score,
            struct_score=struct_score,
            sem_score=sem_score,
            cross_score=cross_score,
            gate=gate,
            struct_nodes=struct_score_nodes,
            sem_nodes=sem_all,
            q_mu=context["q_mu_all"],
            q_logvar=context["q_logvar_all"],
            p_mu=context["p_mu_all"],
            p_logvar=context["p_logvar_all"],
            history_prior_score=history_prior_score,
            history_prior_gate=history_prior_gate,
            struct_feature_score=struct_feature_score,
            relation_entity_prior_score=relation_entity_prior_score,
        )

    def forward(
        self,
        edges: EdgeTensor,
        history: EdgeTensor,
        node_text_emb: torch.Tensor,
        rel_text_emb: torch.Tensor,
        struct_features: torch.Tensor,
        ood_s: torch.Tensor | None = None,
        ood_e: torch.Tensor | None = None,
        node_features: torch.Tensor | None = None,
        history_degree: torch.Tensor | None = None,
        history_prior_features: torch.Tensor | None = None,
    ) -> SSPTGFMOutput:
        device = node_text_emb.device
        context = self.encode_context(
            history,
            edges.time.to(device),
            node_text_emb=node_text_emb,
            rel_text_emb=rel_text_emb,
            node_features=node_features,
            history_degree=history_degree,
        )
        return self.score_edges_from_context(
            edges,
            struct_features,
            ood_s,
            ood_e,
            context,
            history_prior_features=history_prior_features,
        )
