from __future__ import annotations

import math

import torch
from torch import nn


class TimeEncoder(nn.Module):
    def __init__(self, dim: int, mode: str = "fourier", max_period: float = 10_000.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.mode = mode
        self.max_period = float(max_period)
        if mode == "random_fourier":
            self.register_buffer("freq", torch.randn(max(1, dim // 2)) * 0.1)
        elif mode in {"fourier", "wavelet", "identity"}:
            half = max(1, dim // 2)
            freq = torch.exp(-math.log(max_period) * torch.arange(half).float() / max(1, half - 1))
            self.register_buffer("freq", freq)
        else:
            raise ValueError(f"unknown time encoder mode: {mode}")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float().view(-1, 1)
        if self.mode == "identity":
            out = torch.cat([t, torch.log1p(torch.relu(t))], dim=-1)
        elif self.mode == "wavelet":
            x = t * self.freq.view(1, -1)
            out = torch.cos(x) * torch.exp(-0.5 * (x / math.pi).pow(2))
            out = torch.cat([out, torch.sin(x) * torch.exp(-0.5 * (x / math.pi).pow(2))], dim=-1)
        else:
            x = t * self.freq.view(1, -1)
            out = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        if out.size(-1) < self.dim:
            pad = torch.zeros(out.size(0), self.dim - out.size(-1), device=out.device, dtype=out.dtype)
            out = torch.cat([out, pad], dim=-1)
        return out[:, : self.dim]
