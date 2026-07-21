from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn
try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call


def prompt_parameter_names(model: nn.Module) -> list[str]:
    names: list[str] = []
    for name, _ in model.named_parameters():
        if "prompt_" in name or ".prompts" in name or "text_proj" in name or "rel_text_proj" in name:
            names.append(name)
    return names


def freeze_except_prompts_and_adapters(model: nn.Module) -> None:
    allowed = set(prompt_parameter_names(model))
    for name, param in model.named_parameters():
        param.requires_grad = name in allowed


@contextmanager
def temporary_trainable_prompts(model: nn.Module) -> Iterator[None]:
    old_flags = {name: p.requires_grad for name, p in model.named_parameters()}
    freeze_except_prompts_and_adapters(model)
    try:
        yield
    finally:
        for name, param in model.named_parameters():
            param.requires_grad = old_flags[name]


def inner_update_prompts(
    model: nn.Module,
    loss: torch.Tensor,
    lr: float,
    create_graph: bool = False,
) -> dict[str, torch.Tensor]:
    """Return one-step adapted prompt/adapter weights for MAML-style episodes."""
    names = prompt_parameter_names(model)
    params = dict(model.named_parameters())
    grads = torch.autograd.grad(loss, [params[n] for n in names], create_graph=create_graph, allow_unused=True)
    adapted: dict[str, torch.Tensor] = {}
    for name, grad in zip(names, grads):
        if grad is None:
            adapted[name] = params[name]
        else:
            adapted[name] = params[name] - lr * grad
    return adapted


def query_with_adapted_prompts(model: nn.Module, adapted: dict[str, torch.Tensor], *args: object, **kwargs: object) -> object:
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    params.update(adapted)
    return functional_call(model, {**params, **buffers}, args, kwargs)
