from __future__ import annotations

from ssptgfm.model import SSPTGFM


def rough_forward_flops(model: SSPTGFM, num_history_edges: int, batch_edges: int) -> int:
    """Conservative multiply-add estimate for reporting scale, not a profiler trace."""
    d = model.hidden_dim
    layers = model.struct_encoder.num_layers
    rel_rank = model.struct_bilinear.left.size(1)
    prompt_tokens = model.prompt_s.prompts.size(1)
    msg = layers * num_history_edges * (2 * d + model.struct_encoder.time_encoder.dim) * d * 2
    update = layers * model.num_nodes * (2 * d) * d
    bilinear = batch_edges * rel_rank * d * 6
    prompt = batch_edges * prompt_tokens * d * 4
    gate = batch_edges * d * 3
    return int(msg + update + bilinear + prompt + gate)
