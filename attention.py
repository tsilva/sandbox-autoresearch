"""Torch-native attention fallback for unavailable Flash Attention binary variants."""

import torch
from torch.nn import functional as F


def flash_attn_func(q, k, v, *, causal=True, window_size=(-1, -1)):
    """Preserve FA3's BTHD layout, grouped-query heads and local causal windows."""
    left, right = window_size
    mask = None
    is_causal = causal
    if left >= 0 or right >= 0:
        query = torch.arange(q.size(1), device=q.device)[:, None]
        key = torch.arange(k.size(1), device=q.device)[None, :]
        mask = torch.ones((q.size(1), k.size(1)), dtype=torch.bool, device=q.device)
        if causal:
            mask = mask & (key <= query)
        if left >= 0:
            mask = mask & (key >= query - left)
        if right >= 0:
            mask = mask & (key <= query + right)
        is_causal = False
    result = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask, is_causal=is_causal, enable_gqa=q.size(2) != k.size(2),
    )
    return result.transpose(1, 2)
