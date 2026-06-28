"""Product-key lookup for sparse memory retrieval."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductKeyLookup(nn.Module):
    """Efficient top-k lookup over an implicit cartesian-product key space."""

    def __init__(
        self,
        num_heads: int,
        n_keys: int,
        k_dim_per_head: int,
        top_k: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.n_keys = n_keys
        self.k_dim_per_head = k_dim_per_head
        self.top_k = top_k
        self.k_dim_half = k_dim_per_head // 2

        self.keys_1 = nn.Parameter(torch.empty(num_heads, n_keys, self.k_dim_half))
        self.keys_2 = nn.Parameter(torch.empty(num_heads, n_keys, self.k_dim_half))
        self._init_keys()

    def _init_keys(self) -> None:
        std = 1.0 / math.sqrt(self.k_dim_half)
        nn.init.normal_(self.keys_1, mean=0.0, std=std)
        nn.init.normal_(self.keys_2, mean=0.0, std=std)

    def forward(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q1 = query[..., : self.k_dim_half]
        q2 = query[..., self.k_dim_half :]

        scores_1 = torch.einsum("bhd,hnd->bhn", q1, self.keys_1)
        scores_2 = torch.einsum("bhd,hnd->bhn", q2, self.keys_2)

        top_scores_1, top_idx_1 = scores_1.topk(self.top_k, dim=-1)
        top_scores_2, top_idx_2 = scores_2.topk(self.top_k, dim=-1)

        all_scores = top_scores_1.unsqueeze(-1) + top_scores_2.unsqueeze(-2)
        all_indices = top_idx_1.unsqueeze(-1) * self.n_keys + top_idx_2.unsqueeze(-2)

        batch_size, num_heads = query.shape[0], query.shape[1]
        all_scores = all_scores.view(batch_size, num_heads, -1)
        all_indices = all_indices.view(batch_size, num_heads, -1)

        final_scores, final_positions = all_scores.topk(self.top_k, dim=-1)
        final_indices = all_indices.gather(-1, final_positions)

        return final_indices, F.softmax(final_scores, dim=-1)
