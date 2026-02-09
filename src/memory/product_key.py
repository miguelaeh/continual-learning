"""Product key decomposition for efficient top-k memory lookup.

Instead of scoring a query against N keys (O(N)), product keys decompose
each key into two halves and store sqrt(N) sub-keys per half. The lookup
scores each half independently (O(sqrt(N))), takes top-k from each, then
selects the final top-k from the k^2 cartesian product candidates.

Reference: "Large Memory Layers with Product Keys" (Lample et al., NeurIPS 2019)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductKeyLookup(nn.Module):
    """Product key decomposition for efficient sparse memory lookup.

    Stores two sets of sub-keys K1, K2 of shape (num_heads, n_keys, k_dim_half).
    The effective key space is the cartesian product K1 x K2, yielding n_keys^2
    total entries but requiring only O(n_keys) computation for lookup.

    Args:
        num_heads: Number of independent memory heads.
        n_keys: Number of sub-keys per half (total entries = n_keys^2).
        k_dim_per_head: Full key dimension per head (split into 2 x k_dim_half).
        top_k: Number of entries to retrieve per head.
    """

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
        self.top_k = top_k
        self.k_dim_per_head = k_dim_per_head
        self.k_dim_half = k_dim_per_head // 2

        # Sub-key parameters: (num_heads, n_keys, k_dim_half)
        self.keys_1 = nn.Parameter(
            torch.empty(num_heads, n_keys, self.k_dim_half)
        )
        self.keys_2 = nn.Parameter(
            torch.empty(num_heads, n_keys, self.k_dim_half)
        )

        self._init_keys()

    def _init_keys(self):
        std = 1.0 / math.sqrt(self.k_dim_half)
        nn.init.normal_(self.keys_1, mean=0, std=std)
        nn.init.normal_(self.keys_2, mean=0, std=std)

    def forward(
        self, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform product-key top-k lookup.

        Args:
            query: (batch, num_heads, k_dim_per_head)

        Returns:
            indices: (batch, num_heads, top_k) - linear indices into [0, n_keys^2)
            scores: (batch, num_heads, top_k) - softmax attention weights
        """
        # Split query into two halves
        q1 = query[..., : self.k_dim_half]  # (B, H, D/2)
        q2 = query[..., self.k_dim_half :]  # (B, H, D/2)

        # Score each half against its sub-keys
        # (B, H, D/2) @ (H, D/2, n_keys) -> (B, H, n_keys)
        scores_1 = torch.einsum("bhd,hnd->bhn", q1, self.keys_1)
        scores_2 = torch.einsum("bhd,hnd->bhn", q2, self.keys_2)

        # Top-k from each half
        top_scores_1, top_idx_1 = scores_1.topk(self.top_k, dim=-1)  # (B, H, k)
        top_scores_2, top_idx_2 = scores_2.topk(self.top_k, dim=-1)  # (B, H, k)

        # Cartesian product: k^2 candidates
        # (B, H, k, 1) + (B, H, 1, k) -> (B, H, k, k)
        all_scores = top_scores_1.unsqueeze(-1) + top_scores_2.unsqueeze(-2)
        all_indices = (
            top_idx_1.unsqueeze(-1) * self.n_keys + top_idx_2.unsqueeze(-2)
        )

        # Flatten k^2 candidates and select final top-k
        B, H = query.shape[0], query.shape[1]
        all_scores = all_scores.view(B, H, -1)  # (B, H, k^2)
        all_indices = all_indices.view(B, H, -1)  # (B, H, k^2)

        final_scores, final_pos = all_scores.topk(self.top_k, dim=-1)  # (B, H, k)
        final_indices = all_indices.gather(-1, final_pos)  # (B, H, k)

        # Softmax over the final top-k scores
        attn_weights = F.softmax(final_scores, dim=-1)  # (B, H, k)

        return final_indices, attn_weights
