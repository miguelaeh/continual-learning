"""Shared memory store with product keys and EmbeddingBag values.

The key insight from "Memory Layers at Scale" (Berges et al., 2024) is that
multiple memory layers can share the same key-value parameters. This reduces
total parameter count while maintaining memory capacity. Each memory layer
has its own query/gate/value projections but shares the lookup and storage.
"""

import math

import torch
import torch.nn as nn

from src.memory.product_key import ProductKeyLookup


class SharedMemoryStore(nn.Module):
    """Shared key-value store for memory layers.

    Contains:
    - Product keys (K1, K2) for efficient top-k lookup
    - EmbeddingBag values for weighted retrieval

    Shared across all memory layers in the model. Each memory layer
    references this store and has its own query/gate/value projections.

    Args:
        num_heads: Number of memory heads.
        n_keys: Sub-keys per half (total entries = n_keys^2).
        k_dim_per_head: Key dimension per head.
        v_dim: Value embedding dimension.
        top_k: Number of entries retrieved per head.
    """

    def __init__(
        self,
        num_heads: int,
        n_keys: int,
        k_dim_per_head: int,
        v_dim: int,
        top_k: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.n_keys = n_keys
        self.num_entries = n_keys**2
        self.v_dim = v_dim
        self.top_k = top_k

        self.keys = ProductKeyLookup(
            num_heads=num_heads,
            n_keys=n_keys,
            k_dim_per_head=k_dim_per_head,
            top_k=top_k,
        )

        # Value store: EmbeddingBag for efficient weighted sum
        # mode='sum' with per_sample_weights gives us: sum(weight_i * V[idx_i])
        self.values = nn.EmbeddingBag(
            num_embeddings=self.num_entries,
            embedding_dim=v_dim,
            mode="sum",
        )
        self._init_values()

    def _init_values(self):
        nn.init.normal_(
            self.values.weight,
            mean=0,
            std=1.0 / math.sqrt(self.v_dim),
        )
        # EmbeddingBag per_sample_weights backward is not implemented for bf16
        # on CUDA, so values must stay in float32
        self.values.weight.data = self.values.weight.data.float()

    def _apply(self, fn, recurse=True):
        """Override _apply to keep EmbeddingBag values in float32.

        When model.to(dtype=bfloat16) is called, this prevents the values
        from being cast away from float32 (required for backward on CUDA).
        Device changes are still applied normally.
        """
        # Apply to all sub-modules except values
        for module in self.children():
            if module is not self.values:
                module._apply(fn, recurse)

        # For the values module: apply the function but force weight back to float32
        self.values._apply(fn, recurse)
        if self.values.weight.dtype != torch.float32:
            self.values.weight.data = self.values.weight.data.float()

        # Apply to own parameters (non-recursive) - there are none besides children
        return self

    def lookup(
        self, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Product key lookup.

        Args:
            query: (batch, num_heads, k_dim_per_head)

        Returns:
            indices: (batch, num_heads, top_k) - memory slot indices
            scores: (batch, num_heads, top_k) - attention weights (softmax)
        """
        return self.keys(query)

    def retrieve_values(
        self,
        indices: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve weighted sum of values at given indices.

        Args:
            indices: (batch, num_heads * top_k) - flat memory slot indices
            scores: (batch, num_heads * top_k) - flat attention weights

        Returns:
            output: (batch, v_dim) - weighted sum of retrieved values
        """
        B, HK = indices.shape
        input_dtype = scores.dtype

        # EmbeddingBag expects: indices (total,), per_sample_weights (total,),
        # offsets marking sample boundaries
        flat_indices = indices.reshape(-1)  # (B * HK,)
        # EmbeddingBag per_sample_weights backward is not implemented for bf16
        # on CUDA, so always compute in float32
        flat_weights = scores.reshape(-1).float()  # (B * HK,)

        offsets = torch.arange(
            0, B * HK, HK, device=indices.device, dtype=torch.long
        )

        output = self.values(
            flat_indices, per_sample_weights=flat_weights, offsets=offsets
        )
        # output: (B, v_dim) in float32, cast back to model dtype
        return output.to(input_dtype)
