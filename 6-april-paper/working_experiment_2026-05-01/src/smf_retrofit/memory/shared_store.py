"""Shared key-value memory store."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from smf_retrofit.memory.product_key import ProductKeyLookup


class SharedMemoryStore(nn.Module):
    """Shared sparse memory store used by all retrofitted layers."""

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
        self.values = nn.EmbeddingBag(
            num_embeddings=self.num_entries,
            embedding_dim=v_dim,
            mode="sum",
        )
        self._init_values()

    def _init_values(self) -> None:
        nn.init.normal_(self.values.weight, mean=0.0, std=1.0 / math.sqrt(self.v_dim))
        self.values.weight.data = self.values.weight.data.float()

    def _apply(self, fn, recurse: bool = True):  # noqa: ANN001
        for module in self.children():
            if module is not self.values:
                module._apply(fn, recurse)

        self.values._apply(fn, recurse)
        if self.values.weight.dtype != torch.float32:
            self.values.weight.data = self.values.weight.data.float()
        return self

    def lookup(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.keys(query)

    def retrieve_values(
        self,
        indices: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, flat_width = indices.shape
        output_dtype = scores.dtype

        flat_indices = indices.reshape(-1)
        flat_scores = scores.reshape(-1).float()
        offsets = torch.arange(
            0,
            batch_size * flat_width,
            flat_width,
            device=indices.device,
            dtype=torch.long,
        )
        values = self.values(
            flat_indices,
            per_sample_weights=flat_scores,
            offsets=offsets,
        )
        return values.to(output_dtype)
