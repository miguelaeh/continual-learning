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
        use_delta_bank: bool = False,
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
        self.delta_values = (
            nn.EmbeddingBag(
                num_embeddings=self.num_entries,
                embedding_dim=v_dim,
                mode="sum",
            )
            if use_delta_bank
            else None
        )
        self._init_values()
        if self.delta_values is not None:
            self.zero_delta_values_()

    def zero_values_(self) -> None:
        self.values.weight.data.zero_()

    def zero_delta_values_(self) -> None:
        if self.delta_values is not None:
            self.delta_values.weight.data.zero_()

    def _init_values(self) -> None:
        nn.init.normal_(self.values.weight, mean=0.0, std=1.0 / math.sqrt(self.v_dim))
        self.values.weight.data = self.values.weight.data.float()

    def _apply(self, fn, recurse: bool = True):  # noqa: ANN001
        for module in self.children():
            if module not in {self.values, self.delta_values}:
                module._apply(fn, recurse)

        self.values._apply(fn, recurse)
        if self.values.weight.dtype != torch.float32:
            self.values.weight.data = self.values.weight.data.float()
        if self.delta_values is not None:
            self.delta_values._apply(fn, recurse)
            if self.delta_values.weight.dtype != torch.float32:
                self.delta_values.weight.data = self.delta_values.weight.data.float()
        return self

    def lookup(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.keys(query)

    def retrieve_base_values(
        self,
        indices: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        return self._retrieve_embedding(self.values, indices, scores)

    def retrieve_delta_values(
        self,
        indices: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.delta_values is None:
            return None
        return self._retrieve_embedding(self.delta_values, indices, scores)

    def retrieve_values(
        self,
        indices: torch.Tensor,
        scores: torch.Tensor,
        bank: str = "base",
    ) -> torch.Tensor:
        if bank == "base":
            return self.retrieve_base_values(indices, scores)
        if bank == "delta":
            values = self.retrieve_delta_values(indices, scores)
            if values is None:
                raise ValueError("delta bank requested but not initialized")
            return values
        raise ValueError(f"Unknown memory bank: {bank}")

    def _retrieve_embedding(
        self,
        embedding: nn.EmbeddingBag,
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
        values = embedding(
            flat_indices,
            per_sample_weights=flat_scores,
            offsets=offsets,
        )
        return values.to(output_dtype)
