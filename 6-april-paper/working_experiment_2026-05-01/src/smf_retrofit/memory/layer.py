"""Sparse memory layer used to replace a transformer FFN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from smf_retrofit.memory.shared_store import SharedMemoryStore


class SparseMemoryLayer(nn.Module):
    """Memory layer with product-key lookup and optional SiLU gating."""

    def __init__(
        self,
        d_model: int,
        shared_store: SharedMemoryStore,
        num_heads: int,
        k_dim_per_head: int,
        v_dim: int,
        top_k: int,
        use_silu_gating: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.shared_store = shared_store
        self.num_heads = num_heads
        self.k_dim_per_head = k_dim_per_head
        self.v_dim = v_dim
        self.top_k = top_k
        self.use_silu_gating = use_silu_gating

        total_query_dim = num_heads * k_dim_per_head
        self.query_proj = nn.Linear(d_model, total_query_dim, bias=True)
        self.query_norm = nn.LayerNorm(total_query_dim)

        gate_dim = num_heads * v_dim
        if use_silu_gating:
            self.silu_proj = nn.Linear(d_model, gate_dim, bias=False)
        else:
            self.silu_proj = None
        self.value_proj = nn.Linear(gate_dim, d_model, bias=False)

        self._track_indices = False
        self._last_indices: torch.Tensor | None = None

    def enable_index_tracking(self) -> None:
        self._track_indices = True

    def disable_index_tracking(self) -> None:
        self._track_indices = False
        self._last_indices = None

    def get_last_accessed_indices(self) -> torch.Tensor | None:
        return self._last_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)

        query = self.query_norm(self.query_proj(x_flat))
        query = query.view(-1, self.num_heads, self.k_dim_per_head)
        indices, scores = self.shared_store.lookup(query)

        if self._track_indices:
            self._last_indices = indices.detach()

        memory = self._retrieve_per_head(indices, scores)
        if self.silu_proj is not None:
            gate = F.silu(self.silu_proj(x_flat))
            output = self.value_proj(memory * gate)
        else:
            output = self.value_proj(memory)

        return output.view(batch_size, seq_len, d_model)

    def _retrieve_per_head(
        self,
        indices: torch.Tensor,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        outputs = []
        for head_idx in range(indices.shape[1]):
            head_output = self.shared_store.retrieve_values(
                indices[:, head_idx, :],
                scores[:, head_idx, :],
            )
            outputs.append(head_output)
        return torch.cat(outputs, dim=-1)
