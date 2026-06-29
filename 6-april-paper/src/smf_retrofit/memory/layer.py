"""Sparse memory layer used to replace a transformer FFN."""

from __future__ import annotations

import math

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
        use_delta_residual: bool = False,
        use_delta_value_proj: bool = False,
        delta_residual_init_scale: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.shared_store = shared_store
        self.num_heads = num_heads
        self.k_dim_per_head = k_dim_per_head
        self.v_dim = v_dim
        self.top_k = top_k
        self.use_silu_gating = use_silu_gating
        self.use_delta_residual = use_delta_residual
        self.use_delta_value_proj = use_delta_value_proj

        total_query_dim = num_heads * k_dim_per_head
        self.query_proj = nn.Linear(d_model, total_query_dim, bias=True)
        self.query_norm = nn.LayerNorm(k_dim_per_head)

        gate_dim = num_heads * v_dim
        if use_silu_gating:
            self.silu_proj = nn.Linear(d_model, gate_dim, bias=False)
            nn.init.normal_(self.silu_proj.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
        else:
            self.silu_proj = None
        self.value_proj = nn.Linear(gate_dim, d_model, bias=False)
        nn.init.normal_(self.value_proj.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
        if shared_store.delta_values is not None and use_delta_residual:
            self.delta_scale = nn.Parameter(torch.full((1,), float(delta_residual_init_scale)))
            self.delta_value_proj = (
                nn.Linear(gate_dim, d_model, bias=False)
                if use_delta_value_proj
                else None
            )
        else:
            self.register_parameter("delta_scale", None)
            self.delta_value_proj = None

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

        query = self.query_proj(x_flat).view(-1, self.num_heads, self.k_dim_per_head)
        query = self.query_norm(query)
        indices, scores = self.shared_store.lookup(query)

        if self._track_indices:
            self._last_indices = indices.detach()

        base_memory = self._retrieve_per_head(indices, scores, bank="base")
        delta_memory = None
        if self.shared_store.delta_values is not None:
            delta_memory = self._retrieve_per_head(indices, scores, bank="delta")
        if self.silu_proj is not None:
            gate = F.silu(self.silu_proj(x_flat))
            base_output = self.value_proj(base_memory * gate)
            if delta_memory is not None:
                if self.use_delta_residual:
                    delta_output = self._project_delta(delta_memory * gate)
                    output = base_output + (self.delta_scale * delta_output)
                else:
                    output = self.value_proj((base_memory + delta_memory) * gate)
            else:
                output = base_output
        else:
            base_output = self.value_proj(base_memory)
            if delta_memory is not None:
                if self.use_delta_residual:
                    delta_output = self._project_delta(delta_memory)
                    output = base_output + (self.delta_scale * delta_output)
                else:
                    output = self.value_proj(base_memory + delta_memory)
            else:
                output = base_output

        return output.view(batch_size, seq_len, d_model)

    def _project_delta(self, delta_memory: torch.Tensor) -> torch.Tensor:
        if self.delta_value_proj is not None:
            return self.delta_value_proj(delta_memory)
        return self.value_proj(delta_memory)

    def _retrieve_per_head(
        self,
        indices: torch.Tensor,
        scores: torch.Tensor,
        bank: str,
    ) -> torch.Tensor:
        outputs = []
        for head_idx in range(indices.shape[1]):
            head_output = self.shared_store.retrieve_values(
                indices[:, head_idx, :],
                scores[:, head_idx, :],
                bank=bank,
            )
            outputs.append(head_output)
        return torch.cat(outputs, dim=-1)
