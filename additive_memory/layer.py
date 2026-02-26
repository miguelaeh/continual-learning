"""Additive memory layer for zero-cost injection.

Instead of replacing the FFN (which requires expensive pretraining), the
additive memory layer sits in parallel to the original FFN:

    output = original_ffn(x) + memory(x)

Memory values are zero-initialized, so the model is unchanged at injection
time. During the remember step, sparse SGD learns context-dependent steering
vectors that add new knowledge without disrupting existing model behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.memory.shared_memory_store import SharedMemoryStore


class AdditiveMemoryLayer(nn.Module):
    """Memory layer that produces context-dependent steering vectors.

    Same interface as Gemma3MLP: (batch, seq, d_model) -> (batch, seq, d_model).
    Output starts at zero (zero-initialized memory values) and learns
    task-specific perturbations during sparse SGD finetuning.

    The output path is a single trainable projection (no random gating) so that
    the memory signal reaches the residual stream without distortion:
        query → product key lookup → weighted sum of values → output_proj → add to residual

    Args:
        d_model: Model hidden dimension.
        shared_store: Shared key-value memory store (values must be zero-init).
        num_heads: Number of memory heads.
        k_dim_per_head: Key dimension per head.
        v_dim: Value dimension per entry.
        top_k: Number of entries retrieved per head.
    """

    def __init__(
        self,
        d_model: int,
        shared_store: SharedMemoryStore,
        num_heads: int = 4,
        k_dim_per_head: int = 512,
        v_dim: int = 1024,
        top_k: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.shared_store = shared_store
        self.num_heads = num_heads
        self.k_dim_per_head = k_dim_per_head
        self.v_dim = v_dim
        self.top_k = top_k

        # Query projection: d_model -> num_heads * k_dim_per_head
        total_query_dim = num_heads * k_dim_per_head
        self.query_proj = nn.Linear(d_model, total_query_dim, bias=True)
        self.query_norm = nn.LayerNorm(total_query_dim)

        # Output projection: direct path from retrieved values to residual stream.
        # No SiLU gating — random gating distorts the signal across different inputs
        # and makes the memory output input-dependent in an arbitrary way.
        concat_dim = v_dim * num_heads if num_heads > 1 else v_dim
        self.output_proj = nn.Linear(concat_dim, d_model, bias=False)

        # Index tracking for diagnostics
        self._last_indices: torch.Tensor | None = None
        self._track_indices = False

    def enable_index_tracking(self):
        """Enable tracking of accessed memory indices."""
        self._track_indices = True

    def disable_index_tracking(self):
        """Disable index tracking."""
        self._track_indices = False
        self._last_indices = None

    def get_last_accessed_indices(self) -> torch.Tensor | None:
        """Return indices accessed in the last forward pass."""
        return self._last_indices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: produce a steering vector for each token.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model) — steering vectors to add to FFN output
        """
        batch, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)  # (B*T, d_model)

        # 1. Query projection + LayerNorm
        query = self.query_proj(x_flat)  # (B*T, H * k_dim)
        query = self.query_norm(query)
        query = query.view(-1, self.num_heads, self.k_dim_per_head)

        # 2. Product key lookup (sparse: top_k from n_keys^2 entries)
        indices, scores = self.shared_store.lookup(query)
        # indices: (B*T, H, top_k), scores: (B*T, H, top_k)

        if self._track_indices:
            self._last_indices = indices.detach()

        # 3. Retrieve values per head
        mem = self._retrieve_per_head(indices, scores)
        # mem: (B*T, v_dim * num_heads)

        # 4. Output projection (direct, no gating)
        output = self.output_proj(mem)  # (B*T, d_model)

        return output.view(batch, seq_len, d_model)

    def _retrieve_per_head(
        self, indices: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """Retrieve weighted values per head and concatenate.

        Args:
            indices: (B*T, H, top_k)
            scores: (B*T, H, top_k)

        Returns:
            output: (B*T, v_dim * H)
        """
        BT, H, K = indices.shape
        outputs = []

        for h in range(H):
            h_output = self.shared_store.retrieve_values(
                indices[:, h, :], scores[:, h, :]
            )
            outputs.append(h_output)

        return torch.cat(outputs, dim=-1)


class FFNWithMemory(nn.Module):
    """Wrapper that runs the original FFN and memory in parallel.

    output = original_ffn(x) + memory(x)

    At initialization (memory values = 0), this is identical to the original FFN.
    During training, the memory branch learns additive perturbations.
    """

    def __init__(self, original_ffn: nn.Module, memory: AdditiveMemoryLayer):
        super().__init__()
        self.ffn = original_ffn
        self.memory = memory

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x) + self.memory(x)
