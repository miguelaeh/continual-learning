"""Memory+ layer that replaces a transformer FFN.

The Memory+ layer (Berges et al., 2024) replaces the standard FFN with a
sparse memory lookup followed by SiLU-gated projection. It accesses only
top-k entries out of millions per token, keeping compute constant while
scaling capacity with the memory pool size.

Forward pass:
    1. Project input to query space + LayerNorm
    2. Product-key top-k lookup -> indices, attention weights
    3. Weighted sum of values via EmbeddingBag
    4. Memory+ gating: output = (mem_output * SiLU(x @ W1)) @ W2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.memory.shared_memory_store import SharedMemoryStore


class MemoryPlusLayer(nn.Module):
    """Memory+ layer replacing a transformer FFN.

    This module has the same interface as Gemma3MLP: takes (batch, seq, d_model)
    and returns (batch, seq, d_model). The surrounding pre/post feedforward
    LayerNorms in Gemma3DecoderLayer are untouched.

    Args:
        d_model: Model hidden dimension (2560 for Gemma 3 4B).
        shared_store: Shared key-value memory store.
        num_heads: Number of memory heads.
        k_dim_per_head: Key dimension per head.
        v_dim: Value dimension per entry.
        top_k: Number of entries retrieved per head.
        use_silu_gating: Whether to use Memory+ SiLU gating.
    """

    def __init__(
        self,
        d_model: int,
        shared_store: SharedMemoryStore,
        num_heads: int = 4,
        k_dim_per_head: int = 512,
        v_dim: int = 1024,
        top_k: int = 32,
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

        # Query projection: d_model -> num_heads * k_dim_per_head
        total_query_dim = num_heads * k_dim_per_head
        self.query_proj = nn.Linear(d_model, total_query_dim, bias=True)

        # LayerNorm on queries is critical for key utilization (Lample et al.)
        self.query_norm = nn.LayerNorm(total_query_dim)

        if use_silu_gating:
            # Memory+ gating: output = (mem * SiLU(x @ W1)) @ W2
            gate_dim = v_dim * num_heads if num_heads > 1 else v_dim
            # W1: gate projection
            self.silu_proj = nn.Linear(d_model, gate_dim, bias=False)
            # W2: output projection (from gated dim back to d_model)
            self.value_proj = nn.Linear(gate_dim, d_model, bias=False)
        else:
            # Direct projection from v_dim to d_model
            self.value_proj = nn.Linear(v_dim, d_model, bias=False)

        # For tracking accessed indices (used in IDF collection)
        self._last_indices: torch.Tensor | None = None
        self._track_indices = False

        # For gradient masking during continual learning
        self._trainable_mask: torch.Tensor | None = None

    def enable_index_tracking(self):
        """Enable tracking of accessed memory indices (for IDF collection)."""
        self._track_indices = True

    def disable_index_tracking(self):
        """Disable index tracking."""
        self._track_indices = False
        self._last_indices = None

    def get_last_accessed_indices(self) -> torch.Tensor | None:
        """Return indices accessed in the last forward pass."""
        return self._last_indices

    def set_trainable_mask(self, mask: torch.Tensor | None):
        """Set gradient mask for continual learning.

        Args:
            mask: Boolean tensor of shape (num_entries,). True = trainable slot.
                  None to disable masking.
        """
        self._trainable_mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model) - output of pre_feedforward_layernorm

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)  # (B*T, d_model)

        # 1. Query projection + LayerNorm
        query = self.query_proj(x_flat)  # (B*T, H * k_dim)
        query = self.query_norm(query)
        query = query.view(-1, self.num_heads, self.k_dim_per_head)  # (B*T, H, k_dim)

        # 2. Product key lookup
        indices, scores = self.shared_store.lookup(query)
        # indices: (B*T, H, top_k), scores: (B*T, H, top_k)

        # Track indices if enabled (for IDF collection)
        if self._track_indices:
            self._last_indices = indices.detach()

        # 3. Flatten across heads for EmbeddingBag retrieval
        flat_indices = indices.reshape(-1, self.num_heads * self.top_k)
        flat_scores = scores.reshape(-1, self.num_heads * self.top_k)

        # 4. Retrieve values
        mem_output = self.shared_store.retrieve_values(flat_indices, flat_scores)
        # mem_output: (B*T, v_dim)

        # 5. Apply gradient masking if set (for continual learning)
        if self._trainable_mask is not None:
            mem_output = _apply_gradient_mask(
                mem_output, flat_indices, self._trainable_mask
            )

        # 6. Memory+ gating or direct projection
        if self.use_silu_gating:
            # Expand mem_output if multi-head: tile v_dim across heads
            if self.num_heads > 1:
                # mem_output is (B*T, v_dim) from EmbeddingBag sum across heads
                # We need (B*T, v_dim * num_heads) for gating
                # Retrieve per-head values separately
                mem_expanded = self._retrieve_per_head(indices, scores)
                # mem_expanded: (B*T, v_dim * num_heads)
            else:
                mem_expanded = mem_output

            gate = F.silu(self.silu_proj(x_flat))  # (B*T, gate_dim)
            output = self.value_proj(mem_expanded * gate)  # (B*T, d_model)
        else:
            output = self.value_proj(mem_output)  # (B*T, d_model)

        return output.view(batch, seq_len, d_model)

    def _retrieve_per_head(
        self, indices: torch.Tensor, scores: torch.Tensor
    ) -> torch.Tensor:
        """Retrieve values per head (for multi-head gating).

        Instead of summing across heads (EmbeddingBag), retrieve each head's
        weighted sum separately and concatenate.

        Args:
            indices: (B*T, H, top_k)
            scores: (B*T, H, top_k)

        Returns:
            output: (B*T, v_dim * H)
        """
        BT, H, K = indices.shape
        outputs = []

        for h in range(H):
            h_indices = indices[:, h, :]  # (B*T, top_k)
            h_scores = scores[:, h, :]  # (B*T, top_k)
            h_output = self.shared_store.retrieve_values(h_indices, h_scores)
            outputs.append(h_output)  # (B*T, v_dim)

        return torch.cat(outputs, dim=-1)  # (B*T, v_dim * H)


def _apply_gradient_mask(
    mem_output: torch.Tensor,
    indices: torch.Tensor,
    trainable_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply straight-through gradient masking.

    Forward: mem_output passes through unchanged.
    Backward: gradients only flow to slots marked as trainable.

    Uses the identity trick:
        result = mem.detach() + (mem * mask) - (mem * mask).detach()

    This gives result == mem in forward, but grad only flows through mask.
    """
    # Check which accessed indices are trainable
    # indices: (B*T, H*K), trainable_mask: (num_entries,)
    idx_mask = trainable_mask[indices]  # (B*T, H*K) bool
    # If any index in a sample's retrieval is trainable, allow gradient
    sample_mask = idx_mask.any(dim=-1, keepdim=True).float()  # (B*T, 1)

    # Straight-through trick
    masked = mem_output * sample_mask
    return mem_output.detach() + masked - masked.detach()
