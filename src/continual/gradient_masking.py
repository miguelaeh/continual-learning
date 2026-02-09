"""Gradient masking for sparse memory finetuning.

Implements the straight-through gradient masking trick from
"Continual Learning via Sparse Memory Finetuning" (Lin et al., 2025).

Forward: all memory slots contribute normally to the output.
Backward: gradients only flow to selected (trainable) slots.

This is achieved by hooking into the EmbeddingBag's gradient computation
and zeroing out gradients for non-selected slots after each backward pass.
"""

import torch
import torch.nn as nn

from src.config import MemoryConfig
from src.memory.memory_layer import MemoryPlusLayer
from src.model.memory_gemma import get_memory_layers


class GradientMaskManager:
    """Manages gradient masking for continual learning.

    Sets up and tears down gradient masks on memory layers. The mask
    determines which memory value slots receive gradient updates.

    Args:
        model: Model with injected memory layers.
        memory_config: Memory layer configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        memory_config: MemoryConfig,
    ):
        self.model = model
        self.memory_config = memory_config
        self.memory_layers = get_memory_layers(model, memory_config)
        self._hook_handles: list = []

    def set_mask(self, trainable_mask: torch.Tensor):
        """Apply a trainable mask to all memory layers.

        Args:
            trainable_mask: Boolean tensor (num_entries,). True = trainable.
        """
        for layer in self.memory_layers:
            layer.set_trainable_mask(trainable_mask)

        # Also register a gradient hook on the shared value embeddings
        # to zero out gradients for non-trainable slots
        self._remove_hooks()
        shared_store = self.memory_layers[0].shared_store
        handle = shared_store.values.weight.register_hook(
            lambda grad: _mask_embedding_grad(grad, trainable_mask)
        )
        self._hook_handles.append(handle)

    def clear_mask(self):
        """Remove all gradient masks."""
        for layer in self.memory_layers:
            layer.set_trainable_mask(None)
        self._remove_hooks()

    def _remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()


def _mask_embedding_grad(
    grad: torch.Tensor, trainable_mask: torch.Tensor
) -> torch.Tensor:
    """Zero out gradients for non-trainable embedding rows.

    Args:
        grad: Gradient tensor of shape (num_entries, v_dim).
            May be sparse (from EmbeddingBag) or dense.
        trainable_mask: Boolean tensor of shape (num_entries,).

    Returns:
        Masked gradient tensor.
    """
    if grad.is_sparse:
        # For sparse gradients, filter out non-trainable indices
        indices = grad._indices()
        values = grad._values()

        # Get the row indices (first dimension of sparse tensor)
        row_indices = indices[0]
        mask = trainable_mask.to(row_indices.device)[row_indices]

        # Zero out values for non-trainable rows
        masked_values = values * mask.unsqueeze(-1).to(values.dtype)

        return torch.sparse_coo_tensor(
            indices, masked_values, grad.shape, device=grad.device
        ).coalesce()
    else:
        # Dense gradient: directly mask rows
        mask = trainable_mask.to(grad.device).unsqueeze(-1).to(grad.dtype)
        return grad * mask
