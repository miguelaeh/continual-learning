"""Gradient masking for sparse memory updates."""

from __future__ import annotations

import torch
import torch.nn as nn

from smf_retrofit.modeling.qwen import get_memory_layers


class GradientMaskManager:
    """Mask gradients on the shared memory value table."""

    def __init__(self, model: nn.Module, layer_indices: list[int]):
        self.memory_layers = get_memory_layers(model, layer_indices)
        self._handles: list = []

    def _target_embedding(self) -> nn.EmbeddingBag:
        layer = self.memory_layers[0]
        if layer.shared_store.delta_values is not None and layer.shared_store.delta_values.weight.requires_grad:
            return layer.shared_store.delta_values
        return layer.shared_store.values

    def set_mask(self, trainable_mask: torch.Tensor) -> None:
        self.clear_mask()
        embedding = self._target_embedding()
        handle = embedding.weight.register_hook(
            lambda grad: _mask_embedding_grad(grad, trainable_mask)
        )
        self._handles.append(handle)

    def apply_mask_to_grads(self, trainable_mask: torch.Tensor) -> None:
        embedding = self._target_embedding()
        grad = embedding.weight.grad
        if grad is None:
            return
        embedding.weight.grad = _mask_embedding_grad(grad, trainable_mask)

    def clear_mask(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def _mask_embedding_grad(
    grad: torch.Tensor,
    trainable_mask: torch.Tensor,
) -> torch.Tensor:
    if grad.is_sparse:
        grad = grad.coalesce()
        indices = grad.indices()
        values = grad.values()
        row_indices = indices[0]
        mask = trainable_mask.to(row_indices.device)[row_indices]
        masked_values = values * mask.unsqueeze(-1).to(values.dtype)
        return torch.sparse_coo_tensor(indices, masked_values, grad.shape, device=grad.device)

    mask = trainable_mask.to(grad.device).unsqueeze(-1).to(grad.dtype)
    return grad * mask
