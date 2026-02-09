"""Parameter freezing utilities for different training phases."""

import logging

import torch.nn as nn

from src.config import MemoryConfig
from src.memory.memory_layer import MemoryPlusLayer
from src.model.memory_gemma import get_decoder_layers

logger = logging.getLogger(__name__)


def freeze_base_model(model: nn.Module, memory_config: MemoryConfig):
    """Phase 1: Freeze all base model params, only memory layers are trainable.

    Trainable parameters:
    - SharedMemoryStore: keys (K1, K2), values (EmbeddingBag)
    - Per-layer: query_proj, query_norm, silu_proj, value_proj

    Frozen parameters:
    - All attention layers, all non-memory FFN layers
    - All RMSNorm layers (including pre/post feedforward norms)
    - Embeddings, LM head
    """
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze memory layer parameters
    layers = get_decoder_layers(model)
    trainable_count = 0
    for layer_idx in memory_config.memory_layers:
        mem_layer = layers[layer_idx].mlp
        if not isinstance(mem_layer, MemoryPlusLayer):
            raise ValueError(
                f"Layer {layer_idx} MLP is {type(mem_layer).__name__}, "
                "expected MemoryPlusLayer. Did you call inject_memory_layers()?"
            )
        for param in mem_layer.parameters():
            param.requires_grad = True
            trainable_count += param.numel()

    total_count = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable: {trainable_count / 1e6:.1f}M / {total_count / 1e6:.1f}M "
        f"({100 * trainable_count / total_count:.1f}%)"
    )


def freeze_for_continual_learning(model: nn.Module, memory_config: MemoryConfig):
    """Phase 3: Freeze everything. Gradient masking handles sparse updates.

    During continual learning, all parameters are frozen. The gradient
    masking mechanism (straight-through trick) in MemoryPlusLayer handles
    which memory slots receive gradient updates. The optimizer only needs
    to update the shared value embeddings, and gradient masking ensures
    only the selected slots actually change.
    """
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Only unfreeze the shared memory store values
    layers = get_decoder_layers(model)
    # All memory layers share the same store, so we just need one reference
    mem_layer = layers[memory_config.memory_layers[0]].mlp
    if not isinstance(mem_layer, MemoryPlusLayer):
        raise ValueError("Expected MemoryPlusLayer")

    shared_store = mem_layer.shared_store
    shared_store.values.weight.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Continual learning: {trainable / 1e6:.1f}M trainable params "
        f"(gradient masking will restrict to {memory_config.top_k} slots)"
    )
