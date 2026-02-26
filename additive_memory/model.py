"""Model surgery for additive memory injection.

Wraps selected FFN layers with FFNWithMemory, keeping the original FFN
intact and adding a parallel memory branch. No pretraining needed.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from src.memory.shared_memory_store import SharedMemoryStore
from src.model.memory_gemma import get_decoder_layers, load_base_model, load_tokenizer

from additive_memory.layer import AdditiveMemoryLayer, FFNWithMemory

logger = logging.getLogger(__name__)


@dataclass
class AdditiveMemoryConfig:
    """Configuration for additive memory layers."""

    # Base model
    base_model: str = "google/gemma-3-4b-it"
    dtype: str = "bfloat16"

    # Memory architecture
    memory_layers: list[int] = field(default_factory=lambda: [9, 17, 25])
    num_heads: int = 4
    n_keys: int = 1024
    k_dim_per_head: int = 512
    v_dim: int = 1024
    top_k: int = 32

    # Remember training
    learning_rate: float = 2.0
    momentum: float = 0.0
    finetuning_steps: int = 100
    seq_length: int = 512
    repeat_factor: int = 8

    # Paths
    checkpoint_dir: str = "checkpoints/additive_memory"
    memory_checkpoint: str | None = None

    @property
    def num_entries(self) -> int:
        return self.n_keys**2


def inject_additive_memory(
    model: nn.Module,
    config: AdditiveMemoryConfig,
) -> tuple[nn.Module, SharedMemoryStore]:
    """Inject additive memory layers into a Gemma model.

    Wraps target FFN layers with FFNWithMemory. The original FFN is preserved
    and the memory branch starts at zero output (zero-initialized values).

    Args:
        model: Base Gemma model.
        config: Additive memory configuration.

    Returns:
        Tuple of (modified model, shared memory store).
    """
    layers = get_decoder_layers(model)

    # Detect d_model and device/dtype from the first target layer
    sample_mlp = layers[config.memory_layers[0]].mlp
    d_model = sample_mlp.gate_proj.in_features
    target_device = sample_mlp.gate_proj.weight.device
    target_dtype = sample_mlp.gate_proj.weight.dtype

    logger.info(f"Detected d_model={d_model}, device={target_device}, dtype={target_dtype}")

    # Create shared memory store
    shared_store = SharedMemoryStore(
        num_heads=config.num_heads,
        n_keys=config.n_keys,
        k_dim_per_head=config.k_dim_per_head,
        v_dim=config.v_dim,
        top_k=config.top_k,
    )

    # Zero-initialize memory values (model unchanged at injection)
    nn.init.zeros_(shared_store.values.weight)

    # Move shared store to model's device/dtype
    shared_store = shared_store.to(device=target_device, dtype=target_dtype)

    # Wrap target FFN layers
    for layer_idx in config.memory_layers:
        if layer_idx >= len(layers):
            raise ValueError(
                f"Layer index {layer_idx} out of range (model has {len(layers)} layers)"
            )

        original_ffn = layers[layer_idx].mlp

        memory = AdditiveMemoryLayer(
            d_model=d_model,
            shared_store=shared_store,
            num_heads=config.num_heads,
            k_dim_per_head=config.k_dim_per_head,
            v_dim=config.v_dim,
            top_k=config.top_k,
        )
        memory = memory.to(device=target_device, dtype=target_dtype)

        wrapper = FFNWithMemory(original_ffn, memory)
        layers[layer_idx].mlp = wrapper

        logger.info(f"Wrapped layer {layer_idx} FFN with additive memory")

    memory_params = sum(p.numel() for p in shared_store.parameters())
    per_layer_params = sum(
        p.numel()
        for idx in config.memory_layers
        for p in layers[idx].mlp.memory.parameters()
        if p.data_ptr() not in {p2.data_ptr() for p2 in shared_store.parameters()}
    )
    logger.info(
        f"Injected additive memory at layers {config.memory_layers}: "
        f"{(memory_params + per_layer_params) / 1e6:.1f}M new params "
        f"({config.num_entries} entries, {config.num_heads} heads, top_k={config.top_k})"
    )

    return model, shared_store


def freeze_for_remember(model: nn.Module, config: AdditiveMemoryConfig):
    """Freeze everything except shared memory values.

    Only the EmbeddingBag values are trainable. All projections (query, gate,
    value, keys) stay frozen at their random initialization. The original FFN
    is also frozen.

    EmbeddingBag gradients are naturally sparse (only accessed rows get
    non-zero gradients), so no explicit gradient masking is needed.
    """
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze shared memory values
    layers = get_decoder_layers(model)
    wrapper = layers[config.memory_layers[0]].mlp
    wrapper.memory.shared_store.values.weight.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Frozen for remember: {trainable / 1e6:.1f}M trainable / "
        f"{total / 1e6:.1f}M total ({100 * trainable / total:.2f}%)"
    )


def get_memory_layers(
    model: nn.Module, config: AdditiveMemoryConfig
) -> list[AdditiveMemoryLayer]:
    """Get references to all injected AdditiveMemoryLayer instances."""
    layers = get_decoder_layers(model)
    return [layers[i].mlp.memory for i in config.memory_layers]


def save_memory_checkpoint(
    model: nn.Module,
    shared_store: SharedMemoryStore,
    config: AdditiveMemoryConfig,
    save_path: str,
    step: int = 0,
):
    """Save memory-only checkpoint (not the base model).

    Saves the shared store (keys + values) and per-layer projections.
    """
    layers = get_decoder_layers(model)

    checkpoint = {
        "shared_store": shared_store.state_dict(),
        "per_layer_states": {},
        "config": {
            "memory_layers": config.memory_layers,
            "num_heads": config.num_heads,
            "n_keys": config.n_keys,
            "k_dim_per_head": config.k_dim_per_head,
            "v_dim": config.v_dim,
            "top_k": config.top_k,
        },
        "step": step,
        "approach": "additive",
    }

    for layer_idx in config.memory_layers:
        memory = layers[layer_idx].mlp.memory
        per_layer_state = {}
        for name, param in memory.named_parameters():
            if not name.startswith("shared_store."):
                per_layer_state[name] = param.data
        checkpoint["per_layer_states"][layer_idx] = per_layer_state

    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    logger.info(f"Saved memory checkpoint to {save_path} (step {step})")


def load_memory_checkpoint(
    model: nn.Module,
    shared_store: SharedMemoryStore,
    config: AdditiveMemoryConfig,
    load_path: str,
) -> int:
    """Load memory checkpoint."""
    checkpoint = torch.load(load_path, map_location="cpu", weights_only=True)

    shared_store.load_state_dict(checkpoint["shared_store"])

    layers = get_decoder_layers(model)
    for layer_idx, per_layer_state in checkpoint["per_layer_states"].items():
        layer_idx = int(layer_idx)
        memory = layers[layer_idx].mlp.memory
        for name, param_data in per_layer_state.items():
            param = dict(memory.named_parameters())[name]
            param.data.copy_(param_data)

    step = checkpoint.get("step", 0)
    logger.info(f"Loaded memory checkpoint from {load_path} (step {step})")
    return step
