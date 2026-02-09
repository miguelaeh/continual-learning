"""Model surgery: inject memory layers into Gemma 3 4B.

Loads the base Gemma 3 4B IT model, replaces selected FFN layers with
Memory+ layers, and provides utilities for checkpoint management.

The Gemma3DecoderLayer forward path for FFN is:
    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)      # <-- we replace this
    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

We only replace self.mlp; the surrounding norms remain untouched.
"""

import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import MemoryConfig
from src.memory.memory_layer import MemoryPlusLayer
from src.memory.shared_memory_store import SharedMemoryStore

logger = logging.getLogger(__name__)


def load_base_model(
    model_name: str,
    dtype: str = "bfloat16",
    device_map: str | None = "auto",
) -> AutoModelForCausalLM:
    """Load the base Gemma 3 text-only model.

    Args:
        model_name: HuggingFace model ID (e.g., "google/gemma-3-4b-it").
        dtype: Weight dtype ("bfloat16" or "float32").
        device_map: Device map for model placement. For MPS (Apple Silicon),
            pass the device string directly (e.g., "mps") since
            device_map="auto" is not supported.

    Returns:
        Loaded Gemma3ForCausalLM model.
    """
    torch_dtype = getattr(torch, dtype)

    # MPS doesn't support device_map="auto", load to CPU then move
    if device_map == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            attn_implementation="sdpa",
        )
        model = model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="sdpa",
        )

    logger.info(
        f"Loaded {model_name} with {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params"
    )
    return model


def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load the tokenizer for the model."""
    return AutoTokenizer.from_pretrained(model_name)


def get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Get the decoder layer list from either CausalLM or ConditionalGeneration model."""
    # Gemma3ForCausalLM: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Gemma3ForConditionalGeneration: model.model.language_model.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers
    raise ValueError(
        f"Cannot find decoder layers in model of type {type(model).__name__}. "
        "Expected model.model.layers or model.model.language_model.layers to exist."
    )


def inject_memory_layers(
    model: torch.nn.Module,
    memory_config: MemoryConfig,
) -> tuple[torch.nn.Module, SharedMemoryStore]:
    """Replace selected FFN layers with Memory+ layers.

    Creates a SharedMemoryStore (shared K,V across all memory layers) and
    replaces model.model.layers[i].mlp at the specified layer indices.

    Args:
        model: Base Gemma 3 model.
        memory_config: Memory layer configuration.

    Returns:
        Tuple of (modified model, shared memory store).
    """
    layers = get_decoder_layers(model)

    # Determine d_model and device/dtype from the first layer's MLP
    sample_mlp = layers[memory_config.memory_layers[0]].mlp
    if hasattr(sample_mlp, "gate_proj"):
        d_model = sample_mlp.gate_proj.in_features
        target_device = sample_mlp.gate_proj.weight.device
        target_dtype = sample_mlp.gate_proj.weight.dtype
    else:
        d_model = sample_mlp.d_model if hasattr(sample_mlp, "d_model") else 2560
        target_device = next(sample_mlp.parameters()).device
        target_dtype = next(sample_mlp.parameters()).dtype

    logger.info(f"Detected d_model={d_model}, device={target_device}, dtype={target_dtype}")

    # Create shared memory store
    shared_store = SharedMemoryStore(
        num_heads=memory_config.num_heads,
        n_keys=memory_config.n_keys,
        k_dim_per_head=memory_config.k_dim_per_head,
        v_dim=memory_config.v_dim,
        top_k=memory_config.top_k,
    )

    # Move shared store to the same device/dtype as the base model
    shared_store = shared_store.to(device=target_device, dtype=target_dtype)

    # Replace MLPs at specified layers
    for layer_idx in memory_config.memory_layers:
        if layer_idx >= len(layers):
            raise ValueError(
                f"Layer index {layer_idx} out of range (model has {len(layers)} layers)"
            )

        memory_layer = MemoryPlusLayer(
            d_model=d_model,
            shared_store=shared_store,
            num_heads=memory_config.num_heads,
            k_dim_per_head=memory_config.k_dim_per_head,
            v_dim=memory_config.v_dim,
            top_k=memory_config.top_k,
            use_silu_gating=memory_config.use_silu_gating,
        )

        # Move per-layer params to same device/dtype as the base model
        memory_layer = memory_layer.to(device=target_device, dtype=target_dtype)

        # Replace the MLP
        layers[layer_idx].mlp = memory_layer
        logger.info(f"Replaced layer {layer_idx} MLP with MemoryPlusLayer")

    total_memory_params = sum(
        p.numel() for p in shared_store.parameters()
    ) + sum(
        p.numel()
        for idx in memory_config.memory_layers
        for p in layers[idx].mlp.parameters()
        if p.data_ptr()
        not in {p2.data_ptr() for p2 in shared_store.parameters()}
    )
    logger.info(
        f"Injected {len(memory_config.memory_layers)} memory layers "
        f"with {total_memory_params / 1e6:.1f}M new parameters"
    )

    return model, shared_store


def get_memory_layers(
    model: torch.nn.Module,
    memory_config: MemoryConfig,
) -> list[MemoryPlusLayer]:
    """Get references to all injected memory layers."""
    layers = get_decoder_layers(model)
    return [layers[idx].mlp for idx in memory_config.memory_layers]


def save_memory_checkpoint(
    model: torch.nn.Module,
    shared_store: SharedMemoryStore,
    memory_config: MemoryConfig,
    save_path: str,
    step: int | None = None,
):
    """Save only the memory layer parameters (not the base model).

    Saves:
    - SharedMemoryStore state dict (keys + values)
    - Per-layer parameters (query_proj, query_norm, silu_proj, value_proj)
    """
    checkpoint = {
        "shared_store": shared_store.state_dict(),
        "memory_config": {
            "num_heads": memory_config.num_heads,
            "top_k": memory_config.top_k,
            "n_keys": memory_config.n_keys,
            "k_dim_per_head": memory_config.k_dim_per_head,
            "v_dim": memory_config.v_dim,
            "memory_layers": memory_config.memory_layers,
            "use_silu_gating": memory_config.use_silu_gating,
        },
        "per_layer_states": {},
    }

    layers = get_decoder_layers(model)
    for layer_idx in memory_config.memory_layers:
        mem_layer = layers[layer_idx].mlp
        # Save only per-layer params (not shared_store, which is saved separately)
        per_layer_state = {}
        for name, param in mem_layer.named_parameters():
            if not name.startswith("shared_store."):
                per_layer_state[name] = param.data
        checkpoint["per_layer_states"][layer_idx] = per_layer_state

    if step is not None:
        checkpoint["step"] = step

    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    logger.info(f"Saved memory checkpoint to {save_path}")


def load_memory_checkpoint(
    model: torch.nn.Module,
    shared_store: SharedMemoryStore,
    memory_config: MemoryConfig,
    load_path: str,
):
    """Load memory layer parameters from a checkpoint."""
    checkpoint = torch.load(load_path, map_location="cpu", weights_only=True)

    shared_store.load_state_dict(checkpoint["shared_store"])
    logger.info("Loaded shared memory store")

    layers = get_decoder_layers(model)
    for layer_idx, per_layer_state in checkpoint["per_layer_states"].items():
        layer_idx = int(layer_idx)
        mem_layer = layers[layer_idx].mlp
        for name, param_data in per_layer_state.items():
            param = dict(mem_layer.named_parameters())[name]
            param.data.copy_(param_data)
        logger.info(f"Loaded per-layer state for layer {layer_idx}")

    step = checkpoint.get("step", None)
    if step is not None:
        logger.info(f"Checkpoint was saved at step {step}")
    return step
