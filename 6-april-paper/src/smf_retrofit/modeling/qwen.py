"""Qwen2 model surgery for sparse-memory retrofitting."""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from smf_retrofit.config import MemoryConfig, ModelConfig
from smf_retrofit.memory.layer import SparseMemoryLayer
from smf_retrofit.memory.shared_store import SharedMemoryStore
from smf_retrofit.utils import detect_device, ensure_parent_dir, resolve_torch_dtype


logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_config: ModelConfig,
) -> tuple[torch.nn.Module, object]:
    """Load model and tokenizer."""
    dtype = resolve_torch_dtype(model_config.dtype)
    target_device = detect_device(model_config.device_map)
    kwargs = {
        "dtype": dtype,
        "local_files_only": model_config.local_files_only,
    }
    if model_config.attn_implementation:
        kwargs["attn_implementation"] = model_config.attn_implementation

    if target_device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name,
            **kwargs,
        ).to("mps")
    elif model_config.device_map is not None and target_device != "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name,
            device_map=model_config.device_map,
            **kwargs,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_config.model_name, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        local_files_only=model_config.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loaded base model %s", model_config.model_name)
    return model, tokenizer


def get_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"Unsupported model type: {type(model).__name__}")


def select_memory_layers(num_hidden_layers: int, config: MemoryConfig) -> list[int]:
    if config.memory_layers:
        return list(config.memory_layers)

    count = config.num_memory_layers
    if count <= 0:
        raise ValueError("memory.num_memory_layers must be positive.")

    stride = num_hidden_layers / (count + 1)
    selected = []
    for idx in range(count):
        candidate = round((idx + 1) * stride) - 1
        candidate = max(0, min(num_hidden_layers - 1, candidate))
        selected.append(candidate)
    return sorted(set(selected))


def inject_memory_layers(
    model: torch.nn.Module,
    memory_config: MemoryConfig,
) -> tuple[torch.nn.Module, SharedMemoryStore, list[int]]:
    """Replace selected Qwen MLP blocks with sparse memory layers."""
    layers = get_decoder_layers(model)
    layer_indices = select_memory_layers(len(layers), memory_config)

    sample_mlp = layers[layer_indices[0]].mlp
    d_model = sample_mlp.gate_proj.in_features
    target_device = sample_mlp.gate_proj.weight.device
    target_dtype = sample_mlp.gate_proj.weight.dtype

    shared_store = SharedMemoryStore(
        num_heads=memory_config.num_heads,
        n_keys=memory_config.n_keys,
        k_dim_per_head=memory_config.k_dim_per_head,
        v_dim=memory_config.v_dim,
        top_k=memory_config.top_k,
        use_delta_bank=memory_config.use_delta_bank,
    ).to(device=target_device, dtype=target_dtype)

    for layer_idx in layer_indices:
        memory_layer = SparseMemoryLayer(
            d_model=d_model,
            shared_store=shared_store,
            num_heads=memory_config.num_heads,
            k_dim_per_head=memory_config.k_dim_per_head,
            v_dim=memory_config.v_dim,
            top_k=memory_config.top_k,
            use_silu_gating=memory_config.use_silu_gating,
            use_delta_residual=memory_config.use_delta_residual,
            use_delta_value_proj=memory_config.use_delta_value_proj,
            delta_residual_init_scale=memory_config.delta_residual_init_scale,
        ).to(device=target_device, dtype=target_dtype)
        layers[layer_idx].mlp = memory_layer

    logger.info("Injected sparse memory layers at %s", layer_indices)
    return model, shared_store, layer_indices


def get_memory_layers(
    model: torch.nn.Module,
    layer_indices: list[int],
) -> list[SparseMemoryLayer]:
    layers = get_decoder_layers(model)
    return [layers[idx].mlp for idx in layer_indices]


def freeze_for_recovery(model: torch.nn.Module, layer_indices: list[int]) -> None:
    for param in model.parameters():
        param.requires_grad = False

    for layer in get_memory_layers(model, layer_indices):
        for param in layer.parameters():
            param.requires_grad = True
        if layer.shared_store.delta_values is not None:
            layer.shared_store.delta_values.weight.requires_grad = False
        if layer.delta_scale is not None:
            layer.delta_scale.requires_grad = False


def freeze_for_sparse_finetuning(
    model: torch.nn.Module,
    layer_indices: list[int],
    train_memory_mode: str = "values_only",
) -> None:
    for param in model.parameters():
        param.requires_grad = False

    memory_layers = get_memory_layers(model, layer_indices)
    if train_memory_mode == "full_memory":
        for layer in memory_layers:
            for param in layer.parameters():
                param.requires_grad = True
        return
    if train_memory_mode == "delta_values_only":
        for layer in memory_layers:
            if layer.shared_store.delta_values is None:
                raise ValueError("delta_values_only requires memory.use_delta_bank: true")
            layer.shared_store.delta_values.weight.requires_grad = True
        return
    if train_memory_mode == "delta_values_and_scale":
        for layer in memory_layers:
            if layer.shared_store.delta_values is None:
                raise ValueError("delta_values_and_scale requires memory.use_delta_bank: true")
            if layer.delta_scale is None:
                raise ValueError(
                    "delta_values_and_scale requires memory.use_delta_residual: true"
                )
            layer.shared_store.delta_values.weight.requires_grad = True
            layer.delta_scale.requires_grad = True
        return
    if train_memory_mode == "delta_values_scale_and_output":
        for layer in memory_layers:
            if layer.shared_store.delta_values is None:
                raise ValueError(
                    "delta_values_scale_and_output requires memory.use_delta_bank: true"
                )
            if layer.delta_scale is None:
                raise ValueError(
                    "delta_values_scale_and_output requires memory.use_delta_residual: true"
                )
            if layer.delta_value_proj is None:
                raise ValueError(
                    "delta_values_scale_and_output requires memory.use_delta_value_proj: true"
                )
            layer.shared_store.delta_values.weight.requires_grad = True
            layer.delta_scale.requires_grad = True
            layer.delta_value_proj.weight.requires_grad = True
        return
    if train_memory_mode == "delta_values_scale_output_and_gate":
        for layer in memory_layers:
            if layer.shared_store.delta_values is None:
                raise ValueError(
                    "delta_values_scale_output_and_gate requires memory.use_delta_bank: true"
                )
            if layer.delta_scale is None:
                raise ValueError(
                    "delta_values_scale_output_and_gate requires memory.use_delta_residual: true"
                )
            if layer.delta_value_proj is None:
                raise ValueError(
                    "delta_values_scale_output_and_gate requires memory.use_delta_value_proj: true"
                )
            if layer.silu_proj is None:
                raise ValueError(
                    "delta_values_scale_output_and_gate requires memory.use_silu_gating: true"
                )
            layer.shared_store.delta_values.weight.requires_grad = True
            layer.delta_scale.requires_grad = True
            layer.delta_value_proj.weight.requires_grad = True
            layer.silu_proj.weight.requires_grad = True
        return
    if train_memory_mode == "delta_values_and_gate":
        for layer in memory_layers:
            if layer.shared_store.delta_values is None:
                raise ValueError("delta_values_and_gate requires memory.use_delta_bank: true")
            layer.shared_store.delta_values.weight.requires_grad = True
            if layer.silu_proj is not None:
                layer.silu_proj.weight.requires_grad = True
        return
    if train_memory_mode == "values_and_output":
        shared_store = memory_layers[0].shared_store
        shared_store.values.weight.requires_grad = True
        for layer in memory_layers:
            layer.value_proj.weight.requires_grad = True
        return
    if train_memory_mode == "values_gate_and_output":
        shared_store = memory_layers[0].shared_store
        shared_store.values.weight.requires_grad = True
        for layer in memory_layers:
            layer.value_proj.weight.requires_grad = True
            if layer.silu_proj is not None:
                layer.silu_proj.weight.requires_grad = True
        return
    if train_memory_mode != "values_only":
        raise ValueError(
            "train_memory_mode must be one of: values_only, values_and_output, "
            "values_gate_and_output, delta_values_only, delta_values_and_scale, "
            "delta_values_scale_and_output, delta_values_scale_output_and_gate, "
            "delta_values_and_gate, full_memory."
        )

    shared_store = memory_layers[0].shared_store
    shared_store.values.weight.requires_grad = True


def save_memory_checkpoint(
    model: torch.nn.Module,
    layer_indices: list[int],
    path: str,
    step: int | None = None,
) -> None:
    ensure_parent_dir(path)
    layers = get_decoder_layers(model)
    checkpoint = {
        "layer_indices": list(layer_indices),
        "per_layer_states": {},
        "shared_store": layers[layer_indices[0]].mlp.shared_store.state_dict(),
        "step": step,
    }

    for layer_idx in layer_indices:
        memory_layer = layers[layer_idx].mlp
        state = {}
        for name, value in memory_layer.state_dict().items():
            if not name.startswith("shared_store."):
                state[name] = value
        checkpoint["per_layer_states"][layer_idx] = state

    torch.save(checkpoint, path)
    logger.info("Saved memory checkpoint to %s", path)


def load_memory_checkpoint(
    model: torch.nn.Module,
    path: str,
    layer_indices: list[int] | None = None,
) -> list[int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    stored_indices = [int(idx) for idx in checkpoint["layer_indices"]]
    if layer_indices is not None and stored_indices != list(layer_indices):
        raise ValueError(
            f"Checkpoint layer indices {stored_indices} do not match expected {layer_indices}."
        )

    layers = get_decoder_layers(model)
    layers[stored_indices[0]].mlp.shared_store.load_state_dict(
        checkpoint["shared_store"],
        strict=False,
    )
    for layer_idx in stored_indices:
        memory_layer = layers[layer_idx].mlp
        memory_layer.load_state_dict(checkpoint["per_layer_states"][layer_idx], strict=False)
        if memory_layer.delta_value_proj is not None:
            state = checkpoint["per_layer_states"][layer_idx]
            if "delta_value_proj.weight" not in state:
                memory_layer.delta_value_proj.weight.data.copy_(memory_layer.value_proj.weight.data)

    logger.info("Loaded memory checkpoint from %s", path)
    return stored_indices
