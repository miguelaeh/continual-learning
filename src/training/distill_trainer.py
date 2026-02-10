"""Distillation-based initialization of memory layers.

Instead of pretraining memory layers via indirect language modeling loss (128K steps),
this module trains each memory layer to directly mimic its corresponding original FFN
output using MSE loss. This converges in ~2K steps since the gradient signal is direct.

Approach:
1. Keep the original model as a frozen teacher (no memory layers injected)
2. Create standalone memory layers
3. Use forward hooks to capture FFN inputs/outputs from the teacher
4. Train: MSE(memory_layer(ffn_input), ffn_output) for each layer
"""

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import DistillConfig, MemoryConfig
from src.memory.memory_layer import MemoryPlusLayer
from src.memory.shared_memory_store import SharedMemoryStore
from src.model.memory_gemma import get_decoder_layers

logger = logging.getLogger(__name__)


class HiddenStateCapturer:
    """Captures FFN input/output hidden states using forward hooks.

    Registers hooks on the MLP modules at target layer indices. In
    Gemma3DecoderLayer, the FFN path is:
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)  # <-- hooked here

    The hook captures:
    - input[0]: hidden state after pre_feedforward_layernorm (memory layer input)
    - output: FFN output (what the memory layer should learn to produce)
    """

    def __init__(self, model: nn.Module, target_layer_indices: list[int]):
        self.target_indices = target_layer_indices
        self.captured_inputs: dict[int, torch.Tensor] = {}
        self.captured_outputs: dict[int, torch.Tensor] = {}
        self.hooks = []

        layers = get_decoder_layers(model)

        for idx in target_layer_indices:
            mlp = layers[idx].mlp

            def make_hook(layer_idx):
                def hook(module, input, output):
                    self.captured_inputs[layer_idx] = input[0].detach()
                    self.captured_outputs[layer_idx] = output.detach()
                return hook

            handle = mlp.register_forward_hook(make_hook(idx))
            self.hooks.append(handle)

    def get_captured_states(self) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Return captured (input, output) pairs for each layer."""
        return {
            idx: (self.captured_inputs[idx], self.captured_outputs[idx])
            for idx in self.target_indices
            if idx in self.captured_inputs and idx in self.captured_outputs
        }

    def clear(self):
        """Clear captured states for next batch."""
        self.captured_inputs.clear()
        self.captured_outputs.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def create_distill_optimizer(
    memory_layers: dict[int, MemoryPlusLayer],
    shared_store: SharedMemoryStore,
    config: DistillConfig,
) -> torch.optim.Optimizer:
    """Create AdamW optimizer with separate LR for value embeddings.

    Uses 10x higher LRs than pretrain since the distillation signal is direct.
    """
    value_params = []
    other_params = []

    value_param_ids = {id(p) for p in shared_store.values.parameters()}
    seen = set()

    for mem_layer in memory_layers.values():
        for param in mem_layer.parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            if id(param) in value_param_ids:
                value_params.append(param)
            else:
                other_params.append(param)

    param_groups = [
        {
            "params": other_params,
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        },
        {
            "params": value_params,
            "lr": config.value_learning_rate,
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(param_groups)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: DistillConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create linear warmup + cosine decay LR scheduler."""

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(
            1, config.total_steps - config.warmup_steps
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_distilled_checkpoint(
    memory_layers: dict[int, MemoryPlusLayer],
    shared_store: SharedMemoryStore,
    memory_config: MemoryConfig,
    checkpoint_dir: str,
    step: int | None = None,
):
    """Save distilled checkpoint in the same format as pretrain checkpoints.

    This ensures compatibility with Phase 2 (IDF) and Phase 3 (continual learning).
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

    for layer_idx, mem_layer in memory_layers.items():
        per_layer_state = {}
        for name, param in mem_layer.named_parameters():
            if not name.startswith("shared_store."):
                per_layer_state[name] = param.data
        checkpoint["per_layer_states"][layer_idx] = per_layer_state

    if step is not None:
        checkpoint["step"] = step

    save_dir = Path(checkpoint_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if step is not None:
        save_path = save_dir / f"memory_step_{step}.pt"
    else:
        save_path = save_dir / "memory_layers.pt"

    torch.save(checkpoint, str(save_path))
    logger.info(f"Saved distilled checkpoint to {save_path}")


def distill_memory_layers(
    teacher_model: nn.Module,
    memory_layers: dict[int, MemoryPlusLayer],
    shared_store: SharedMemoryStore,
    memory_config: MemoryConfig,
    train_config: DistillConfig,
    dataloader: DataLoader,
    device: torch.device | str = "cuda",
):
    """Distill memory layers to mimic original FFN outputs.

    For each batch:
    1. Forward through frozen teacher to capture FFN inputs/outputs via hooks
    2. Forward captured inputs through standalone memory layers
    3. Loss = MSE(memory_output, ffn_output) averaged across layers
    4. Backward + update memory layer params only

    Args:
        teacher_model: Original Gemma 3 4B (frozen, no memory layers injected).
        memory_layers: Dict mapping layer_idx -> standalone MemoryPlusLayer.
        shared_store: Shared memory store (referenced by all memory layers).
        memory_config: Memory layer configuration.
        train_config: Distillation training configuration.
        dataloader: FineWeb-Edu streaming dataloader.
        device: Training device.
    """
    teacher_model.eval()
    for mem_layer in memory_layers.values():
        mem_layer.train()

    optimizer = create_distill_optimizer(memory_layers, shared_store, train_config)
    scheduler = get_lr_scheduler(optimizer, train_config)

    capturer = HiddenStateCapturer(
        teacher_model, memory_config.memory_layers
    )

    accum_steps = train_config.gradient_accumulation_steps
    accum_loss = 0.0
    global_step = 0
    micro_step = 0

    logger.info(
        f"Starting distillation for {train_config.total_steps} steps "
        f"(batch_size={train_config.batch_size}, accum={accum_steps})"
    )

    for batch in dataloader:
        if global_step >= train_config.total_steps:
            break

        input_ids = batch["input_ids"].to(device)
        token_type_ids = torch.zeros_like(input_ids)

        # 1. Forward through teacher to capture FFN hidden states
        with torch.no_grad():
            teacher_model(input_ids=input_ids, token_type_ids=token_type_ids)

        captured = capturer.get_captured_states()
        capturer.clear()

        # 2. Compute distillation loss for each memory layer
        total_loss = torch.tensor(0.0, device=device)
        for layer_idx, (ffn_input, ffn_output) in captured.items():
            mem_layer = memory_layers[layer_idx]
            mem_output = mem_layer(ffn_input)
            layer_loss = F.mse_loss(mem_output, ffn_output)
            total_loss = total_loss + layer_loss

        total_loss = total_loss / len(memory_config.memory_layers)

        # 3. Gradient accumulation
        loss = total_loss / accum_steps
        loss.backward()
        accum_loss += loss.item()
        micro_step += 1

        if micro_step % accum_steps == 0:
            # Gradient clipping
            all_params = []
            seen = set()
            for mem_layer in memory_layers.values():
                for p in mem_layer.parameters():
                    if p.requires_grad and id(p) not in seen:
                        all_params.append(p)
                        seen.add(id(p))
            torch.nn.utils.clip_grad_norm_(all_params, train_config.gradient_clip)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Logging
            if global_step % train_config.log_every_steps == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {global_step}/{train_config.total_steps} | "
                    f"Distill MSE: {accum_loss:.6f} | LR: {lr:.2e}"
                )
                accum_loss = 0.0

            # Checkpointing
            if global_step % train_config.save_every_steps == 0:
                save_distilled_checkpoint(
                    memory_layers, shared_store, memory_config,
                    train_config.checkpoint_dir, step=global_step,
                )

    # Cleanup
    capturer.remove_hooks()

    # Final checkpoint
    save_distilled_checkpoint(
        memory_layers, shared_store, memory_config,
        train_config.checkpoint_dir, step=global_step,
    )
    # Also save as memory_layers.pt for easy reference
    final_path = Path(train_config.checkpoint_dir) / "memory_layers.pt"
    step_path = Path(train_config.checkpoint_dir) / f"memory_step_{global_step}.pt"
    if step_path.exists() and not final_path.exists():
        import shutil
        shutil.copy2(str(step_path), str(final_path))
        logger.info(f"Copied final checkpoint to {final_path}")

    logger.info(f"Distillation complete after {global_step} steps")
