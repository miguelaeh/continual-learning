"""Phase 1 training loop: pretrain memory layers with base model frozen.

Trains the memory layer parameters (shared keys, values, per-layer projections)
on FineWeb-Edu using cross-entropy language modeling loss. The base Gemma 3 4B
parameters remain frozen throughout.
"""

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import MemoryConfig, PretrainConfig
from src.memory.shared_memory_store import SharedMemoryStore
from src.model.memory_gemma import save_memory_checkpoint

logger = logging.getLogger(__name__)


def save_training_checkpoint(
    model: nn.Module,
    shared_store: SharedMemoryStore,
    memory_config: MemoryConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    global_step: int,
    micro_step: int,
    save_path: str,
):
    """Save full training state for resuming."""
    from src.model.memory_gemma import get_decoder_layers

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
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "micro_step": micro_step,
        "step": global_step,
    }

    layers = get_decoder_layers(model)
    for layer_idx in memory_config.memory_layers:
        mem_layer = layers[layer_idx].mlp
        per_layer_state = {}
        for name, param in mem_layer.named_parameters():
            if not name.startswith("shared_store."):
                per_layer_state[name] = param.data
        checkpoint["per_layer_states"][layer_idx] = per_layer_state

    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    logger.info(f"Saved training checkpoint to {save_path} (step {global_step})")


def load_training_checkpoint(
    model: nn.Module,
    shared_store: SharedMemoryStore,
    memory_config: MemoryConfig,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    load_path: str,
) -> tuple[int, int]:
    """Load full training state for resuming.

    Returns:
        Tuple of (global_step, micro_step).
    """
    from src.model.memory_gemma import get_decoder_layers

    checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)

    # Restore memory weights
    shared_store.load_state_dict(checkpoint["shared_store"])
    layers = get_decoder_layers(model)
    for layer_idx, per_layer_state in checkpoint["per_layer_states"].items():
        layer_idx = int(layer_idx)
        mem_layer = layers[layer_idx].mlp
        for name, param_data in per_layer_state.items():
            param = dict(mem_layer.named_parameters())[name]
            param.data.copy_(param_data)

    # Restore optimizer and scheduler
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    global_step = checkpoint["global_step"]
    micro_step = checkpoint.get("micro_step", global_step * 8)

    logger.info(f"Resumed from checkpoint at step {global_step}")
    return global_step, micro_step


def create_optimizer(
    model: nn.Module,
    shared_store: SharedMemoryStore,
    config: PretrainConfig,
) -> torch.optim.Optimizer:
    """Create optimizer with separate LR for value embeddings.

    Supports AdamW (default) and SGD. The paper notes that SGD with high LR
    works better for sparse memory — AdamW's adaptive step sizes and momentum
    can interact with sparsity in unexpected ways.
    """
    value_params = []
    other_params = []

    value_param_ids = {id(p) for p in shared_store.values.parameters()}

    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in value_param_ids:
            value_params.append(param)
        else:
            other_params.append(param)

    optimizer_type = getattr(config, "optimizer", "adamw").lower()

    if optimizer_type == "sgd":
        param_groups = [
            {
                "params": other_params,
                "lr": config.learning_rate,
            },
            {
                "params": value_params,
                "lr": config.value_learning_rate,
            },
        ]
        momentum = getattr(config, "momentum", 0.0)
        logger.info(f"Using SGD optimizer (lr={config.learning_rate}, "
                    f"value_lr={config.value_learning_rate}, momentum={momentum})")
        return torch.optim.SGD(param_groups, momentum=momentum)
    else:
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
    config: PretrainConfig,
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


def train_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device | str,
) -> torch.Tensor:
    """Single training step. Returns the loss."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    token_type_ids = torch.zeros_like(input_ids)

    outputs = model(input_ids=input_ids, labels=labels, token_type_ids=token_type_ids)
    return outputs.loss


def pretrain_memory_layers(
    model: nn.Module,
    shared_store: SharedMemoryStore,
    memory_config: MemoryConfig,
    train_config: PretrainConfig,
    dataloader: DataLoader,
    device: torch.device | str = "cuda",
    resume_path: str | None = None,
):
    """Main Phase 1 training loop.

    Args:
        model: Model with injected and frozen base + trainable memory layers.
        shared_store: Shared memory store (for checkpoint saving).
        memory_config: Memory layer configuration.
        train_config: Training configuration.
        dataloader: FineWeb-Edu streaming dataloader.
        device: Training device.
        resume_path: Path to a training checkpoint to resume from.
    """
    model.train()

    optimizer = create_optimizer(model, shared_store, train_config)
    scheduler = get_lr_scheduler(optimizer, train_config)

    # Gradient accumulation
    accum_steps = train_config.gradient_accumulation_steps
    accum_loss = 0.0

    global_step = 0
    micro_step = 0

    # Resume from checkpoint
    if resume_path:
        global_step, micro_step = load_training_checkpoint(
            model, shared_store, memory_config, optimizer, scheduler, resume_path
        )

    logger.info(
        f"Starting Phase 1 pretraining for {train_config.total_steps} steps "
        f"(batch_size={train_config.batch_size}, accum={accum_steps}, "
        f"resuming from step {global_step})"
    )

    # Skip batches already consumed before the checkpoint
    batches_to_skip = micro_step
    skipped = 0

    for batch in dataloader:
        if global_step >= train_config.total_steps:
            break

        # Fast-forward past already-consumed batches
        if skipped < batches_to_skip:
            skipped += 1
            if skipped % 1000 == 0:
                logger.info(f"  Skipping batch {skipped}/{batches_to_skip}...")
            continue

        loss = train_step(model, batch, device)
        loss = loss / accum_steps
        loss.backward()
        accum_loss += loss.item()
        micro_step += 1

        if micro_step % accum_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                train_config.gradient_clip,
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Logging
            if global_step % train_config.log_every_steps == 0:
                avg_loss = accum_loss / train_config.log_every_steps
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {global_step}/{train_config.total_steps} | "
                    f"Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                )
                accum_loss = 0.0

            # Checkpointing — save full training state for resume
            if global_step % train_config.save_every_steps == 0:
                save_path = str(
                    Path(train_config.checkpoint_dir) / f"memory_step_{global_step}.pt"
                )
                save_training_checkpoint(
                    model, shared_store, memory_config,
                    optimizer, scheduler, global_step, micro_step, save_path,
                )

    # Final checkpoint (memory-only, compatible with Phase 2/3)
    save_path = str(Path(train_config.checkpoint_dir) / "memory_layers.pt")
    save_memory_checkpoint(
        model, shared_store, memory_config, save_path, step=global_step
    )
    logger.info(f"Phase 1 pretraining complete after {global_step} steps")
