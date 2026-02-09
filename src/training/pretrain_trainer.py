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


def create_optimizer(
    model: nn.Module,
    shared_store: SharedMemoryStore,
    config: PretrainConfig,
) -> torch.optim.Optimizer:
    """Create AdamW optimizer with separate LR for value embeddings.

    The value embeddings (EmbeddingBag) use a fixed higher learning rate
    (1e-3) as recommended in "Memory Layers at Scale", while other memory
    parameters use the standard learning rate.
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

    param_groups = [
        {
            "params": other_params,
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        },
        {
            "params": value_params,
            "lr": config.value_learning_rate,
            "weight_decay": 0.0,  # no weight decay on embeddings
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
):
    """Main Phase 1 training loop.

    Args:
        model: Model with injected and frozen base + trainable memory layers.
        shared_store: Shared memory store (for checkpoint saving).
        memory_config: Memory layer configuration.
        train_config: Training configuration.
        dataloader: FineWeb-Edu streaming dataloader.
        device: Training device.
    """
    model.train()

    optimizer = create_optimizer(model, shared_store, train_config)
    scheduler = get_lr_scheduler(optimizer, train_config)

    # Gradient accumulation
    accum_steps = train_config.gradient_accumulation_steps
    accum_loss = 0.0

    global_step = 0
    micro_step = 0

    logger.info(
        f"Starting Phase 1 pretraining for {train_config.total_steps} steps "
        f"(batch_size={train_config.batch_size}, accum={accum_steps})"
    )

    for batch in dataloader:
        if global_step >= train_config.total_steps:
            break

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
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {global_step}/{train_config.total_steps} | "
                    f"Loss: {accum_loss:.4f} | LR: {lr:.2e}"
                )
                accum_loss = 0.0

            # Checkpointing
            if global_step % train_config.save_every_steps == 0:
                save_path = str(
                    Path(train_config.checkpoint_dir) / f"memory_step_{global_step}.pt"
                )
                save_memory_checkpoint(
                    model, shared_store, memory_config, save_path, step=global_step
                )

    # Final checkpoint
    save_path = str(Path(train_config.checkpoint_dir) / "memory_layers.pt")
    save_memory_checkpoint(
        model, shared_store, memory_config, save_path, step=global_step
    )
    logger.info(f"Phase 1 pretraining complete after {global_step} steps")
