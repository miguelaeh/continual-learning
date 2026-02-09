"""Phase 3 training loop: continual learning via sparse memory finetuning.

Uses TF-IDF slot selection and gradient masking to update only a small
subset of memory slots. SGD optimizer with high learning rate and no momentum
is critical for minimizing forgetting.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import ContinualLearningConfig, MemoryConfig
from src.continual.gradient_masking import GradientMaskManager
from src.continual.tfidf_selector import TFIDFSelector
from src.memory.memory_layer import MemoryPlusLayer
from src.model.memory_gemma import (
    get_memory_layers,
    save_memory_checkpoint,
)

logger = logging.getLogger(__name__)


def create_continual_optimizer(
    model: nn.Module,
    config: ContinualLearningConfig,
) -> torch.optim.Optimizer:
    """Create SGD optimizer for continual learning.

    SGD with no momentum is critical: yields 11% forgetting vs ~30% with AdamW.
    High learning rates (2.0 or even 10.0) work well for sparse updates.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if config.optimizer == "sgd":
        return torch.optim.SGD(
            trainable_params,
            lr=config.learning_rate,
            momentum=config.momentum,
        )
    elif config.optimizer == "adamw":
        logger.warning(
            "Using AdamW for continual learning. SGD is recommended "
            "for much less forgetting (11% vs ~30%)."
        )
        return torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def continual_learning_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    memory_layers: list[MemoryPlusLayer],
    selector: TFIDFSelector,
    mask_manager: GradientMaskManager,
    memory_config: MemoryConfig,
    device: torch.device | str,
) -> tuple[torch.Tensor, int]:
    """Single continual learning step with TF-IDF selection.

    1. Forward pass with index tracking to collect access counts
    2. TF-IDF ranking to select top-t trainable slots
    3. Apply gradient mask
    4. Backward pass (gradients only flow to selected slots)

    Returns:
        Tuple of (loss, number of unique slots selected).
    """
    # Enable index tracking
    for layer in memory_layers:
        layer.enable_index_tracking()

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    # Forward pass
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss

    # Collect access counts from all memory layers
    indices_list = []
    for layer in memory_layers:
        idx = layer.get_last_accessed_indices()
        if idx is not None:
            indices_list.append(idx)

    # Disable tracking
    for layer in memory_layers:
        layer.disable_index_tracking()

    # TF-IDF slot selection
    num_entries = memory_config.num_entries
    access_counts = selector.compute_access_counts(indices_list, num_entries)
    trainable_indices = selector.select(access_counts)

    # Create and apply gradient mask
    trainable_mask = selector.create_trainable_mask(
        trainable_indices, num_entries, device=device
    )
    mask_manager.set_mask(trainable_mask)

    # Backward pass (gradients masked to selected slots only)
    loss.backward()

    return loss, len(trainable_indices)


def continual_learn(
    model: nn.Module,
    shared_store: nn.Module,
    memory_config: MemoryConfig,
    cl_config: ContinualLearningConfig,
    dataloader: DataLoader,
    idf_statistics: dict,
    device: torch.device | str = "cuda",
    eval_fn=None,
):
    """Main Phase 3 continual learning loop.

    Args:
        model: Model with pretrained memory layers.
        shared_store: Shared memory store.
        memory_config: Memory layer configuration.
        cl_config: Continual learning configuration.
        dataloader: Task data dataloader.
        idf_statistics: Background IDF statistics from Phase 2.
        device: Training device.
        eval_fn: Optional evaluation function called periodically.
    """
    model.train()

    memory_layers = get_memory_layers(model, memory_config)
    selector = TFIDFSelector(idf_statistics, top_t=cl_config.top_t)
    mask_manager = GradientMaskManager(model, memory_config)
    optimizer = create_continual_optimizer(model, cl_config)

    logger.info(
        f"Starting Phase 3 continual learning for {cl_config.total_steps} steps "
        f"(top_t={cl_config.top_t}, optimizer={cl_config.optimizer}, "
        f"lr={cl_config.learning_rate})"
    )

    global_step = 0
    running_loss = 0.0

    for batch in dataloader:
        if global_step >= cl_config.total_steps:
            break

        optimizer.zero_grad()

        loss, num_selected = continual_learning_step(
            model=model,
            batch=batch,
            memory_layers=memory_layers,
            selector=selector,
            mask_manager=mask_manager,
            memory_config=memory_config,
            device=device,
        )

        optimizer.step()
        global_step += 1
        running_loss += loss.item()

        # Logging
        if global_step % cl_config.log_every_steps == 0:
            avg_loss = running_loss / cl_config.log_every_steps
            logger.info(
                f"Step {global_step}/{cl_config.total_steps} | "
                f"Loss: {avg_loss:.4f} | Slots updated: {num_selected}"
            )
            running_loss = 0.0

        # Evaluation
        if eval_fn and global_step % cl_config.eval_every_steps == 0:
            model.eval()
            eval_fn(model, global_step)
            model.train()

        # Checkpointing
        if global_step % cl_config.save_every_steps == 0:
            save_path = str(
                Path(cl_config.checkpoint_dir) / f"continual_step_{global_step}.pt"
            )
            save_memory_checkpoint(
                model, shared_store, memory_config, save_path, step=global_step
            )

    # Clean up
    mask_manager.clear_mask()

    # Final checkpoint
    save_path = str(Path(cl_config.checkpoint_dir) / "continual_final.pt")
    save_memory_checkpoint(
        model, shared_store, memory_config, save_path, step=global_step
    )
    logger.info(f"Phase 3 continual learning complete after {global_step} steps")
