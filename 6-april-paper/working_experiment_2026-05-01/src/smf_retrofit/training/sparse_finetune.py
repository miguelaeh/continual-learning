"""Sparse memory finetuning loop."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from smf_retrofit.config import ContinualConfig
from smf_retrofit.continual.gradient_masking import GradientMaskManager
from smf_retrofit.continual.selection import (
    KLSlotSelector,
    TFIDFSlotSelector,
    compute_access_counts,
    create_trainable_mask,
)
from smf_retrofit.modeling.qwen import get_memory_layers, save_memory_checkpoint


logger = logging.getLogger(__name__)


def create_sparse_optimizer(
    model: torch.nn.Module,
    config: ContinualConfig,
) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    if config.optimizer == "sgd":
        return torch.optim.SGD(params, lr=config.learning_rate, momentum=config.momentum)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=config.learning_rate)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def _build_selector(stats: dict, config: ContinualConfig):
    df = stats["slot_document_frequency"]
    total_batches = int(stats["total_batches"])
    if config.selector.lower() == "tfidf":
        return TFIDFSlotSelector(df, total_batches=total_batches, top_t=config.top_t)
    if config.selector.lower() == "kl":
        return KLSlotSelector(df, total_batches=total_batches, top_t=config.top_t)
    raise ValueError(f"Unsupported selector: {config.selector}")


def run_sparse_finetuning(
    model: torch.nn.Module,
    dataloader,
    config: ContinualConfig,
    background_stats: dict,
    layer_indices: list[int],
    num_entries: int,
    device: str,
) -> str:
    model.train()
    memory_layers = get_memory_layers(model, layer_indices)
    selector = _build_selector(background_stats, config)
    mask_manager = GradientMaskManager(model, layer_indices)
    optimizer = create_sparse_optimizer(model, config)

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    global_step = 0
    running_loss = 0.0

    while global_step < config.total_steps:
        progressed = False
        for batch in dataloader:
            if global_step >= config.total_steps:
                break

            progressed = True
            optimizer.zero_grad(set_to_none=True)

            for layer in memory_layers:
                layer.enable_index_tracking()

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            indices_list = []
            for layer in memory_layers:
                indices = layer.get_last_accessed_indices()
                if indices is not None:
                    indices_list.append(indices)
                layer.disable_index_tracking()

            access_counts = compute_access_counts(indices_list, num_entries)
            trainable_indices = selector.select(access_counts)
            trainable_mask = create_trainable_mask(trainable_indices, num_entries, device=device)
            mask_manager.set_mask(trainable_mask)

            loss = outputs.loss
            loss.backward()

            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    config.gradient_clip,
                )

            optimizer.step()
            global_step += 1
            running_loss += loss.item()

            if global_step % config.log_every_steps == 0:
                logger.info(
                    "Sparse finetune step %s/%s | loss %.4f | selected slots %s",
                    global_step,
                    config.total_steps,
                    running_loss / config.log_every_steps,
                    int(trainable_indices.numel()),
                )
                running_loss = 0.0

            if global_step % config.save_every_steps == 0:
                checkpoint_path = str(Path(config.output_dir) / f"memory_step_{global_step}.pt")
                save_memory_checkpoint(model, layer_indices, checkpoint_path, step=global_step)

        if not progressed:
            raise ValueError(
                "Continual dataloader produced zero batches. "
                "Reduce data.seq_length or provide more continual-learning text."
            )

    mask_manager.clear_mask()
    final_path = str(Path(config.output_dir) / "memory.pt")
    save_memory_checkpoint(model, layer_indices, final_path, step=global_step)
    return final_path
