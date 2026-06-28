"""Sparse memory finetuning loop."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from smf_retrofit.config import ContinualConfig
from smf_retrofit.continual.gradient_masking import GradientMaskManager
from smf_retrofit.continual.selection import (
    KLSlotSelector,
    TFIDFSlotSelector,
    aggregate_access_counts,
    select_top_neighbor_indices,
    compute_contrastive_access_counts,
    compute_access_counts,
    create_trainable_mask,
)
from smf_retrofit.modeling.qwen import get_memory_layers, save_memory_checkpoint


logger = logging.getLogger(__name__)


def _collect_tracked_indices(memory_layers) -> list[torch.Tensor]:
    indices_list = []
    for layer in memory_layers:
        indices = layer.get_last_accessed_indices()
        if indices is not None:
            indices_list.append(indices)
        layer.disable_index_tracking()
    return indices_list


def _selection_token_mask(batch: dict[str, torch.Tensor], supervised_only: bool) -> torch.Tensor | None:
    if not supervised_only:
        return None
    return batch["labels"].ne(-100).detach()


def _next_replay_batch(replay_iterator, replay_dataloader):
    try:
        return next(replay_iterator), replay_iterator
    except StopIteration:
        replay_iterator = iter(replay_dataloader)
        return next(replay_iterator), replay_iterator


def _top_gradient_rows(grad: torch.Tensor | None, top_k: int) -> torch.Tensor:
    if grad is None or top_k <= 0:
        return torch.zeros(0, dtype=torch.long)

    if grad.is_sparse:
        sparse = grad.coalesce()
        row_ids = sparse.indices()[0]
        row_norms = sparse.values().float().norm(dim=-1)
        if row_ids.numel() == 0:
            return torch.zeros(0, dtype=torch.long)
        unique_rows = row_ids.unique()
        per_row = torch.zeros(unique_rows.shape[0], dtype=row_norms.dtype, device=row_norms.device)
        for idx, row in enumerate(unique_rows):
            per_row[idx] = row_norms[row_ids == row].sum()
        k = min(top_k, unique_rows.numel())
        top_pos = per_row.topk(k).indices
        return unique_rows[top_pos].detach().cpu()

    row_norms = grad.float().norm(dim=-1)
    nonzero = torch.nonzero(row_norms > 0, as_tuple=False).flatten()
    if nonzero.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    scores = row_norms[nonzero]
    k = min(top_k, nonzero.numel())
    top_pos = scores.topk(k).indices
    return nonzero[top_pos].detach().cpu()


def _active_value_weight(memory_layers) -> torch.nn.Parameter:
    layer = memory_layers[0]
    if layer.shared_store.delta_values is not None and layer.shared_store.delta_values.weight.requires_grad:
        return layer.shared_store.delta_values.weight
    return layer.shared_store.values.weight


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
    replay_dataloader=None,
    replay_teacher_model: torch.nn.Module | None = None,
) -> str:
    model.train()
    memory_layers = get_memory_layers(model, layer_indices)
    selector = _build_selector(background_stats, config)
    mask_manager = GradientMaskManager(model, layer_indices)
    optimizer = create_sparse_optimizer(model, config)
    replay_iterator = iter(replay_dataloader) if replay_dataloader is not None else None

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    global_step = 0
    running_loss = 0.0
    running_task_loss = 0.0
    running_replay_loss = 0.0

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
            task_indices_list = _collect_tracked_indices(memory_layers)
            task_token_mask = _selection_token_mask(batch, config.selection_supervised_only)
            task_access_counts = compute_access_counts(
                task_indices_list,
                num_entries,
                token_mask=task_token_mask,
            )
            effective_access_counts = task_access_counts
            replay_batches: list[dict[str, torch.Tensor]] = []
            replay_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
            replay_access_counts = None

            task_loss = outputs.loss
            replay_loss = None
            loss = task_loss

            replay_active = (
                replay_iterator is not None
                and config.replay_weight > 0
                and global_step >= config.replay_start_step
            )
            replay_selection_active = (
                replay_iterator is not None
                and (config.replay_select_union or config.replay_select_difference)
            )
            if replay_active or replay_selection_active:
                candidate_count = max(1, config.replay_neighbor_candidates)
                for _ in range(candidate_count):
                    replay_batch, replay_iterator = _next_replay_batch(
                        replay_iterator,
                        replay_dataloader,
                    )
                    replay_batches.append(replay_batch)
                    replay_input_ids = replay_batch["input_ids"].to(device)
                    replay_labels = replay_batch["labels"].to(device)
                    replay_attention_mask = replay_batch.get("attention_mask")
                    if replay_attention_mask is not None:
                        replay_attention_mask = replay_attention_mask.to(device)
                    replay_inputs.append(
                        (replay_input_ids, replay_labels, replay_attention_mask)
                    )

            selected_neighbor_indices: list[int] = []
            candidate_replay_access_counts: list[torch.Tensor] = []
            if replay_selection_active and replay_batches:
                if config.replay_select_union or config.replay_select_difference:
                    for replay_batch, replay_input in zip(replay_batches, replay_inputs):
                        replay_input_ids, _, replay_attention_mask = replay_input
                        for layer in memory_layers:
                            layer.enable_index_tracking()
                        with torch.no_grad():
                            model(
                                input_ids=replay_input_ids,
                                attention_mask=replay_attention_mask,
                                labels=None,
                            )
                        replay_indices_list = _collect_tracked_indices(memory_layers)
                        replay_token_mask = _selection_token_mask(
                            replay_batch,
                            config.selection_supervised_only,
                        )
                        candidate_replay_access_counts.append(
                            compute_access_counts(
                                replay_indices_list,
                                num_entries,
                                token_mask=replay_token_mask,
                            )
                        )

                    selected_neighbor_indices = select_top_neighbor_indices(
                        task_access_counts,
                        candidate_replay_access_counts,
                        top_k=max(1, config.replay_neighbor_top_k),
                        metric=config.replay_neighbor_similarity,
                    )
                    replay_access_counts = aggregate_access_counts(
                        candidate_replay_access_counts,
                        selected_neighbor_indices,
                    )

            if config.replay_select_difference and replay_access_counts is not None:
                effective_access_counts = compute_contrastive_access_counts(
                    task_access_counts,
                    replay_access_counts,
                    negative_scale=config.replay_select_difference_scale,
                )

            trainable_indices = selector.select(effective_access_counts)

            if replay_active and replay_batches:
                active_neighbor_indices = (
                    selected_neighbor_indices
                    if selected_neighbor_indices
                    else list(range(min(max(1, config.replay_neighbor_top_k), len(replay_batches))))
                )
                replay_losses = []
                for idx in active_neighbor_indices:
                    replay_input_ids, replay_labels, replay_attention_mask = replay_inputs[idx]
                    if config.replay_distill_to_recovery:
                        replay_outputs = model(
                            input_ids=replay_input_ids,
                            attention_mask=replay_attention_mask,
                        )
                        if replay_teacher_model is None:
                            raise ValueError(
                                "replay_distill_to_recovery requires replay_teacher_model."
                            )
                        with torch.no_grad():
                            teacher_outputs = replay_teacher_model(
                                input_ids=replay_input_ids,
                                attention_mask=replay_attention_mask,
                            )
                        replay_losses.append(
                            _masked_forward_kl(
                                replay_outputs.logits,
                                teacher_outputs.logits,
                                replay_labels.ne(-100),
                            )
                        )
                    else:
                        replay_outputs = model(
                            input_ids=replay_input_ids,
                            attention_mask=replay_attention_mask,
                            labels=replay_labels,
                        )
                        replay_losses.append(replay_outputs.loss)
                replay_loss = torch.stack(replay_losses).mean()
                if config.replay_select_union:
                    if replay_access_counts is None:
                        candidate_replay_access_counts = []
                        for replay_batch, replay_input in zip(replay_batches, replay_inputs):
                            replay_input_ids, _, replay_attention_mask = replay_input
                            for layer in memory_layers:
                                layer.enable_index_tracking()
                            with torch.no_grad():
                                model(
                                    input_ids=replay_input_ids,
                                    attention_mask=replay_attention_mask,
                                    labels=None,
                                )
                            replay_indices_list = _collect_tracked_indices(memory_layers)
                            replay_token_mask = _selection_token_mask(
                                replay_batch,
                                config.selection_supervised_only,
                            )
                            candidate_replay_access_counts.append(
                                compute_access_counts(
                                    replay_indices_list,
                                    num_entries,
                                    token_mask=replay_token_mask,
                                )
                            )
                        replay_access_counts = aggregate_access_counts(
                            candidate_replay_access_counts,
                            active_neighbor_indices,
                        )
                    replay_trainable_indices = selector.select(replay_access_counts)
                    trainable_indices = torch.unique(
                        torch.cat([trainable_indices, replay_trainable_indices]).long()
                    )
                loss = task_loss + (config.replay_weight * replay_loss)

            loss.backward()

            if config.post_backward_grad_top_k > 0:
                grad_top_rows = _top_gradient_rows(
                    _active_value_weight(memory_layers).grad,
                    config.post_backward_grad_top_k,
                )
                if grad_top_rows.numel() > 0:
                    trainable_indices = torch.unique(
                        torch.cat([trainable_indices.cpu(), grad_top_rows]).long()
                    )

            trainable_mask = create_trainable_mask(trainable_indices, num_entries, device=device)
            mask_manager.apply_mask_to_grads(trainable_mask)

            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    config.gradient_clip,
                )

            optimizer.step()
            global_step += 1
            running_loss += loss.item()
            running_task_loss += task_loss.item()
            if replay_loss is not None:
                running_replay_loss += replay_loss.item()

            if global_step % config.log_every_steps == 0:
                if replay_loss is None:
                    logger.info(
                        "Sparse finetune step %s/%s | loss %.4f | selected slots %s",
                        global_step,
                        config.total_steps,
                        running_loss / config.log_every_steps,
                        int(trainable_indices.numel()),
                    )
                else:
                    logger.info(
                        "Sparse finetune step %s/%s | total %.4f | task %.4f | replay %.4f | selected slots %s",
                        global_step,
                        config.total_steps,
                        running_loss / config.log_every_steps,
                        running_task_loss / config.log_every_steps,
                        running_replay_loss / config.log_every_steps,
                        int(trainable_indices.numel()),
                    )
                running_loss = 0.0
                running_task_loss = 0.0
                running_replay_loss = 0.0

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


def _masked_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    active = token_mask.bool()
    if not bool(active.any()):
        return student_logits.new_zeros(())

    student = student_logits[active].float()
    teacher = teacher_logits[active].float()
    log_probs = F.log_softmax(student, dim=-1)
    probs = F.softmax(teacher, dim=-1)
    return F.kl_div(log_probs, probs, reduction="batchmean")
