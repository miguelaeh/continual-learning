from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from self_building_brain.config import TrainingConfig
from self_building_brain.data.synthetic import SyntheticBatch
from self_building_brain.models.teacher import SyntheticTeacherTargets


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: SyntheticBatch,
    teacher_targets: SyntheticTeacherTargets,
    config: TrainingConfig,
) -> dict[str, torch.Tensor]:
    answer_loss = F.cross_entropy(outputs["answer_logits"], batch.answer_ids)
    state_loss = F.mse_loss(outputs["step_states"], teacher_targets.teacher_states)
    trace_loss = F.mse_loss(outputs["step_write_vectors"], teacher_targets.trace_embeddings)
    read_value_loss = F.cross_entropy(
        outputs["step_value_logits"].reshape(-1, outputs["step_value_logits"].size(-1)),
        batch.read_value_ids.reshape(-1),
    )
    routing_loss = F.cross_entropy(
        outputs["step_slot_logits"].reshape(-1, outputs["step_slot_logits"].size(-1)),
        teacher_targets.step_slot_ids.reshape(-1),
    )
    query_slot_loss = F.cross_entropy(outputs["query_slot_logits"], teacher_targets.query_slot_ids)

    slot_weights = outputs["step_slot_weights"].clamp_min(1e-8)
    entropy = -(slot_weights * slot_weights.log()).sum(dim=-1)
    sparsity_loss = entropy.mean() / math.log(slot_weights.size(-1))

    total_loss = (
        answer_loss
        + config.state_loss_weight * state_loss
        + config.trace_loss_weight * trace_loss
        + config.read_value_loss_weight * read_value_loss
        + config.routing_loss_weight * routing_loss
        + config.query_slot_loss_weight * query_slot_loss
        + config.sparsity_loss_weight * sparsity_loss
    )

    with torch.no_grad():
        answer_accuracy = (outputs["answer_logits"].argmax(dim=-1) == batch.answer_ids).float().mean()
        query_slot_accuracy = (outputs["query_slot_logits"].argmax(dim=-1) == teacher_targets.query_slot_ids).float().mean()
        routing_accuracy = (
            outputs["step_slot_logits"].argmax(dim=-1) == teacher_targets.step_slot_ids
        ).float().mean()

    return {
        "loss": total_loss,
        "answer_loss": answer_loss,
        "state_loss": state_loss,
        "trace_loss": trace_loss,
        "read_value_loss": read_value_loss,
        "routing_loss": routing_loss,
        "query_slot_loss": query_slot_loss,
        "sparsity_loss": sparsity_loss,
        "answer_accuracy": answer_accuracy,
        "query_slot_accuracy": query_slot_accuracy,
        "routing_accuracy": routing_accuracy,
    }
