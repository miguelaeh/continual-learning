from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from self_building_brain.config import TrainingConfig
from self_building_brain.data.synthetic import SyntheticBatch


def build_transition_targets(read_key_ids: torch.Tensor, num_nodes: int) -> torch.Tensor:
    batch_size, num_steps = read_key_ids.shape
    targets = torch.zeros(batch_size, num_nodes, num_nodes, device=read_key_ids.device)
    if num_steps < 2:
        return targets

    batch_indices = torch.arange(batch_size, device=read_key_ids.device).unsqueeze(1).expand(-1, num_steps - 1)
    source_ids = read_key_ids[:, :-1]
    target_ids = read_key_ids[:, 1:]
    targets[batch_indices, source_ids, target_ids] = 1.0
    return targets


def compute_graph_losses(
    outputs: dict[str, torch.Tensor],
    batch: SyntheticBatch,
    config: TrainingConfig,
) -> dict[str, torch.Tensor]:
    answer_loss = F.cross_entropy(outputs["answer_logits"], batch.answer_ids)
    node_loss = F.cross_entropy(
        outputs["step_node_logits"].reshape(-1, outputs["step_node_logits"].size(-1)),
        batch.read_key_ids.reshape(-1),
    )
    query_node_loss = F.cross_entropy(outputs["query_node_logits"], batch.query_key_ids)
    read_value_loss = F.cross_entropy(
        outputs["step_value_logits"].reshape(-1, outputs["step_value_logits"].size(-1)),
        batch.read_value_ids.reshape(-1),
    )

    transition_targets = build_transition_targets(batch.read_key_ids, num_nodes=outputs["final_edge_weights"].size(-1))
    edge_loss = F.binary_cross_entropy(outputs["final_edge_weights"].clamp(1e-6, 1 - 1e-6), transition_targets)

    node_weights = outputs["step_node_weights"].clamp_min(1e-8)
    node_entropy = -(node_weights * node_weights.log()).sum(dim=-1)
    node_sparsity_loss = node_entropy.mean() / math.log(node_weights.size(-1))
    edge_sparsity_loss = outputs["final_edge_weights"].mean()

    total_loss = (
        answer_loss
        + config.graph_node_loss_weight * (node_loss + query_node_loss)
        + config.read_value_loss_weight * read_value_loss
        + config.edge_loss_weight * edge_loss
        + config.node_sparsity_loss_weight * node_sparsity_loss
        + config.edge_sparsity_loss_weight * edge_sparsity_loss
    )

    with torch.no_grad():
        answer_accuracy = (outputs["answer_logits"].argmax(dim=-1) == batch.answer_ids).float().mean()
        node_accuracy = (outputs["step_node_logits"].argmax(dim=-1) == batch.read_key_ids).float().mean()
        query_node_accuracy = (outputs["query_node_logits"].argmax(dim=-1) == batch.query_key_ids).float().mean()

    return {
        "loss": total_loss,
        "answer_loss": answer_loss,
        "node_loss": node_loss,
        "query_node_loss": query_node_loss,
        "read_value_loss": read_value_loss,
        "edge_loss": edge_loss,
        "node_sparsity_loss": node_sparsity_loss,
        "edge_sparsity_loss": edge_sparsity_loss,
        "answer_accuracy": answer_accuracy,
        "node_accuracy": node_accuracy,
        "query_node_accuracy": query_node_accuracy,
    }
