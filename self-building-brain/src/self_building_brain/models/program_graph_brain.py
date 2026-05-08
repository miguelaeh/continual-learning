from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_distribution(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.clamp_min(1e-8)
    return tensor / tensor.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _apply_operator_bank(distribution: torch.Tensor) -> torch.Tensor:
    shifted_left = torch.roll(distribution, shifts=-1, dims=-1)
    shifted_right = torch.roll(distribution, shifts=1, dims=-1)
    sharpened = _normalize_distribution(distribution.pow(2))
    smoothed = _normalize_distribution(torch.sqrt(distribution.clamp_min(1e-8)))
    complemented = _normalize_distribution(1.0 - distribution + 1e-4)
    return torch.stack(
        [
            distribution,
            sharpened,
            smoothed,
            shifted_left,
            shifted_right,
            complemented,
        ],
        dim=-2,
    )


@dataclass
class ProgramGraphState:
    node_key_logits: torch.Tensor
    node_value_logits: torch.Tensor
    node_operator_logits: torch.Tensor
    edge_weights: torch.Tensor
    edge_operator_logits: torch.Tensor
    node_active: torch.Tensor
    last_target_nodes: torch.Tensor

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        num_nodes: int,
        num_keys: int,
        num_values: int,
        num_operators: int,
        device: torch.device | str,
    ) -> "ProgramGraphState":
        return cls(
            node_key_logits=torch.zeros(batch_size, num_nodes, num_keys, device=device),
            node_value_logits=torch.zeros(batch_size, num_nodes, num_values, device=device),
            node_operator_logits=torch.zeros(batch_size, num_nodes, num_operators, device=device),
            edge_weights=torch.zeros(batch_size, num_nodes, num_nodes, device=device),
            edge_operator_logits=torch.zeros(batch_size, num_nodes, num_nodes, num_operators, device=device),
            node_active=torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device),
            last_target_nodes=torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device),
        )


class ProgramGraphGenerator(nn.Module):
    """
    Learns to update a self-contained graph program.

    The graph stores:
    - node keys
    - node values
    - node-local operator selections
    - edge weights
    - edge-local operator selections
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_nodes: int,
        num_keys: int,
        num_values: int,
        num_operators: int,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_keys = num_keys
        self.num_values = num_values
        self.num_operators = num_operators

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.context_projection = nn.Linear(num_keys + num_values + num_operators, hidden_dim)
        self.target_router = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes),
        )
        self.source_router = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes),
        )
        self.key_head = nn.Linear(hidden_dim * 2, num_keys)
        self.value_head = nn.Linear(hidden_dim * 2, num_values)
        self.node_operator_head = nn.Linear(hidden_dim * 2, num_operators)
        self.edge_operator_head = nn.Linear(hidden_dim * 2, num_operators)
        self.node_blend_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_step(self, read_tokens: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embedding(read_tokens)
        _, hidden = self.encoder(token_embeddings)
        return hidden[-1]

    def summarize_graph(self, state: ProgramGraphState) -> torch.Tensor:
        key_probs = F.softmax(state.node_key_logits, dim=-1)
        value_probs = F.softmax(state.node_value_logits, dim=-1)
        operator_probs = F.softmax(state.node_operator_logits, dim=-1)
        node_summary = torch.cat([key_probs, value_probs, operator_probs], dim=-1)
        masked_summary = node_summary * state.node_active.unsqueeze(-1).float()
        denom = state.node_active.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_summary = masked_summary.sum(dim=1) / denom
        return self.context_projection(mean_summary)

    def step(
        self,
        state: ProgramGraphState,
        read_tokens: torch.Tensor,
    ) -> tuple[ProgramGraphState, dict[str, torch.Tensor]]:
        step_repr = self.encode_step(read_tokens)
        graph_context = self.summarize_graph(state)
        fused = torch.cat([step_repr, graph_context], dim=-1)

        target_logits = self.target_router(fused)
        source_logits = self.source_router(fused)
        source_logits = source_logits.masked_fill(~state.node_active, -1e4)

        has_active = state.node_active.any(dim=-1)
        safe_source_logits = torch.where(has_active.unsqueeze(-1), source_logits, torch.zeros_like(source_logits))
        selected_targets = target_logits.argmax(dim=-1)
        selected_sources = torch.where(
            has_active,
            safe_source_logits.argmax(dim=-1),
            selected_targets,
        )

        key_logits = self.key_head(fused)
        value_logits = self.value_head(fused)
        node_operator_logits = self.node_operator_head(fused)
        edge_operator_logits = self.edge_operator_head(fused)
        node_blend = torch.sigmoid(self.node_blend_gate(fused)).squeeze(-1)
        edge_gate = torch.sigmoid(self.edge_gate(fused)).squeeze(-1)

        target_one_hot = F.one_hot(selected_targets, num_classes=self.num_nodes).float()
        source_one_hot = F.one_hot(selected_sources, num_classes=self.num_nodes).float()
        node_mask = target_one_hot.unsqueeze(-1)
        batch_size = read_tokens.size(0)

        new_key_logits = key_logits.unsqueeze(1).expand(-1, self.num_nodes, -1)
        new_value_logits = value_logits.unsqueeze(1).expand(-1, self.num_nodes, -1)
        new_node_operator_logits = node_operator_logits.unsqueeze(1).expand(-1, self.num_nodes, -1)
        blend = node_blend.view(batch_size, 1, 1)

        updated_node_key_logits = state.node_key_logits * (1.0 - node_mask) + (
            blend * state.node_key_logits + (1.0 - blend) * new_key_logits
        ) * node_mask
        updated_node_value_logits = state.node_value_logits * (1.0 - node_mask) + (
            blend * state.node_value_logits + (1.0 - blend) * new_value_logits
        ) * node_mask
        updated_node_operator_logits = state.node_operator_logits * (1.0 - node_mask) + (
            blend * state.node_operator_logits + (1.0 - blend) * new_node_operator_logits
        ) * node_mask

        edge_mask = source_one_hot.unsqueeze(2) * target_one_hot.unsqueeze(1)
        new_edge_operator_logits = edge_operator_logits.view(batch_size, 1, 1, self.num_operators)
        updated_edge_operator_logits = (
            state.edge_operator_logits * (1.0 - edge_mask.unsqueeze(-1))
            + new_edge_operator_logits * edge_mask.unsqueeze(-1)
        )
        updated_edge_weights = torch.maximum(state.edge_weights, edge_gate.view(batch_size, 1, 1) * edge_mask)

        next_state = ProgramGraphState(
            node_key_logits=updated_node_key_logits,
            node_value_logits=updated_node_value_logits,
            node_operator_logits=updated_node_operator_logits,
            edge_weights=updated_edge_weights,
            edge_operator_logits=updated_edge_operator_logits,
            node_active=state.node_active | target_one_hot.bool(),
            last_target_nodes=selected_targets,
        )
        return next_state, {
            "target_logits": target_logits,
            "source_logits": safe_source_logits,
            "key_logits": key_logits,
            "value_logits": value_logits,
            "node_operator_logits": node_operator_logits,
            "edge_operator_logits": edge_operator_logits,
            "selected_targets": selected_targets,
            "selected_sources": selected_sources,
            "edge_gates": edge_gate,
        }


class FixedProgramGraphVM(nn.Module):
    """
    Fixed, parameter-free virtual machine for executing a stored graph.

    All learned semantics are stored in the graph itself. The VM only applies:
    - key-based seeding
    - operator-bank transforms
    - edge-weighted propagation
    - fixed readout
    """

    def __init__(
        self,
        num_entities: int,
        num_attributes: int,
        entity_offset: int,
        attribute_offset: int,
        num_values: int,
        propagation_steps: int = 3,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_attributes = num_attributes
        self.entity_offset = entity_offset
        self.attribute_offset = attribute_offset
        self.num_values = num_values
        self.propagation_steps = propagation_steps

    def query_key_ids(self, query_tokens: torch.Tensor) -> torch.Tensor:
        entity_ids = query_tokens[:, 1] - self.entity_offset
        attribute_ids = query_tokens[:, 2] - self.attribute_offset
        return entity_ids * self.num_attributes + attribute_ids

    def apply_node_ops(self, value_probs: torch.Tensor, operator_logits: torch.Tensor) -> torch.Tensor:
        operator_probs = F.softmax(operator_logits, dim=-1)
        bank = _apply_operator_bank(value_probs)
        return (operator_probs.unsqueeze(-1) * bank).sum(dim=-2)

    def apply_edge_ops(self, value_probs: torch.Tensor, operator_logits: torch.Tensor) -> torch.Tensor:
        operator_probs = F.softmax(operator_logits, dim=-1)
        bank = _apply_operator_bank(value_probs)
        return (operator_probs.unsqueeze(-1) * bank).sum(dim=-2)

    def forward(self, state: ProgramGraphState, query_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        query_key_ids = self.query_key_ids(query_tokens)
        query_key_one_hot = F.one_hot(query_key_ids, num_classes=state.node_key_logits.size(-1)).float()

        node_key_probs = F.softmax(state.node_key_logits, dim=-1)
        node_value_probs = F.softmax(state.node_value_logits, dim=-1)
        node_programmed_values = self.apply_node_ops(node_value_probs, state.node_operator_logits)

        seed_strength = torch.matmul(node_key_probs, query_key_one_hot.unsqueeze(-1)).squeeze(-1)
        seed_strength = seed_strength * state.node_active.float()
        activations = _normalize_distribution(seed_strength + 1e-8)
        activations = activations * state.node_active.float()
        activations = _normalize_distribution(activations + 1e-8)
        dynamic_values = node_programmed_values * state.node_active.unsqueeze(-1).float()

        for _ in range(self.propagation_steps):
            per_source_edge_values = self.apply_edge_ops(
                dynamic_values.unsqueeze(2).expand(-1, -1, state.edge_weights.size(2), -1),
                state.edge_operator_logits,
            )
            weighted_messages = (
                activations.unsqueeze(-1).unsqueeze(-1)
                * state.edge_weights.unsqueeze(-1)
                * per_source_edge_values
            )
            incoming_values = weighted_messages.sum(dim=1)
            propagated_activations = (activations.unsqueeze(-1) * state.edge_weights).sum(dim=1)
            activations = _normalize_distribution(seed_strength + propagated_activations + 1e-8)
            activations = activations * state.node_active.float()
            activations = _normalize_distribution(activations + 1e-8)
            dynamic_values = _normalize_distribution(node_programmed_values + incoming_values)
            dynamic_values = dynamic_values * state.node_active.unsqueeze(-1).float()
            dynamic_values = _normalize_distribution(dynamic_values + 1e-8) * state.node_active.unsqueeze(-1).float()

        answer_logits = torch.sum(activations.unsqueeze(-1) * dynamic_values, dim=1)
        return {
            "answer_logits": answer_logits.clamp_min(1e-8).log(),
            "node_activations": activations,
            "dynamic_values": dynamic_values,
            "query_key_ids": query_key_ids,
        }


class SelfContainedProgramGraphBrain(nn.Module):
    """
    Generator-updated graph program plus a fixed VM.

    The generator learns to update the graph from new inputs.
    The VM is parameter-free and simply executes the graph on a query.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_nodes: int,
        num_keys: int,
        num_values: int,
        num_operators: int,
        num_entities: int,
        num_attributes: int,
        entity_offset: int,
        attribute_offset: int,
        propagation_steps: int = 3,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_keys = num_keys
        self.num_values = num_values
        self.num_operators = num_operators
        self.generator = ProgramGraphGenerator(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_keys=num_keys,
            num_values=num_values,
            num_operators=num_operators,
        )
        self.vm = FixedProgramGraphVM(
            num_entities=num_entities,
            num_attributes=num_attributes,
            entity_offset=entity_offset,
            attribute_offset=attribute_offset,
            num_values=num_values,
            propagation_steps=propagation_steps,
        )

    def initial_state(self, batch_size: int, device: torch.device | str) -> ProgramGraphState:
        return ProgramGraphState.zeros(
            batch_size=batch_size,
            num_nodes=self.num_nodes,
            num_keys=self.num_keys,
            num_values=self.num_values,
            num_operators=self.num_operators,
            device=device,
        )

    def forward(self, read_tokens: torch.Tensor, query_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = read_tokens.size(0)
        state = self.initial_state(batch_size=batch_size, device=read_tokens.device)

        target_logits_history = []
        source_logits_history = []
        key_logits_history = []
        value_logits_history = []
        node_operator_logits_history = []
        edge_operator_logits_history = []
        selected_targets_history = []
        selected_sources_history = []
        edge_gate_history = []

        for step in range(read_tokens.size(1)):
            state, aux = self.generator.step(state, read_tokens[:, step])
            target_logits_history.append(aux["target_logits"])
            source_logits_history.append(aux["source_logits"])
            key_logits_history.append(aux["key_logits"])
            value_logits_history.append(aux["value_logits"])
            node_operator_logits_history.append(aux["node_operator_logits"])
            edge_operator_logits_history.append(aux["edge_operator_logits"])
            selected_targets_history.append(aux["selected_targets"])
            selected_sources_history.append(aux["selected_sources"])
            edge_gate_history.append(aux["edge_gates"])

        vm_outputs = self.vm(state, query_tokens)
        return {
            "final_node_key_logits": state.node_key_logits,
            "final_node_value_logits": state.node_value_logits,
            "final_node_operator_logits": state.node_operator_logits,
            "final_edge_weights": state.edge_weights,
            "final_edge_operator_logits": state.edge_operator_logits,
            "final_node_active": state.node_active,
            "step_target_logits": torch.stack(target_logits_history, dim=1),
            "step_source_logits": torch.stack(source_logits_history, dim=1),
            "step_key_logits": torch.stack(key_logits_history, dim=1),
            "step_value_logits": torch.stack(value_logits_history, dim=1),
            "step_node_operator_logits": torch.stack(node_operator_logits_history, dim=1),
            "step_edge_operator_logits": torch.stack(edge_operator_logits_history, dim=1),
            "step_selected_targets": torch.stack(selected_targets_history, dim=1),
            "step_selected_sources": torch.stack(selected_sources_history, dim=1),
            "step_edge_gates": torch.stack(edge_gate_history, dim=1),
            **vm_outputs,
        }
