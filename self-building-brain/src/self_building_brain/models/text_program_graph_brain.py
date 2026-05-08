from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_normalize(vectors: torch.Tensor) -> torch.Tensor:
    return F.normalize(vectors, dim=-1, eps=1e-8)


def _continuous_operator_bank(vectors: torch.Tensor) -> torch.Tensor:
    centered = vectors - vectors.mean(dim=-1, keepdim=True)
    signed_square = torch.sign(vectors) * vectors.pow(2)
    return torch.stack(
        [
            vectors,
            torch.tanh(vectors),
            F.relu(vectors),
            _safe_normalize(vectors),
            _safe_normalize(centered),
            _safe_normalize(signed_square),
        ],
        dim=-2,
    )


def _topk_softmax(logits: torch.Tensor, temperature: float, k: int) -> torch.Tensor:
    if k >= logits.size(-1):
        return F.softmax(logits / temperature, dim=-1)
    topk_values, topk_indices = torch.topk(logits, k=k, dim=-1)
    sparse_logits = torch.full_like(logits, fill_value=-1e4)
    sparse_logits.scatter_(dim=-1, index=topk_indices, src=topk_values)
    return F.softmax(sparse_logits / temperature, dim=-1)


@dataclass
class TextProgramGraphState:
    node_key_vectors: torch.Tensor
    node_value_vectors: torch.Tensor
    node_operator_logits: torch.Tensor
    edge_weights: torch.Tensor
    edge_operator_logits: torch.Tensor
    node_active_mass: torch.Tensor
    last_target_nodes: torch.Tensor

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        num_nodes: int,
        input_dim: int,
        num_operators: int,
        device: torch.device | str,
    ) -> "TextProgramGraphState":
        return cls(
            node_key_vectors=torch.zeros(batch_size, num_nodes, input_dim, device=device),
            node_value_vectors=torch.zeros(batch_size, num_nodes, input_dim, device=device),
            node_operator_logits=torch.zeros(batch_size, num_nodes, num_operators, device=device),
            edge_weights=torch.zeros(batch_size, num_nodes, num_nodes, device=device),
            edge_operator_logits=torch.zeros(batch_size, num_nodes, num_nodes, num_operators, device=device),
            node_active_mass=torch.zeros(batch_size, num_nodes, device=device),
            last_target_nodes=torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device),
        )


class TextProgramGraphGenerator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_nodes: int,
        num_operators: int,
        routing_temperature: float = 0.35,
        routing_topk: int = 2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_operators = num_operators
        self.routing_temperature = routing_temperature
        self.routing_topk = routing_topk
        self.summary_projection = nn.Sequential(
            nn.Linear(input_dim * 2 + num_operators, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.target_router = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes),
        )
        self.source_router = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes),
        )
        self.key_head = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, input_dim),
            nn.Tanh(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, input_dim),
            nn.Tanh(),
        )
        self.node_operator_head = nn.Linear(input_dim + hidden_dim, num_operators)
        self.edge_operator_head = nn.Linear(input_dim + hidden_dim, num_operators)
        self.node_blend_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def summarize_graph(self, state: TextProgramGraphState) -> torch.Tensor:
        active = state.node_active_mass.unsqueeze(-1)
        denom = active.sum(dim=1).clamp_min(1.0)
        key_mean = (state.node_key_vectors * active).sum(dim=1) / denom
        value_mean = (state.node_value_vectors * active).sum(dim=1) / denom
        operator_mean = (F.softmax(state.node_operator_logits, dim=-1) * active).sum(dim=1) / denom
        return self.summary_projection(torch.cat([key_mean, value_mean, operator_mean], dim=-1))

    def forward_step(
        self,
        state: TextProgramGraphState,
        chunk_embeddings: torch.Tensor,
    ) -> tuple[TextProgramGraphState, dict[str, torch.Tensor]]:
        graph_summary = self.summarize_graph(state)
        fused = torch.cat([chunk_embeddings, graph_summary], dim=-1)

        target_logits = self.target_router(fused)
        source_logits = self.source_router(fused)
        target_probs = _topk_softmax(target_logits, temperature=self.routing_temperature, k=self.routing_topk)
        raw_source_probs = _topk_softmax(source_logits, temperature=self.routing_temperature, k=self.routing_topk)
        active_mass = state.node_active_mass
        has_active = active_mass.sum(dim=-1, keepdim=True) > 1e-6
        masked_source = raw_source_probs * active_mass
        normalized_source = masked_source / masked_source.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        source_probs = torch.where(has_active, normalized_source, target_probs)

        candidate_keys = _safe_normalize(self.key_head(fused))
        candidate_values = _safe_normalize(self.value_head(fused))
        candidate_node_operators = self.node_operator_head(fused)
        candidate_edge_operators = self.edge_operator_head(fused)
        node_blend = torch.sigmoid(self.node_blend_gate(fused)).squeeze(-1)
        edge_gate = torch.sigmoid(self.edge_gate(fused)).squeeze(-1)

        batch_size, input_dim = candidate_keys.size(0), candidate_keys.size(1)
        target_mass = target_probs.unsqueeze(-1)
        source_mass = source_probs.unsqueeze(-1)
        update_gate = (1.0 - node_blend).view(batch_size, 1, 1)

        key_updates = candidate_keys.view(batch_size, 1, input_dim)
        value_updates = candidate_values.view(batch_size, 1, input_dim)
        operator_updates = candidate_node_operators.view(batch_size, 1, self.num_operators)

        next_node_key_vectors = state.node_key_vectors + target_mass * update_gate * (key_updates - state.node_key_vectors)
        next_node_value_vectors = state.node_value_vectors + target_mass * update_gate * (value_updates - state.node_value_vectors)
        next_node_operator_logits = state.node_operator_logits + target_probs.unsqueeze(-1) * update_gate * (
            operator_updates - state.node_operator_logits
        )

        edge_mass = source_probs.unsqueeze(2) * target_probs.unsqueeze(1)
        edge_operator_updates = candidate_edge_operators.view(batch_size, 1, 1, self.num_operators)
        next_edge_operator_logits = state.edge_operator_logits + edge_mass.unsqueeze(-1) * (
            edge_operator_updates - state.edge_operator_logits
        )
        next_edge_weights = state.edge_weights + edge_gate.view(batch_size, 1, 1) * edge_mass * (1.0 - state.edge_weights)
        next_node_active_mass = 1.0 - (1.0 - state.node_active_mass) * (1.0 - target_probs)

        next_state = TextProgramGraphState(
            node_key_vectors=next_node_key_vectors,
            node_value_vectors=next_node_value_vectors,
            node_operator_logits=next_node_operator_logits,
            edge_weights=next_edge_weights,
            edge_operator_logits=next_edge_operator_logits,
            node_active_mass=next_node_active_mass,
            last_target_nodes=target_probs.argmax(dim=-1),
        )
        return next_state, {
            "target_logits": target_logits,
            "source_logits": source_logits,
            "target_probs": target_probs,
            "source_probs": source_probs,
            "candidate_keys": candidate_keys,
            "candidate_values": candidate_values,
            "candidate_node_operators": candidate_node_operators,
            "candidate_edge_operators": candidate_edge_operators,
            "selected_targets": target_probs.argmax(dim=-1),
            "selected_sources": source_probs.argmax(dim=-1),
            "edge_gates": edge_gate,
        }


class FixedTextProgramGraphVM(nn.Module):
    def __init__(
        self,
        num_operators: int,
        propagation_steps: int = 3,
        choice_temperature: float = 12.0,
        seed_weight: float = 0.6,
        message_activation_weight: float = 2.0,
        message_value_weight: float = 2.0,
    ):
        super().__init__()
        self.num_operators = num_operators
        self.propagation_steps = propagation_steps
        self.choice_temperature = choice_temperature
        self.seed_weight = seed_weight
        self.message_activation_weight = message_activation_weight
        self.message_value_weight = message_value_weight

    def apply_ops(self, vectors: torch.Tensor, operator_logits: torch.Tensor) -> torch.Tensor:
        operator_probs = F.softmax(operator_logits, dim=-1)
        bank = _continuous_operator_bank(vectors)
        return (operator_probs.unsqueeze(-1) * bank).sum(dim=-2)

    def forward(
        self,
        state: TextProgramGraphState,
        query_embeddings: torch.Tensor,
        choice_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_keys = _safe_normalize(state.node_key_vectors)
        query = _safe_normalize(query_embeddings)
        seed_scores = torch.matmul(node_keys, query.unsqueeze(-1)).squeeze(-1)
        active_mass = state.node_active_mass
        safe_scores = seed_scores + torch.log(active_mass.clamp_min(1e-8))
        activations = F.softmax(safe_scores, dim=-1) * active_mass
        activations = activations / activations.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        node_values = self.apply_ops(state.node_value_vectors, state.node_operator_logits)
        node_values = _safe_normalize(node_values) * active_mass.unsqueeze(-1)
        dynamic_values = node_values

        for _ in range(self.propagation_steps):
            per_edge_values = self.apply_ops(
                dynamic_values.unsqueeze(2).expand(-1, -1, state.edge_weights.size(2), -1),
                state.edge_operator_logits,
            )
            weighted_messages = (
                activations.unsqueeze(-1).unsqueeze(-1)
                * state.edge_weights.unsqueeze(-1)
                * per_edge_values
            )
            incoming_values = weighted_messages.sum(dim=1)
            propagated_activations = (activations.unsqueeze(-1) * state.edge_weights).sum(dim=1)
            activations = F.softmax(
                self.seed_weight * seed_scores
                + self.message_activation_weight * propagated_activations
                + torch.log(active_mass.clamp_min(1e-8)),
                dim=-1,
            ) * active_mass
            activations = activations / activations.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            dynamic_values = _safe_normalize(node_values + self.message_value_weight * incoming_values)
            dynamic_values = dynamic_values * active_mass.unsqueeze(-1)

        context = torch.sum(activations.unsqueeze(-1) * dynamic_values, dim=1)
        context = _safe_normalize(context)
        normalized_choices = _safe_normalize(choice_embeddings)
        choice_logits = self.choice_temperature * torch.matmul(normalized_choices, context.unsqueeze(-1)).squeeze(-1)
        return {
            "choice_logits": choice_logits,
            "node_activations": activations,
            "dynamic_values": dynamic_values,
            "context": context,
        }


class SelfContainedTextProgramGraphBrain(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_nodes: int,
        num_operators: int = 6,
        propagation_steps: int = 3,
        choice_temperature: float = 12.0,
        routing_temperature: float = 0.35,
        routing_topk: int = 2,
        seed_weight: float = 0.6,
        message_activation_weight: float = 2.0,
        message_value_weight: float = 2.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_operators = num_operators
        self.generator = TextProgramGraphGenerator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_nodes=num_nodes,
            num_operators=num_operators,
            routing_temperature=routing_temperature,
            routing_topk=routing_topk,
        )
        self.vm = FixedTextProgramGraphVM(
            num_operators=num_operators,
            propagation_steps=propagation_steps,
            choice_temperature=choice_temperature,
            seed_weight=seed_weight,
            message_activation_weight=message_activation_weight,
            message_value_weight=message_value_weight,
        )

    def initial_state(self, batch_size: int, device: torch.device | str) -> TextProgramGraphState:
        return TextProgramGraphState.zeros(
            batch_size=batch_size,
            num_nodes=self.num_nodes,
            input_dim=self.input_dim,
            num_operators=self.num_operators,
            device=device,
        )

    def forward(
        self,
        chunk_embeddings: torch.Tensor,
        chunk_mask: torch.Tensor,
        query_embeddings: torch.Tensor,
        choice_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_chunks, _ = chunk_embeddings.shape
        state = self.initial_state(batch_size=batch_size, device=chunk_embeddings.device)

        target_logits_history = []
        source_logits_history = []
        target_prob_history = []
        source_prob_history = []
        candidate_key_history = []
        candidate_value_history = []
        node_operator_history = []
        edge_operator_history = []
        selected_target_history = []
        selected_source_history = []
        edge_gate_history = []

        for chunk_index in range(num_chunks):
            valid_mask = chunk_mask[:, chunk_index] > 0
            next_state, aux = self.generator.forward_step(state=state, chunk_embeddings=chunk_embeddings[:, chunk_index])

            valid_nodes = valid_mask.view(batch_size, 1, 1).float()
            valid_edges = valid_mask.view(batch_size, 1, 1, 1).float()
            valid_mass = valid_mask.view(batch_size, 1).float()
            state = TextProgramGraphState(
                node_key_vectors=valid_nodes * next_state.node_key_vectors + (1.0 - valid_nodes) * state.node_key_vectors,
                node_value_vectors=valid_nodes * next_state.node_value_vectors + (1.0 - valid_nodes) * state.node_value_vectors,
                node_operator_logits=valid_nodes * next_state.node_operator_logits + (1.0 - valid_nodes) * state.node_operator_logits,
                edge_weights=valid_nodes * next_state.edge_weights + (1.0 - valid_nodes) * state.edge_weights,
                edge_operator_logits=valid_edges * next_state.edge_operator_logits + (1.0 - valid_edges) * state.edge_operator_logits,
                node_active_mass=valid_mass * next_state.node_active_mass + (1.0 - valid_mass) * state.node_active_mass,
                last_target_nodes=torch.where(valid_mask, next_state.last_target_nodes, state.last_target_nodes),
            )

            target_logits_history.append(aux["target_logits"])
            source_logits_history.append(aux["source_logits"])
            target_prob_history.append(aux["target_probs"])
            source_prob_history.append(aux["source_probs"])
            candidate_key_history.append(aux["candidate_keys"])
            candidate_value_history.append(aux["candidate_values"])
            node_operator_history.append(aux["candidate_node_operators"])
            edge_operator_history.append(aux["candidate_edge_operators"])
            selected_target_history.append(aux["selected_targets"])
            selected_source_history.append(aux["selected_sources"])
            edge_gate_history.append(aux["edge_gates"])

        vm_outputs = self.vm(state=state, query_embeddings=query_embeddings, choice_embeddings=choice_embeddings)
        return {
            "final_node_key_vectors": state.node_key_vectors,
            "final_node_value_vectors": state.node_value_vectors,
            "final_node_operator_logits": state.node_operator_logits,
            "final_edge_weights": state.edge_weights,
            "final_edge_operator_logits": state.edge_operator_logits,
            "final_node_active_mass": state.node_active_mass,
            "final_node_active": state.node_active_mass >= 0.1,
            "step_target_logits": torch.stack(target_logits_history, dim=1),
            "step_source_logits": torch.stack(source_logits_history, dim=1),
            "step_target_probs": torch.stack(target_prob_history, dim=1),
            "step_source_probs": torch.stack(source_prob_history, dim=1),
            "step_candidate_keys": torch.stack(candidate_key_history, dim=1),
            "step_candidate_values": torch.stack(candidate_value_history, dim=1),
            "step_candidate_node_operators": torch.stack(node_operator_history, dim=1),
            "step_candidate_edge_operators": torch.stack(edge_operator_history, dim=1),
            "step_selected_targets": torch.stack(selected_target_history, dim=1),
            "step_selected_sources": torch.stack(selected_source_history, dim=1),
            "step_edge_gates": torch.stack(edge_gate_history, dim=1),
            **vm_outputs,
        }
