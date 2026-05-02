from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GraphBrainState:
    node_states: torch.Tensor
    edge_weights: torch.Tensor
    edge_operator_weights: torch.Tensor
    node_usage: torch.Tensor
    last_node_weights: torch.Tensor

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        num_nodes: int,
        node_dim: int,
        num_operators: int,
        device: torch.device | str,
    ) -> "GraphBrainState":
        return cls(
            node_states=torch.zeros(batch_size, num_nodes, node_dim, device=device),
            edge_weights=torch.zeros(batch_size, num_nodes, num_nodes, device=device),
            edge_operator_weights=torch.zeros(batch_size, num_nodes, num_nodes, num_operators, device=device),
            node_usage=torch.zeros(batch_size, num_nodes, device=device),
            last_node_weights=torch.zeros(batch_size, num_nodes, device=device),
        )


class GraphBrainUpdater(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, node_dim: int, num_nodes: int, num_operators: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_operators = num_operators
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.target_node_router = nn.Linear(hidden_dim, num_nodes)
        self.operator_selector = nn.Linear(hidden_dim, num_operators)
        self.write_gate_projection = nn.Linear(hidden_dim, 1)
        self.edge_gate_projection = nn.Linear(hidden_dim, 1)
        self.operator_modules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim + 2 * node_dim, node_dim),
                    nn.Tanh(),
                )
                for _ in range(num_operators)
            ]
        )

    def forward(self, brain_state: GraphBrainState, read_tokens: torch.Tensor) -> tuple[GraphBrainState, dict[str, torch.Tensor]]:
        token_embeddings = self.token_embedding(read_tokens)
        _, hidden = self.encoder(token_embeddings)
        step_repr = hidden[-1]

        target_node_logits = self.target_node_router(step_repr)
        target_node_weights = F.softmax(target_node_logits, dim=-1)
        source_node_weights = brain_state.last_node_weights

        source_context = torch.bmm(source_node_weights.unsqueeze(1), brain_state.node_states).squeeze(1)
        target_context = torch.bmm(target_node_weights.unsqueeze(1), brain_state.node_states).squeeze(1)
        operator_input = torch.cat([step_repr, source_context, target_context], dim=-1)

        operator_logits = self.operator_selector(step_repr)
        operator_weights = F.softmax(operator_logits, dim=-1)
        operator_candidates = torch.stack([module(operator_input) for module in self.operator_modules], dim=1)
        mixed_candidate = torch.sum(operator_weights.unsqueeze(-1) * operator_candidates, dim=1)

        write_gate = torch.sigmoid(self.write_gate_projection(step_repr))
        write_strength = target_node_weights.unsqueeze(-1) * write_gate.unsqueeze(-1)
        updated_nodes = brain_state.node_states * (1.0 - write_strength) + write_strength * mixed_candidate.unsqueeze(1)

        edge_gate = torch.sigmoid(self.edge_gate_projection(step_repr)).squeeze(-1)
        pair_update = (
            source_node_weights.unsqueeze(2) * target_node_weights.unsqueeze(1) * edge_gate.view(-1, 1, 1)
        )
        updated_edge_weights = brain_state.edge_weights * (1.0 - pair_update) + pair_update
        operator_pair_update = pair_update.unsqueeze(-1) * operator_weights.unsqueeze(1).unsqueeze(1)
        updated_edge_operator_weights = (
            brain_state.edge_operator_weights * (1.0 - pair_update.unsqueeze(-1)) + operator_pair_update
        )

        next_state = GraphBrainState(
            node_states=updated_nodes,
            edge_weights=updated_edge_weights,
            edge_operator_weights=updated_edge_operator_weights,
            node_usage=brain_state.node_usage + target_node_weights,
            last_node_weights=target_node_weights,
        )
        aux = {
            "step_repr": step_repr,
            "target_node_logits": target_node_logits,
            "target_node_weights": target_node_weights,
            "operator_logits": operator_logits,
            "operator_weights": operator_weights,
            "write_vector": mixed_candidate,
        }
        return next_state, aux


class GraphBrainExecutor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        node_dim: int,
        num_nodes: int,
        num_operators: int,
        message_passing_steps: int,
        num_values: int,
    ):
        super().__init__()
        self.message_passing_steps = message_passing_steps
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.query_node_classifier = nn.Linear(hidden_dim, num_nodes)
        self.message_operator_modules = nn.ModuleList([nn.Linear(node_dim, node_dim) for _ in range(num_operators)])
        self.propagation_norm = nn.LayerNorm(node_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim + node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_values),
        )

    def forward(self, brain_state: GraphBrainState, query_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        token_embeddings = self.token_embedding(query_tokens)
        _, hidden = self.encoder(token_embeddings)
        query_repr = hidden[-1]
        query_node_logits = self.query_node_classifier(query_repr)
        propagated_nodes = brain_state.node_states

        for _ in range(self.message_passing_steps):
            transformed_sources = torch.stack(
                [torch.tanh(module(propagated_nodes)) for module in self.message_operator_modules],
                dim=2,
            )
            edge_operator_denominator = brain_state.edge_operator_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            edge_operator_probs = brain_state.edge_operator_weights / edge_operator_denominator
            messages = (
                edge_operator_probs.unsqueeze(-1) * transformed_sources.unsqueeze(2)
            ).sum(dim=3)
            messages = messages * brain_state.edge_weights.unsqueeze(-1)
            incoming = messages.sum(dim=1)
            propagated_nodes = self.propagation_norm(propagated_nodes + incoming)

        query_node_weights = F.softmax(query_node_logits, dim=-1)
        context = torch.sum(query_node_weights.unsqueeze(-1) * propagated_nodes, dim=1)
        answer_logits = self.output(torch.cat([query_repr, context], dim=-1))
        return {
            "answer_logits": answer_logits,
            "query_node_logits": query_node_logits,
            "query_node_weights": query_node_weights,
            "propagated_nodes": propagated_nodes,
        }


class GraphStructuredBrain(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        node_dim: int,
        num_nodes: int,
        num_operators: int,
        message_passing_steps: int,
        num_values: int,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.num_operators = num_operators
        self.updater = GraphBrainUpdater(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            node_dim=node_dim,
            num_nodes=num_nodes,
            num_operators=num_operators,
        )
        self.read_value_head = nn.Linear(node_dim, num_values)
        self.executor = GraphBrainExecutor(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            node_dim=node_dim,
            num_nodes=num_nodes,
            num_operators=num_operators,
            message_passing_steps=message_passing_steps,
            num_values=num_values,
        )

    def initial_state(self, batch_size: int, device: torch.device | str) -> GraphBrainState:
        return GraphBrainState.zeros(
            batch_size=batch_size,
            num_nodes=self.num_nodes,
            node_dim=self.node_dim,
            num_operators=self.num_operators,
            device=device,
        )

    def forward(self, read_tokens: torch.Tensor, query_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = read_tokens.size(0)
        brain_state = self.initial_state(batch_size=batch_size, device=read_tokens.device)

        step_node_logits = []
        step_node_weights = []
        step_operator_logits = []
        step_operator_weights = []
        step_write_vectors = []

        for step in range(read_tokens.size(1)):
            brain_state, aux = self.updater(brain_state, read_tokens[:, step])
            step_node_logits.append(aux["target_node_logits"])
            step_node_weights.append(aux["target_node_weights"])
            step_operator_logits.append(aux["operator_logits"])
            step_operator_weights.append(aux["operator_weights"])
            step_write_vectors.append(aux["write_vector"])

        executor_outputs = self.executor(brain_state, query_tokens)
        return {
            "final_node_states": brain_state.node_states,
            "final_edge_weights": brain_state.edge_weights,
            "final_edge_operator_weights": brain_state.edge_operator_weights,
            "step_node_logits": torch.stack(step_node_logits, dim=1),
            "step_node_weights": torch.stack(step_node_weights, dim=1),
            "step_operator_logits": torch.stack(step_operator_logits, dim=1),
            "step_operator_weights": torch.stack(step_operator_weights, dim=1),
            "step_write_vectors": torch.stack(step_write_vectors, dim=1),
            "step_value_logits": self.read_value_head(torch.stack(step_write_vectors, dim=1)),
            **executor_outputs,
        }
