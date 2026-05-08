from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical

from catdog.vocab import VOCAB_SIZE


def one_hot(token_id: int, device: torch.device | str) -> torch.Tensor:
    vector = torch.zeros(VOCAB_SIZE, dtype=torch.float32, device=device)
    vector[token_id] = 1.0
    return vector


def safe_normalize(vector: torch.Tensor) -> torch.Tensor:
    norm = vector.norm(p=2)
    if float(norm.item()) < 1e-8:
        return vector
    return vector / norm


@dataclass
class GraphState:
    node_states: torch.Tensor
    edge_weights: torch.Tensor
    pointer: int


class PolicyGenerator(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.target_scorer = nn.Sequential(
            nn.Linear(VOCAB_SIZE * 4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, token_vector: torch.Tensor, graph: GraphState) -> torch.Tensor:
        graph_mean = graph.node_states.mean(dim=0)
        pointer_state = graph.node_states[graph.pointer]

        target_logits = []
        for node_index in range(self.num_nodes):
            target_input = torch.cat([token_vector, graph_mean, pointer_state, graph.node_states[node_index]], dim=0)
            target_logits.append(self.target_scorer(target_input).squeeze(0))
        target_logits = torch.stack(target_logits, dim=0)
        return target_logits

    def sample_action(self, token_vector: torch.Tensor, graph: GraphState) -> dict[str, torch.Tensor | int]:
        target_logits = self.forward(token_vector, graph)
        target_dist = Categorical(logits=target_logits)
        target = target_dist.sample()
        log_prob = target_dist.log_prob(target)
        entropy = target_dist.entropy()
        return {
            "target": int(target.item()),
            "log_prob": log_prob,
            "entropy": entropy,
            "target_logits": target_logits,
        }


class FixedGraphExecutor:
    def __init__(self, propagation_steps: int = 2, seed_scale: float = 6.0, message_scale: float = 1.0):
        self.propagation_steps = propagation_steps
        self.seed_scale = seed_scale
        self.message_scale = message_scale

    def execute(self, graph: GraphState, query_vector: torch.Tensor) -> torch.Tensor:
        seed_scores = self.seed_scale * torch.mv(graph.node_states, query_vector)
        activations = torch.softmax(seed_scores, dim=0)
        edge_probs = graph.edge_weights / graph.edge_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        context = torch.zeros(VOCAB_SIZE, dtype=torch.float32, device=query_vector.device)
        current = activations
        for _ in range(self.propagation_steps + 1):
            context = context + torch.mv(graph.node_states.T, current)
            current = torch.mv(edge_probs.T, current)
            current = current / current.sum().clamp_min(1e-6)
        logits = self.message_scale * context
        return logits


class GraphEditEnvironment:
    def __init__(self, num_nodes: int, device: torch.device | str):
        self.num_nodes = num_nodes
        self.device = device

    def initial_graph(self) -> GraphState:
        node_states = torch.zeros(self.num_nodes, VOCAB_SIZE, dtype=torch.float32, device=self.device)
        edge_weights = torch.eye(self.num_nodes, dtype=torch.float32, device=self.device)
        for index in range(self.num_nodes):
            edge_weights[index, (index + 1) % self.num_nodes] += 0.1
        return GraphState(node_states=node_states, edge_weights=edge_weights, pointer=0)

    def apply_action(self, graph: GraphState, token_id: int, target: int) -> GraphState:
        token_vector = one_hot(token_id, graph.node_states.device)
        next_nodes = graph.node_states.clone()
        next_edges = graph.edge_weights.clone()
        next_nodes[target] = safe_normalize(next_nodes[target] + token_vector)
        next_edges[graph.pointer, target] += 1.0

        return GraphState(node_states=next_nodes, edge_weights=next_edges, pointer=target)
