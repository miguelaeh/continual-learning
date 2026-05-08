from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from catdog.vocab import VOCAB, VOCAB_SIZE


DEFAULT_STATE_DIM = 16
NUM_NODE_OPS = 4
NUM_ACTION_TYPES = 4
INPUT_NODE_INDEX = 0
OUTPUT_NODE_INDEX = 1
WORK_NODE_INDEX = 2

ACTION_CREATE_NODE = 0
ACTION_UPDATE_NODE = 1
ACTION_CONNECT = 2
ACTION_SET_NODE_OP = 3


def build_token_embeddings(state_dim: int, device: torch.device | str) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    embeddings = torch.randn(VOCAB_SIZE, state_dim, generator=generator, dtype=torch.float32)
    embeddings = F.normalize(embeddings, dim=-1)
    return embeddings.to(device)


def token_embedding(token_id: int, token_embeddings: torch.Tensor) -> torch.Tensor:
    return token_embeddings[token_id]


def project_to_vocab(state: torch.Tensor, token_embeddings: torch.Tensor) -> torch.Tensor:
    return torch.matmul(token_embeddings, state)


def safe_normalize(vector: torch.Tensor) -> torch.Tensor:
    norm = vector.norm(p=2)
    if float(norm.item()) < 1e-8:
        return vector
    return vector / norm


def apply_node_op(vector: torch.Tensor, op_id: int) -> torch.Tensor:
    if op_id == 0:
        return vector
    if op_id == 1:
        return torch.tanh(vector)
    if op_id == 2:
        return F.relu(vector)
    centered = vector - vector.mean()
    return safe_normalize(centered)


@dataclass
class CompGraphState:
    node_states: torch.Tensor
    node_ops: torch.Tensor
    edge_weights: torch.Tensor
    active_mask: torch.Tensor
    pointer: int


class ComputationalPolicy(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int, state_dim: int = DEFAULT_STATE_DIM):
        super().__init__()
        self.num_nodes = num_nodes
        self.state_dim = state_dim
        self.action_head = nn.Sequential(
            nn.Linear(state_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, NUM_ACTION_TYPES),
        )
        self.source_scorer = nn.Sequential(
            nn.Linear(state_dim * 4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.target_scorer = nn.Sequential(
            nn.Linear(state_dim * 4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.op_head = nn.Sequential(
            nn.Linear(state_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, NUM_NODE_OPS),
        )
        self.write_mean_head = nn.Sequential(
            nn.Linear(state_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.write_log_std = nn.Parameter(torch.full((state_dim,), -0.5))

    def forward(
        self, token_vector: torch.Tensor, graph: CompGraphState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        graph_mean = graph.node_states.mean(dim=0)
        pointer_state = graph.node_states[graph.pointer]
        global_features = torch.cat([token_vector, graph_mean, pointer_state], dim=0)

        source_logits = []
        target_logits = []
        for node_index in range(self.num_nodes):
            source_input = torch.cat([token_vector, graph_mean, pointer_state, graph.node_states[node_index]], dim=0)
            target_input = torch.cat([token_vector, graph_mean, pointer_state, graph.node_states[node_index]], dim=0)
            source_logits.append(self.source_scorer(source_input).squeeze(0))
            target_logits.append(self.target_scorer(target_input).squeeze(0))
        source_logits = torch.stack(source_logits, dim=0)
        target_logits = torch.stack(target_logits, dim=0)
        op_logits = self.op_head(global_features)
        action_logits = self.action_head(global_features)
        write_mean = self.write_mean_head(global_features)
        return action_logits, source_logits, target_logits, op_logits, write_mean

    def sample_action(self, token_vector: torch.Tensor, graph: CompGraphState) -> dict[str, torch.Tensor | int]:
        action_logits, source_logits, target_logits, op_logits, write_mean = self.forward(token_vector, graph)
        action_dist = Categorical(logits=action_logits)
        source_dist = Categorical(logits=source_logits)
        target_dist = Categorical(logits=target_logits)
        op_dist = Categorical(logits=op_logits)
        write_std = self.write_log_std.exp().clamp_min(1e-4)
        write_dist = Normal(write_mean, write_std)
        action_type = action_dist.sample()
        source = source_dist.sample()
        target = target_dist.sample()
        op = op_dist.sample()
        raw_write_vector = write_dist.rsample()
        write_vector = torch.tanh(raw_write_vector)
        log_prob = (
            action_dist.log_prob(action_type)
            + source_dist.log_prob(source)
            + target_dist.log_prob(target)
            + op_dist.log_prob(op)
            + write_dist.log_prob(raw_write_vector).sum()
        )
        entropy = (
            action_dist.entropy()
            + source_dist.entropy()
            + target_dist.entropy()
            + op_dist.entropy()
            + write_dist.entropy().sum()
        )
        return {
            "action_type": int(action_type.item()),
            "source": int(source.item()),
            "target": int(target.item()),
            "op": int(op.item()),
            "write_vector": write_vector,
            "log_prob": log_prob,
            "entropy": entropy,
        }


class FixedComputationalExecutor:
    def __init__(self, propagation_steps: int = 3, residual_weight: float = 0.2):
        self.propagation_steps = propagation_steps
        self.residual_weight = residual_weight

    def execute(self, graph: CompGraphState, query_vector: torch.Tensor) -> torch.Tensor:
        working_states = graph.node_states.clone()
        working_states[INPUT_NODE_INDEX] = query_vector
        active_mask = graph.active_mask.clone()
        active_mask[INPUT_NODE_INDEX] = 1.0
        active_mask[OUTPUT_NODE_INDEX] = 1.0
        edge_probs = graph.edge_weights / graph.edge_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        for _ in range(self.propagation_steps):
            transformed = torch.stack(
                [apply_node_op(working_states[idx], int(graph.node_ops[idx].item())) for idx in range(graph.node_states.size(0))],
                dim=0,
            )
            transformed = transformed * active_mask.unsqueeze(-1)
            incoming = torch.matmul(edge_probs.transpose(0, 1), transformed)
            next_states = self.residual_weight * working_states + (1.0 - self.residual_weight) * incoming
            # Keep magnitude information. Per-node normalization was turning tiny
            # leaked copies of the input into full-strength output echoes.
            next_states = torch.tanh(next_states)
            next_states = next_states * active_mask.unsqueeze(-1)
            next_states[INPUT_NODE_INDEX] = query_vector
            working_states = next_states
        output_state = apply_node_op(working_states[OUTPUT_NODE_INDEX], int(graph.node_ops[OUTPUT_NODE_INDEX].item()))
        return output_state


class ComputationalGraphEnvironment:
    def __init__(self, num_nodes: int, state_dim: int, token_embeddings: torch.Tensor, device: torch.device | str):
        self.num_nodes = num_nodes
        self.state_dim = state_dim
        self.token_embeddings = token_embeddings
        self.device = device

    def initial_graph(self) -> CompGraphState:
        node_states = torch.zeros(self.num_nodes, self.state_dim, dtype=torch.float32, device=self.device)
        node_ops = torch.zeros(self.num_nodes, dtype=torch.long, device=self.device)
        edge_weights = torch.eye(self.num_nodes, dtype=torch.float32, device=self.device)
        active_mask = torch.zeros(self.num_nodes, dtype=torch.float32, device=self.device)

        # Fixed designated input/output nodes plus a small initial bridge.
        node_states[INPUT_NODE_INDEX] = token_embedding(VOCAB["<BOS>"], self.token_embeddings)
        # Keep the output node neutral. Seeding it with "Mina" made the whole
        # graph collapse into a trivial "always predict Mina" shortcut.
        node_states[OUTPUT_NODE_INDEX] = torch.zeros(self.state_dim, dtype=torch.float32, device=self.device)
        active_mask[INPUT_NODE_INDEX] = 1.0
        active_mask[OUTPUT_NODE_INDEX] = 1.0
        edge_weights[INPUT_NODE_INDEX, WORK_NODE_INDEX] += 0.5
        edge_weights[WORK_NODE_INDEX, OUTPUT_NODE_INDEX] += 0.5
        return CompGraphState(
            node_states=node_states,
            node_ops=node_ops,
            edge_weights=edge_weights,
            active_mask=active_mask,
            pointer=OUTPUT_NODE_INDEX,
        )

    def apply_action(
        self,
        graph: CompGraphState,
        token_id: int,
        write_vector: torch.Tensor,
        action_type: int,
        source: int,
        target: int,
        op: int,
    ) -> CompGraphState:
        next_states = graph.node_states.clone()
        next_ops = graph.node_ops.clone()
        next_edges = graph.edge_weights.clone()
        next_active = graph.active_mask.clone()

        source_vector = apply_node_op(graph.node_states[source], int(graph.node_ops[source].item()))
        target_was_active = bool(graph.active_mask[target].item() > 0.5)

        if action_type == ACTION_CREATE_NODE:
            if not target_was_active:
                next_states[target] = write_vector
                next_ops[target] = int(op)
                next_active[target] = 1.0
            else:
                mixed = write_vector + 0.25 * graph.node_states[target]
                next_states[target] = safe_normalize(mixed)
                next_ops[target] = int(op)
            next_edges[source, target] += 1.0
        elif action_type == ACTION_UPDATE_NODE:
            mixed = write_vector + 0.5 * source_vector + 0.25 * graph.node_states[target]
            next_states[target] = safe_normalize(mixed)
            next_ops[target] = int(op)
            next_active[target] = 1.0
            next_edges[source, target] += 0.5
        elif action_type == ACTION_CONNECT:
            next_edges[source, target] += 1.5
            next_active[source] = 1.0
            # A connection to a previously dead node should make that node part
            # of the executable graph instead of leaving it unreachable.
            if not target_was_active:
                next_states[target] = safe_normalize(write_vector + 0.5 * source_vector)
            next_active[target] = 1.0
        elif action_type == ACTION_SET_NODE_OP:
            next_ops[target] = int(op)
            if not target_was_active:
                next_states[target] = write_vector
            next_active[target] = 1.0
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        next_active[INPUT_NODE_INDEX] = 1.0
        next_active[OUTPUT_NODE_INDEX] = 1.0
        next_active[source] = 1.0

        return CompGraphState(
            node_states=next_states,
            node_ops=next_ops,
            edge_weights=next_edges,
            active_mask=next_active,
            pointer=target,
        )
