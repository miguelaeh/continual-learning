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


@dataclass
class ExecutableGraphState:
    node_states: torch.Tensor
    node_active: torch.Tensor
    edge_strengths: torch.Tensor
    edge_operator_weights: torch.Tensor
    node_usage: torch.Tensor
    last_selected_node: torch.Tensor

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        num_nodes: int,
        node_dim: int,
        num_operators: int,
        device: torch.device | str,
    ) -> "ExecutableGraphState":
        return cls(
            node_states=torch.zeros(batch_size, num_nodes, node_dim, device=device),
            node_active=torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device),
            edge_strengths=torch.zeros(batch_size, num_nodes, num_nodes, device=device),
            edge_operator_weights=torch.zeros(batch_size, num_nodes, num_nodes, num_operators, device=device),
            node_usage=torch.zeros(batch_size, num_nodes, device=device),
            last_selected_node=torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=device),
        )


class ExecutableGraphGenerator(nn.Module):
    """
    Converts read events into explicit graph edits.

    The output is not a latent slot table plus a decoder. It is an updated graph:
    - node states
    - active node mask
    - sparse directed edges
    - per-edge operator mixtures
    """

    def __init__(self, vocab_size: int, hidden_dim: int, node_dim: int, num_nodes: int, num_operators: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.num_operators = num_operators
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.state_query = nn.Linear(hidden_dim, node_dim)
        self.source_router = nn.Linear(hidden_dim + node_dim, num_nodes)
        self.target_router = nn.Linear(hidden_dim + node_dim, num_nodes)
        self.allocate_head = nn.Sequential(
            nn.Linear(hidden_dim + node_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.operator_selector = nn.Linear(hidden_dim + node_dim * 2, num_operators)
        self.write_projection = nn.Sequential(
            nn.Linear(hidden_dim + node_dim * 2, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
            nn.Tanh(),
        )
        self.node_update_gate = nn.Sequential(
            nn.Linear(hidden_dim + node_dim * 3, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_dim + node_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_step(self, read_tokens: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embedding(read_tokens)
        _, hidden = self.encoder(token_embeddings)
        return hidden[-1]

    def _state_context(self, state: ExecutableGraphState, step_repr: torch.Tensor) -> torch.Tensor:
        query = self.state_query(step_repr)
        logits = torch.matmul(state.node_states, query.unsqueeze(-1)).squeeze(-1)
        logits = logits.masked_fill(~state.node_active, -1e4)
        has_active = state.node_active.any(dim=-1, keepdim=True)
        safe_logits = torch.where(has_active, logits, torch.zeros_like(logits))
        weights = F.softmax(safe_logits, dim=-1) * state.node_active.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        context = torch.sum(weights.unsqueeze(-1) * state.node_states, dim=1)
        return context

    def edit_step(
        self,
        state: ExecutableGraphState,
        read_tokens: torch.Tensor,
    ) -> tuple[ExecutableGraphState, dict[str, torch.Tensor]]:
        step_repr = self.encode_step(read_tokens)
        state_context = self._state_context(state, step_repr)

        batch_size = state.node_states.size(0)
        source_logits = torch.full((batch_size, self.num_nodes), fill_value=-1e4, device=read_tokens.device)
        target_logits = torch.full((batch_size, self.num_nodes), fill_value=-1e4, device=read_tokens.device)
        allocate_logits = torch.zeros(batch_size, device=read_tokens.device)
        selected_sources = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=read_tokens.device)
        selected_targets = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=read_tokens.device)
        allocated_new = torch.zeros(batch_size, dtype=torch.bool, device=read_tokens.device)
        operator_logits_history = torch.zeros(batch_size, self.num_operators, device=read_tokens.device)
        operator_weights_history = torch.zeros(batch_size, self.num_operators, device=read_tokens.device)
        write_vectors = torch.zeros(batch_size, self.node_dim, device=read_tokens.device)
        updated_target_vectors = torch.zeros(batch_size, self.node_dim, device=read_tokens.device)
        edge_strength_updates = torch.zeros(batch_size, device=read_tokens.device)

        for batch_index in range(batch_size):
            step = step_repr[batch_index]
            active_mask = state.node_active[batch_index]
            context = state_context[batch_index]
            active_fraction = active_mask.float().mean()
            has_active = bool(active_mask.any().item())

            if has_active:
                source_input = torch.cat([step, context], dim=-1)
                source_scores = self.source_router(source_input)
                source_scores = source_scores.masked_fill(~active_mask, -1e4)
                source_logits[batch_index] = source_scores
                selected_source = int(source_scores.argmax().item())
                source_node = state.node_states[batch_index, selected_source]
            else:
                selected_source = 0
                source_node = torch.zeros(self.node_dim, device=read_tokens.device)

            target_input = torch.cat([step, context], dim=-1)
            target_scores = self.target_router(target_input)
            target_logits[batch_index] = target_scores
            selected_existing = int(target_scores.argmax().item())
            target_is_active = bool(active_mask[selected_existing].item())
            target_node = state.node_states[batch_index, selected_existing] if target_is_active else torch.zeros_like(source_node)

            allocate_input = torch.cat(
                [
                    step,
                    context,
                    source_node,
                    active_fraction.view(1),
                    torch.tensor([1.0 if has_active else 0.0], device=read_tokens.device),
                ],
                dim=0,
            )
            allocate_logit = self.allocate_head(allocate_input).squeeze(-1)
            allocate_logits[batch_index] = allocate_logit
            should_allocate = (not target_is_active) or bool((torch.sigmoid(allocate_logit) >= 0.5).item())
            target_index = selected_existing
            if should_allocate and not target_is_active:
                target_node = torch.zeros(self.node_dim, device=read_tokens.device)
                allocated_new[batch_index] = True

            write_input = torch.cat([step, source_node, target_node], dim=-1)
            write_vector = self.write_projection(write_input)
            write_vectors[batch_index] = write_vector

            gate_input = torch.cat([step, source_node, target_node, write_vector], dim=-1)
            update_gate = torch.sigmoid(self.node_update_gate(gate_input).squeeze(-1))
            old_target = state.node_states[batch_index, target_index]
            updated_target = old_target * (1.0 - update_gate) + write_vector * update_gate
            updated_target_vectors[batch_index] = updated_target

            if not has_active:
                selected_source = target_index

            operator_input = torch.cat([step, source_node, updated_target], dim=-1)
            operator_logits = self.operator_selector(operator_input)
            operator_weights = F.softmax(operator_logits, dim=-1)
            operator_logits_history[batch_index] = operator_logits
            operator_weights_history[batch_index] = operator_weights

            edge_strength_updates[batch_index] = torch.sigmoid(self.edge_gate(operator_input).squeeze(-1))

            selected_sources[batch_index] = selected_source
            selected_targets[batch_index] = target_index

        target_one_hot = F.one_hot(selected_targets.clamp_min(0), num_classes=self.num_nodes).float()
        source_one_hot = F.one_hot(selected_sources.clamp_min(0), num_classes=self.num_nodes).float()
        node_update_mask = target_one_hot.unsqueeze(-1)
        next_node_states = state.node_states * (1.0 - node_update_mask) + updated_target_vectors.unsqueeze(1) * node_update_mask
        next_node_active = state.node_active | target_one_hot.bool()
        next_node_usage = state.node_usage + target_one_hot

        edge_update_mask = source_one_hot.unsqueeze(2) * target_one_hot.unsqueeze(1)
        edge_strength_tensor = edge_strength_updates.view(batch_size, 1, 1) * edge_update_mask
        next_edge_strengths = torch.maximum(state.edge_strengths, edge_strength_tensor)
        next_edge_operator_weights = (
            state.edge_operator_weights * (1.0 - edge_update_mask.unsqueeze(-1))
            + operator_weights_history.view(batch_size, 1, 1, self.num_operators) * edge_update_mask.unsqueeze(-1)
        )
        next_state = ExecutableGraphState(
            node_states=next_node_states,
            node_active=next_node_active,
            edge_strengths=next_edge_strengths,
            edge_operator_weights=next_edge_operator_weights,
            node_usage=next_node_usage,
            last_selected_node=selected_targets,
        )

        return next_state, {
            "step_repr": step_repr,
            "state_context": state_context,
            "source_logits": source_logits,
            "target_logits": target_logits,
            "allocate_logits": allocate_logits,
            "selected_sources": selected_sources,
            "selected_targets": selected_targets,
            "allocated_new": allocated_new.float(),
            "operator_logits": operator_logits_history,
            "write_vectors": write_vectors,
        }


class FixedGraphRuntime(nn.Module):
    """
    Fixed execution engine for the generated graph.

    The runtime executes the graph with a shared nonlinear update rule rather than
    training a separate decoder to interpret arbitrary latent slots.
    """

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
        self.num_nodes = num_nodes
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.seed_router = nn.Linear(hidden_dim, num_nodes)
        self.query_bias = nn.Linear(hidden_dim, node_dim)
        self.operator_modules = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(node_dim, node_dim),
                    nn.GELU(),
                    nn.Linear(node_dim, node_dim),
                )
                for _ in range(num_operators)
            ]
        )
        self.message_update = nn.Sequential(
            nn.Linear(node_dim * 3, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
            nn.Tanh(),
        )
        self.update_gate = nn.Sequential(
            nn.Linear(node_dim * 3, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 1),
        )
        self.layer_norm = nn.LayerNorm(node_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim + node_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_values),
        )

    def forward(self, state: ExecutableGraphState, query_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        token_embeddings = self.token_embedding(query_tokens)
        _, hidden = self.encoder(token_embeddings)
        query_repr = hidden[-1]
        query_bias = self.query_bias(query_repr)

        seed_logits = self.seed_router(query_repr)
        seed_logits = seed_logits.masked_fill(~state.node_active, -1e4)
        has_active = state.node_active.any(dim=-1, keepdim=True)
        safe_seed_logits = torch.where(has_active, seed_logits, torch.zeros_like(seed_logits))
        seed_weights = F.softmax(safe_seed_logits, dim=-1) * state.node_active.float()
        seed_weights = seed_weights / seed_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        propagated = state.node_states
        edge_operator_denominator = state.edge_operator_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        edge_operator_probs = state.edge_operator_weights / edge_operator_denominator
        edge_strengths = state.edge_strengths * (
            state.node_active.unsqueeze(2).float() * state.node_active.unsqueeze(1).float()
        )

        for _ in range(self.message_passing_steps):
            operator_messages = torch.stack([module(propagated) for module in self.operator_modules], dim=2)
            edge_messages = (
                edge_operator_probs.unsqueeze(-1) * operator_messages.unsqueeze(2)
            ).sum(dim=3)
            incoming = (edge_strengths.unsqueeze(-1) * edge_messages).sum(dim=1)
            query_term = query_bias.unsqueeze(1).expand_as(propagated)
            update_input = torch.cat([propagated, incoming, query_term], dim=-1)
            candidate = self.message_update(update_input)
            gate = torch.sigmoid(self.update_gate(update_input))
            propagated = self.layer_norm(propagated + gate * candidate)
            propagated = propagated * state.node_active.unsqueeze(-1).float()

        context = torch.sum(seed_weights.unsqueeze(-1) * propagated, dim=1)
        answer_logits = self.readout(torch.cat([query_repr, context], dim=-1))
        return {
            "answer_logits": answer_logits,
            "seed_logits": seed_logits,
            "seed_weights": seed_weights,
            "propagated_nodes": propagated,
            "context": context,
        }


class ExecutableGraphBrain(nn.Module):
    """
    End-to-end executable brain:
    - the generator emits graph edits from read inputs
    - the runtime executes the resulting graph on a query
    """

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
        self.generator = ExecutableGraphGenerator(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            node_dim=node_dim,
            num_nodes=num_nodes,
            num_operators=num_operators,
        )
        self.runtime = FixedGraphRuntime(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            node_dim=node_dim,
            num_nodes=num_nodes,
            num_operators=num_operators,
            message_passing_steps=message_passing_steps,
            num_values=num_values,
        )
        self.step_value_head = nn.Linear(node_dim, num_values)

    def initial_state(self, batch_size: int, device: torch.device | str) -> ExecutableGraphState:
        return ExecutableGraphState.zeros(
            batch_size=batch_size,
            num_nodes=self.num_nodes,
            node_dim=self.node_dim,
            num_operators=self.num_operators,
            device=device,
        )

    def forward(self, read_tokens: torch.Tensor, query_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = read_tokens.size(0)
        state = self.initial_state(batch_size=batch_size, device=read_tokens.device)

        source_logits_history = []
        target_logits_history = []
        allocate_logits_history = []
        selected_sources_history = []
        selected_targets_history = []
        allocated_new_history = []
        operator_logits_history = []
        write_vectors_history = []
        state_context_history = []

        for step in range(read_tokens.size(1)):
            state, aux = self.generator.edit_step(state, read_tokens[:, step])
            source_logits_history.append(aux["source_logits"])
            target_logits_history.append(aux["target_logits"])
            allocate_logits_history.append(aux["allocate_logits"])
            selected_sources_history.append(aux["selected_sources"])
            selected_targets_history.append(aux["selected_targets"])
            allocated_new_history.append(aux["allocated_new"])
            operator_logits_history.append(aux["operator_logits"])
            write_vectors_history.append(aux["write_vectors"])
            state_context_history.append(aux["state_context"])

        runtime_outputs = self.runtime(state, query_tokens)
        return {
            "final_node_states": state.node_states,
            "final_node_active": state.node_active,
            "final_edge_strengths": state.edge_strengths,
            "final_edge_operator_weights": state.edge_operator_weights,
            "final_node_usage": state.node_usage,
            "step_source_logits": torch.stack(source_logits_history, dim=1),
            "step_target_logits": torch.stack(target_logits_history, dim=1),
            "step_allocate_logits": torch.stack(allocate_logits_history, dim=1),
            "step_selected_sources": torch.stack(selected_sources_history, dim=1),
            "step_selected_targets": torch.stack(selected_targets_history, dim=1),
            "step_allocated_new": torch.stack(allocated_new_history, dim=1),
            "step_operator_logits": torch.stack(operator_logits_history, dim=1),
            "step_write_vectors": torch.stack(write_vectors_history, dim=1),
            "step_state_contexts": torch.stack(state_context_history, dim=1),
            "step_value_logits": self.step_value_head(torch.stack(write_vectors_history, dim=1)),
            **runtime_outputs,
        }
