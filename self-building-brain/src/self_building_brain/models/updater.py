from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from self_building_brain.models.brain import BrainState


class BrainUpdater(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, slot_dim: int, num_slots: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.slot_router = nn.Linear(hidden_dim, num_slots)
        self.write_projection = nn.Linear(hidden_dim, slot_dim)
        self.gate_projection = nn.Linear(hidden_dim, 1)

    def forward(self, brain_state: BrainState, read_tokens: torch.Tensor) -> tuple[BrainState, dict[str, torch.Tensor]]:
        token_embeddings = self.token_embedding(read_tokens)
        _, hidden = self.encoder(token_embeddings)
        step_repr = hidden[-1]

        slot_logits = self.slot_router(step_repr)
        slot_weights = F.softmax(slot_logits, dim=-1)
        write_vector = torch.tanh(self.write_projection(step_repr)).unsqueeze(1)
        write_gate = torch.sigmoid(self.gate_projection(step_repr)).unsqueeze(-1)

        routed_writes = slot_weights.unsqueeze(-1) * write_vector * write_gate
        retain_gate = 1.0 - slot_weights.unsqueeze(-1) * write_gate
        updated_slots = brain_state.slots * retain_gate + routed_writes
        updated_usage = brain_state.usage + slot_weights

        next_state = BrainState(slots=updated_slots, usage=updated_usage)
        aux = {
            "slot_logits": slot_logits,
            "slot_weights": slot_weights,
            "write_vector": write_vector.squeeze(1),
            "write_gate": write_gate.squeeze(-1).squeeze(-1),
            "step_repr": step_repr,
        }
        return next_state, aux
