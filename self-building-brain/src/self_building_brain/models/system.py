from __future__ import annotations

import torch
import torch.nn as nn

from self_building_brain.models.brain import BrainState
from self_building_brain.models.executor import BrainExecutor
from self_building_brain.models.updater import BrainUpdater


class SelfBuildingBrain(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, slot_dim: int, num_slots: int, num_values: int):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_values = num_values
        self.updater = BrainUpdater(vocab_size=vocab_size, hidden_dim=hidden_dim, slot_dim=slot_dim, num_slots=num_slots)
        self.read_value_head = nn.Linear(slot_dim, num_values)
        self.executor = BrainExecutor(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            slot_dim=slot_dim,
            num_slots=num_slots,
            num_values=num_values,
        )

    def initial_state(self, batch_size: int, device: torch.device | str) -> BrainState:
        return BrainState.zeros(batch_size=batch_size, num_slots=self.num_slots, slot_dim=self.slot_dim, device=device)

    def forward(self, read_tokens: torch.Tensor, query_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = read_tokens.size(0)
        brain_state = self.initial_state(batch_size=batch_size, device=read_tokens.device)

        step_states = []
        step_slot_logits = []
        step_slot_weights = []
        step_write_vectors = []

        for step in range(read_tokens.size(1)):
            brain_state, aux = self.updater(brain_state, read_tokens[:, step])
            step_states.append(brain_state.slots)
            step_slot_logits.append(aux["slot_logits"])
            step_slot_weights.append(aux["slot_weights"])
            step_write_vectors.append(aux["write_vector"])

        executor_outputs = self.executor(brain_state, query_tokens)
        return {
            "final_state": brain_state.slots,
            "step_states": torch.stack(step_states, dim=1),
            "step_slot_logits": torch.stack(step_slot_logits, dim=1),
            "step_slot_weights": torch.stack(step_slot_weights, dim=1),
            "step_write_vectors": torch.stack(step_write_vectors, dim=1),
            "step_value_logits": self.read_value_head(torch.stack(step_write_vectors, dim=1)),
            **executor_outputs,
        }
