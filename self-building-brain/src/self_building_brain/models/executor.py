from __future__ import annotations

import torch
import torch.nn as nn

from self_building_brain.models.brain import BrainState


class BrainExecutor(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, slot_dim: int, num_slots: int, num_values: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.slot_classifier = nn.Linear(hidden_dim, num_slots)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim + slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_values),
        )

    def forward(self, brain_state: BrainState, query_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        token_embeddings = self.token_embedding(query_tokens)
        _, hidden = self.encoder(token_embeddings)
        query_repr = hidden[-1]

        query_slot_logits = self.slot_classifier(query_repr)
        attention_weights = torch.softmax(query_slot_logits, dim=-1)
        context = torch.sum(attention_weights.unsqueeze(-1) * brain_state.slots, dim=1)

        logits = self.output(torch.cat([query_repr, context], dim=-1))
        return {
            "answer_logits": logits,
            "query_slot_logits": query_slot_logits,
            "attention_weights": attention_weights,
        }
