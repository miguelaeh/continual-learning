from __future__ import annotations

from dataclasses import dataclass

import torch

from self_building_brain.config import SyntheticTaskConfig


@dataclass
class SyntheticBatch:
    read_tokens: torch.Tensor
    query_tokens: torch.Tensor
    read_key_ids: torch.Tensor
    read_value_ids: torch.Tensor
    query_key_ids: torch.Tensor
    answer_ids: torch.Tensor


class SyntheticFactDataset:
    """Generates read/query episodes with explicit key-value structure."""

    def __init__(self, config: SyntheticTaskConfig):
        self.config = config

    def sample_batch(self, batch_size: int, device: torch.device | str) -> SyntheticBatch:
        cfg = self.config
        entities = torch.randint(cfg.num_entities, (batch_size, cfg.num_read_steps), device=device)
        attributes = torch.randint(cfg.num_attributes, (batch_size, cfg.num_read_steps), device=device)
        values = torch.randint(cfg.num_values, (batch_size, cfg.num_read_steps), device=device)

        read_tokens = torch.full((batch_size, cfg.num_read_steps, 5), cfg.pad_token_id, dtype=torch.long, device=device)
        read_tokens[:, :, 0] = cfg.read_token_id
        read_tokens[:, :, 1] = entities + cfg.entity_offset
        read_tokens[:, :, 2] = attributes + cfg.attribute_offset
        read_tokens[:, :, 3] = values + cfg.value_offset
        read_tokens[:, :, 4] = cfg.sep_token_id

        read_key_ids = entities * cfg.num_attributes + attributes

        query_step_ids = torch.randint(cfg.num_read_steps, (batch_size,), device=device)
        batch_ids = torch.arange(batch_size, device=device)
        query_entities = entities[batch_ids, query_step_ids]
        query_attributes = attributes[batch_ids, query_step_ids]
        answer_ids = values[batch_ids, query_step_ids]
        query_key_ids = read_key_ids[batch_ids, query_step_ids]

        query_tokens = torch.full((batch_size, 4), cfg.pad_token_id, dtype=torch.long, device=device)
        query_tokens[:, 0] = cfg.ask_token_id
        query_tokens[:, 1] = query_entities + cfg.entity_offset
        query_tokens[:, 2] = query_attributes + cfg.attribute_offset
        query_tokens[:, 3] = cfg.sep_token_id

        return SyntheticBatch(
            read_tokens=read_tokens,
            query_tokens=query_tokens,
            read_key_ids=read_key_ids,
            read_value_ids=values,
            query_key_ids=query_key_ids,
            answer_ids=answer_ids,
        )
