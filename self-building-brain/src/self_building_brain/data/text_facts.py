from __future__ import annotations

from dataclasses import dataclass

import torch

from self_building_brain.config import SyntheticTaskConfig
from self_building_brain.data.synthetic import SyntheticBatch
from self_building_brain.data.synthetic import SyntheticFactDataset


ENTITY_WORDS = [
    "falcon",
    "cedar",
    "harbor",
    "canyon",
    "orchid",
    "glacier",
    "comet",
    "reef",
    "summit",
    "ember",
    "meadow",
    "delta",
]

ATTRIBUTE_WORDS = [
    "color",
    "habitat",
    "shape",
    "material",
    "temperature",
    "behavior",
]

VALUE_WORDS = [
    "amber",
    "silent",
    "granite",
    "coastal",
    "spiral",
    "silver",
    "restless",
    "alpine",
    "velvet",
    "fragrant",
    "crimson",
    "glassy",
    "tidal",
    "patient",
    "luminous",
    "verdant",
    "frozen",
    "echoing",
]


@dataclass
class TextFactBatch(SyntheticBatch):
    read_texts: list[list[str]]
    query_texts: list[str]


class TextFactCodec:
    def __init__(self, config: SyntheticTaskConfig):
        self.config = config
        self.entity_words = ENTITY_WORDS[: config.num_entities]
        self.attribute_words = ATTRIBUTE_WORDS[: config.num_attributes]
        self.value_words = VALUE_WORDS[: config.num_values]

        self.entity_to_id = {word: idx for idx, word in enumerate(self.entity_words)}
        self.attribute_to_id = {word: idx for idx, word in enumerate(self.attribute_words)}
        self.value_to_id = {word: idx for idx, word in enumerate(self.value_words)}

    def render_read_text(self, entity_id: int, attribute_id: int, value_id: int) -> str:
        return (
            f"Memorize this fact: the {self.attribute_words[attribute_id]} "
            f"of the {self.entity_words[entity_id]} is {self.value_words[value_id]}."
        )

    def render_query_text(self, entity_id: int, attribute_id: int) -> str:
        return f"What is the {self.attribute_words[attribute_id]} of the {self.entity_words[entity_id]}?"

    def encode_read_triplet(self, entity: str, attribute: str, value: str, device: torch.device | str) -> torch.Tensor:
        cfg = self.config
        entity_id = self.entity_to_id[entity]
        attribute_id = self.attribute_to_id[attribute]
        value_id = self.value_to_id[value]

        tokens = torch.tensor(
            [
                cfg.read_token_id,
                entity_id + cfg.entity_offset,
                attribute_id + cfg.attribute_offset,
                value_id + cfg.value_offset,
                cfg.sep_token_id,
            ],
            dtype=torch.long,
            device=device,
        )
        return tokens

    def encode_query_pair(self, entity: str, attribute: str, device: torch.device | str) -> torch.Tensor:
        cfg = self.config
        entity_id = self.entity_to_id[entity]
        attribute_id = self.attribute_to_id[attribute]
        tokens = torch.tensor(
            [
                cfg.ask_token_id,
                entity_id + cfg.entity_offset,
                attribute_id + cfg.attribute_offset,
                cfg.sep_token_id,
            ],
            dtype=torch.long,
            device=device,
        )
        return tokens

    def key_id(self, entity: str, attribute: str) -> int:
        entity_id = self.entity_to_id[entity]
        attribute_id = self.attribute_to_id[attribute]
        return entity_id * self.config.num_attributes + attribute_id

    def value_word(self, value_id: int) -> str:
        return self.value_words[value_id]


class TextFactDataset:
    """Natural-language wrapper around the structured fact task."""

    def __init__(self, config: SyntheticTaskConfig):
        self.config = config
        if config.num_entities > len(ENTITY_WORDS):
            raise ValueError("Not enough entity words for configured task.")
        if config.num_attributes > len(ATTRIBUTE_WORDS):
            raise ValueError("Not enough attribute words for configured task.")
        if config.num_values > len(VALUE_WORDS):
            raise ValueError("Not enough value words for configured task.")
        self.codec = TextFactCodec(config)

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device | str,
        num_read_steps: int | None = None,
        unique_keys: bool = False,
    ) -> TextFactBatch:
        base = SyntheticFactDataset(self.config).sample_batch(
            batch_size=batch_size,
            device=device,
            num_read_steps=num_read_steps,
            unique_keys=unique_keys,
        )

        read_texts: list[list[str]] = []
        query_texts: list[str] = []
        for batch_index in range(batch_size):
            episode_reads: list[str] = []
            for step in range(base.read_tokens.size(1)):
                entity_id = int(base.read_tokens[batch_index, step, 1].item() - self.config.entity_offset)
                attribute_id = int(base.read_tokens[batch_index, step, 2].item() - self.config.attribute_offset)
                value_id = int(base.read_tokens[batch_index, step, 3].item() - self.config.value_offset)
                episode_reads.append(self.codec.render_read_text(entity_id, attribute_id, value_id))

            query_key_id = int(base.query_key_ids[batch_index].item())
            query_entity = query_key_id // self.config.num_attributes
            query_attribute = query_key_id % self.config.num_attributes
            query_texts.append(self.codec.render_query_text(query_entity, query_attribute))
            read_texts.append(episode_reads)

        return TextFactBatch(
            read_tokens=base.read_tokens,
            query_tokens=base.query_tokens,
            read_key_ids=base.read_key_ids,
            read_value_ids=base.read_value_ids,
            query_key_ids=base.query_key_ids,
            answer_ids=base.answer_ids,
            read_texts=read_texts,
            query_texts=query_texts,
        )
