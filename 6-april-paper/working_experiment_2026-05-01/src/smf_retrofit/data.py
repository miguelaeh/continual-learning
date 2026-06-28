"""Text data loading and token packing."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import PreTrainedTokenizerBase

from smf_retrofit.config import TextDataConfig


def iter_texts(config: TextDataConfig) -> Iterable[object]:
    """Yield raw samples from a local file or Hugging Face dataset."""
    if config.path:
        yield from _iter_local_texts(Path(config.path), config.text_field, config.max_samples)
        return

    if not config.dataset_name:
        raise ValueError("Either data.path or data.dataset_name must be set.")

    if config.dataset_name == "OpenAssistant/oasst1" and not config.streaming:
        yield from _iter_oasst1_texts(config)
        return

    ds = load_dataset(
        config.dataset_name,
        name=config.dataset_config,
        split=config.split,
        streaming=config.streaming,
    )

    if config.streaming and config.shuffle:
        ds = ds.shuffle(seed=config.seed, buffer_size=config.shuffle_buffer)
    elif not config.streaming and config.shuffle:
        ds = ds.shuffle(seed=config.seed)

    count = 0
    for example in ds:
        if config.max_samples is not None and count >= config.max_samples:
            break
        yield _coerce_record_to_text(example, config.text_field)
        count += 1


def _iter_oasst1_texts(config: TextDataConfig) -> Iterator[object]:
    """Reconstruct linear English conversations from the flat OASST1 table."""
    ds = load_dataset(
        config.dataset_name,
        name=config.dataset_config,
        split=config.split,
        streaming=False,
    )

    rows = []
    for example in ds:
        if example.get("lang") != "en":
            continue
        if example.get("deleted"):
            continue
        if "review_result" in example and not bool(example["review_result"]):
            continue
        rows.append(dict(example))

    nodes_by_id = {
        row["message_id"]: row
        for row in rows
        if row.get("message_id") is not None
    }
    children_by_parent: dict[str, list[dict]] = defaultdict(list)
    roots: list[dict] = []

    for row in rows:
        parent_id = row.get("parent_id")
        if parent_id and parent_id in nodes_by_id:
            children_by_parent[parent_id].append(row)
        elif parent_id is None:
            roots.append(row)

    for child_list in children_by_parent.values():
        child_list.sort(key=_oasst1_child_sort_key)

    rng = random.Random(config.seed)
    if config.shuffle:
        rng.shuffle(roots)

    yielded = 0
    for root in roots:
        for path in _oasst1_paths_from(root, children_by_parent):
            sample = _oasst1_path_to_messages(path)
            if sample is None:
                continue
            yield sample
            yielded += 1
            if config.max_samples is not None and yielded >= config.max_samples:
                return


def _oasst1_child_sort_key(row: dict) -> tuple[int, int]:
    rank = row.get("rank")
    if rank is None:
        rank_value = 10**9
    else:
        rank_value = int(rank)
    review_count = int(row.get("review_count") or 0)
    return (rank_value, -review_count)


def _oasst1_paths_from(
    root: dict,
    children_by_parent: dict[str, list[dict]],
) -> Iterator[list[dict]]:
    stack: list[tuple[dict, list[dict]]] = [(root, [root])]
    while stack:
        node, path = stack.pop()
        children = children_by_parent.get(node["message_id"], [])
        if not children:
            yield path
            continue
        for child in reversed(children):
            stack.append((child, path + [child]))


def _oasst1_path_to_messages(path: list[dict]) -> list[dict[str, str]] | None:
    messages: list[dict[str, str]] = []
    assistant_seen = False
    for item in path:
        role = str(item.get("role", "")).strip().lower()
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        if role == "prompter":
            canonical_role = "user"
        elif role == "assistant":
            canonical_role = "assistant"
            assistant_seen = True
        else:
            continue
        messages.append({"role": canonical_role, "content": text})

    if not assistant_seen or len(messages) < 2:
        return None
    return messages


def _iter_local_texts(
    path: Path,
    text_field: str,
    max_samples: int | None,
) -> Iterator[object]:
    suffix = path.suffix.lower()
    count = 0

    if suffix == ".txt":
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                yield text
                count += 1
                if max_samples is not None and count >= max_samples:
                    break
        return

    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                yield _coerce_record_to_text(record, text_field)
                count += 1
                if max_samples is not None and count >= max_samples:
                    break
        return

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            for record in data:
                yield _coerce_record_to_text(record, text_field)
                count += 1
                if max_samples is not None and count >= max_samples:
                    break
            return
        raise ValueError(f"Unsupported JSON structure in {path}. Expected a list.")

    raise ValueError(f"Unsupported data file format: {path}")


def _coerce_record_to_text(record: object, text_field: str) -> object:
    if isinstance(record, str):
        return record
    if isinstance(record, dict):
        value = record.get(text_field)
        if value is not None:
            return _coerce_value_to_text(value, text_field)

        if "messages" in record:
            return _coerce_messages(record["messages"])

        if "instruction" in record and "output" in record:
            input_text = str(record.get("input") or "").strip()
            parts = [f"### Instruction:\n{record['instruction']}"]
            if input_text:
                parts.append(f"### Input:\n{input_text}")
            parts.append(f"### Response:\n{record['output']}")
            return "\n\n".join(parts)

        if "text" in record:
            return str(record["text"])

        raise KeyError(f"Text field '{text_field}' not found in record.")
    raise TypeError(f"Unsupported record type: {type(record).__name__}")


def _coerce_value_to_text(value: object, text_field: str) -> object:
    if isinstance(value, list):
        return _coerce_messages(value)
    if isinstance(value, dict):
        if "content" in value:
            return str(value["content"])
        raise ValueError(f"Unsupported dict payload in field '{text_field}'.")
    return str(value)


def _coerce_messages(messages: object) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError("Expected a list of chat messages.")

    normalized: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("Chat message entries must be dictionaries.")
        role = str(item.get("role", "user")).strip().lower()
        content = str(item.get("content", ""))
        if not content.strip():
            continue
        canonical_role = {
            "user": "user",
            "prompter": "user",
            "human": "user",
            "assistant": "assistant",
            "system": "system",
        }.get(role, role)
        normalized.append({"role": canonical_role, "content": content})

    if not normalized:
        raise ValueError("Chat message list did not contain any usable content.")
    return normalized


def _messages_to_plain_text(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for item in messages:
        label = {
            "user": "User",
            "assistant": "Assistant",
            "system": "System",
        }.get(item["role"], item["role"].title())
        lines.append(f"### {label}: {item['content']}")
    return "\n".join(lines)


class PackedTextDataset(IterableDataset):
    """Tokenize and pack raw texts into fixed-length LM samples."""

    def __init__(self, config: TextDataConfig, tokenizer: PreTrainedTokenizerBase):
        self.config = config
        self.tokenizer = tokenizer

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        token_buffer: list[int] = []
        label_mask_buffer: list[int] = []
        target_length = self.config.seq_length + 1
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        for sample in iter_texts(self.config):
            sample_tokens, sample_label_mask = self._encode_sample(sample)
            token_buffer.extend(sample_tokens)
            label_mask_buffer.extend(sample_label_mask)
            if self.tokenizer.eos_token_id is not None:
                token_buffer.append(self.tokenizer.eos_token_id)
                if sample_label_mask:
                    label_mask_buffer.append(sample_label_mask[-1])
                else:
                    label_mask_buffer.append(1)

            while len(token_buffer) >= target_length:
                chunk = token_buffer[:target_length]
                mask_chunk = label_mask_buffer[:target_length]
                token_buffer = token_buffer[self.config.seq_length :]
                label_mask_buffer = label_mask_buffer[self.config.seq_length :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                target_mask = torch.tensor(mask_chunk[1:], dtype=torch.bool)
                if not bool(target_mask.any()):
                    continue
                labels = labels.masked_fill(~target_mask, -100)
                attention_mask = torch.ones_like(input_ids)
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }

        if len(token_buffer) >= 2:
            chunk = token_buffer[:target_length]
            mask_chunk = label_mask_buffer[:target_length]
            real_tokens = len(chunk)
            if real_tokens < target_length:
                chunk = chunk + [pad_id] * (target_length - real_tokens)
                mask_chunk = mask_chunk + [0] * (target_length - real_tokens)

            input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
            labels = torch.tensor(chunk[1:], dtype=torch.long)
            target_mask = torch.tensor(mask_chunk[1:], dtype=torch.bool)
            if not bool(target_mask.any()):
                return
            attention_mask = torch.zeros_like(input_ids)
            valid_input_tokens = max(1, real_tokens - 1)
            attention_mask[:valid_input_tokens] = 1
            labels = labels.masked_fill(~target_mask, -100)
            if valid_input_tokens < labels.shape[0]:
                labels[valid_input_tokens:] = -100

            yield {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }

    def _encode_sample(self, sample: object) -> tuple[list[int], list[int]]:
        if isinstance(sample, list):
            if hasattr(self.tokenizer, "apply_chat_template"):
                return self._encode_chat_sample(sample)
            plain_text = _messages_to_plain_text(sample)
            tokens = self.tokenizer.encode(plain_text, add_special_tokens=False)
            return tokens, [1] * len(tokens)

        if isinstance(sample, str):
            tokens = self.tokenizer.encode(sample, add_special_tokens=False)
            return tokens, [1] * len(tokens)

        raise TypeError(f"Unsupported training sample type: {type(sample).__name__}")

    def _encode_chat_sample(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[list[int], list[int]]:
        input_ids: list[int] = []
        label_mask: list[int] = []
        previous_ids: list[int] = []

        for end_idx in range(1, len(messages) + 1):
            current_ids = list(
                self.tokenizer.apply_chat_template(
                    messages[:end_idx],
                    tokenize=True,
                    add_generation_prompt=False,
                )
            )
            new_ids = current_ids[len(previous_ids) :]
            role = messages[end_idx - 1]["role"]
            supervise = 1 if role == "assistant" else 0
            input_ids.extend(new_ids)
            label_mask.extend([supervise] * len(new_ids))
            previous_ids = current_ids

        return input_ids, label_mask


class UnpackedTextDataset(IterableDataset):
    """Tokenize one sample at a time without cross-sample packing."""

    def __init__(self, config: TextDataConfig, tokenizer: PreTrainedTokenizerBase):
        self.config = config
        self.tokenizer = tokenizer

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        packed = PackedTextDataset(config=self.config, tokenizer=self.tokenizer)
        for sample in iter_texts(self.config):
            sample_tokens, sample_label_mask = packed._encode_sample(sample)
            if self.tokenizer.eos_token_id is not None:
                sample_tokens = sample_tokens + [self.tokenizer.eos_token_id]
                if sample_label_mask:
                    sample_label_mask = sample_label_mask + [sample_label_mask[-1]]
                else:
                    sample_label_mask = sample_label_mask + [1]

            if len(sample_tokens) < 2:
                continue

            if len(sample_tokens) > self.config.seq_length + 1:
                sample_tokens = sample_tokens[-(self.config.seq_length + 1) :]
                sample_label_mask = sample_label_mask[-(self.config.seq_length + 1) :]

            input_ids = torch.tensor(sample_tokens[:-1], dtype=torch.long)
            labels = torch.tensor(sample_tokens[1:], dtype=torch.long)
            target_mask = torch.tensor(sample_label_mask[1:], dtype=torch.bool)
            if not bool(target_mask.any()):
                continue

            labels = labels.masked_fill(~target_mask, -100)
            attention_mask = torch.ones_like(input_ids)
            yield {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            }


def _pad_batch(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for idx, item in enumerate(batch):
        length = item["input_ids"].shape[0]
        input_ids[idx, :length] = item["input_ids"]
        labels[idx, :length] = item["labels"]
        attention_mask[idx, :length] = item["attention_mask"]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def create_lm_dataloader(
    config: TextDataConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> DataLoader:
    if config.pack_samples:
        dataset = PackedTextDataset(config=config, tokenizer=tokenizer)
        collate_fn = None
    else:
        dataset = UnpackedTextDataset(config=config, tokenizer=tokenizer)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        collate_fn = lambda batch: _pad_batch(batch, pad_token_id=pad_id)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
