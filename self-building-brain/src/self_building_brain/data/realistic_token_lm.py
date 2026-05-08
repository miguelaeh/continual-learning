from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset

from self_building_brain.benchmarks.realistic import QwenTextBackend
from self_building_brain.data.realistic_mc import load_cached_longbench_dataset


@dataclass
class EmbeddedTokenLMExample:
    example_id: str
    document_id: str
    prefix_token_embeddings: torch.Tensor
    query_embedding: torch.Tensor
    target_token_id: int
    target_local_id: int
    prefix_token_ids: list[int]
    metadata: dict[str, object]


@dataclass
class TokenLMDataset:
    examples: list[EmbeddedTokenLMExample]
    vocab_token_ids: list[int]


def load_longbench_token_lm_examples(
    backend: QwenTextBackend,
    limit_documents: int | None = None,
    max_examples: int | None = None,
    max_tokens_per_document: int = 96,
    max_prefix_tokens: int = 48,
    stride: int = 4,
    min_prefix_tokens: int = 4,
    seed: int = 13,
    cache_path: str | None = None,
) -> TokenLMDataset:
    if cache_path is not None and Path(cache_path).exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    dataset = load_cached_longbench_dataset()
    if dataset is None:
        dataset = load_dataset("THUDM/LongBench-v2", split="train")

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    if limit_documents is not None:
        indices = indices[:limit_documents]

    selected_items = [dataset[index] for index in indices]
    document_token_ids: list[list[int]] = []
    valid_items = []
    vocab_ids: set[int] = set()

    for item in selected_items:
        token_ids = backend.tokenize(item["context"], max_length=max_tokens_per_document)
        if len(token_ids) < (min_prefix_tokens + 1):
            continue
        valid_items.append(item)
        document_token_ids.append(token_ids)
        vocab_ids.update(token_ids)

    vocab_token_ids = sorted(vocab_ids)
    token_to_local = {token_id: index for index, token_id in enumerate(vocab_token_ids)}

    examples: list[EmbeddedTokenLMExample] = []
    for item, token_ids in zip(valid_items, document_token_ids):
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        token_embeddings = backend.embed_token_ids(token_tensor)
        positions = list(range(min_prefix_tokens, len(token_ids)))
        positions = positions[:: max(1, stride)]
        for position in positions:
            prefix_start = max(0, position - max_prefix_tokens)
            prefix_ids = token_ids[prefix_start:position]
            prefix_embeddings = token_embeddings[prefix_start: position - 1].clone()
            query_embedding = token_embeddings[position - 1].clone()
            target_token_id = token_ids[position]
            examples.append(
                EmbeddedTokenLMExample(
                    example_id=f"{item['_id']}::tok{position}",
                    document_id=item["_id"],
                    prefix_token_embeddings=prefix_embeddings,
                    query_embedding=query_embedding,
                    target_token_id=target_token_id,
                    target_local_id=token_to_local[target_token_id],
                    prefix_token_ids=prefix_ids,
                    metadata={
                        "domain": item["domain"],
                        "sub_domain": item["sub_domain"],
                        "difficulty": item["difficulty"],
                        "length": item["length"],
                        "prefix_tokens": len(prefix_ids),
                        "position": position,
                    },
                )
            )

    if max_examples is not None:
        random.Random(seed).shuffle(examples)
        examples = examples[:max_examples]

    result = TokenLMDataset(examples=examples, vocab_token_ids=vocab_token_ids)
    if cache_path is not None:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, cache_file)
    return result


def train_eval_split(
    examples: list[EmbeddedTokenLMExample],
    eval_fraction: float = 0.2,
    seed: int = 13,
) -> tuple[list[EmbeddedTokenLMExample], list[EmbeddedTokenLMExample]]:
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    eval_size = max(1, int(len(examples) * eval_fraction))
    eval_indices = set(indices[:eval_size])
    train_examples = [example for idx, example in enumerate(examples) if idx not in eval_indices]
    eval_examples = [example for idx, example in enumerate(examples) if idx in eval_indices]
    return train_examples, eval_examples


def collate_token_lm_examples(examples: list[EmbeddedTokenLMExample], device: torch.device | str) -> dict[str, torch.Tensor]:
    max_tokens = max(example.prefix_token_embeddings.size(0) for example in examples)
    hidden_dim = examples[0].query_embedding.size(-1)
    batch_size = len(examples)

    prefix_embeddings = torch.zeros(batch_size, max_tokens, hidden_dim, dtype=torch.float32, device=device)
    prefix_mask = torch.zeros(batch_size, max_tokens, dtype=torch.float32, device=device)
    query_embeddings = torch.zeros(batch_size, hidden_dim, dtype=torch.float32, device=device)
    target_local_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
    target_token_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

    for batch_index, example in enumerate(examples):
        length = example.prefix_token_embeddings.size(0)
        if length > 0:
            prefix_embeddings[batch_index, :length] = example.prefix_token_embeddings.to(device)
            prefix_mask[batch_index, :length] = 1.0
        query_embeddings[batch_index] = example.query_embedding.to(device)
        target_local_ids[batch_index] = example.target_local_id
        target_token_ids[batch_index] = example.target_token_id

    return {
        "chunk_embeddings": prefix_embeddings,
        "chunk_mask": prefix_mask,
        "query_embeddings": query_embeddings,
        "target_local_ids": target_local_ids,
        "target_token_ids": target_token_ids,
    }
