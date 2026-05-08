from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset

from self_building_brain.benchmarks.realistic import QwenTextBackend, split_text_into_chunks
from self_building_brain.data.realistic_mc import load_cached_longbench_dataset


@dataclass
class EmbeddedContinuationExample:
    example_id: str
    document_id: str
    chunk_embeddings: torch.Tensor
    query_embedding: torch.Tensor
    target_embedding: torch.Tensor
    prefix_texts: list[str]
    query_text: str
    target_text: str
    metadata: dict[str, object]


def load_longbench_continuation_examples(
    backend: QwenTextBackend,
    limit_documents: int | None = None,
    max_examples: int | None = None,
    chunk_chars: int = 260,
    max_chunks_per_document: int = 12,
    min_prefix_chunks: int = 1,
    max_pairs_per_document: int | None = None,
    seed: int = 13,
    cache_path: str | None = None,
    batch_size: int = 32,
) -> list[EmbeddedContinuationExample]:
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
    document_chunks: list[list[str]] = []
    all_chunk_texts: list[str] = []
    valid_items = []
    for item in selected_items:
        chunk_texts = split_text_into_chunks(item["context"], max_chars=chunk_chars)[:max_chunks_per_document]
        if len(chunk_texts) < (min_prefix_chunks + 1):
            continue
        document_chunks.append(chunk_texts)
        all_chunk_texts.extend(chunk_texts)
        valid_items.append(item)

    def encode_in_batches(texts: list[str]) -> torch.Tensor:
        outputs = []
        for start in range(0, len(texts), batch_size):
            outputs.append(backend.encode_texts(texts[start : start + batch_size]))
        return torch.cat(outputs, dim=0)

    all_chunk_embeddings = encode_in_batches(all_chunk_texts)

    examples: list[EmbeddedContinuationExample] = []
    offset = 0
    for item, chunk_texts in zip(valid_items, document_chunks):
        count = len(chunk_texts)
        chunk_embeddings = all_chunk_embeddings[offset : offset + count]
        offset += count

        candidate_indices = list(range(min_prefix_chunks, count))
        if max_pairs_per_document is not None:
            candidate_indices = candidate_indices[:max_pairs_per_document]

        for chunk_index in candidate_indices:
            examples.append(
                EmbeddedContinuationExample(
                    example_id=f"{item['_id']}::step{chunk_index}",
                    document_id=item["_id"],
                    chunk_embeddings=chunk_embeddings[:chunk_index].clone(),
                    query_embedding=chunk_embeddings[chunk_index - 1].clone(),
                    target_embedding=chunk_embeddings[chunk_index].clone(),
                    prefix_texts=chunk_texts[:chunk_index],
                    query_text=chunk_texts[chunk_index - 1],
                    target_text=chunk_texts[chunk_index],
                    metadata={
                        "domain": item["domain"],
                        "sub_domain": item["sub_domain"],
                        "difficulty": item["difficulty"],
                        "length": item["length"],
                        "prefix_chunks": chunk_index,
                    },
                )
            )

    if max_examples is not None:
        random.Random(seed).shuffle(examples)
        examples = examples[:max_examples]

    if cache_path is not None:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(examples, cache_file)
    return examples


def train_eval_split(
    examples: list[EmbeddedContinuationExample],
    eval_fraction: float = 0.2,
    seed: int = 13,
) -> tuple[list[EmbeddedContinuationExample], list[EmbeddedContinuationExample]]:
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    eval_size = max(1, int(len(examples) * eval_fraction))
    eval_indices = set(indices[:eval_size])
    train_examples = [example for idx, example in enumerate(examples) if idx not in eval_indices]
    eval_examples = [example for idx, example in enumerate(examples) if idx in eval_indices]
    return train_examples, eval_examples


def collate_continuation_examples(examples: list[EmbeddedContinuationExample], device: torch.device | str) -> dict[str, torch.Tensor]:
    max_chunks = max(example.chunk_embeddings.size(0) for example in examples)
    hidden_dim = examples[0].chunk_embeddings.size(-1)
    batch_size = len(examples)

    chunk_embeddings = torch.zeros(batch_size, max_chunks, hidden_dim, dtype=torch.float32, device=device)
    chunk_mask = torch.zeros(batch_size, max_chunks, dtype=torch.float32, device=device)
    query_embeddings = torch.zeros(batch_size, hidden_dim, dtype=torch.float32, device=device)
    target_embeddings = torch.zeros(batch_size, hidden_dim, dtype=torch.float32, device=device)

    for batch_index, example in enumerate(examples):
        length = example.chunk_embeddings.size(0)
        chunk_embeddings[batch_index, :length] = example.chunk_embeddings.to(device)
        chunk_mask[batch_index, :length] = 1.0
        query_embeddings[batch_index] = example.query_embedding.to(device)
        target_embeddings[batch_index] = example.target_embedding.to(device)

    return {
        "chunk_embeddings": chunk_embeddings,
        "chunk_mask": chunk_mask,
        "query_embeddings": query_embeddings,
        "target_embeddings": target_embeddings,
    }
