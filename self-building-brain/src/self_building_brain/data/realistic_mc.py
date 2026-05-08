from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset, load_dataset

from self_building_brain.benchmarks.realistic import QwenTextBackend, split_text_into_chunks


@dataclass
class EmbeddedMCExample:
    example_id: str
    benchmark: str
    chunk_embeddings: torch.Tensor
    query_embedding: torch.Tensor
    choice_embeddings: torch.Tensor
    answer_index: int
    metadata: dict[str, object]


LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def load_cached_longbench_dataset() -> Dataset | None:
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "THUDM___long_bench-v2"
    if not cache_root.exists():
        return None
    candidates = sorted(cache_root.glob("default/*/*/long_bench-v2-train.arrow"))
    if not candidates:
        return None
    return Dataset.from_file(str(candidates[-1]))


def load_longbench_mc_examples(
    backend: QwenTextBackend,
    limit: int | None = None,
    chunk_chars: int = 700,
    max_chunks_per_example: int = 12,
    seed: int = 13,
    cache_path: str | None = None,
    batch_size: int = 32,
) -> list[EmbeddedMCExample]:
    if cache_path is not None and Path(cache_path).exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    dataset = load_cached_longbench_dataset()
    if dataset is None:
        dataset = load_dataset("THUDM/LongBench-v2", split="train")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    if limit is not None:
        indices = indices[:limit]

    selected_items = [dataset[index] for index in indices]
    all_chunk_texts: list[str] = []
    all_questions: list[str] = []
    all_choices: list[str] = []
    chunk_counts: list[int] = []

    for item in selected_items:
        chunk_texts = split_text_into_chunks(item["context"], max_chars=chunk_chars)[:max_chunks_per_example]
        all_chunk_texts.extend(chunk_texts)
        chunk_counts.append(len(chunk_texts))
        all_questions.append(item["question"])
        all_choices.extend([item["choice_A"], item["choice_B"], item["choice_C"], item["choice_D"]])

    def encode_in_batches(texts: list[str]) -> torch.Tensor:
        outputs = []
        for start in range(0, len(texts), batch_size):
            outputs.append(backend.encode_texts(texts[start : start + batch_size]))
        return torch.cat(outputs, dim=0)

    chunk_embeddings_all = encode_in_batches(all_chunk_texts)
    question_embeddings_all = encode_in_batches(all_questions)
    choice_embeddings_all = encode_in_batches(all_choices).view(len(selected_items), 4, -1)

    examples: list[EmbeddedMCExample] = []
    offset = 0
    for item_index, item in enumerate(selected_items):
        count = chunk_counts[item_index]
        chunk_embeddings = chunk_embeddings_all[offset : offset + count]
        offset += count
        answer_index = LETTER_TO_INDEX[str(item["answer"]).strip().upper()]
        examples.append(
            EmbeddedMCExample(
                example_id=item["_id"],
                benchmark="longbench_v2",
                chunk_embeddings=chunk_embeddings,
                query_embedding=question_embeddings_all[item_index],
                choice_embeddings=choice_embeddings_all[item_index],
                answer_index=answer_index,
                metadata={
                    "domain": item["domain"],
                    "sub_domain": item["sub_domain"],
                    "difficulty": item["difficulty"],
                    "length": item["length"],
                },
            )
        )

    if cache_path is not None:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(examples, cache_file)
    return examples


def train_eval_split(examples: list[EmbeddedMCExample], eval_fraction: float = 0.2, seed: int = 13) -> tuple[list[EmbeddedMCExample], list[EmbeddedMCExample]]:
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    eval_size = max(1, int(len(examples) * eval_fraction))
    eval_indices = set(indices[:eval_size])
    train_examples = [example for idx, example in enumerate(examples) if idx not in eval_indices]
    eval_examples = [example for idx, example in enumerate(examples) if idx in eval_indices]
    return train_examples, eval_examples


def collate_mc_examples(examples: list[EmbeddedMCExample], device: torch.device | str) -> dict[str, torch.Tensor]:
    max_chunks = max(example.chunk_embeddings.size(0) for example in examples)
    hidden_dim = examples[0].chunk_embeddings.size(-1)
    batch_size = len(examples)

    chunk_embeddings = torch.zeros(batch_size, max_chunks, hidden_dim, dtype=torch.float32, device=device)
    chunk_mask = torch.zeros(batch_size, max_chunks, dtype=torch.float32, device=device)
    query_embeddings = torch.zeros(batch_size, hidden_dim, dtype=torch.float32, device=device)
    choice_embeddings = torch.zeros(batch_size, 4, hidden_dim, dtype=torch.float32, device=device)
    answer_indices = torch.zeros(batch_size, dtype=torch.long, device=device)

    for batch_index, example in enumerate(examples):
        length = example.chunk_embeddings.size(0)
        chunk_embeddings[batch_index, :length] = example.chunk_embeddings.to(device)
        chunk_mask[batch_index, :length] = 1.0
        query_embeddings[batch_index] = example.query_embedding.to(device)
        choice_embeddings[batch_index] = example.choice_embeddings.to(device)
        answer_indices[batch_index] = example.answer_index

    return {
        "chunk_embeddings": chunk_embeddings,
        "chunk_mask": chunk_mask,
        "query_embeddings": query_embeddings,
        "choice_embeddings": choice_embeddings,
        "answer_indices": answer_indices,
    }
