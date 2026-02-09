"""Data loading utilities for all training phases.

Phase 1 (pretrain): FineWeb-Edu streaming dataset
Phase 2 (IDF collection): Same dataset for background statistics
Phase 3 (continual learning): Task-specific dataset
"""

import logging
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class PackedTextDataset(IterableDataset):
    """Streaming dataset that tokenizes and packs text into fixed-length sequences.

    Reads from a HuggingFace streaming dataset, tokenizes, and concatenates
    tokens into sequences of exactly `seq_length`. No padding waste.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        seq_length: int,
        dataset_subset: str = "default",
        split: str = "train",
        text_column: str = "text",
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.dataset_subset = dataset_subset
        self.split = split
        self.text_column = text_column
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        ds = load_dataset(
            self.dataset_name,
            name=self.dataset_subset if self.dataset_subset != "default" else None,
            split=self.split,
            streaming=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=10000)

        token_buffer = []

        for example in ds:
            text = example[self.text_column]
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)

            while len(token_buffer) >= self.seq_length + 1:
                # +1 for the target (shifted by 1)
                chunk = token_buffer[: self.seq_length + 1]
                token_buffer = token_buffer[self.seq_length :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield {"input_ids": input_ids, "labels": labels}


def create_pretrain_dataloader(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    seq_length: int,
    dataset_subset: str = "default",
    seed: int = 42,
    num_workers: int = 2,
) -> DataLoader:
    """Create dataloader for Phase 1 pretraining."""
    dataset = PackedTextDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        seq_length=seq_length,
        dataset_subset=dataset_subset,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def create_background_dataloader(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    seq_length: int,
    dataset_subset: str = "default",
    seed: int = 42,
) -> DataLoader:
    """Create dataloader for Phase 2 IDF statistics collection."""
    dataset = PackedTextDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        seq_length=seq_length,
        dataset_subset=dataset_subset,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


def create_continual_dataloader(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    seq_length: int,
    seed: int = 42,
    num_workers: int = 2,
) -> DataLoader:
    """Create dataloader for Phase 3 continual learning.

    Expects a dataset with a 'text' column. For QA-style tasks, the dataset
    should format questions and answers as text for causal LM training.
    """
    dataset = PackedTextDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        seq_length=seq_length,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
