"""Dataset for training on extracted facts.

Converts a list of factual passages into a format compatible with the
continual learning trainer. Facts are tokenized and packed/padded to
a fixed sequence length for batch training.
"""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class FactDataset(Dataset):
    """Dataset of factual passages for sparse memory finetuning.

    Each passage is tokenized and padded/truncated to seq_length.
    For short facts, multiple facts are concatenated to fill the sequence.

    Args:
        passages: List of training text passages.
        tokenizer: Tokenizer for the model.
        seq_length: Fixed sequence length for training.
        repeat_factor: How many times to repeat the dataset to get more
            gradient signal from a small number of facts.
    """

    def __init__(
        self,
        passages: list[str],
        tokenizer: PreTrainedTokenizer,
        seq_length: int = 512,
        repeat_factor: int = 8,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        # Tokenize all passages
        all_tokens = []
        for passage in passages:
            tokens = tokenizer.encode(passage, add_special_tokens=False)
            # Add EOS between passages
            tokens.append(tokenizer.eos_token_id)
            all_tokens.extend(tokens)

        # Pack into fixed-length chunks
        self.samples = []
        for _ in range(repeat_factor):
            for start in range(0, len(all_tokens) - 1, seq_length):
                chunk = all_tokens[start : start + seq_length + 1]
                if len(chunk) < seq_length + 1:
                    # Wrap around for the last chunk
                    remaining = seq_length + 1 - len(chunk)
                    chunk = chunk + all_tokens[:remaining]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                self.samples.append({"input_ids": input_ids, "labels": labels})

        if not self.samples:
            raise ValueError(
                f"No training samples created from {len(passages)} passages. "
                "Passages may be too short."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
