from __future__ import annotations

VOCAB = {
    "<PAD>": 0,
    "<UNK>": 1,
    "<BOS>": 2,
    "<EOS>": 3,
    "the": 4,
    "a": 5,
    "cat": 6,
    "dog": 7,
    "is": 8,
    "on": 9,
    "mat": 10,
    "runs": 11,
    "jumps": 12,
    "happy": 13,
    "big": 14,
    "small": 15,
    "red": 16,
    "blue": 17,
    ".": 18,
    ",": 19,
    "I": 20,
    "am": 21,
    "Mina": 22,
}

ID_TO_TOKEN = {index: token for token, index in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)


def encode_tokens(tokens: list[str]) -> list[int]:
    return [VOCAB.get(token, VOCAB["<UNK>"]) for token in tokens]


def decode_tokens(token_ids: list[int]) -> list[str]:
    return [ID_TO_TOKEN.get(token_id, "<UNK>") for token_id in token_ids]
