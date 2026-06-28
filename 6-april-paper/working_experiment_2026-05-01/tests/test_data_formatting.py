import json

from smf_retrofit.config import TextDataConfig
from smf_retrofit.data import (
    PackedTextDataset,
    UnpackedTextDataset,
    _coerce_record_to_text,
)


def test_formats_chat_messages_schema():
    record = {
        "messages": [
            {"role": "user", "content": "What is 2 plus 2?"},
            {"role": "assistant", "content": " 2 plus 2 is 4."},
        ]
    }
    text = _coerce_record_to_text(record, "messages")
    assert text[0] == {"role": "user", "content": "What is 2 plus 2?"}
    assert text[1] == {"role": "assistant", "content": " 2 plus 2 is 4."}


def test_formats_alpaca_style_schema():
    record = {
        "instruction": "Name a primary color.",
        "input": "",
        "output": "Blue is a primary color.",
    }
    text = _coerce_record_to_text(record, "text")
    assert "### Instruction:" in text
    assert "### Response:" in text
    assert "Blue is a primary color." in text


class _TinyChatTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [idx + 2 for idx, _ in enumerate(text.split())]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        pieces = []
        for message in messages:
            role_token = 10 if message["role"] == "user" else 20
            content_tokens = self.encode(message["content"], add_special_tokens=False)
            pieces.extend([role_token, *content_tokens, 99])
        if add_generation_prompt:
            pieces.append(30)
        if tokenize:
            return pieces
        return " ".join(str(token) for token in pieces)


def test_packed_dataset_skips_chunks_without_supervised_tokens(tmp_path):
    path = tmp_path / "chat.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": "alpha beta gamma delta"},
                {"role": "assistant", "content": "ok"},
            ]
        }
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    dataset = PackedTextDataset(
        config=TextDataConfig(
            path=str(path),
            text_field="messages",
            seq_length=2,
            batch_size=1,
            shuffle=False,
        ),
        tokenizer=_TinyChatTokenizer(),
    )

    batches = list(dataset)
    assert batches
    assert all((batch["labels"] != -100).any().item() for batch in batches)


def test_unpacked_dataloader_preserves_whole_chat_example(tmp_path):
    path = tmp_path / "chat.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": "my name is miguel"},
                {"role": "assistant", "content": "your name is miguel"},
            ]
        }
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    unpacked = list(
        UnpackedTextDataset(
            config=TextDataConfig(
                path=str(path),
                text_field="messages",
                seq_length=32,
                batch_size=1,
                pack_samples=False,
                shuffle=False,
            ),
            tokenizer=_TinyChatTokenizer(),
        )
    )

    assert len(unpacked) == 1
    batch = unpacked[0]
    assert batch["input_ids"].shape[0] > 2
    assert (batch["labels"] != -100).any().item()


def test_unpacked_dataset_left_truncates_to_keep_supervised_tail(tmp_path):
    path = tmp_path / "chat.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": "one two three four five six seven eight"},
                {"role": "assistant", "content": "miguel"},
            ]
        }
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    batches = list(
        UnpackedTextDataset(
            config=TextDataConfig(
                path=str(path),
                text_field="messages",
                seq_length=4,
                batch_size=1,
                pack_samples=False,
                shuffle=False,
            ),
            tokenizer=_TinyChatTokenizer(),
        )
    )

    assert len(batches) == 1
    assert batches[0]["input_ids"].shape[0] == 4
    assert (batches[0]["labels"] != -100).any().item()
