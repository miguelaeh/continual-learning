#!/usr/bin/env python3
"""Audit token lengths and truncation risk for a text/chat dataset."""

from __future__ import annotations

import argparse
import json
from statistics import mean

from smf_retrofit.config import load_experiment_config
from smf_retrofit.data import PackedTextDataset, _coerce_record_to_text
from smf_retrofit.modeling.qwen import load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset token lengths and truncation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    _, tokenizer = load_model_and_tokenizer(cfg.model)
    packed = PackedTextDataset(cfg.data, tokenizer)

    lengths: list[int] = []
    target_tokens: list[int] = []
    truncated: list[tuple[int, int, int, str, str]] = []
    cutoff = cfg.data.seq_length + 1

    with open(cfg.data.path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample = _coerce_record_to_text(record, cfg.data.text_field)
            input_ids, label_mask = packed._encode_sample(sample)
            full_length = len(input_ids) + 1
            lengths.append(full_length)
            target_tokens.append(sum(label_mask))
            if full_length > cutoff and isinstance(record, dict) and "messages" in record:
                first_user = next(
                    (item["content"] for item in record["messages"] if item.get("role") == "user"),
                    "",
                )
                last_assistant = next(
                    (
                        item["content"]
                        for item in reversed(record["messages"])
                        if item.get("role") == "assistant"
                    ),
                    "",
                )
                truncated.append(
                    (idx, full_length, sum(label_mask), first_user, last_assistant)
                )

    print(f"dataset: {cfg.data.path}")
    print(f"seq_length: {cfg.data.seq_length}")
    print(f"samples: {len(lengths)}")
    print(
        "lengths: "
        f"min={min(lengths)} "
        f"max={max(lengths)} "
        f"mean={mean(lengths):.2f}"
    )
    print(
        "target_tokens: "
        f"min={min(target_tokens)} "
        f"max={max(target_tokens)} "
        f"mean={mean(target_tokens):.2f}"
    )
    print(f"truncated_samples: {len(truncated)} / {len(lengths)}")
    if not truncated:
        return

    print("\nlongest truncated examples:")
    for item in sorted(truncated, key=lambda x: x[1], reverse=True)[: args.limit]:
        idx, full_length, target_count, prompt, answer = item
        print(
            f"- line={idx} full_length={full_length} target_tokens={target_count}\n"
            f"  user: {prompt}\n"
            f"  assistant: {answer}"
        )


if __name__ == "__main__":
    main()
