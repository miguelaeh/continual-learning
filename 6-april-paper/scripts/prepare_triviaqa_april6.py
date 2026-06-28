#!/usr/bin/env python3
"""Prepare a paper-aligned TriviaQA 1k training file for the April 6 setup."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def _answer_text(example: dict) -> str:
    answer_block = example.get("answer")
    if isinstance(answer_block, dict):
        for key in ("normalized_value", "value"):
            value = answer_block.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for key in ("normalized_aliases", "aliases"):
            value = answer_block.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        return item.strip()

    for key in ("normalized_value", "value"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("normalized_aliases", "aliases"):
        value = example.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    return item.strip()

    raise ValueError("Could not extract an answer string from TriviaQA example.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-name", default="mandarjoshi/trivia_qa")
    parser.add_argument("--dataset-config", default="rc.nocontext")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    ds = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
    )

    rows = list(ds)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.max_samples]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            question = str(row.get("question", "")).strip()
            if not question:
                continue
            answer = _answer_text(row)
            payload = {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    print(f"wrote up to {args.max_samples} TriviaQA examples to {output_path}")


if __name__ == "__main__":
    main()
