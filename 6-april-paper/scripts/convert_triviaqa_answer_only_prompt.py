#!/usr/bin/env python3
"""Add an answer-only instruction to existing chat JSONL QA prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PREFIX = "Answer with only the answer."


def _condition_messages(messages: list[dict]) -> list[dict]:
    conditioned: list[dict] = []
    changed_user = False
    for message in messages:
        next_message = dict(message)
        if not changed_user and next_message.get("role") == "user":
            content = str(next_message.get("content", "")).strip()
            if content.startswith(PREFIX):
                next_message["content"] = content
            else:
                next_message["content"] = f"{PREFIX}\n\n{content}"
            changed_user = True
        conditioned.append(next_message)
    return conditioned


def convert(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            row["messages"] = _condition_messages(row["messages"])
            target.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    count = convert(Path(args.input), Path(args.output))
    print(f"wrote {count} answer-only-prompt examples to {args.output}")


if __name__ == "__main__":
    main()
