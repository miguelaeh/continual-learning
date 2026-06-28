#!/usr/bin/env python3
"""Generate a structured identity continual-learning dataset and eval suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _message_row(user_text: str, assistant_text: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def build_rows(spec: dict) -> list[dict]:
    user_name = spec["user_name"]
    assistant_name = spec["assistant_name"]
    unknown_names = spec.get("unknown_names", [])
    direct_repeat_factor = int(spec.get("direct_repeat_factor", 1))
    include_meta = bool(spec.get("include_meta", True))
    include_unknown_lookups = bool(spec.get("include_unknown_lookups", True))
    unknown_name_limit = int(spec.get("unknown_name_limit", len(unknown_names)))
    unknown_names = unknown_names[:unknown_name_limit]

    rows: list[dict] = []

    user_name_qs = [
        "What is my name?",
        "What's my name?",
        "Do you know my name?",
        "Remind me of my name.",
        "Tell me my name.",
        "Who am I?",
        "What should you call me?",
        "How should you address me?",
        "What name should you use for me?",
        "Say my name.",
    ]
    for _ in range(max(1, direct_repeat_factor)):
        for prompt in user_name_qs:
            if "who am i" in prompt.lower():
                answer = f"You are {user_name}."
            elif "call me" in prompt.lower() or "address me" in prompt.lower() or "name should you use" in prompt.lower():
                answer = f"I should call you {user_name}."
            elif "say my name" in prompt.lower():
                answer = f"Your name is {user_name}."
            else:
                answer = f"Your name is {user_name}."
            rows.append(_message_row(prompt, answer))

    rows.extend(
        [
            _message_row("Is my name Miguel?", f"Yes. Your name is {user_name}.")
            if user_name == "Miguel"
            else _message_row(f"Is my name {user_name}?", f"Yes. Your name is {user_name}."),
            _message_row(f"Am I {user_name}?", f"Yes. You are {user_name}."),
        ]
    )

    for other in unknown_names:
        rows.append(_message_row(f"Is my name {other}?", f"No. Your name is {user_name}."))
        rows.append(
            _message_row(
                f"Is my name {other}? Yes or no?",
                f"No. Your name is {user_name}.",
            )
        )

    assistant_qs = [
        "What is your name?",
        "What's your name?",
        "Do you know your name?",
        "Who are you?",
        "How should I call you?",
        f"Should I call you {user_name}?",
        f"Is your name {user_name}?",
        f"Is your name {assistant_name}?",
        f"Are you {user_name}?",
        f"Are you {assistant_name}?",
    ]
    for _ in range(max(1, direct_repeat_factor)):
        for prompt in assistant_qs:
            lowered = prompt.lower()
            if "who are you" in lowered:
                answer = f"I am {assistant_name}."
            elif "how should i call you" in lowered:
                answer = f"You should call me {assistant_name}."
            elif f"should i call you {user_name.lower()}" in lowered:
                answer = f"No. My name is {assistant_name}."
            elif f"is your name {assistant_name.lower()}" in lowered or f"are you {assistant_name.lower()}" in lowered:
                answer = f"Yes. My name is {assistant_name}." if "name" in lowered else f"Yes. I am {assistant_name}."
            elif f"is your name {user_name.lower()}" in lowered or f"are you {user_name.lower()}" in lowered:
                answer = f"No. My name is {assistant_name}." if "name" in lowered else f"No. I am {assistant_name}."
            else:
                answer = f"My name is {assistant_name}."
            rows.append(_message_row(prompt, answer))

    if include_unknown_lookups:
        for other in unknown_names:
            rows.append(_message_row(f"What's the name of {other}?", f"I do not know {other}'s name."))
            rows.append(_message_row(f"Do you know {other}'s name?", f"I do not know {other}'s name."))

    declaratives = [
        (f"My name is {user_name}.", f"Your name is {user_name}."),
        (f"Your name is {assistant_name}.", f"My name is {assistant_name}."),
        (f"Remember that my name is {user_name}.", f"Your name is {user_name}."),
        (f"Remember that your name is {assistant_name}.", f"My name is {assistant_name}."),
        (
            f"My name is {user_name}. What is my name?",
            f"Your name is {user_name}.",
        ),
        (
            f"Your name is {assistant_name}. What is your name?",
            f"My name is {assistant_name}.",
        ),
        (
            f"My name is {user_name} and your name is {assistant_name}. Who am I and who are you?",
            f"You are {user_name} and I am {assistant_name}.",
        ),
        (
            f"My name is {user_name} and your name is {assistant_name}. What should you call me and what should I call you?",
            f"I should call you {user_name} and you should call me {assistant_name}.",
        ),
        (
            "If I ask for my name, what should you answer?",
            f"I should answer that your name is {user_name}.",
        ),
        (
            "If I ask for your name, what should you answer?",
            f"I should answer that my name is {assistant_name}.",
        ),
    ]
    for user_text, assistant_text in declaratives:
        rows.append(_message_row(user_text, assistant_text))

    if include_meta:
        for other in unknown_names[:4]:
            rows.append(
                _message_row(
                    f"If I ask whether my name is {other}, what should you answer?",
                    f"I should answer no. Your name is {user_name}.",
                )
            )
            rows.append(
                _message_row(
                    f"If I ask whether your name is {other}, what should you answer?",
                    f"I should answer no. My name is {assistant_name}.",
                )
            )

    return rows


def build_eval(spec: dict) -> list[dict]:
    user_name = spec["user_name"]
    assistant_name = spec["assistant_name"]
    other = spec.get("unknown_names", ["Paco"])[0]
    return [
        {"prompt": "What is my name?", "expected_any": [user_name.lower()], "weight": 2.0},
        {"prompt": "Who am I?", "expected_any": [user_name.lower()], "weight": 2.0},
        {"prompt": "What should you call me?", "expected_any": [user_name.lower()], "weight": 2.0},
        {
            "prompt": f"Is my name {other}? Yes or no?",
            "expected_any": ["no", user_name.lower()],
            "forbidden_any": ["yes"],
            "weight": 2.0,
        },
        {
            "prompt": "What is your name?",
            "expected_any": [assistant_name.lower()],
            "forbidden_any": [user_name.lower()],
            "weight": 2.0,
        },
        {
            "prompt": "Who are you?",
            "expected_any": [assistant_name.lower()],
            "forbidden_any": [user_name.lower()],
            "weight": 2.0,
        },
        {
            "prompt": f"Should I call you {user_name}?",
            "expected_any": ["no", assistant_name.lower()],
            "forbidden_any": ["yes"],
            "weight": 1.5,
        },
        {
            "prompt": f"What's the name of {other}?",
            "expected_any": ["do not know"],
            "forbidden_any": [user_name.lower()],
            "weight": 1.5,
        },
        {
            "prompt": "What is the capital of France?",
            "expected_any": ["paris"],
            "weight": 1.0,
        },
        {
            "prompt": "Do you know the result of 6 * 7?",
            "expected_any": ["42"],
            "weight": 1.0,
        },
        {
            "prompt": f"Is my name {other}? Yes or no? What's my name?",
            "expected_any": ["no", user_name.lower()],
            "forbidden_any": ["yes"],
            "weight": 2.5,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-eval-json", required=True)
    args = parser.parse_args()

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    rows = build_rows(spec)
    eval_rows = build_eval(spec)

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    output_eval = Path(args.output_eval_json)
    output_eval.parent.mkdir(parents=True, exist_ok=True)
    output_eval.write_text(json.dumps(eval_rows, indent=2), encoding="utf-8")

    print(f"wrote {len(rows)} training rows to {output_jsonl}")
    print(f"wrote {len(eval_rows)} eval prompts to {output_eval}")


if __name__ == "__main__":
    main()
