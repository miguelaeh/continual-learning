#!/usr/bin/env python3
"""Evaluate memory checkpoints on a TriviaQA-style JSONL validation set."""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

from smf_retrofit.config import load_experiment_config
from smf_retrofit.eval import generate_text
from smf_retrofit.modeling.qwen import load_memory_checkpoint
from smf_retrofit.eval import build_model
from smf_retrofit.utils import detect_device


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, ground_truth: str) -> bool:
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _load_jsonl_examples(path: str, max_samples: int | None) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            user = next((item for item in messages if item.get("role") == "user"), None)
            assistant = next((item for item in messages if item.get("role") == "assistant"), None)
            if not user or not assistant:
                continue
            prompt = str(user.get("content", "")).strip()
            answer = str(assistant.get("content", "")).strip()
            if not prompt or not answer:
                continue
            examples.append({"prompt": prompt, "answer": answer})
            if max_samples is not None and len(examples) >= max_samples:
                break
    if not examples:
        raise ValueError(f"No valid examples found in {path}")
    return examples


def _assistant_answer(decoded: str) -> str:
    marker = "assistant\n"
    if marker in decoded:
        return decoded.rsplit(marker, 1)[-1].strip()
    return decoded.strip()


def _step_key(path: Path) -> tuple[int, str]:
    match = re.search(r"memory_step_(\d+)\.pt$", path.name)
    if match:
        return (int(match.group(1)), path.name)
    if path.name == "memory.pt":
        return (10**9, path.name)
    return (-1, path.name)


def _resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint:
        return [Path(item) for item in args.checkpoint]
    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoints = sorted(checkpoints_dir.glob("memory*.pt"), key=_step_key)
    if args.steps:
        wanted = {f"memory_step_{step}.pt" for step in args.steps}
        checkpoints = [path for path in checkpoints if path.name in wanted]
    if not checkpoints:
        raise ValueError("No checkpoints matched the requested selection.")
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TriviaQA EM/F1 for checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--steps", type=int, nargs="*", default=[])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    if not args.checkpoint and not args.checkpoints_dir:
        raise ValueError("Provide --checkpoint or --checkpoints-dir.")

    cfg = load_experiment_config(args.config)
    device = args.device or detect_device(cfg.model.device_map) or "cpu"
    examples = _load_jsonl_examples(args.data_path, args.max_samples)
    checkpoints = _resolve_checkpoints(args)

    model, tokenizer = build_model(cfg, checkpoint=str(checkpoints[0]))
    results = []
    for checkpoint in checkpoints:
        load_memory_checkpoint(model, str(checkpoint))
        rows = []
        exact_count = 0
        f1_total = 0.0
        contains_count = 0
        for example in examples:
            decoded = generate_text(
                model,
                tokenizer,
                example["prompt"],
                device=device,
                max_new_tokens=args.max_new_tokens,
            )
            prediction = _assistant_answer(decoded)
            gold = example["answer"]
            exact = _exact_match(prediction, gold)
            f1 = _token_f1(prediction, gold)
            contains = _normalize_answer(gold) in _normalize_answer(prediction)
            exact_count += int(exact)
            contains_count += int(contains)
            f1_total += f1
            rows.append(
                {
                    "prompt": example["prompt"],
                    "gold": gold,
                    "prediction": prediction,
                    "exact": exact,
                    "f1": f1,
                    "contains": contains,
                }
            )
        summary = {
            "checkpoint": str(checkpoint),
            "samples": len(examples),
            "exact_match": exact_count / len(examples),
            "contains": contains_count / len(examples),
            "f1": f1_total / len(examples),
            "rows": rows,
        }
        results.append(summary)
        print(
            f"{checkpoint.name}: EM {summary['exact_match']:.3f} | "
            f"contains {summary['contains']:.3f} | F1 {summary['f1']:.3f}"
        )

    best = max(results, key=lambda item: (item["f1"], item["exact_match"], item["contains"]))
    print(
        "best: "
        f"{Path(best['checkpoint']).name} | EM {best['exact_match']:.3f} | "
        f"contains {best['contains']:.3f} | F1 {best['f1']:.3f}"
    )

    if args.output_json:
        payload = {"best": best, "results": results}
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
