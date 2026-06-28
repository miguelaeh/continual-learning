#!/usr/bin/env python3
"""Evaluate memory checkpoints on MMLU Pro via next-token logit scoring."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset

from smf_retrofit.config import load_experiment_config
from smf_retrofit.eval import build_model, format_prompt
from smf_retrofit.modeling.qwen import load_memory_checkpoint
from smf_retrofit.utils import detect_device

OPTION_LETTERS = "ABCDEFGHIJ"


def _format_mmlu_pro_prompt(question: str, options: list[str]) -> str:
    lines = [question, ""]
    for i, opt in enumerate(options):
        lines.append(f"{OPTION_LETTERS[i]}. {opt}")
    lines.append("")
    lines.append("Answer with the letter only.")
    return "\n".join(lines)


@torch.no_grad()
def _score_example(model, tokenizer, question: str, options: list[str], device: str) -> int:
    """Return the predicted answer index (0-based) via logit scoring."""
    prompt = _format_mmlu_pro_prompt(question, options)
    prompt_text = format_prompt(tokenizer, prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    # logits for the next token after the full prompt
    next_token_logits = outputs.logits[0, -1, :]

    option_token_ids = []
    for letter in OPTION_LETTERS[: len(options)]:
        # encode with and without leading space — pick whichever exists as a single token
        ids_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
        ids_plain = tokenizer.encode(letter, add_special_tokens=False)
        # prefer the single-token representation; fall back to first token if multi
        token_id = ids_space[0] if len(ids_space) == 1 else ids_plain[0]
        option_token_ids.append(token_id)

    scores = torch.tensor(
        [next_token_logits[tid].item() for tid in option_token_ids],
        dtype=torch.float32,
    )
    return int(scores.argmax().item())


def _step_key(path: Path) -> tuple[int, str]:
    match = re.search(r"memory_step_(\d+)\.pt$", path.name)
    if match:
        return (int(match.group(1)), path.name)
    if path.name == "memory.pt":
        return (10**9, path.name)
    return (-1, path.name)


def _resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint:
        return [Path(c) for c in args.checkpoint]
    checkpoints_dir = Path(args.checkpoints_dir)
    checkpoints = sorted(checkpoints_dir.glob("memory*.pt"), key=_step_key)
    if args.steps:
        wanted = {f"memory_step_{s}.pt" for s in args.steps}
        checkpoints = [p for p in checkpoints if p.name in wanted]
    if not checkpoints:
        raise ValueError("No checkpoints matched the requested selection.")
    return checkpoints


def _evaluate_checkpoint(model, tokenizer, examples: list[dict], device: str) -> dict:
    model.eval()
    correct = 0
    per_category: dict[str, list[bool]] = defaultdict(list)

    for i, ex in enumerate(examples):
        pred_idx = _score_example(
            model, tokenizer, ex["question"], ex["options"], device
        )
        is_correct = pred_idx == ex["answer_index"]
        correct += int(is_correct)
        per_category[ex["category"]].append(is_correct)
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(examples)}] running acc: {correct / (i + 1):.3f}")

    overall = correct / len(examples)
    category_acc = {
        cat: sum(hits) / len(hits) for cat, hits in sorted(per_category.items())
    }
    return {"overall": overall, "n": len(examples), "per_category": category_acc}


def main() -> None:
    parser = argparse.ArgumentParser(description="MMLU Pro evaluation for memory checkpoints.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument("--checkpoints-dir", default=None)
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="Explicit checkpoint path(s). Repeatable.")
    parser.add_argument("--steps", type=int, nargs="*", default=[])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap total examples (useful for quick smoke tests).")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Subset of categories to evaluate.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--also-eval-base", action="store_true",
                        help="Also score the base model (no memory) for comparison.")
    parser.add_argument("--base-only", action="store_true",
                        help="Evaluate only the base model (no memory checkpoint needed).")
    args = parser.parse_args()

    if args.base_only:
        args.also_eval_base = True
    if not args.base_only and not args.checkpoint and not args.checkpoints_dir:
        raise ValueError("Provide --checkpoint, --checkpoints-dir, or --base-only.")

    cfg = load_experiment_config(args.config)
    device = args.device or detect_device(cfg.model.device_map) or "cpu"
    checkpoints = [] if args.base_only else _resolve_checkpoints(args)

    print("Loading MMLU Pro test split…")
    raw_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    examples = list(raw_ds)
    if args.categories:
        examples = [e for e in examples if e["category"] in args.categories]
    if args.max_samples:
        examples = examples[: args.max_samples]
    print(f"Evaluating on {len(examples)} examples.")

    if not args.base_only:
        model, tokenizer = build_model(cfg, checkpoint=str(checkpoints[0]))
        model = model.to(device)
    else:
        from smf_retrofit.modeling.qwen import load_model_and_tokenizer
        _, tokenizer = load_model_and_tokenizer(cfg.model)

    all_results = []

    if args.also_eval_base:
        from smf_retrofit.modeling.qwen import load_model_and_tokenizer
        base_model, _ = load_model_and_tokenizer(cfg.model)
        base_model = base_model.to(device)
        print("\n=== BASE MODEL (no memory) ===")
        base_stats = _evaluate_checkpoint(base_model, tokenizer, examples, device)
        print(f"Base overall accuracy: {base_stats['overall']:.4f} ({base_stats['n']} examples)")
        for cat, acc in base_stats["per_category"].items():
            print(f"  {cat}: {acc:.3f}")
        all_results.append({"checkpoint": "base", **base_stats})
        del base_model
        torch.cuda.empty_cache() if device.startswith("cuda") else None

    for ckpt in checkpoints:
        load_memory_checkpoint(model, str(ckpt))
        print(f"\n=== {ckpt.name} ===")
        stats = _evaluate_checkpoint(model, tokenizer, examples, device)
        print(f"Overall accuracy: {stats['overall']:.4f} ({stats['n']} examples)")
        for cat, acc in stats["per_category"].items():
            print(f"  {cat}: {acc:.3f}")
        all_results.append({"checkpoint": str(ckpt), **stats})

    non_base = [r for r in all_results if r["checkpoint"] != "base"]
    if non_base:
        best = max(non_base, key=lambda r: r["overall"])
        print(
            f"\nBest checkpoint: {Path(best['checkpoint']).name} "
            f"| accuracy {best['overall']:.4f}"
        )

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(all_results, indent=2), encoding="utf-8"
        )
        print(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()
