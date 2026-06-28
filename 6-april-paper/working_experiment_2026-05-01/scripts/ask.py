#!/usr/bin/env python3
"""Ask the base, recovery, or continual model a single prompt."""

from __future__ import annotations

import argparse

import torch

from smf_retrofit.config import load_experiment_config
from smf_retrofit.modeling.qwen import (
    inject_memory_layers,
    load_memory_checkpoint,
    load_model_and_tokenizer,
)
from smf_retrofit.utils import configure_logging, detect_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a sparse-memory model checkpoint.")
    parser.add_argument("--config", default="configs/continual.yaml")
    parser.add_argument(
        "--mode",
        choices=["base", "recovery", "continual"],
        default="continual",
        help="Which model state to load.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit checkpoint path. Overrides --mode for non-base models.",
    )
    parser.add_argument(
        "--prompt",
        default="What is my name?",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Use the prompt as-is instead of wrapping it with the tokenizer chat template.",
    )
    return parser.parse_args()


def resolve_checkpoint(args: argparse.Namespace) -> str | None:
    if args.checkpoint:
        return args.checkpoint
    if args.mode == "recovery":
        return "checkpoints/recovery/memory.pt"
    if args.mode == "continual":
        return "checkpoints/continual/memory.pt"
    return None


@torch.no_grad()
def main() -> None:
    args = parse_args()
    configure_logging()
    cfg = load_experiment_config(args.config)
    device = detect_device(cfg.model.device_map) or "cpu"

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    checkpoint = resolve_checkpoint(args)

    if checkpoint is not None:
        model, _, _ = inject_memory_layers(model, cfg.memory)
        load_memory_checkpoint(model, checkpoint)

    model.eval()
    prompt_text = args.prompt
    if not args.raw_prompt and hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()
