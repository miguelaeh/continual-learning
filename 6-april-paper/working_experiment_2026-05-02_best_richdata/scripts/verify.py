#!/usr/bin/env python3
"""Verify sparse memory checkpoints by comparing loss and generation."""

from __future__ import annotations

import argparse
import json

from smf_retrofit.config import load_experiment_config
from smf_retrofit.eval import (
    build_model,
    compare_models_on_loss,
    generate_text,
)
from smf_retrofit.utils import configure_logging, detect_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify sparse memory checkpoints.")
    parser.add_argument("--config", default="configs/continual.yaml")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional override for evaluation data. Defaults to config data.path.",
    )
    parser.add_argument(
        "--continual-checkpoint",
        default="checkpoints/continual/memory.pt",
    )
    parser.add_argument(
        "--recovery-checkpoint",
        default="checkpoints/recovery/memory.pt",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to generate from. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    cfg = load_experiment_config(args.config)
    set_seed(cfg.data.seed)

    device = detect_device(cfg.model.device_map) or "cpu"
    base_model, tokenizer = build_model(cfg, checkpoint=None)
    recovery_model, _ = build_model(cfg, checkpoint=args.recovery_checkpoint)
    continual_model, _ = build_model(cfg, checkpoint=args.continual_checkpoint)
    loss_results = compare_models_on_loss(
        cfg=cfg,
        device=device,
        eval_data_path=args.data_path,
        max_batches=args.max_batches,
        recovery_checkpoint=args.recovery_checkpoint,
        continual_checkpoint=args.continual_checkpoint,
    )

    results = {
        "loss": loss_results,
        "generations": [],
    }

    prompts = args.prompt or [
        "What is my name?",
        "Introduce yourself briefly.",
    ]
    for prompt in prompts:
        results["generations"].append(
            {
                "prompt": prompt,
                "base": generate_text(base_model, tokenizer, prompt, device, args.max_new_tokens),
                "recovery": generate_text(
                    recovery_model, tokenizer, prompt, device, args.max_new_tokens
                ),
                "continual": generate_text(
                    continual_model, tokenizer, prompt, device, args.max_new_tokens
                ),
            }
        )

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print("Loss comparison")
    print(f"  base:      {results['loss']['base']:.4f}")
    print(f"  recovery:  {results['loss']['recovery']:.4f}")
    print(f"  continual: {results['loss']['continual']:.4f}")
    print()
    print("Generation comparison")
    for item in results["generations"]:
        print("- prompt:")
        print(item["prompt"])
        print("  base:")
        print(item["base"])
        print("  recovery:")
        print(item["recovery"])
        print("  continual:")
        print(item["continual"])
        print()


if __name__ == "__main__":
    main()
