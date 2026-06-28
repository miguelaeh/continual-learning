#!/usr/bin/env python3
"""Check whether recovery preserved enough of the base model to proceed."""

from __future__ import annotations

import argparse
import json
import sys

from smf_retrofit.config import load_experiment_config
from smf_retrofit.eval import (
    build_model,
    compare_models_on_loss,
    evaluate_prompt_specs,
    generate_text,
    load_prompt_specs,
    recovery_gate_status,
)
from smf_retrofit.utils import configure_logging, detect_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check recovery quality against base.")
    parser.add_argument("--config", default="configs/recovery.yaml")
    parser.add_argument(
        "--recovery-checkpoint",
        default=None,
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Optional evaluation corpus override.",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--prompts-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    cfg = load_experiment_config(args.config)
    set_seed(cfg.data.seed)
    device = detect_device(cfg.model.device_map) or "cpu"
    recovery_checkpoint = args.recovery_checkpoint or f"{cfg.recovery.output_dir}/memory.pt"

    eval_data_path = args.data_path or cfg.recovery.eval_data_path
    max_batches = args.max_batches or cfg.recovery.eval_max_batches
    prompts_path = args.prompts_path or cfg.recovery.sanity_prompts_path
    prompt_specs = load_prompt_specs(prompts_path)

    losses = compare_models_on_loss(
        cfg=cfg,
        device=device,
        eval_data_path=eval_data_path,
        max_batches=max_batches,
        recovery_checkpoint=recovery_checkpoint,
    )
    gate = recovery_gate_status(
        base_loss=losses["base"],
        recovery_loss=losses["recovery"],
        max_loss_delta_vs_base=cfg.recovery.max_loss_delta_vs_base,
        max_loss_ratio_vs_base=cfg.recovery.max_loss_ratio_vs_base,
    )

    base_model, tokenizer = build_model(cfg, checkpoint=None)
    recovery_model, _ = build_model(cfg, checkpoint=recovery_checkpoint)
    generations = evaluate_prompt_specs(
        base_model=base_model,
        recovery_model=recovery_model,
        tokenizer=tokenizer,
        prompt_specs=prompt_specs,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )
    prompt_checks_passed = (
        all(item["passed"] for item in generations if item["expected_any"])
        if cfg.recovery.require_prompt_checks
        else True
    )
    gate = recovery_gate_status(
        base_loss=losses["base"],
        recovery_loss=losses["recovery"],
        max_loss_delta_vs_base=cfg.recovery.max_loss_delta_vs_base,
        max_loss_ratio_vs_base=cfg.recovery.max_loss_ratio_vs_base,
        prompt_checks_passed=prompt_checks_passed,
        prompt_checks_summary=generations,
    )
    results = {"loss": losses, "gate": gate, "generations": generations}
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("Recovery gate")
        print(f"  passed:    {gate['passed']}")
        print(f"  loss ok:   {gate['loss_checks_passed']}")
        print(f"  prompts ok:{gate['prompt_checks_passed']}")
        print(f"  base loss: {losses['base']:.4f}")
        print(f"  rec loss:  {losses['recovery']:.4f}")
        print(f"  delta:     {gate['loss_delta_vs_base']:.4f}")
        print(f"  ratio:     {gate['loss_ratio_vs_base']:.4f}")
        print()
        print("Generation comparison")
        for item in generations:
            print("- prompt:")
            print(item["prompt"])
            if item["expected_any"]:
                print(f"  expected any: {item['expected_any']}")
                print(f"  passed: {item['passed']}")
            print("  base:")
            print(item["base"])
            print("  recovery:")
            print(item["recovery"])
            print()

    if not gate["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
