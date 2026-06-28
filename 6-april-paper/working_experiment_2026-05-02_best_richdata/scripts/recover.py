#!/usr/bin/env python3
"""Stage 2 recovery/healing for retrofitted sparse memory layers."""

from __future__ import annotations

import argparse
import sys

from smf_retrofit.config import load_experiment_config
from smf_retrofit.data import create_lm_dataloader
from smf_retrofit.eval import recovery_gate_status
from smf_retrofit.modeling.qwen import (
    freeze_for_recovery,
    inject_memory_layers,
    load_model_and_tokenizer,
)
from smf_retrofit.training.recovery import evaluate_recovery_checkpoint, run_recovery
from smf_retrofit.utils import configure_logging, detect_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the recovery/healing stage.")
    parser.add_argument("--config", default="configs/recovery.yaml")
    args = parser.parse_args()

    configure_logging()
    cfg = load_experiment_config(args.config)
    set_seed(cfg.data.seed)

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    model, _, layer_indices = inject_memory_layers(model, cfg.memory)
    freeze_for_recovery(model, layer_indices)

    dataloader = create_lm_dataloader(cfg.data, tokenizer)
    device = detect_device(cfg.model.device_map) or "cpu"
    checkpoint_path = run_recovery(model, dataloader, cfg.recovery, layer_indices, device=device)
    evaluation = evaluate_recovery_checkpoint(
        cfg=cfg,
        recovery_checkpoint=checkpoint_path,
        device=device,
    )
    losses = evaluation["losses"]
    prompt_results = evaluation["prompt_results"]
    prompt_checks_passed = (
        all(item["passed"] for item in prompt_results if item["expected_any"])
        if cfg.recovery.require_prompt_checks
        else True
    )
    gate = recovery_gate_status(
        base_loss=losses["base"],
        recovery_loss=losses["recovery"],
        max_loss_delta_vs_base=cfg.recovery.max_loss_delta_vs_base,
        max_loss_ratio_vs_base=cfg.recovery.max_loss_ratio_vs_base,
        prompt_checks_passed=prompt_checks_passed,
        prompt_checks_summary=prompt_results,
    )
    print("Recovery gate")
    print(f"  passed:    {gate['passed']}")
    print(f"  loss ok:   {gate['loss_checks_passed']}")
    print(f"  prompts ok:{gate['prompt_checks_passed']}")
    print(f"  base loss: {losses['base']:.4f}")
    print(f"  rec loss:  {losses['recovery']:.4f}")
    print(f"  delta:     {gate['loss_delta_vs_base']:.4f}")
    print(f"  ratio:     {gate['loss_ratio_vs_base']:.4f}")
    for item in prompt_results:
        if item["expected_any"]:
            print(
                f"  prompt check: {item['prompt']!r} -> {item['passed']} "
                f"(expected any of {item['expected_any']})"
            )
    if not gate["passed"]:
        print(
            "Recovery quality is below the configured threshold. "
            "Use a broader recovery corpus or more steps before continual learning.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
