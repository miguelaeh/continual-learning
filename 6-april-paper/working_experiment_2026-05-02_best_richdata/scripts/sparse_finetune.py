#!/usr/bin/env python3
"""Stage 3 sparse memory finetuning."""

from __future__ import annotations

import argparse
import copy
import sys

import torch

from smf_retrofit.config import TextDataConfig, load_experiment_config
from smf_retrofit.data import create_lm_dataloader
from smf_retrofit.eval import (
    build_model,
    compare_models_on_loss,
    evaluate_prompt_specs,
    load_prompt_specs,
    recovery_gate_status,
)
from smf_retrofit.modeling.qwen import (
    freeze_for_sparse_finetuning,
    inject_memory_layers,
    load_memory_checkpoint,
    load_model_and_tokenizer,
)
from smf_retrofit.training.sparse_finetune import run_sparse_finetuning
from smf_retrofit.utils import configure_logging, detect_device, set_seed


def _resolve_train_memory_mode(cfg) -> str:
    if cfg.continual.train_memory_mode:
        return cfg.continual.train_memory_mode
    if cfg.continual.train_full_memory:
        return "full_memory"
    return "values_only"


def _build_replay_data_config(cfg) -> TextDataConfig | None:
    if (
        cfg.continual.replay_weight <= 0
        or (cfg.continual.replay_data_path is None and cfg.continual.replay_dataset_name is None)
    ):
        return None

    replay_cfg = copy.deepcopy(cfg.data)
    replay_cfg.path = cfg.continual.replay_data_path
    replay_cfg.dataset_name = cfg.continual.replay_dataset_name
    replay_cfg.dataset_config = cfg.continual.replay_dataset_config
    replay_cfg.split = cfg.continual.replay_split
    replay_cfg.text_field = cfg.continual.replay_text_field
    replay_cfg.seq_length = cfg.continual.replay_seq_length
    replay_cfg.batch_size = cfg.continual.replay_batch_size
    replay_cfg.pack_samples = cfg.continual.replay_pack_samples
    replay_cfg.max_samples = cfg.continual.replay_max_samples
    replay_cfg.shuffle = cfg.continual.replay_shuffle
    return replay_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sparse memory finetuning.")
    parser.add_argument("--config", default="configs/continual.yaml")
    args = parser.parse_args()

    configure_logging()
    cfg = load_experiment_config(args.config)
    set_seed(cfg.data.seed)
    device = detect_device(cfg.model.device_map) or "cpu"

    if cfg.continual.require_recovery_pass:
        losses = compare_models_on_loss(
            cfg=cfg,
            device=device,
            eval_data_path=cfg.continual.recovery_eval_data_path or cfg.recovery.eval_data_path,
            max_batches=cfg.continual.recovery_eval_max_batches,
            recovery_checkpoint=cfg.continual.recovery_checkpoint,
        )
        prompt_specs = load_prompt_specs(cfg.recovery.sanity_prompts_path)
        base_model, tokenizer = build_model(cfg, checkpoint=None)
        recovery_model, _ = build_model(cfg, checkpoint=cfg.continual.recovery_checkpoint)
        prompt_results = evaluate_prompt_specs(
            base_model=base_model,
            recovery_model=recovery_model,
            tokenizer=tokenizer,
            prompt_specs=prompt_specs,
            device=device,
            max_new_tokens=48,
        )
        prompt_checks_passed = (
            all(item["passed"] for item in prompt_results if item["expected_any"])
            if cfg.continual.require_recovery_prompt_checks
            else True
        )
        gate = recovery_gate_status(
            base_loss=losses["base"],
            recovery_loss=losses["recovery"],
            max_loss_delta_vs_base=cfg.continual.recovery_max_loss_delta_vs_base,
            max_loss_ratio_vs_base=cfg.continual.recovery_max_loss_ratio_vs_base,
            prompt_checks_passed=prompt_checks_passed,
            prompt_checks_summary=prompt_results,
        )
        if not gate["passed"]:
            print(
                "Refusing to run continual finetuning because recovery quality is below the configured threshold.\n"
                f"base loss={losses['base']:.4f}, recovery loss={losses['recovery']:.4f}, "
                f"delta={gate['loss_delta_vs_base']:.4f}, ratio={gate['loss_ratio_vs_base']:.4f}, "
                f"prompt_checks_passed={gate['prompt_checks_passed']}\n"
                "Run scripts/check_recovery.py to inspect the failure, then improve the recovery dataset or steps.",
                file=sys.stderr,
            )
            sys.exit(1)

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    model, _, layer_indices = inject_memory_layers(model, cfg.memory)
    load_memory_checkpoint(model, cfg.continual.recovery_checkpoint, layer_indices=layer_indices)
    train_memory_mode = _resolve_train_memory_mode(cfg)
    freeze_for_sparse_finetuning(
        model,
        layer_indices,
        train_memory_mode=train_memory_mode,
    )

    dataloader = create_lm_dataloader(cfg.data, tokenizer)
    replay_cfg = _build_replay_data_config(cfg)
    replay_dataloader = (
        create_lm_dataloader(replay_cfg, tokenizer) if replay_cfg is not None else None
    )
    replay_teacher_model = None
    if replay_dataloader is not None and cfg.continual.replay_distill_to_recovery:
        replay_teacher_model, _ = build_model(
            cfg,
            checkpoint=cfg.continual.recovery_checkpoint,
        )
        replay_teacher_model.eval()
    background_stats = torch.load(
        cfg.continual.background_stats_path,
        map_location="cpu",
        weights_only=True,
    )
    run_sparse_finetuning(
        model=model,
        dataloader=dataloader,
        config=cfg.continual,
        background_stats=background_stats,
        layer_indices=layer_indices,
        num_entries=cfg.memory.num_entries,
        device=device,
        replay_dataloader=replay_dataloader,
        replay_teacher_model=replay_teacher_model,
    )


if __name__ == "__main__":
    main()
