#!/usr/bin/env python3
"""Collect background slot-usage statistics after recovery."""

from __future__ import annotations

import argparse

from smf_retrofit.config import load_experiment_config
from smf_retrofit.data import create_lm_dataloader
from smf_retrofit.modeling.qwen import (
    inject_memory_layers,
    load_memory_checkpoint,
    load_model_and_tokenizer,
)
from smf_retrofit.training.background import collect_background_statistics
from smf_retrofit.utils import configure_logging, detect_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect background slot statistics.")
    parser.add_argument("--config", default="configs/background.yaml")
    args = parser.parse_args()

    configure_logging()
    cfg = load_experiment_config(args.config)
    set_seed(cfg.data.seed)

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    model, _, layer_indices = inject_memory_layers(model, cfg.memory)
    load_memory_checkpoint(model, cfg.continual.recovery_checkpoint, layer_indices=layer_indices)

    dataloader = create_lm_dataloader(cfg.data, tokenizer)
    device = detect_device(cfg.model.device_map) or "cpu"
    collect_background_statistics(
        model=model,
        dataloader=dataloader,
        layer_indices=layer_indices,
        num_entries=cfg.memory.num_entries,
        num_batches=cfg.background.num_batches,
        output_path=cfg.background.output_path,
        device=device,
    )


if __name__ == "__main__":
    main()
