#!/usr/bin/env python3
"""Profile one recovery step to find the actual bottleneck.

Usage: python scripts/profile_recovery.py --config <recovery.yaml>
"""

from __future__ import annotations

import argparse
import time

import torch

from smf_retrofit.config import load_experiment_config
from smf_retrofit.data import create_lm_dataloader
from smf_retrofit.modeling.qwen import (
    freeze_for_recovery,
    get_memory_layers,
    inject_memory_layers,
    load_model_and_tokenizer,
)
from smf_retrofit.utils import configure_logging, detect_device, set_seed


def cuda_sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    configure_logging()
    cfg = load_experiment_config(args.config)
    set_seed(cfg.data.seed)

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    model, _, layer_indices = inject_memory_layers(model, cfg.memory)
    freeze_for_recovery(model, layer_indices)
    device = detect_device(cfg.model.device_map) or "cpu"
    model.train()

    # Report where the values table actually lives — this is the thing we
    # tried to keep on GPU. If it says cpu on a cuda run, that's the bug.
    store = get_memory_layers(model, layer_indices)[0].shared_store
    print(f"device            : {device}")
    print(f"model first param : {next(model.parameters()).device}")
    print(f"values.weight     : {store.values.weight.device} ({store.values.weight.dtype})")
    print(f"memory layers     : {layer_indices}")
    print(f"batch_size        : {cfg.data.batch_size}, seq_len: {cfg.data.seq_length}")
    print()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.recovery.learning_rate
    )

    dataloader = create_lm_dataloader(cfg.data, tokenizer)
    it = iter(dataloader)

    def run_step():
        batch = next(it)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Warmup
    for _ in range(args.warmup):
        run_step()
    cuda_sync(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
        for _ in range(args.steps):
            run_step()
        cuda_sync(device)

    sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=25))


if __name__ == "__main__":
    main()
