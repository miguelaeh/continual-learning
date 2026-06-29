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

    totals = {"dataload": 0.0, "to_device": 0.0, "forward": 0.0, "backward": 0.0, "optim": 0.0}
    n = 0

    for step in range(args.warmup + args.steps):
        record = step >= args.warmup

        cuda_sync(device)
        t0 = time.perf_counter()
        batch = next(it)
        cuda_sync(device)
        t1 = time.perf_counter()

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        cuda_sync(device)
        t2 = time.perf_counter()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        cuda_sync(device)
        t3 = time.perf_counter()

        loss.backward()
        cuda_sync(device)
        t4 = time.perf_counter()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        cuda_sync(device)
        t5 = time.perf_counter()

        if record:
            totals["dataload"] += t1 - t0
            totals["to_device"] += t2 - t1
            totals["forward"] += t3 - t2
            totals["backward"] += t4 - t3
            totals["optim"] += t5 - t4
            n += 1

    print(f"Per-step averages over {n} steps (ms):")
    grand = sum(totals.values())
    for name, total in totals.items():
        ms = 1000 * total / n
        pct = 100 * total / grand
        print(f"  {name:10s}: {ms:7.1f} ms  ({pct:4.1f}%)")
    print(f"  {'TOTAL':10s}: {1000 * grand / n:7.1f} ms")


if __name__ == "__main__":
    main()
