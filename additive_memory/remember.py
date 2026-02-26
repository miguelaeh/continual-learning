#!/usr/bin/env python3
"""Remember facts via additive memory — no pretraining needed.

Usage:
    # Remember facts directly
    python additive_memory/remember.py \\
        --facts "My name is Miguel" "I work on continual learning"

    # From a text file (one fact per line)
    python additive_memory/remember.py --facts-file facts.txt

    # Load existing memory and add more facts
    python additive_memory/remember.py \\
        --facts "My favorite color is blue" \\
        --checkpoint checkpoints/additive_memory/memory.pt

    # Use a smaller model for testing
    python additive_memory/remember.py \\
        --facts "The sky is green" \\
        --base-model google/gemma-3-1b-it \\
        --layers 9 17 --n-keys 256
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.model.memory_gemma import load_base_model, load_tokenizer
from src.remember.fact_dataset import FactDataset
from src.utils import get_device

from additive_memory.model import (
    AdditiveMemoryConfig,
    freeze_for_remember,
    inject_additive_memory,
    load_memory_checkpoint,
    save_memory_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_remember(model, config, facts, tokenizer, device):
    """Run sparse SGD to remember facts.

    No gradient masking needed — EmbeddingBag gradients are naturally sparse
    (only accessed rows get non-zero gradients) and memory starts empty
    (nothing to protect).

    Returns:
        Number of training steps completed.
    """
    dataset = FactDataset(
        passages=facts,
        tokenizer=tokenizer,
        seq_length=config.seq_length,
        repeat_factor=config.repeat_factor,
    )
    dataloader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)

    logger.info(
        f"Training on {len(dataset)} samples "
        f"({len(facts)} facts x {config.repeat_factor} repeats)"
    )

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        momentum=config.momentum,
    )

    model.train()
    step = 0
    running_loss = 0.0

    while step < config.finetuning_steps:
        for batch in dataloader:
            if step >= config.finetuning_steps:
                break

            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = torch.zeros_like(input_ids)

            outputs = model(
                input_ids=input_ids, labels=labels, token_type_ids=token_type_ids
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % 10 == 0:
                avg = running_loss / 10
                logger.info(f"  Step {step}/{config.finetuning_steps} | Loss: {avg:.4f}")
                running_loss = 0.0

    return step


def main():
    parser = argparse.ArgumentParser(
        description="Remember facts via additive memory (no pretraining needed)"
    )

    # Input sources
    parser.add_argument("--facts", nargs="+", type=str, help="Facts to remember")
    parser.add_argument(
        "--facts-file", type=str, help="Text file with one fact per line"
    )

    # Model
    parser.add_argument(
        "--base-model", default="google/gemma-3-4b-it", help="Base model name"
    )
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")

    # Memory architecture
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[9, 17, 25],
        help="Layer indices for memory injection",
    )
    parser.add_argument("--n-keys", type=int, default=1024, help="Sub-keys per half")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of memory heads")
    parser.add_argument("--top-k", type=int, default=32, help="Top-k entries per head")
    parser.add_argument("--v-dim", type=int, default=1024, help="Value dimension")
    parser.add_argument(
        "--k-dim-per-head", type=int, default=512, help="Key dimension per head"
    )

    # Training
    parser.add_argument("--steps", type=int, default=100, help="Finetuning steps")
    parser.add_argument("--lr", type=float, default=2.0, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.0, help="SGD momentum")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument(
        "--repeat-factor", type=int, default=8, help="Dataset repeat factor"
    )

    # Checkpoints
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Load existing memory checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/additive_memory",
        help="Output directory",
    )

    args = parser.parse_args()

    # Gather facts
    all_facts = []
    if args.facts:
        all_facts.extend(args.facts)
    if args.facts_file:
        path = Path(args.facts_file)
        lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
        logger.info(f"Loaded {len(lines)} facts from {args.facts_file}")
        all_facts.extend(lines)

    if not all_facts:
        parser.error("Provide at least one of: --facts or --facts-file")

    logger.info(f"Facts to remember ({len(all_facts)}):")
    for i, fact in enumerate(all_facts):
        logger.info(f"  [{i + 1}] {fact}")

    # Build config
    config = AdditiveMemoryConfig(
        base_model=args.base_model,
        dtype=args.dtype,
        memory_layers=args.layers,
        num_heads=args.num_heads,
        n_keys=args.n_keys,
        k_dim_per_head=args.k_dim_per_head,
        v_dim=args.v_dim,
        top_k=args.top_k,
        learning_rate=args.lr,
        momentum=args.momentum,
        finetuning_steps=args.steps,
        seq_length=args.seq_length,
        repeat_factor=args.repeat_factor,
        checkpoint_dir=args.output,
        memory_checkpoint=args.checkpoint,
    )

    device = get_device()
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model: {config.base_model}")
    tokenizer = load_tokenizer(config.base_model)
    model = load_base_model(config.base_model, dtype=config.dtype, device_map=device)

    # Inject additive memory
    model, shared_store = inject_additive_memory(model, config)

    # Load existing checkpoint if provided
    if config.memory_checkpoint:
        load_memory_checkpoint(model, shared_store, config, config.memory_checkpoint)

    # Freeze everything except memory values
    freeze_for_remember(model, config)

    # Remember
    logger.info("=" * 60)
    logger.info("REMEMBERING")
    logger.info("=" * 60)

    steps = run_remember(model, config, all_facts, tokenizer, device)

    # Save checkpoint
    output_dir = Path(config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(output_dir / "memory.pt")
    save_memory_checkpoint(model, shared_store, config, save_path, step=steps)

    # Save facts for reference
    facts_path = str(output_dir / "facts.json")
    with open(facts_path, "w") as f:
        json.dump({"facts": all_facts, "steps": steps}, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Done! Remembered {len(all_facts)} facts in {steps} steps.")
    logger.info(f"Checkpoint: {save_path}")
    logger.info(f"Facts log:  {facts_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
