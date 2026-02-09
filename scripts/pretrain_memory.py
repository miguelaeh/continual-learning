#!/usr/bin/env python3
"""Phase 1: Pretrain memory layers with base Gemma 3 4B frozen.

Usage:
    python scripts/pretrain_memory.py --config configs/pretrain_memory.yaml

This script:
1. Loads Gemma 3 4B IT (text-only CausalLM)
2. Injects Memory+ layers at specified positions [9, 17, 25]
3. Freezes all base model parameters
4. Trains memory layer parameters on FineWeb-Edu
5. Saves memory layer checkpoints
"""

import argparse
import logging
import sys

import torch

from src.config import load_config, make_pretrain_config
from src.data.datasets import create_pretrain_dataloader
from src.model.freeze_utils import freeze_base_model
from src.model.memory_gemma import inject_memory_layers, load_base_model, load_tokenizer
from src.training.pretrain_trainer import pretrain_memory_layers
from src.utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Pretrain memory layers")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_memory.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    memory_config, train_config = make_pretrain_config(cfg)

    logger.info(f"Memory layers at: {memory_config.memory_layers}")
    logger.info(f"Memory entries: {memory_config.num_entries:,}")
    logger.info(f"Dataset: {train_config.dataset}")
    logger.info(f"Total steps: {train_config.total_steps:,}")

    # Determine device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = load_tokenizer(train_config.base_model)

    # Load base model
    logger.info(f"Loading base model: {train_config.base_model}")
    model = load_base_model(
        train_config.base_model,
        dtype=train_config.dtype,
        device_map=device,
    )

    # Inject memory layers (automatically placed on same device/dtype as base model)
    model, shared_store = inject_memory_layers(model, memory_config)

    # Freeze base model, keep memory layers trainable
    freeze_base_model(model, memory_config)

    # Create dataloader
    dataloader = create_pretrain_dataloader(
        dataset_name=train_config.dataset,
        tokenizer=tokenizer,
        batch_size=train_config.batch_size,
        seq_length=train_config.seq_length,
        dataset_subset=train_config.dataset_subset,
        seed=train_config.seed,
    )

    # Train
    pretrain_memory_layers(
        model=model,
        shared_store=shared_store,
        memory_config=memory_config,
        train_config=train_config,
        dataloader=dataloader,
        device=device,
    )

    logger.info("Phase 1 complete!")


if __name__ == "__main__":
    main()
