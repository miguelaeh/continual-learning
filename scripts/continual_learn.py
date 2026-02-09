#!/usr/bin/env python3
"""Phase 3: Continual learning via sparse memory finetuning.

Usage:
    python scripts/continual_learn.py --config configs/continual_learning.yaml

This script:
1. Loads Gemma 3 4B IT with pretrained memory layers
2. Loads IDF statistics from Phase 2
3. For each training batch:
   a. Forward pass with index tracking
   b. TF-IDF ranking to select top-t=500 trainable slots
   c. Apply gradient masking (straight-through trick)
   d. Backward pass (gradients only flow to selected slots)
   e. SGD update (lr=2.0, no momentum)
"""

import argparse
import logging
import sys

import torch

from src.config import load_config, make_continual_config
from src.data.datasets import create_continual_dataloader
from src.model.freeze_utils import freeze_for_continual_learning
from src.model.memory_gemma import (
    inject_memory_layers,
    load_base_model,
    load_memory_checkpoint,
    load_tokenizer,
)
from src.training.continual_trainer import continual_learn
from src.utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Continual learning via sparse memory finetuning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/continual_learning.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    memory_config, cl_config = make_continual_config(cfg)

    if not cl_config.dataset:
        logger.error(
            "No dataset specified for continual learning. "
            "Set 'dataset' in the config or pass a HuggingFace dataset name."
        )
        sys.exit(1)

    logger.info(f"Memory checkpoint: {cl_config.memory_checkpoint}")
    logger.info(f"IDF statistics: {cl_config.idf_statistics_path}")
    logger.info(f"Top-t slots: {cl_config.top_t}")
    logger.info(f"Optimizer: {cl_config.optimizer}, LR: {cl_config.learning_rate}")
    logger.info(f"Dataset: {cl_config.dataset}")

    # Determine device
    device = get_device()

    # Load tokenizer
    tokenizer = load_tokenizer(cl_config.base_model)

    # Load model and inject memory layers
    model = load_base_model(
        cl_config.base_model,
        dtype=cl_config.dtype,
        device_map=device,
    )
    model, shared_store = inject_memory_layers(model, memory_config)

    # Load pretrained memory checkpoint
    load_memory_checkpoint(model, shared_store, memory_config, cl_config.memory_checkpoint)

    # Freeze for continual learning (only value embeddings are trainable)
    freeze_for_continual_learning(model, memory_config)

    # Load IDF statistics
    idf_statistics = torch.load(
        cl_config.idf_statistics_path, map_location="cpu", weights_only=True
    )
    logger.info(
        f"Loaded IDF statistics: {idf_statistics['total_batches']} batches, "
        f"{(idf_statistics['slot_document_frequency'] > 0).sum().item()} active slots"
    )

    # Create continual learning dataloader
    dataloader = create_continual_dataloader(
        dataset_name=cl_config.dataset,
        tokenizer=tokenizer,
        batch_size=cl_config.batch_size,
        seq_length=cl_config.seq_length,
        seed=cl_config.seed,
    )

    # Run continual learning
    continual_learn(
        model=model,
        shared_store=shared_store,
        memory_config=memory_config,
        cl_config=cl_config,
        dataloader=dataloader,
        idf_statistics=idf_statistics,
        device=device,
    )

    logger.info("Phase 3 complete!")


if __name__ == "__main__":
    main()
