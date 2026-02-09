#!/usr/bin/env python3
"""Phase 2: Collect IDF background statistics for TF-IDF slot selection.

Usage:
    python scripts/collect_statistics.py --config configs/collect_statistics.yaml

This script:
1. Loads Gemma 3 4B IT with pretrained memory layers
2. Runs inference on 1000 batches of background data
3. Tracks which memory slots are accessed per batch
4. Computes document frequency (batches where each slot appears)
5. Saves statistics to disk for use in Phase 3
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from src.config import load_config, make_collection_config
from src.continual.idf_collector import IDFCollector
from src.data.datasets import create_background_dataloader
from src.model.memory_gemma import (
    inject_memory_layers,
    load_base_model,
    load_memory_checkpoint,
    load_tokenizer,
)
from src.utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Collect IDF statistics")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/collect_statistics.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    memory_config, coll_config = make_collection_config(cfg)

    logger.info(f"Memory checkpoint: {coll_config.memory_checkpoint}")
    logger.info(f"Background batches: {coll_config.num_background_batches}")
    logger.info(f"Output: {coll_config.output_path}")

    # Determine device
    device = get_device()

    # Load tokenizer
    tokenizer = load_tokenizer(coll_config.base_model)

    # Load model and inject memory layers
    model = load_base_model(
        coll_config.base_model,
        dtype=coll_config.dtype,
        device_map=device,
    )
    model, shared_store = inject_memory_layers(model, memory_config)

    # Load pretrained memory checkpoint
    load_memory_checkpoint(model, shared_store, memory_config, coll_config.memory_checkpoint)

    # Create background dataloader
    dataloader = create_background_dataloader(
        dataset_name=coll_config.dataset,
        tokenizer=tokenizer,
        batch_size=coll_config.batch_size,
        seq_length=coll_config.seq_length,
        dataset_subset=coll_config.dataset_subset,
        seed=coll_config.seed,
    )

    # Collect IDF statistics
    collector = IDFCollector(
        model=model,
        memory_config=memory_config,
        num_batches=coll_config.num_background_batches,
        device=device,
    )
    idf_statistics = collector.collect(dataloader)

    # Save statistics
    output_path = Path(coll_config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(idf_statistics, str(output_path))
    logger.info(f"Saved IDF statistics to {coll_config.output_path}")

    # Print summary
    df = idf_statistics["slot_document_frequency"]
    active = (df > 0).sum().item()
    total = memory_config.num_entries
    logger.info(
        f"Summary: {active}/{total} slots accessed "
        f"({100 * active / total:.1f}%), "
        f"mean df={df[df > 0].float().mean():.1f}, "
        f"max df={df.max().item()}"
    )

    logger.info("Phase 2 complete!")


if __name__ == "__main__":
    main()
