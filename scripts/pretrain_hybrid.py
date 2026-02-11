#!/usr/bin/env python3
"""Hybrid pretraining: LM loss + MSE distillation for memory layers.

Solves the "residual bypass" problem where LM-only pretraining causes the model
to ignore the memory layer via the residual connection (loss plateaus at ~5.0).

The MSE component gives direct gradient signal to the memory layer (matching
the original FFN output), while the LM component trains end-to-end for proper
slot structure.

Usage:
    python scripts/pretrain_hybrid.py --config configs/pretrain_hybrid.yaml

    # Resume from checkpoint
    python scripts/pretrain_hybrid.py --config configs/pretrain_hybrid.yaml \
        --resume checkpoints/pretrain_hybrid/memory_step_1000.pt
"""

import argparse
import logging
import sys

from omegaconf import OmegaConf

from src.config import MemoryConfig, load_config
from src.data.datasets import create_pretrain_dataloader
from src.model.freeze_utils import freeze_base_model
from src.model.memory_gemma import (
    get_decoder_layers,
    inject_memory_layers,
    load_base_model,
    load_tokenizer,
)
from src.training.hybrid_trainer import hybrid_pretrain
from src.utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid pretraining: LM loss + MSE distillation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_hybrid.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a training checkpoint",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    memory_config = MemoryConfig(**OmegaConf.to_container(cfg.memory, resolve=True))
    train_cfg = OmegaConf.to_container(cfg.hybrid_pretrain, resolve=True)

    device = get_device()
    logger.info(f"Using device: {device}")

    base_model_name = train_cfg["base_model"]
    tokenizer = load_tokenizer(base_model_name)

    # Load base model
    logger.info(f"Loading base model: {base_model_name}")
    model = load_base_model(
        base_model_name, dtype=train_cfg["dtype"], device_map=device
    )

    # Save original FFNs BEFORE injection (for MSE targets)
    logger.info("Saving original FFN modules as distillation targets...")
    layers = get_decoder_layers(model)
    original_ffns = {}
    for layer_idx in memory_config.memory_layers:
        original_ffn = layers[layer_idx].mlp
        original_ffn.eval()
        for p in original_ffn.parameters():
            p.requires_grad = False
        original_ffns[layer_idx] = original_ffn
        logger.info(f"  Saved FFN at layer {layer_idx}")

    # Inject memory layers (replaces the MLPs we just saved)
    model, shared_store = inject_memory_layers(model, memory_config)

    # Freeze base model
    freeze_base_model(model, memory_config)

    # Create dataloader
    dataloader = create_pretrain_dataloader(
        dataset_name=train_cfg["dataset"],
        tokenizer=tokenizer,
        batch_size=train_cfg["batch_size"],
        seq_length=train_cfg["seq_length"],
        dataset_subset=train_cfg.get("dataset_subset", "default"),
        seed=train_cfg["seed"],
    )

    # Train
    hybrid_pretrain(
        model=model,
        shared_store=shared_store,
        memory_config=memory_config,
        config=train_cfg,
        original_ffns=original_ffns,
        dataloader=dataloader,
        device=device,
        resume_path=args.resume,
    )

    logger.info("Hybrid pretraining complete!")


if __name__ == "__main__":
    main()
