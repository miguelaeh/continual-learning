#!/usr/bin/env python3
"""Distillation-based initialization of memory layers (alternative to Phase 1).

Usage:
    python scripts/distill_memory.py --config configs/distill_memory.yaml

This script:
1. Loads Gemma 3 4B IT as a frozen teacher (original FFN layers intact)
2. Creates standalone Memory+ layers (NOT injected into the model)
3. Uses forward hooks to capture FFN inputs/outputs from the teacher
4. Trains memory layers to match FFN outputs via MSE loss
5. Saves checkpoint compatible with Phase 2 (IDF) and Phase 3 (continual learning)

Advantages over pretraining (scripts/pretrain_memory.py):
- Direct training signal (MSE on FFN output) vs indirect (LM loss)
- Converges in ~2K steps vs 128K steps
- Each memory layer sees correct hidden state distribution from the original model
"""

import argparse
import logging
import sys

import torch

from src.config import load_config, make_distill_config
from src.data.datasets import create_pretrain_dataloader
from src.memory.memory_layer import MemoryPlusLayer
from src.memory.shared_memory_store import SharedMemoryStore
from src.model.memory_gemma import get_decoder_layers, load_base_model, load_tokenizer
from src.training.distill_trainer import distill_memory_layers
from src.utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Distillation-based memory layer initialization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/distill_memory.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    memory_config, train_config = make_distill_config(cfg)

    logger.info(f"Memory layers at: {memory_config.memory_layers}")
    logger.info(f"Memory entries: {memory_config.num_entries:,}")
    logger.info(f"Dataset: {train_config.dataset}")
    logger.info(f"Total steps: {train_config.total_steps:,}")
    logger.info("Method: Distillation (direct FFN output matching via MSE)")

    # Determine device
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = load_tokenizer(train_config.base_model)

    # Load base model as TEACHER (no memory layers injected)
    logger.info(f"Loading teacher model: {train_config.base_model}")
    teacher_model = load_base_model(
        train_config.base_model,
        dtype=train_config.dtype,
        device_map=device,
    )
    teacher_model.eval()

    # Freeze teacher completely
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Get model dimensions from teacher
    layers = get_decoder_layers(teacher_model)
    sample_mlp = layers[memory_config.memory_layers[0]].mlp
    if hasattr(sample_mlp, "gate_proj"):
        d_model = sample_mlp.gate_proj.in_features
        target_device = sample_mlp.gate_proj.weight.device
        target_dtype = sample_mlp.gate_proj.weight.dtype
    else:
        d_model = 2560
        target_device = next(sample_mlp.parameters()).device
        target_dtype = next(sample_mlp.parameters()).dtype

    logger.info(f"Detected d_model={d_model}, device={target_device}, dtype={target_dtype}")

    # Create shared memory store (standalone)
    logger.info("Creating standalone memory layers for distillation")
    shared_store = SharedMemoryStore(
        num_heads=memory_config.num_heads,
        n_keys=memory_config.n_keys,
        k_dim_per_head=memory_config.k_dim_per_head,
        v_dim=memory_config.v_dim,
        top_k=memory_config.top_k,
    )
    shared_store = shared_store.to(device=target_device, dtype=target_dtype)

    # Create standalone memory layers (NOT injected into teacher)
    memory_layers = {}
    for layer_idx in memory_config.memory_layers:
        mem_layer = MemoryPlusLayer(
            d_model=d_model,
            shared_store=shared_store,
            num_heads=memory_config.num_heads,
            k_dim_per_head=memory_config.k_dim_per_head,
            v_dim=memory_config.v_dim,
            top_k=memory_config.top_k,
            use_silu_gating=memory_config.use_silu_gating,
        )
        mem_layer = mem_layer.to(device=target_device, dtype=target_dtype)
        memory_layers[layer_idx] = mem_layer
        logger.info(f"Created standalone MemoryPlusLayer for layer {layer_idx}")

    total_memory_params = sum(
        p.numel()
        for mem_layer in memory_layers.values()
        for p in mem_layer.parameters()
    )
    # Subtract double-counted shared store params
    shared_params = sum(p.numel() for p in shared_store.parameters())
    unique_params = total_memory_params - shared_params * (len(memory_layers) - 1)
    logger.info(f"Total unique memory parameters: {unique_params / 1e6:.1f}M")

    # Create dataloader
    dataloader = create_pretrain_dataloader(
        dataset_name=train_config.dataset,
        tokenizer=tokenizer,
        batch_size=train_config.batch_size,
        seq_length=train_config.seq_length,
        dataset_subset=train_config.dataset_subset,
        seed=train_config.seed,
    )

    # Run distillation
    distill_memory_layers(
        teacher_model=teacher_model,
        memory_layers=memory_layers,
        shared_store=shared_store,
        memory_config=memory_config,
        train_config=train_config,
        dataloader=dataloader,
        device=device,
    )

    logger.info("Distillation complete!")
    logger.info(
        f"Checkpoint saved to {train_config.checkpoint_dir}/memory_layers.pt"
    )
    logger.info(
        "This checkpoint is compatible with Phase 2 (IDF collection) "
        "and Phase 3 (continual learning / remember)"
    )


if __name__ == "__main__":
    main()
