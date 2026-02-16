#!/usr/bin/env python3
"""Staged pretraining: train memory layers one at a time to avoid cascading corruption.

Stage 1: Inject only the middle memory layer (layer 17), train with LM loss
Stage 2: Add the early layer (layer 9), train both
Stage 3: Add the late layer (layer 25), train all three

Each new layer gets clean inputs from already-trained layers and the original FFN,
avoiding the cascading corruption problem of training all layers simultaneously.

Usage:
    # Run all stages
    python scripts/pretrain_staged.py --config configs/pretrain_staged.yaml

    # Resume from stage 2 (loads stage 1 checkpoint automatically)
    python scripts/pretrain_staged.py --config configs/pretrain_staged.yaml --start-stage 2

    # Resume stage 2 from a training checkpoint (optimizer/scheduler restored)
    python scripts/pretrain_staged.py --config configs/pretrain_staged.yaml --start-stage 2 \
        --resume checkpoints/pretrain_staged/stage2_step_500.pt
"""

import argparse
import logging
import sys

from omegaconf import OmegaConf

from src.config import MemoryConfig, PretrainConfig, load_config
from src.data.datasets import create_pretrain_dataloader
from src.model.freeze_utils import freeze_base_model
from src.model.memory_gemma import (
    inject_memory_layers,
    load_base_model,
    load_memory_checkpoint,
    load_tokenizer,
)
from src.training.pretrain_trainer import pretrain_memory_layers
from src.utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def build_stages(memory_layers: list[int]) -> list[list[int]]:
    """Build training stages: start with middle layer, add outward.

    For [9, 17, 25] -> [[17], [9, 17], [9, 17, 25]]
    """
    sorted_layers = sorted(memory_layers)
    middle_idx = len(sorted_layers) // 2
    middle = sorted_layers[middle_idx]

    stages = [[middle]]

    # Add remaining layers one at a time (earlier first, then later)
    remaining = sorted_layers[:middle_idx] + sorted_layers[middle_idx + 1 :]
    current = [middle]
    for layer in remaining:
        current = sorted(current + [layer])
        stages.append(list(current))

    return stages


def run_stage(
    stage_num: int,
    stage_layers: list[int],
    memory_config: MemoryConfig,
    train_cfg: dict,
    prev_checkpoint: str | None,
    resume_path: str | None,
    device: str,
):
    """Run one stage of training."""
    stage_name = f"stage{stage_num}"
    checkpoint_dir = f"{train_cfg['checkpoint_dir']}/{stage_name}"

    logger.info("=" * 60)
    logger.info(f"STAGE {stage_num}: Training memory layers {stage_layers}")
    logger.info("=" * 60)

    # Create a MemoryConfig for this stage's layers
    stage_memory_config = MemoryConfig(
        num_heads=memory_config.num_heads,
        top_k=memory_config.top_k,
        n_keys=memory_config.n_keys,
        k_dim_per_head=memory_config.k_dim_per_head,
        v_dim=memory_config.v_dim,
        memory_layers=stage_layers,
        use_silu_gating=memory_config.use_silu_gating,
    )

    # Create PretrainConfig for this stage
    stage_train_config = PretrainConfig(
        base_model=train_cfg["base_model"],
        optimizer=train_cfg.get("optimizer", "adamw"),
        learning_rate=train_cfg["learning_rate"],
        value_learning_rate=train_cfg["value_learning_rate"],
        momentum=train_cfg.get("momentum", 0.0),
        warmup_steps=train_cfg["warmup_steps"],
        total_steps=train_cfg["steps_per_stage"],
        batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        seq_length=train_cfg["seq_length"],
        weight_decay=train_cfg.get("weight_decay", 0.1),
        gradient_clip=train_cfg["gradient_clip"],
        dtype=train_cfg["dtype"],
        dataset=train_cfg["dataset"],
        dataset_subset=train_cfg.get("dataset_subset", "default"),
        checkpoint_dir=checkpoint_dir,
        save_every_steps=train_cfg["save_every_steps"],
        log_every_steps=train_cfg["log_every_steps"],
        seed=train_cfg["seed"] + stage_num,  # different data order per stage
    )

    # Load tokenizer
    tokenizer = load_tokenizer(stage_train_config.base_model)

    # Load base model fresh for each stage
    logger.info(f"Loading base model: {stage_train_config.base_model}")
    model = load_base_model(
        stage_train_config.base_model,
        dtype=stage_train_config.dtype,
        device_map=device,
    )

    # Inject memory layers for this stage only
    model, shared_store = inject_memory_layers(model, stage_memory_config)

    # Load previous stage checkpoint (if not resuming — resume restores everything)
    if not resume_path and prev_checkpoint:
        logger.info(f"Loading previous stage checkpoint: {prev_checkpoint}")
        load_memory_checkpoint(model, shared_store, stage_memory_config, prev_checkpoint)
        logger.info(
            f"Loaded trained weights for previous layers. "
            f"New layer(s) start from random init."
        )

    # Freeze base model
    freeze_base_model(model, stage_memory_config)

    # Create dataloader
    dataloader = create_pretrain_dataloader(
        dataset_name=stage_train_config.dataset,
        tokenizer=tokenizer,
        batch_size=stage_train_config.batch_size,
        seq_length=stage_train_config.seq_length,
        dataset_subset=stage_train_config.dataset_subset,
        seed=stage_train_config.seed,
    )

    # Train
    pretrain_memory_layers(
        model=model,
        shared_store=shared_store,
        memory_config=stage_memory_config,
        train_config=stage_train_config,
        dataloader=dataloader,
        device=device,
        resume_path=resume_path,
    )

    # Return path to final checkpoint for next stage
    final_checkpoint = f"{checkpoint_dir}/memory_layers.pt"
    logger.info(f"Stage {stage_num} complete. Checkpoint: {final_checkpoint}")
    return final_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Staged pretraining: train memory layers one at a time"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain_staged.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        help="Stage to start from (1, 2, or 3). Previous stage checkpoint loaded automatically.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume a specific stage from a training checkpoint (optimizer/scheduler restored)",
    )
    parser.add_argument(
        "--memory-checkpoint",
        type=str,
        default=None,
        help="Warm-start the starting stage from a memory checkpoint (fresh optimizer/scheduler)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    memory_config = MemoryConfig(**OmegaConf.to_container(cfg.memory, resolve=True))
    train_cfg = OmegaConf.to_container(cfg.staged_pretrain, resolve=True)

    device = get_device()
    logger.info(f"Using device: {device}")

    # Build stages from memory_layers
    stages = build_stages(memory_config.memory_layers)
    logger.info(f"Training stages: {stages}")

    # Run stages
    prev_checkpoint = args.memory_checkpoint

    for i, stage_layers in enumerate(stages):
        stage_num = i + 1

        if stage_num < args.start_stage:
            # Skip but set prev_checkpoint for the next stage
            if not prev_checkpoint:
                prev_checkpoint = (
                    f"{train_cfg['checkpoint_dir']}/stage{stage_num}/memory_layers.pt"
                )
            logger.info(f"Skipping stage {stage_num} (start-stage={args.start_stage})")
            continue

        # Only use --resume for the first stage we actually run
        resume_path = args.resume if stage_num == args.start_stage else None

        prev_checkpoint = run_stage(
            stage_num=stage_num,
            stage_layers=stage_layers,
            memory_config=memory_config,
            train_cfg=train_cfg,
            prev_checkpoint=prev_checkpoint,
            resume_path=resume_path,
            device=device,
        )

    logger.info("=" * 60)
    logger.info("All stages complete!")
    logger.info(f"Final checkpoint: {prev_checkpoint}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
