#!/usr/bin/env python3
"""Remember: extract facts from a conversation and finetune memory slots.

Usage:
    # From a saved conversation JSON file
    python scripts/remember.py --conversation chat.json

    # From a directory of conversation files
    python scripts/remember.py --conversation-dir conversations/

    # From raw text facts (skip extraction)
    python scripts/remember.py --facts "My name is Miguel" "I work on ML research"

    # From a text file with one fact per line
    python scripts/remember.py --facts-file facts.txt

This script:
1. Loads the conversation(s)
2. Extracts factual statements via self-distillation (LLM reads the chat)
3. Expands facts into natural training passages
4. Runs sparse memory finetuning (Phase 3) on the passages
5. Saves the updated memory checkpoint

The conversation JSON should be a list of {"role": "user"|"assistant", "content": "..."}
objects (OpenAI chat format), which is what most tools export.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import MemoryConfig, ContinualLearningConfig, load_config
from src.continual.gradient_masking import GradientMaskManager
from src.continual.tfidf_selector import TFIDFSelector
from src.model.freeze_utils import freeze_for_continual_learning
from src.model.memory_gemma import (
    get_memory_layers,
    inject_memory_layers,
    load_base_model,
    load_memory_checkpoint,
    load_tokenizer,
    save_memory_checkpoint,
)
from src.remember.fact_dataset import FactDataset
from src.remember.fact_extractor import (
    extract_facts_api,
    extract_facts_local,
    facts_to_training_text,
    load_conversation,
)
from src.utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_config(cfg):
    """Parse the remember config into MemoryConfig + remember settings."""
    from omegaconf import OmegaConf

    memory_config = MemoryConfig(**OmegaConf.to_container(cfg.memory, resolve=True))
    rem = cfg.remember
    return memory_config, rem


def gather_facts(args, rem_config, model=None, tokenizer=None, device="cuda"):
    """Gather facts from all input sources."""
    all_facts = []

    # From direct --facts arguments
    if args.facts:
        logger.info(f"Using {len(args.facts)} directly provided facts")
        all_facts.extend(args.facts)

    # From --facts-file
    if args.facts_file:
        path = Path(args.facts_file)
        lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
        logger.info(f"Loaded {len(lines)} facts from {args.facts_file}")
        all_facts.extend(lines)

    # From --conversation (single file)
    if args.conversation:
        facts = extract_from_file(
            args.conversation, rem_config, model, tokenizer, device
        )
        all_facts.extend(facts)

    # From --conversation-dir (directory of files)
    if args.conversation_dir:
        conv_dir = Path(args.conversation_dir)
        for conv_file in sorted(conv_dir.glob("*.json")):
            facts = extract_from_file(
                str(conv_file), rem_config, model, tokenizer, device
            )
            all_facts.extend(facts)

    return all_facts


def extract_from_file(path, rem_config, model, tokenizer, device):
    """Extract facts from a single conversation file."""
    logger.info(f"Extracting facts from: {path}")
    messages = load_conversation(path)
    logger.info(f"  Loaded {len(messages)} messages")

    backend = rem_config.extraction_backend

    if backend == "api":
        facts = extract_facts_api(
            messages,
            api_base=rem_config.api_base,
            model_name=rem_config.api_model,
            api_key=rem_config.api_key,
        )
    elif backend == "local":
        facts = extract_facts_local(
            messages, model=model, tokenizer=tokenizer, device=device
        )
    else:
        raise ValueError(f"Unknown extraction backend: {backend}")

    logger.info(f"  Extracted {len(facts)} facts")
    for i, fact in enumerate(facts):
        logger.info(f"    [{i+1}] {fact}")

    return facts


def run_remember_finetuning(
    model,
    shared_store,
    memory_config,
    rem_config,
    passages,
    tokenizer,
    idf_statistics,
    device,
):
    """Run sparse memory finetuning on the extracted passages."""
    # Create dataset
    dataset = FactDataset(
        passages=passages,
        tokenizer=tokenizer,
        seq_length=rem_config.seq_length,
        repeat_factor=rem_config.repeat_factor,
    )
    dataloader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)

    logger.info(
        f"Training on {len(dataset)} samples "
        f"({len(passages)} passages x {rem_config.repeat_factor} repeats)"
    )

    # Setup
    memory_layers = get_memory_layers(model, memory_config)
    selector = TFIDFSelector(idf_statistics, top_t=rem_config.top_t)
    mask_manager = GradientMaskManager(model, memory_config)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=rem_config.learning_rate,
        momentum=rem_config.momentum,
    )

    # Training loop
    model.train()
    total_steps = rem_config.finetuning_steps
    step = 0
    running_loss = 0.0

    while step < total_steps:
        for batch in dataloader:
            if step >= total_steps:
                break

            optimizer.zero_grad()

            # Enable index tracking
            for layer in memory_layers:
                layer.enable_index_tracking()

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = torch.zeros_like(input_ids)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels, token_type_ids=token_type_ids)
            loss = outputs.loss

            # Collect access counts
            indices_list = []
            for layer in memory_layers:
                idx = layer.get_last_accessed_indices()
                if idx is not None:
                    indices_list.append(idx)
                layer.disable_index_tracking()

            # TF-IDF slot selection
            num_entries = memory_config.num_entries
            access_counts = selector.compute_access_counts(indices_list, num_entries)
            trainable_indices = selector.select(access_counts)
            trainable_mask = selector.create_trainable_mask(
                trainable_indices, num_entries, device=device
            )
            mask_manager.set_mask(trainable_mask)

            # Backward + update
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % 10 == 0:
                avg = running_loss / 10
                logger.info(f"  Step {step}/{total_steps} | Loss: {avg:.4f}")
                running_loss = 0.0

    mask_manager.clear_mask()
    return step


def main():
    parser = argparse.ArgumentParser(
        description="Remember: extract facts from conversations and finetune memory"
    )
    parser.add_argument("--config", default="configs/remember.yaml", help="Config file")
    parser.add_argument("--conversation", type=str, help="Path to a conversation JSON")
    parser.add_argument(
        "--conversation-dir", type=str, help="Directory of conversation JSONs"
    )
    parser.add_argument(
        "--facts", nargs="+", type=str, help="Direct facts to remember"
    )
    parser.add_argument(
        "--facts-file", type=str, help="Text file with one fact per line"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract facts only, don't finetune",
    )
    args = parser.parse_args()

    if not any([args.conversation, args.conversation_dir, args.facts, args.facts_file]):
        parser.error(
            "Provide at least one of: --conversation, --conversation-dir, "
            "--facts, or --facts-file"
        )

    # Load config
    cfg = load_config(args.config)
    memory_config, rem_config = parse_config(cfg)

    device = get_device()
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = load_tokenizer(rem_config.base_model)

    # Load model + memory layers (needed for both extraction and finetuning)
    logger.info(f"Loading model: {rem_config.base_model}")
    model = load_base_model(
        rem_config.base_model, dtype=rem_config.dtype, device_map=device
    )
    model, shared_store = inject_memory_layers(model, memory_config)

    load_memory_checkpoint(
        model, shared_store, memory_config, rem_config.memory_checkpoint
    )

    # Step 1: Extract facts
    logger.info("=" * 60)
    logger.info("STEP 1: Extracting facts from conversations")
    logger.info("=" * 60)
    facts = gather_facts(args, rem_config, model, tokenizer, device)

    if not facts:
        logger.warning("No facts extracted. Nothing to remember.")
        return

    logger.info(f"\nTotal facts extracted: {len(facts)}")
    for i, fact in enumerate(facts):
        logger.info(f"  [{i+1}] {fact}")

    if args.dry_run:
        logger.info("Dry run — skipping finetuning.")
        return

    # Step 2: Expand facts into training passages
    logger.info("=" * 60)
    logger.info("STEP 2: Expanding facts into training passages")
    logger.info("=" * 60)

    if rem_config.expand_facts:
        if rem_config.extraction_backend == "api":
            passages = facts_to_training_text(
                facts,
                api_base=rem_config.api_base,
                model_name=rem_config.api_model,
                api_key=rem_config.api_key,
            )
        else:
            passages = facts_to_training_text(
                facts, model=model, tokenizer=tokenizer, device=device
            )
    else:
        passages = facts

    logger.info(f"Training passages: {len(passages)}")
    for i, p in enumerate(passages):
        logger.info(f"  [{i+1}] {p[:100]}{'...' if len(p) > 100 else ''}")

    # Step 3: Sparse memory finetuning
    logger.info("=" * 60)
    logger.info("STEP 3: Sparse memory finetuning")
    logger.info("=" * 60)

    # Load IDF statistics
    idf_statistics = torch.load(
        rem_config.idf_statistics_path, map_location="cpu", weights_only=True
    )

    # Freeze base model, keep all memory params trainable
    # We use freeze_base_model (not freeze_for_continual_learning) because
    # the remember step needs to update projections too — not just values.
    # The paper's Phase 3 only unfreezes values because projections are
    # well-trained after 128K pretrain steps. With distillation, projections
    # learned to mimic the FFN but need adjustment to encode new facts.
    from src.model.freeze_utils import freeze_base_model
    freeze_base_model(model, memory_config)

    steps = run_remember_finetuning(
        model=model,
        shared_store=shared_store,
        memory_config=memory_config,
        rem_config=rem_config,
        passages=passages,
        tokenizer=tokenizer,
        idf_statistics=idf_statistics,
        device=device,
    )

    # Step 4: Save checkpoint
    checkpoint_dir = Path(rem_config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(checkpoint_dir / "memory_layers.pt")
    save_memory_checkpoint(model, shared_store, memory_config, save_path, step=steps)

    # Also save the facts for reference
    facts_path = str(checkpoint_dir / "remembered_facts.json")
    with open(facts_path, "w") as f:
        json.dump({"facts": facts, "passages": passages}, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Done! Remembered {len(facts)} facts in {steps} steps.")
    logger.info(f"Checkpoint: {save_path}")
    logger.info(f"Facts log:  {facts_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
