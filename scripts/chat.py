#!/usr/bin/env python3
"""Interactive chat with the memory-augmented Gemma model.

Usage:
    # Chat with the model after running the remember pipeline
    python scripts/chat.py --memory-checkpoint checkpoints/remembered/memory_layers.pt

    # Chat with the pretrained memory (before remember)
    python scripts/chat.py --memory-checkpoint checkpoints/pretrain/memory_layers.pt

    # Chat with base Gemma (no memory layers) for comparison
    python scripts/chat.py --no-memory

    # Single-shot query (non-interactive)
    python scripts/chat.py --query "What is my name?" --memory-checkpoint checkpoints/remembered/memory_layers.pt

    # Compare base vs remembered model on the same query
    python scripts/chat.py --compare --query "What is my name?" --memory-checkpoint checkpoints/remembered/memory_layers.pt

This loads the model, optionally injects + loads memory layer weights,
and runs interactive generation so you can verify the model recalls facts.
"""

import argparse
import logging
import sys

import torch
from transformers import TextStreamer

from src.config import MemoryConfig, load_config
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


def generate_response(
    model, tokenizer, messages, device, max_new_tokens=512, temperature=0.7
):
    """Generate a response given chat messages."""
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            streamer=streamer,
        )

    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_model_with_memory(config_path, memory_checkpoint, device, dtype="float32"):
    """Load the base model and inject + load memory layers."""
    from omegaconf import OmegaConf

    cfg = load_config(config_path)
    memory_config = MemoryConfig(**OmegaConf.to_container(cfg.memory, resolve=True))

    # Determine base model name from config
    for section in ["remember", "pretrain", "collection", "continual"]:
        if hasattr(cfg, section) and hasattr(cfg[section], "base_model"):
            base_model = cfg[section].base_model
            break
    else:
        base_model = "google/gemma-3-4b-it"

    logger.info(f"Loading base model: {base_model}")
    model = load_base_model(base_model, dtype=dtype, device_map=device)

    logger.info("Injecting memory layers...")
    model, shared_store = inject_memory_layers(model, memory_config)

    logger.info(f"Loading memory checkpoint: {memory_checkpoint}")
    load_memory_checkpoint(model, shared_store, memory_config, memory_checkpoint)

    model.eval()
    return model, memory_config


def load_model_base(base_model, device, dtype="float32"):
    """Load the base model without memory layers (for comparison)."""
    logger.info(f"Loading base model (no memory): {base_model}")
    model = load_base_model(base_model, dtype=dtype, device_map=device)
    model.eval()
    return model


def run_interactive(model, tokenizer, device, system_prompt=None):
    """Run an interactive chat loop."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    print("\n" + "=" * 60)
    print("Interactive chat (type 'quit' to exit, 'reset' to clear history)")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Bye!")
            break
        if user_input.lower() == "reset":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            print("[Chat history cleared]\n")
            continue

        messages.append({"role": "user", "content": user_input})

        print("Assistant: ", end="", flush=True)
        response = generate_response(model, tokenizer, messages, device)
        print()  # newline after streamed output

        messages.append({"role": "assistant", "content": response})


def run_single_query(model, tokenizer, device, query, label=""):
    """Run a single query and return the response."""
    messages = [{"role": "user", "content": query}]

    if label:
        print(f"\n--- {label} ---")
    print(f"Query: {query}")
    print("Response: ", end="", flush=True)
    response = generate_response(model, tokenizer, messages, device)
    print("\n")
    return response


def main():
    parser = argparse.ArgumentParser(
        description="Chat with the memory-augmented Gemma model"
    )
    parser.add_argument(
        "--config",
        default="configs/remember_mac.yaml",
        help="Config file (for memory layer dimensions)",
    )
    parser.add_argument(
        "--memory-checkpoint",
        type=str,
        default=None,
        help="Path to memory checkpoint (e.g., checkpoints/remembered/memory_layers.pt)",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Load base model without memory layers (for comparison)",
    )
    parser.add_argument(
        "--base-model",
        default="google/gemma-3-4b-it",
        help="Base model name",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query (non-interactive mode)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare base model vs memory model on the same query",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Model dtype (float32 for MPS, bfloat16 for CUDA)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )

    args = parser.parse_args()

    if not args.no_memory and args.memory_checkpoint is None:
        parser.error("Provide --memory-checkpoint or use --no-memory")

    device = get_device()
    logger.info(f"Using device: {device}")

    tokenizer = load_tokenizer(args.base_model)

    if args.compare and args.query:
        # Load base model first
        base_model = load_model_base(args.base_model, device, args.dtype)
        run_single_query(base_model, tokenizer, device, args.query, "Base Gemma (no memory)")
        del base_model
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

        # Load memory model
        mem_model, _ = load_model_with_memory(
            args.config, args.memory_checkpoint, device, args.dtype
        )
        run_single_query(mem_model, tokenizer, device, args.query, "Memory-augmented Gemma")
        return

    if args.no_memory:
        model = load_model_base(args.base_model, device, args.dtype)
    else:
        model, _ = load_model_with_memory(
            args.config, args.memory_checkpoint, device, args.dtype
        )

    if args.query:
        run_single_query(model, tokenizer, device, args.query)
    else:
        run_interactive(model, tokenizer, device)


if __name__ == "__main__":
    main()
