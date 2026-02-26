#!/usr/bin/env python3
"""Generate text from a model with additive memory layers.

Usage:
    python additive_memory/generate.py \
        --checkpoint checkpoints/additive_memory/memory.pt \
        --prompt "What is my name?"

    # Without a checkpoint (baseline, memory outputs zero)
    python additive_memory/generate.py \
        --prompt "What is the capital of France?" \
        --base-model google/gemma-3-1b-it \
        --layers 9 17 --n-keys 256
"""

import argparse
import logging
import sys

import torch

from src.model.memory_gemma import load_base_model, load_tokenizer
from src.utils import get_device

from additive_memory.model import (
    AdditiveMemoryConfig,
    inject_additive_memory,
    load_memory_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def generate(model, tokenizer, prompt, device, max_new_tokens=256):
    """Generate text from a prompt."""
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the generated tokens (skip the prompt)
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with additive memory model"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Memory checkpoint to load"
    )
    parser.add_argument(
        "--base-model", default="google/gemma-3-4b-it", help="Base model name"
    )
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument(
        "--layers", nargs="+", type=int, default=[9, 17, 25], help="Memory layer indices"
    )
    parser.add_argument("--n-keys", type=int, default=1024, help="Sub-keys per half")
    parser.add_argument("--num-heads", type=int, default=4, help="Memory heads")
    parser.add_argument("--top-k", type=int, default=32, help="Top-k per head")
    parser.add_argument("--v-dim", type=int, default=1024, help="Value dimension")
    parser.add_argument("--k-dim-per-head", type=int, default=512, help="Key dim per head")
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Max tokens to generate"
    )
    args = parser.parse_args()

    config = AdditiveMemoryConfig(
        base_model=args.base_model,
        dtype=args.dtype,
        memory_layers=args.layers,
        num_heads=args.num_heads,
        n_keys=args.n_keys,
        k_dim_per_head=args.k_dim_per_head,
        v_dim=args.v_dim,
        top_k=args.top_k,
    )

    device = get_device()
    logger.info(f"Using device: {device}")

    tokenizer = load_tokenizer(config.base_model)
    model = load_base_model(config.base_model, dtype=config.dtype, device_map=device)
    model, shared_store = inject_additive_memory(model, config)

    if args.checkpoint:
        load_memory_checkpoint(model, shared_store, config, args.checkpoint)
    else:
        logger.info("No checkpoint — memory outputs zero (baseline)")

    logger.info(f"Prompt: {args.prompt}")
    response = generate(model, tokenizer, args.prompt, device, args.max_tokens)
    print(f"\n{'='*60}")
    print(f"Prompt:   {args.prompt}")
    print(f"Response: {response}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
