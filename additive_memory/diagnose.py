#!/usr/bin/env python3
"""Diagnose memory layer behavior: slot overlap, output magnitudes, value norms.

Usage:
    python additive_memory/diagnose.py \
        --checkpoint checkpoints/additive_memory/memory.pt \
        --train-text "User: What is my name? Assistant: Your name is Miguel." \
        --inference-text "What is my name?" \
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
    get_memory_layers,
    inject_additive_memory,
    load_memory_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def get_accessed_slots(model, config, input_ids, device):
    """Run a forward pass and collect which slots each memory layer accessed."""
    memory_layers = get_memory_layers(model, config)
    for layer in memory_layers:
        layer.enable_index_tracking()

    token_type_ids = torch.zeros_like(input_ids)
    with torch.no_grad():
        model(input_ids=input_ids, token_type_ids=token_type_ids)

    all_indices = {}
    for i, (layer_idx, layer) in enumerate(zip(config.memory_layers, memory_layers)):
        idx = layer.get_last_accessed_indices()  # (B*T, H, top_k)
        if idx is not None:
            all_indices[layer_idx] = idx.clone()
        layer.disable_index_tracking()

    return all_indices


def get_memory_and_ffn_outputs(model, config, input_ids, device):
    """Hook into FFNWithMemory to capture FFN and memory output magnitudes."""
    memory_outputs = {}
    ffn_outputs = {}

    from additive_memory.layer import FFNWithMemory

    hooks = []
    layers = model.model.layers

    for layer_idx in config.memory_layers:
        wrapper = layers[layer_idx].mlp
        if not isinstance(wrapper, FFNWithMemory):
            continue

        def make_hook(idx):
            def hook_fn(module, input, output):
                x = input[0]
                with torch.no_grad():
                    ffn_out = module.ffn(x)
                    mem_out = module.memory(x)
                ffn_outputs[idx] = ffn_out.detach()
                memory_outputs[idx] = mem_out.detach()
            return hook_fn

        h = wrapper.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    token_type_ids = torch.zeros_like(input_ids)
    with torch.no_grad():
        model(input_ids=input_ids, token_type_ids=token_type_ids)

    for h in hooks:
        h.remove()

    return ffn_outputs, memory_outputs


def compute_overlap(indices_a, indices_b):
    """Compute slot overlap statistics between two sets of accessed indices."""
    # Flatten all indices to sets
    set_a = set(indices_a.reshape(-1).tolist())
    set_b = set(indices_b.reshape(-1).tolist())

    intersection = set_a & set_b
    union = set_a | set_b

    return {
        "slots_a": len(set_a),
        "slots_b": len(set_b),
        "intersection": len(intersection),
        "union": len(union),
        "jaccard": len(intersection) / len(union) if union else 0,
        "recall_a_in_b": len(intersection) / len(set_a) if set_a else 0,
        "recall_b_in_a": len(intersection) / len(set_b) if set_b else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose additive memory behavior")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--train-text", type=str, required=True,
                        help="Text that was used for training")
    parser.add_argument("--inference-text", type=str, required=True,
                        help="Inference prompt to test")
    parser.add_argument("--base-model", default="google/gemma-3-4b-it")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layers", nargs="+", type=int, default=[9, 17, 25])
    parser.add_argument("--n-keys", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--v-dim", type=int, default=1024)
    parser.add_argument("--k-dim-per-head", type=int, default=512)
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
    tokenizer = load_tokenizer(config.base_model)
    model = load_base_model(config.base_model, dtype=config.dtype, device_map=device)
    model, shared_store = inject_additive_memory(model, config)

    if args.checkpoint:
        load_memory_checkpoint(model, shared_store, config, args.checkpoint)

    model.eval()

    # Tokenize inputs
    train_ids = tokenizer.encode(args.train_text, return_tensors="pt").to(device)

    # Format inference text as chat (matching generate.py)
    messages = [{"role": "user", "content": args.inference_text}]
    inf_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inf_ids = tokenizer(inf_text, return_tensors="pt")["input_ids"].to(device)

    print(f"\n{'='*60}")
    print("DIAGNOSTIC REPORT")
    print(f"{'='*60}")
    print(f"Train text:     {args.train_text[:80]}...")
    print(f"Inference text: {args.inference_text}")
    print(f"Train tokens:   {train_ids.shape[1]}")
    print(f"Inference tokens: {inf_ids.shape[1]}")
    print(f"Memory entries: {config.num_entries}")
    print(f"Top-k:          {config.top_k}")
    print(f"Num heads:      {config.num_heads}")
    print(f"Checkpoint:     {args.checkpoint or 'None (zero-init)'}")

    # 1. Slot overlap analysis
    print(f"\n{'='*60}")
    print("1. SLOT OVERLAP ANALYSIS")
    print(f"{'='*60}")

    train_indices = get_accessed_slots(model, config, train_ids, device)
    inf_indices = get_accessed_slots(model, config, inf_ids, device)

    for layer_idx in config.memory_layers:
        if layer_idx in train_indices and layer_idx in inf_indices:
            overlap = compute_overlap(train_indices[layer_idx], inf_indices[layer_idx])
            print(f"\n  Layer {layer_idx}:")
            print(f"    Train slots:     {overlap['slots_a']}")
            print(f"    Inference slots: {overlap['slots_b']}")
            print(f"    Intersection:    {overlap['intersection']}")
            print(f"    Jaccard:         {overlap['jaccard']:.4f}")
            print(f"    Train→Inf recall: {overlap['recall_a_in_b']:.4f} "
                  f"(fraction of train slots also hit at inference)")
            print(f"    Inf→Train recall: {overlap['recall_b_in_a']:.4f} "
                  f"(fraction of inference slots that were trained)")

    # 2. Memory vs FFN output magnitudes
    print(f"\n{'='*60}")
    print("2. OUTPUT MAGNITUDE ANALYSIS (inference prompt)")
    print(f"{'='*60}")

    ffn_outs, mem_outs = get_memory_and_ffn_outputs(model, config, inf_ids, device)

    for layer_idx in config.memory_layers:
        if layer_idx in ffn_outs and layer_idx in mem_outs:
            ffn_norm = ffn_outs[layer_idx].float().norm().item()
            mem_norm = mem_outs[layer_idx].float().norm().item()
            ratio = mem_norm / ffn_norm if ffn_norm > 0 else 0
            print(f"\n  Layer {layer_idx}:")
            print(f"    FFN output L2 norm:    {ffn_norm:.4f}")
            print(f"    Memory output L2 norm: {mem_norm:.4f}")
            print(f"    Memory/FFN ratio:      {ratio:.6f}")

    # 3. Value embedding statistics
    print(f"\n{'='*60}")
    print("3. MEMORY VALUE STATISTICS")
    print(f"{'='*60}")

    values = shared_store.values.weight.data
    nonzero_rows = (values.abs().sum(dim=1) > 1e-8).sum().item()
    total_rows = values.shape[0]
    print(f"  Total slots:        {total_rows}")
    print(f"  Non-zero slots:     {nonzero_rows} ({100*nonzero_rows/total_rows:.2f}%)")
    print(f"  Value norm (mean):  {values.norm(dim=1).mean().item():.6f}")
    print(f"  Value norm (max):   {values.norm(dim=1).max().item():.6f}")

    if nonzero_rows > 0:
        nonzero_mask = values.abs().sum(dim=1) > 1e-8
        nonzero_norms = values[nonzero_mask].norm(dim=1)
        print(f"  Non-zero norm (mean): {nonzero_norms.mean().item():.6f}")
        print(f"  Non-zero norm (max):  {nonzero_norms.max().item():.6f}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
