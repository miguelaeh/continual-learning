"""Hybrid training: LM loss + MSE distillation for memory layer pretraining.

Combines language modeling loss (end-to-end) with direct MSE distillation
against the original FFN output. The MSE component provides direct gradient
signal to the memory layer, preventing the "residual bypass" problem where
the model learns to ignore the memory layer via the residual connection.

total_loss = lm_loss + alpha * mse_loss

The LM loss trains proper slot structure (end-to-end), while the MSE loss
ensures the memory layer actually produces useful hidden states.
"""

import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import MemoryConfig
from src.memory.shared_memory_store import SharedMemoryStore
from src.model.memory_gemma import get_decoder_layers, save_memory_checkpoint

logger = logging.getLogger(__name__)


class FFNDistillationHelper:
    """Captures memory layer I/O via hooks and computes MSE against original FFNs."""

    def __init__(self, model, memory_config, original_ffns):
        """
        Args:
            model: Model with injected memory layers.
            memory_config: Memory layer configuration.
            original_ffns: Dict mapping layer_idx -> original frozen FFN module.
        """
        self.original_ffns = original_ffns
        self.captures = {}
        self.hooks = []

        layers = get_decoder_layers(model)
        for layer_idx in memory_config.memory_layers:
            mem_layer = layers[layer_idx].mlp
            hook = mem_layer.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            self.captures[layer_idx] = {
                "input": input[0],  # FFN input (after pre_feedforward_layernorm)
                "output": output,  # Memory layer output
            }

        return hook_fn

    def compute_mse_loss(self):
        """Compute MSE between memory layer output and original FFN output."""
        total_mse = 0.0
        count = 0
        for layer_idx, captured in self.captures.items():
            with torch.no_grad():
                target = self.original_ffns[layer_idx](captured["input"])
            mse = F.mse_loss(captured["output"], target)
            total_mse += mse
            count += 1
        self.captures.clear()
        return total_mse / max(count, 1)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def create_hybrid_optimizer(model, shared_store, config):
    """Create AdamW optimizer with separate LR for value embeddings."""
    value_params = []
    other_params = []

    value_param_ids = {id(p) for p in shared_store.values.parameters()}

    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in value_param_ids:
            value_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {
            "params": other_params,
            "lr": config["learning_rate"],
            "weight_decay": config.get("weight_decay", 0.1),
        },
        {
            "params": value_params,
            "lr": config["value_learning_rate"],
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(param_groups)


def get_hybrid_lr_scheduler(optimizer, config):
    """Create linear warmup + cosine decay LR scheduler."""
    warmup_steps = config["warmup_steps"]
    total_steps = config["total_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_hybrid_checkpoint(
    model, shared_store, memory_config, optimizer, scheduler,
    global_step, micro_step, save_path,
):
    """Save full training state for resuming."""
    checkpoint = {
        "shared_store": shared_store.state_dict(),
        "memory_config": {
            "num_heads": memory_config.num_heads,
            "top_k": memory_config.top_k,
            "n_keys": memory_config.n_keys,
            "k_dim_per_head": memory_config.k_dim_per_head,
            "v_dim": memory_config.v_dim,
            "memory_layers": memory_config.memory_layers,
            "use_silu_gating": memory_config.use_silu_gating,
        },
        "per_layer_states": {},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
        "micro_step": micro_step,
        "step": global_step,
    }

    layers = get_decoder_layers(model)
    for layer_idx in memory_config.memory_layers:
        mem_layer = layers[layer_idx].mlp
        per_layer_state = {}
        for name, param in mem_layer.named_parameters():
            if not name.startswith("shared_store."):
                per_layer_state[name] = param.data
        checkpoint["per_layer_states"][layer_idx] = per_layer_state

    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    logger.info(f"Saved hybrid checkpoint to {save_path} (step {global_step})")


def load_hybrid_checkpoint(model, shared_store, memory_config, optimizer, scheduler, load_path):
    """Load full training state for resuming."""
    checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)

    shared_store.load_state_dict(checkpoint["shared_store"])
    layers = get_decoder_layers(model)
    for layer_idx, per_layer_state in checkpoint["per_layer_states"].items():
        layer_idx = int(layer_idx)
        mem_layer = layers[layer_idx].mlp
        for name, param_data in per_layer_state.items():
            param = dict(mem_layer.named_parameters())[name]
            param.data.copy_(param_data)

    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    global_step = checkpoint["global_step"]
    micro_step = checkpoint.get("micro_step", global_step * 8)

    logger.info(f"Resumed from hybrid checkpoint at step {global_step}")
    return global_step, micro_step


def hybrid_pretrain(
    model,
    shared_store,
    memory_config,
    config,
    original_ffns,
    dataloader,
    device,
    resume_path=None,
):
    """Main hybrid training loop.

    Args:
        model: Model with injected memory layers (base frozen).
        shared_store: Shared memory store.
        memory_config: Memory layer configuration.
        config: Training config dict.
        original_ffns: Dict mapping layer_idx -> original frozen FFN module.
        dataloader: Streaming dataloader.
        device: Training device.
        resume_path: Optional path to resume from.
    """
    model.train()

    # Ensure original FFNs stay in eval mode
    for ffn in original_ffns.values():
        ffn.eval()

    optimizer = create_hybrid_optimizer(model, shared_store, config)
    scheduler = get_hybrid_lr_scheduler(optimizer, config)

    distill_helper = FFNDistillationHelper(model, memory_config, original_ffns)

    accum_steps = config["gradient_accumulation_steps"]
    mse_alpha_start = config.get("mse_alpha_start", config.get("mse_alpha", 1.0))
    mse_alpha_end = config.get("mse_alpha_end", 0.0)
    mse_warmup_steps = config.get("mse_warmup_steps", 0)
    total_steps = config["total_steps"]
    log_every = config["log_every_steps"]
    save_every = config["save_every_steps"]
    checkpoint_dir = config["checkpoint_dir"]

    accum_lm_loss = 0.0
    accum_mse_loss = 0.0
    accum_total_loss = 0.0

    global_step = 0
    micro_step = 0

    if resume_path:
        global_step, micro_step = load_hybrid_checkpoint(
            model, shared_store, memory_config, optimizer, scheduler, resume_path
        )

    logger.info(
        f"Starting hybrid pretraining for {total_steps} steps "
        f"(batch={config['batch_size']}, accum={accum_steps}, "
        f"mse_alpha={mse_alpha_start}->{mse_alpha_end} over {mse_warmup_steps} warmup, "
        f"resuming from step {global_step})"
    )

    batches_to_skip = micro_step
    skipped = 0

    for batch in dataloader:
        if global_step >= total_steps:
            break

        if skipped < batches_to_skip:
            skipped += 1
            if skipped % 1000 == 0:
                logger.info(f"  Skipping batch {skipped}/{batches_to_skip}...")
            continue

        # Compute annealed alpha
        if global_step < mse_warmup_steps:
            # Constant high alpha during warmup (bootstrap phase)
            mse_alpha = mse_alpha_start
        elif mse_alpha_end == mse_alpha_start:
            mse_alpha = mse_alpha_start
        else:
            # Linear decay from alpha_start to alpha_end after warmup
            anneal_progress = (global_step - mse_warmup_steps) / max(
                1, total_steps - mse_warmup_steps
            )
            anneal_progress = min(1.0, anneal_progress)
            mse_alpha = mse_alpha_start + (mse_alpha_end - mse_alpha_start) * anneal_progress

        # Forward pass
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        token_type_ids = torch.zeros_like(input_ids)

        outputs = model(
            input_ids=input_ids, labels=labels, token_type_ids=token_type_ids
        )
        lm_loss = outputs.loss

        # MSE loss from hooks
        mse_loss = distill_helper.compute_mse_loss()

        # Combined loss
        total_loss = (lm_loss + mse_alpha * mse_loss) / accum_steps
        total_loss.backward()

        accum_lm_loss += lm_loss.item()
        accum_mse_loss += mse_loss.item()
        accum_total_loss += (lm_loss.item() + mse_alpha * mse_loss.item())
        micro_step += 1

        if micro_step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config["gradient_clip"],
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % log_every == 0:
                avg_lm = accum_lm_loss / (log_every * accum_steps)
                avg_mse = accum_mse_loss / (log_every * accum_steps)
                avg_total = accum_total_loss / (log_every * accum_steps)
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {global_step}/{total_steps} | "
                    f"LM: {avg_lm:.4f} | MSE: {avg_mse:.4f} | "
                    f"Total: {avg_total:.4f} | α: {mse_alpha:.1f} | LR: {lr:.2e}"
                )
                accum_lm_loss = 0.0
                accum_mse_loss = 0.0
                accum_total_loss = 0.0

            if global_step % save_every == 0:
                save_path = str(
                    Path(checkpoint_dir) / f"memory_step_{global_step}.pt"
                )
                save_hybrid_checkpoint(
                    model, shared_store, memory_config,
                    optimizer, scheduler, global_step, micro_step, save_path,
                )

    # Final checkpoint (memory-only, compatible with Phase 2/3)
    distill_helper.remove_hooks()
    save_path = str(Path(checkpoint_dir) / "memory_layers.pt")
    save_memory_checkpoint(model, shared_store, memory_config, save_path, step=global_step)
    logger.info(f"Hybrid pretraining complete after {global_step} steps")
