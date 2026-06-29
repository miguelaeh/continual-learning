"""Recovery/healing training loop."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch

from smf_retrofit.config import ExperimentConfig, RecoveryConfig
from smf_retrofit.eval import (
    build_model,
    compare_models_on_loss,
    evaluate_prompt_specs,
    load_prompt_specs,
)
from smf_retrofit.modeling.qwen import load_memory_checkpoint, save_memory_checkpoint


logger = logging.getLogger(__name__)


def create_recovery_optimizer(
    model: torch.nn.Module,
    config: RecoveryConfig,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def _find_latest_checkpoint(output_dir: str) -> tuple[str, int] | None:
    checkpoints = sorted(Path(output_dir).glob("memory_step_*.pt"))
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    ckpt = torch.load(str(latest), map_location="cpu", weights_only=True)
    step = int(ckpt.get("step", 0))
    return str(latest), step


def run_recovery(
    model: torch.nn.Module,
    dataloader,
    config: RecoveryConfig,
    layer_indices: list[int],
    device: str,
    resume_from: str | None = None,
) -> str:
    model.train()
    optimizer, scheduler = create_recovery_optimizer(model, config)

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    global_step = 0
    running_loss = 0.0

    if resume_from:
        load_memory_checkpoint(model, resume_from, layer_indices)
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=True)
        global_step = int(ckpt.get("step", 0))
        for _ in range(global_step):
            scheduler.step()
        logger.info("Resumed recovery from %s at step %d", resume_from, global_step)

    running_loss_acc = torch.tensor(0.0, device=device)

    while global_step < config.total_steps:
        progressed = False
        for batch in dataloader:
            if global_step >= config.total_steps:
                break

            progressed = True
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % config.gradient_accumulation_steps == 0:
                if config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [param for param in model.parameters() if param.requires_grad],
                        config.gradient_clip,
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            running_loss_acc += loss.detach() * config.gradient_accumulation_steps

            if global_step % config.log_every_steps == 0:
                logger.info(
                    "Recovery step %s/%s | loss %.4f",
                    global_step,
                    config.total_steps,
                    running_loss_acc.item() / config.log_every_steps,
                )
                running_loss_acc.zero_()

            if global_step % config.save_every_steps == 0:
                checkpoint_path = str(Path(config.output_dir) / f"memory_step_{global_step}.pt")
                save_memory_checkpoint(model, layer_indices, checkpoint_path, step=global_step)

        if not progressed:
            raise ValueError(
                "Recovery dataloader produced zero batches. "
                "Reduce data.seq_length or provide more recovery text."
            )

    final_path = str(Path(config.output_dir) / "memory.pt")
    save_memory_checkpoint(model, layer_indices, final_path, step=global_step)
    return final_path


def evaluate_recovery_checkpoint(
    cfg: ExperimentConfig,
    recovery_checkpoint: str,
    device: str,
) -> dict:
    losses = compare_models_on_loss(
        cfg=cfg,
        device=device,
        eval_data_path=cfg.recovery.eval_data_path,
        max_batches=cfg.recovery.eval_max_batches,
        recovery_checkpoint=recovery_checkpoint,
    )
    prompt_specs = load_prompt_specs(cfg.recovery.sanity_prompts_path)
    base_model, tokenizer = build_model(cfg, checkpoint=None)
    recovery_model, _ = build_model(cfg, checkpoint=recovery_checkpoint)
    prompt_results = evaluate_prompt_specs(
        base_model=base_model,
        recovery_model=recovery_model,
        tokenizer=tokenizer,
        prompt_specs=prompt_specs,
        device=device,
        max_new_tokens=48,
    )
    return {"losses": losses, "prompt_results": prompt_results}
