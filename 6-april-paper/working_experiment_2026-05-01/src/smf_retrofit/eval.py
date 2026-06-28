"""Evaluation helpers for recovery and continual checkpoints."""

from __future__ import annotations

import copy
import json

import torch

from smf_retrofit.config import ExperimentConfig, TextDataConfig
from smf_retrofit.data import create_lm_dataloader
from smf_retrofit.modeling.qwen import (
    inject_memory_layers,
    load_memory_checkpoint,
    load_model_and_tokenizer,
)


def build_model(cfg: ExperimentConfig, checkpoint: str | None):
    model, tokenizer = load_model_and_tokenizer(cfg.model)
    if checkpoint is not None:
        model, _, _ = inject_memory_layers(model, cfg.memory)
        load_memory_checkpoint(model, checkpoint)
    return model, tokenizer


def make_eval_data_config(
    cfg: ExperimentConfig,
    data_path: str | None = None,
) -> TextDataConfig:
    eval_cfg = copy.deepcopy(cfg.data)
    if data_path is not None:
        eval_cfg.path = data_path
    return eval_cfg


@torch.no_grad()
def evaluate_loss(model, dataloader, device: str, max_batches: int) -> float:
    model.eval()
    total_loss = 0.0
    count = 0

    for batch in dataloader:
        if count >= max_batches:
            break
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
        total_loss += float(outputs.loss.item())
        count += 1

    if count == 0:
        raise ValueError("Evaluation dataloader produced zero batches.")
    return total_loss / count


def format_prompt(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    use_chat_template: bool = True,
) -> str:
    model.eval()
    prompt_text = format_prompt(tokenizer, prompt) if use_chat_template else prompt
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def load_prompts(path: str | None) -> list[str]:
    if path is None:
        return [
            "Introduce yourself briefly.",
            "What is 2 plus 2?",
            "Explain in one sentence what a transformer is.",
        ]
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"Prompt file {path} must be a JSON list of strings.")
    return raw


def load_prompt_specs(path: str | None) -> list[dict]:
    raw_default = [
        {"prompt": "Introduce yourself briefly."},
        {"prompt": "What is 2 plus 2?", "expected_any": ["4", "four"]},
        {"prompt": "Explain in one sentence what a transformer is."},
    ]
    if path is None:
        return raw_default

    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"Prompt file {path} must be a JSON list.")

    specs = []
    for item in raw:
        if isinstance(item, str):
            specs.append({"prompt": item})
            continue
        if isinstance(item, dict) and isinstance(item.get("prompt"), str):
            normalized = {"prompt": item["prompt"]}
            if "expected_any" in item:
                expected_any = item["expected_any"]
                if not isinstance(expected_any, list) or not all(
                    isinstance(value, str) for value in expected_any
                ):
                    raise ValueError(
                        f"Prompt spec for '{item['prompt']}' has invalid expected_any."
                    )
                normalized["expected_any"] = expected_any
            specs.append(normalized)
            continue
        raise ValueError(f"Invalid prompt entry in {path}: {item!r}")
    return specs


def compare_models_on_loss(
    cfg: ExperimentConfig,
    device: str,
    eval_data_path: str | None,
    max_batches: int,
    recovery_checkpoint: str,
    continual_checkpoint: str | None = None,
) -> dict:
    eval_cfg = make_eval_data_config(cfg, data_path=eval_data_path)
    base_model, tokenizer = build_model(cfg, checkpoint=None)
    recovery_model, _ = build_model(cfg, checkpoint=recovery_checkpoint)

    results = {
        "base": evaluate_loss(
            base_model,
            create_lm_dataloader(eval_cfg, tokenizer),
            device,
            max_batches,
        ),
        "recovery": evaluate_loss(
            recovery_model,
            create_lm_dataloader(eval_cfg, tokenizer),
            device,
            max_batches,
        ),
    }

    if continual_checkpoint is not None:
        continual_model, _ = build_model(cfg, checkpoint=continual_checkpoint)
        results["continual"] = evaluate_loss(
            continual_model,
            create_lm_dataloader(eval_cfg, tokenizer),
            device,
            max_batches,
        )
    return results


def recovery_gate_status(
    base_loss: float,
    recovery_loss: float,
    max_loss_delta_vs_base: float,
    max_loss_ratio_vs_base: float,
    prompt_checks_passed: bool = True,
    prompt_checks_summary: list[dict] | None = None,
) -> dict:
    loss_delta = recovery_loss - base_loss
    loss_ratio = recovery_loss / max(base_loss, 1e-8)
    loss_checks_passed = (
        loss_delta <= max_loss_delta_vs_base
        and loss_ratio <= max_loss_ratio_vs_base
    )
    passed = loss_checks_passed and prompt_checks_passed
    return {
        "passed": passed,
        "loss_checks_passed": loss_checks_passed,
        "prompt_checks_passed": prompt_checks_passed,
        "loss_delta_vs_base": loss_delta,
        "loss_ratio_vs_base": loss_ratio,
        "max_loss_delta_vs_base": max_loss_delta_vs_base,
        "max_loss_ratio_vs_base": max_loss_ratio_vs_base,
        "prompt_checks_summary": prompt_checks_summary or [],
    }


def evaluate_prompt_specs(
    base_model,
    recovery_model,
    tokenizer,
    prompt_specs: list[dict],
    device: str,
    max_new_tokens: int,
) -> list[dict]:
    results = []
    for spec in prompt_specs:
        prompt = spec["prompt"]
        expected_any = spec.get("expected_any", [])
        base_text = generate_text(base_model, tokenizer, prompt, device, max_new_tokens)
        recovery_text = generate_text(
            recovery_model, tokenizer, prompt, device, max_new_tokens
        )
        passed = True
        if expected_any:
            lowered = recovery_text.lower()
            passed = any(candidate.lower() in lowered for candidate in expected_any)
        results.append(
            {
                "prompt": prompt,
                "expected_any": expected_any,
                "passed": passed,
                "base": base_text,
                "recovery": recovery_text,
            }
        )
    return results
