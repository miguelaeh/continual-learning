from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from self_building_brain.benchmarks.realistic import QwenTextBackend
from self_building_brain.data.realistic_mc import collate_mc_examples, load_longbench_mc_examples, train_eval_split
from self_building_brain.models.text_memory_reasoners import (
    GrowingSlotTextReasoner,
    SlotTextReasoner,
    StateConditionedGrowingSlotTextReasoner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fixed-slot, growing-slot, and state-conditioned slot memory on LongBench v2 MC.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--chunk-chars", type=int, default=1500)
    parser.add_argument("--max-chunks-per-example", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--memory-dim", type=int, default=256)
    parser.add_argument("--num-slots", type=int, default=24)
    parser.add_argument("--allocation-threshold", type=float, default=0.55)
    parser.add_argument("--update-momentum", type=float, default=0.7)
    parser.add_argument("--novelty-loss-weight", type=float, default=0.2)
    parser.add_argument("--separation-loss-weight", type=float, default=0.05)
    parser.add_argument("--key-contrastive-loss-weight", type=float, default=0.1)
    parser.add_argument("--positive-sim-threshold", type=float, default=0.8)
    parser.add_argument("--negative-sim-threshold", type=float, default=0.55)
    parser.add_argument("--negative-key-margin", type=float, default=0.35)
    parser.add_argument("--cache-path", type=str, default="outputs/longbench_embeddings_48_coarse.pt")
    parser.add_argument("--output-path", type=str, default="outputs/growing_slot_longbench_results.json")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/growing_slot_checkpoints")
    return parser.parse_args()


def iterate_batches(examples, batch_size: int, seed: int):
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [examples[index] for index in indices[start : start + batch_size]]


def compute_key_contrastive_loss(
    outputs: dict[str, torch.Tensor],
    chunk_embeddings: torch.Tensor,
    chunk_mask: torch.Tensor,
    positive_sim_threshold: float,
    negative_sim_threshold: float,
    negative_key_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    metrics: dict[str, float] = {}
    if "key_vectors" not in outputs:
        return torch.zeros((), device=chunk_mask.device), metrics

    normalized_input = F.normalize(chunk_embeddings, dim=-1, eps=1e-8)
    normalized_keys = F.normalize(outputs["key_vectors"], dim=-1, eps=1e-8)
    input_similarity = torch.matmul(normalized_input, normalized_input.transpose(1, 2))
    key_similarity = torch.matmul(normalized_keys, normalized_keys.transpose(1, 2))

    valid_pairs = (chunk_mask.unsqueeze(1) > 0) & (chunk_mask.unsqueeze(2) > 0)
    diagonal = torch.eye(chunk_mask.size(1), dtype=torch.bool, device=chunk_mask.device).unsqueeze(0)
    valid_pairs = valid_pairs & ~diagonal

    positive_pairs = valid_pairs & (input_similarity >= positive_sim_threshold)
    negative_pairs = valid_pairs & (input_similarity <= negative_sim_threshold)

    positive_loss = torch.zeros((), device=chunk_mask.device)
    negative_loss = torch.zeros((), device=chunk_mask.device)

    if bool(positive_pairs.any().item()):
        positive_loss = (1.0 - key_similarity.masked_select(positive_pairs)).pow(2).mean()
        metrics["positive_pair_fraction"] = float(positive_pairs.float().mean().item())
    if bool(negative_pairs.any().item()):
        negative_scores = key_similarity.masked_select(negative_pairs)
        negative_loss = F.relu(negative_scores - negative_key_margin).pow(2).mean()
        metrics["negative_pair_fraction"] = float(negative_pairs.float().mean().item())

    contrastive_loss = positive_loss + negative_loss
    if bool(valid_pairs.any().item()):
        metrics["mean_key_similarity"] = float(key_similarity.masked_select(valid_pairs).mean().item())
        metrics["mean_input_similarity"] = float(input_similarity.masked_select(valid_pairs).mean().item())
    return contrastive_loss, metrics


def compute_auxiliary_losses(
    outputs: dict[str, torch.Tensor],
    chunk_embeddings: torch.Tensor,
    chunk_mask: torch.Tensor,
    positive_sim_threshold: float,
    negative_sim_threshold: float,
    negative_key_margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    metrics: dict[str, float] = {}
    novelty_loss = torch.zeros((), device=chunk_mask.device)
    separation_loss = torch.zeros((), device=chunk_mask.device)
    contrastive_loss = torch.zeros((), device=chunk_mask.device)

    if "novelty_logits" in outputs:
        valid = chunk_mask > 0
        novelty_logits = outputs["novelty_logits"][valid]
        novelty_targets = outputs["novelty_targets"][valid]
        if novelty_logits.numel() > 0:
            novelty_loss = F.binary_cross_entropy_with_logits(novelty_logits, novelty_targets)
            novelty_predictions = (torch.sigmoid(novelty_logits) >= 0.5).float()
            metrics["novelty_accuracy"] = float((novelty_predictions == novelty_targets).float().mean().item())
            metrics["mean_max_similarity"] = float(outputs["max_similarities"][valid].mean().item())

    if "slot_separation_loss" in outputs:
        separation_loss = outputs["slot_separation_loss"]
        metrics["slot_separation_loss"] = float(separation_loss.item())

    contrastive_loss, contrastive_metrics = compute_key_contrastive_loss(
        outputs=outputs,
        chunk_embeddings=chunk_embeddings,
        chunk_mask=chunk_mask,
        positive_sim_threshold=positive_sim_threshold,
        negative_sim_threshold=negative_sim_threshold,
        negative_key_margin=negative_key_margin,
    )
    metrics.update(contrastive_metrics)

    return novelty_loss, separation_loss, contrastive_loss, metrics


def evaluate_model(
    model,
    examples,
    batch_size: int,
    device: torch.device,
    novelty_loss_weight: float,
    separation_loss_weight: float,
    key_contrastive_loss_weight: float,
    positive_sim_threshold: float,
    negative_sim_threshold: float,
    negative_key_margin: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    total_active_slots = 0.0
    total_allocations = 0.0
    total_write_steps = 0.0
    total_novelty_accuracy = 0.0
    total_similarity = 0.0
    total_separation = 0.0
    total_key_similarity = 0.0
    novelty_batches = 0

    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            batch = collate_mc_examples(batch_examples, device=device)
            outputs = model(
                chunk_embeddings=batch["chunk_embeddings"],
                chunk_mask=batch["chunk_mask"],
                query_embeddings=batch["query_embeddings"],
                choice_embeddings=batch["choice_embeddings"],
            )
            answer_loss = F.cross_entropy(outputs["choice_logits"], batch["answer_indices"])
            novelty_loss, separation_loss, contrastive_loss, aux_metrics = compute_auxiliary_losses(
                outputs,
                batch["chunk_embeddings"],
                batch["chunk_mask"],
                positive_sim_threshold=positive_sim_threshold,
                negative_sim_threshold=negative_sim_threshold,
                negative_key_margin=negative_key_margin,
            )
            loss = (
                answer_loss
                + novelty_loss_weight * novelty_loss
                + separation_loss_weight * separation_loss
                + key_contrastive_loss_weight * contrastive_loss
            )
            predictions = outputs["choice_logits"].argmax(dim=-1)
            batch_size_actual = len(batch_examples)
            total_loss += float(loss.item()) * batch_size_actual
            total_correct += float((predictions == batch["answer_indices"]).float().sum().item())
            total_count += batch_size_actual

            if "active_counts" in outputs:
                total_active_slots += float(outputs["active_counts"].float().sum().item())
            if "allocation_mask" in outputs:
                total_allocations += float(outputs["allocation_mask"].sum().item())
                total_write_steps += float(batch["chunk_mask"].sum().item())
            if "novelty_accuracy" in aux_metrics:
                total_novelty_accuracy += aux_metrics["novelty_accuracy"] * batch_size_actual
                total_similarity += aux_metrics["mean_max_similarity"] * batch_size_actual
                novelty_batches += batch_size_actual
            if "slot_separation_loss" in aux_metrics:
                total_separation += aux_metrics["slot_separation_loss"] * batch_size_actual
            if "mean_key_similarity" in aux_metrics:
                total_key_similarity += aux_metrics["mean_key_similarity"] * batch_size_actual

    metrics = {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
    }
    if total_active_slots > 0:
        metrics["mean_active_slots"] = total_active_slots / total_count
    if total_write_steps > 0:
        metrics["allocation_rate"] = total_allocations / total_write_steps
        metrics["mean_allocations_per_example"] = total_allocations / total_count
    if novelty_batches > 0:
        metrics["novelty_accuracy"] = total_novelty_accuracy / novelty_batches
        metrics["mean_max_similarity"] = total_similarity / novelty_batches
    if total_separation > 0:
        metrics["slot_separation_loss"] = total_separation / total_count
    if total_key_similarity != 0.0:
        metrics["mean_key_similarity"] = total_key_similarity / total_count
    return metrics


def train_model(
    model,
    train_examples,
    eval_examples,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    novelty_loss_weight: float,
    separation_loss_weight: float,
    key_contrastive_loss_weight: float,
    positive_sim_threshold: float,
    negative_sim_threshold: float,
    negative_key_margin: float,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_examples in iterate_batches(train_examples, batch_size=batch_size, seed=epoch):
            batch = collate_mc_examples(batch_examples, device=device)
            outputs = model(
                chunk_embeddings=batch["chunk_embeddings"],
                chunk_mask=batch["chunk_mask"],
                query_embeddings=batch["query_embeddings"],
                choice_embeddings=batch["choice_embeddings"],
            )
            answer_loss = F.cross_entropy(outputs["choice_logits"], batch["answer_indices"])
            novelty_loss, separation_loss, contrastive_loss, _ = compute_auxiliary_losses(
                outputs,
                batch["chunk_embeddings"],
                batch["chunk_mask"],
                positive_sim_threshold=positive_sim_threshold,
                negative_sim_threshold=negative_sim_threshold,
                negative_key_margin=negative_key_margin,
            )
            loss = (
                answer_loss
                + novelty_loss_weight * novelty_loss
                + separation_loss_weight * separation_loss
                + key_contrastive_loss_weight * contrastive_loss
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_model(
            model,
            train_examples,
            batch_size=batch_size,
            device=device,
            novelty_loss_weight=novelty_loss_weight,
            separation_loss_weight=separation_loss_weight,
            key_contrastive_loss_weight=key_contrastive_loss_weight,
            positive_sim_threshold=positive_sim_threshold,
            negative_sim_threshold=negative_sim_threshold,
            negative_key_margin=negative_key_margin,
        )
        eval_metrics = evaluate_model(
            model,
            eval_examples,
            batch_size=batch_size,
            device=device,
            novelty_loss_weight=novelty_loss_weight,
            separation_loss_weight=separation_loss_weight,
            key_contrastive_loss_weight=key_contrastive_loss_weight,
            positive_sim_threshold=positive_sim_threshold,
            negative_sim_threshold=negative_sim_threshold,
            negative_key_margin=negative_key_margin,
        )
        history.append({"epoch": epoch, "train": train_metrics, "eval": eval_metrics})
        print(
            f"epoch={epoch:02d} "
            f"train_acc={train_metrics['accuracy']:.3f} train_loss={train_metrics['loss']:.4f} "
            f"eval_acc={eval_metrics['accuracy']:.3f} eval_loss={eval_metrics['loss']:.4f}"
        )
        if "mean_active_slots" in eval_metrics:
            print(
                f"           eval_mean_active_slots={eval_metrics['mean_active_slots']:.3f} "
                f"eval_allocation_rate={eval_metrics['allocation_rate']:.3f}"
            )
        if "novelty_accuracy" in eval_metrics:
            print(
                f"           eval_novelty_acc={eval_metrics['novelty_accuracy']:.3f} "
                f"eval_mean_max_similarity={eval_metrics['mean_max_similarity']:.3f} "
                f"eval_slot_sep={eval_metrics.get('slot_separation_loss', 0.0):.4f} "
                f"eval_key_sim={eval_metrics.get('mean_key_similarity', 0.0):.3f}"
            )
    return history


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    backend = QwenTextBackend(model_name=args.model_name, device=device)
    examples = load_longbench_mc_examples(
        backend=backend,
        limit=args.limit,
        chunk_chars=args.chunk_chars,
        max_chunks_per_example=args.max_chunks_per_example,
        cache_path=args.cache_path,
    )
    train_examples, eval_examples = train_eval_split(examples, eval_fraction=0.2)
    input_dim = examples[0].chunk_embeddings.size(-1)

    models = {
        "slot": SlotTextReasoner(input_dim=input_dim, memory_dim=args.memory_dim, num_slots=args.num_slots).to(device),
        "growing_slot": GrowingSlotTextReasoner(
            input_dim=input_dim,
            memory_dim=args.memory_dim,
            max_slots=args.num_slots,
            allocation_threshold=args.allocation_threshold,
            update_momentum=args.update_momentum,
        ).to(device),
        "state_conditioned_slot": StateConditionedGrowingSlotTextReasoner(
            input_dim=input_dim,
            memory_dim=args.memory_dim,
            max_slots=args.num_slots,
            allocation_threshold=args.allocation_threshold,
        ).to(device),
    }

    results = {
        "config": vars(args),
        "dataset": {
            "num_examples": len(examples),
            "num_train": len(train_examples),
            "num_eval": len(eval_examples),
            "mean_chunks_per_example": sum(example.chunk_embeddings.size(0) for example in examples) / len(examples),
        },
        "models": {},
    }

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        print(f"training {model_name} model")
        history = train_model(
            model=model,
            train_examples=train_examples,
            eval_examples=eval_examples,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device,
            novelty_loss_weight=args.novelty_loss_weight,
            separation_loss_weight=args.separation_loss_weight,
            key_contrastive_loss_weight=args.key_contrastive_loss_weight,
            positive_sim_threshold=args.positive_sim_threshold,
            negative_sim_threshold=args.negative_sim_threshold,
            negative_key_margin=args.negative_key_margin,
        )
        final_eval = evaluate_model(
            model,
            eval_examples,
            batch_size=args.batch_size,
            device=device,
            novelty_loss_weight=args.novelty_loss_weight,
            separation_loss_weight=args.separation_loss_weight,
            key_contrastive_loss_weight=args.key_contrastive_loss_weight,
            positive_sim_threshold=args.positive_sim_threshold,
            negative_sim_threshold=args.negative_sim_threshold,
            negative_key_margin=args.negative_key_margin,
        )
        checkpoint_path = checkpoint_dir / f"{model_name}.pt"
        torch.save(
            {
                "model_name": model_name,
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "config": vars(args),
            },
            checkpoint_path,
        )
        results["models"][model_name] = {
            "history": history,
            "final_eval": final_eval,
            "checkpoint_path": str(checkpoint_path),
        }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"saved results to {output_path}")


if __name__ == "__main__":
    main()
