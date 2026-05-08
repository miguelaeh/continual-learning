from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from self_building_brain.benchmarks.realistic import QwenTextBackend
from self_building_brain.data.realistic_mc import EmbeddedMCExample, collate_mc_examples, load_longbench_mc_examples, train_eval_split
from self_building_brain.models.text_memory_reasoners import StateConditionedGrowingSlotTextReasoner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the state-conditioned brain generator on larger mixed-length LongBench streams.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--chunk-chars", type=int, default=400)
    parser.add_argument("--max-chunks-per-example", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--memory-dim", type=int, default=256)
    parser.add_argument("--num-slots", type=int, default=24)
    parser.add_argument("--allocation-threshold", type=float, default=0.9)
    parser.add_argument("--novelty-loss-weight", type=float, default=0.2)
    parser.add_argument("--separation-loss-weight", type=float, default=0.05)
    parser.add_argument("--key-contrastive-loss-weight", type=float, default=0.1)
    parser.add_argument("--positive-sim-threshold", type=float, default=0.8)
    parser.add_argument("--negative-sim-threshold", type=float, default=0.55)
    parser.add_argument("--negative-key-margin", type=float, default=0.35)
    parser.add_argument("--min-train-chunks", type=int, default=2)
    parser.add_argument("--length-bias-power", type=float, default=2.0)
    parser.add_argument("--eval-lengths", type=str, default="2,4,6,8,12")
    parser.add_argument("--early-stop-length", type=int, default=12)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--min-epochs-before-stop", type=int, default=4)
    parser.add_argument("--cache-path", type=str, default="outputs/longbench_embeddings_120_long12.pt")
    parser.add_argument("--output-path", type=str, default="outputs/state_conditioned_generator_mixed_lengths.json")
    parser.add_argument("--checkpoint-path", type=str, default="outputs/state_conditioned_generator_mixed_lengths.pt")
    return parser.parse_args()


def iterate_batches(examples: list[EmbeddedMCExample], batch_size: int, seed: int):
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [examples[index] for index in indices[start : start + batch_size]]


def parse_eval_lengths(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def truncate_example(example: EmbeddedMCExample, target_length: int) -> EmbeddedMCExample:
    clipped = min(target_length, example.chunk_embeddings.size(0))
    return replace(example, chunk_embeddings=example.chunk_embeddings[:clipped])


def sample_train_length(available: int, min_train_chunks: int, length_bias_power: float) -> int:
    if available <= min_train_chunks:
        return available
    span = available - min_train_chunks
    biased = 1.0 - random.random() ** length_bias_power
    return min_train_chunks + int(round(span * biased))


def build_training_batch(
    batch_examples: list[EmbeddedMCExample],
    min_train_chunks: int,
    length_bias_power: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    truncated = [
        truncate_example(example, sample_train_length(example.chunk_embeddings.size(0), min_train_chunks, length_bias_power))
        for example in batch_examples
    ]
    return collate_mc_examples(truncated, device=device)


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


def compute_total_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    novelty_loss_weight: float,
    separation_loss_weight: float,
    key_contrastive_loss_weight: float,
    positive_sim_threshold: float,
    negative_sim_threshold: float,
    negative_key_margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
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
    aux_metrics["answer_loss"] = float(answer_loss.item())
    return loss, aux_metrics


def evaluate_examples(
    model: StateConditionedGrowingSlotTextReasoner,
    examples: list[EmbeddedMCExample],
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
    total_key_similarity = 0.0
    novelty_weight_count = 0

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
            loss, aux_metrics = compute_total_loss(
                outputs,
                batch,
                novelty_loss_weight=novelty_loss_weight,
                separation_loss_weight=separation_loss_weight,
                key_contrastive_loss_weight=key_contrastive_loss_weight,
                positive_sim_threshold=positive_sim_threshold,
                negative_sim_threshold=negative_sim_threshold,
                negative_key_margin=negative_key_margin,
            )
            predictions = outputs["choice_logits"].argmax(dim=-1)
            batch_size_actual = len(batch_examples)
            total_loss += float(loss.item()) * batch_size_actual
            total_correct += float((predictions == batch["answer_indices"]).float().sum().item())
            total_count += batch_size_actual
            total_active_slots += float(outputs["active_counts"].float().sum().item())
            total_allocations += float(outputs["allocation_mask"].sum().item())
            total_write_steps += float(batch["chunk_mask"].sum().item())

            if "novelty_accuracy" in aux_metrics:
                total_novelty_accuracy += aux_metrics["novelty_accuracy"] * batch_size_actual
                total_similarity += aux_metrics["mean_max_similarity"] * batch_size_actual
                total_key_similarity += aux_metrics.get("mean_key_similarity", 0.0) * batch_size_actual
                novelty_weight_count += batch_size_actual

    metrics = {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
        "mean_active_slots": total_active_slots / max(total_count, 1),
        "allocation_rate": total_allocations / max(total_write_steps, 1.0),
        "mean_allocations_per_example": total_allocations / max(total_count, 1),
    }
    if novelty_weight_count > 0:
        metrics["novelty_accuracy"] = total_novelty_accuracy / novelty_weight_count
        metrics["mean_max_similarity"] = total_similarity / novelty_weight_count
        metrics["mean_key_similarity"] = total_key_similarity / novelty_weight_count
    return metrics


def evaluate_by_length(
    model: StateConditionedGrowingSlotTextReasoner,
    eval_examples: list[EmbeddedMCExample],
    eval_lengths: list[int],
    batch_size: int,
    device: torch.device,
    novelty_loss_weight: float,
    separation_loss_weight: float,
    key_contrastive_loss_weight: float,
    positive_sim_threshold: float,
    negative_sim_threshold: float,
    negative_key_margin: float,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for length in eval_lengths:
        bucket_examples = [truncate_example(example, length) for example in eval_examples if example.chunk_embeddings.size(0) >= length]
        if not bucket_examples:
            continue
        results[str(length)] = evaluate_examples(
            model,
            bucket_examples,
            batch_size=batch_size,
            device=device,
            novelty_loss_weight=novelty_loss_weight,
            separation_loss_weight=separation_loss_weight,
            key_contrastive_loss_weight=key_contrastive_loss_weight,
            positive_sim_threshold=positive_sim_threshold,
            negative_sim_threshold=negative_sim_threshold,
            negative_key_margin=negative_key_margin,
        )
    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    eval_lengths = parse_eval_lengths(args.eval_lengths)

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

    model = StateConditionedGrowingSlotTextReasoner(
        input_dim=input_dim,
        memory_dim=args.memory_dim,
        max_slots=args.num_slots,
        allocation_threshold=args.allocation_threshold,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    best_epoch = 0
    best_score = float("-inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    early_stop_length = str(args.early_stop_length)
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_examples in iterate_batches(train_examples, batch_size=args.batch_size, seed=epoch):
            batch = build_training_batch(
                batch_examples,
                min_train_chunks=args.min_train_chunks,
                length_bias_power=args.length_bias_power,
                device=device,
            )
            outputs = model(
                chunk_embeddings=batch["chunk_embeddings"],
                chunk_mask=batch["chunk_mask"],
                query_embeddings=batch["query_embeddings"],
                choice_embeddings=batch["choice_embeddings"],
            )
            loss, _ = compute_total_loss(
                outputs,
                batch,
                novelty_loss_weight=args.novelty_loss_weight,
                separation_loss_weight=args.separation_loss_weight,
                key_contrastive_loss_weight=args.key_contrastive_loss_weight,
                positive_sim_threshold=args.positive_sim_threshold,
                negative_sim_threshold=args.negative_sim_threshold,
                negative_key_margin=args.negative_key_margin,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_examples(
            model,
            train_examples,
            batch_size=args.batch_size,
            device=device,
            novelty_loss_weight=args.novelty_loss_weight,
            separation_loss_weight=args.separation_loss_weight,
            key_contrastive_loss_weight=args.key_contrastive_loss_weight,
            positive_sim_threshold=args.positive_sim_threshold,
            negative_sim_threshold=args.negative_sim_threshold,
            negative_key_margin=args.negative_key_margin,
        )
        eval_by_length = evaluate_by_length(
            model,
            eval_examples,
            eval_lengths=eval_lengths,
            batch_size=args.batch_size,
            device=device,
            novelty_loss_weight=args.novelty_loss_weight,
            separation_loss_weight=args.separation_loss_weight,
            key_contrastive_loss_weight=args.key_contrastive_loss_weight,
            positive_sim_threshold=args.positive_sim_threshold,
            negative_sim_threshold=args.negative_sim_threshold,
            negative_key_margin=args.negative_key_margin,
        )
        history.append({"epoch": epoch, "train": train_metrics, "eval_by_length": eval_by_length})

        summary_length = early_stop_length if early_stop_length in eval_by_length else str(max(eval_lengths))
        summary = eval_by_length.get(summary_length) or next(iter(eval_by_length.values()))
        score = float(summary["accuracy"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:02d} "
            f"train_acc={train_metrics['accuracy']:.3f} train_loss={train_metrics['loss']:.4f} "
            f"eval_len={summary_length} acc={summary['accuracy']:.3f} loss={summary['loss']:.4f} "
            f"active={summary['mean_active_slots']:.3f} alloc_rate={summary['allocation_rate']:.3f}"
        )
        if epoch >= args.min_epochs_before_stop and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"early stopping at epoch={epoch:02d} "
                f"best_epoch={best_epoch:02d} best_len={summary_length} best_acc={best_score:.3f}"
            )
            break

    model.load_state_dict(best_state_dict)

    final_eval_by_length = evaluate_by_length(
        model,
        eval_examples,
        eval_lengths=eval_lengths,
        batch_size=args.batch_size,
        device=device,
        novelty_loss_weight=args.novelty_loss_weight,
        separation_loss_weight=args.separation_loss_weight,
        key_contrastive_loss_weight=args.key_contrastive_loss_weight,
        positive_sim_threshold=args.positive_sim_threshold,
        negative_sim_threshold=args.negative_sim_threshold,
        negative_key_margin=args.negative_key_margin,
    )

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "state_conditioned_slot",
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "config": vars(args),
        },
        checkpoint_path,
    )

    results = {
        "config": vars(args),
        "dataset": {
            "num_examples": len(examples),
            "num_train": len(train_examples),
            "num_eval": len(eval_examples),
            "mean_chunks_per_example": sum(example.chunk_embeddings.size(0) for example in examples) / len(examples),
        },
        "history": history,
        "best_epoch": best_epoch,
        "best_eval_length": args.early_stop_length,
        "best_accuracy": best_score,
        "final_eval_by_length": final_eval_by_length,
        "checkpoint_path": str(checkpoint_path),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"saved results to {output_path}")


if __name__ == "__main__":
    main()
