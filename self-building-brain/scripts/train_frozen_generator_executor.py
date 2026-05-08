from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from self_building_brain.benchmarks.realistic import QwenTextBackend
from self_building_brain.data.realistic_mc import EmbeddedMCExample, collate_mc_examples, load_longbench_mc_examples, train_eval_split
from self_building_brain.models.text_memory_reasoners import FrozenBrainExecutor, StateConditionedGrowingSlotTextReasoner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stronger executor on top of a frozen brain generator.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--chunk-chars", type=int, default=400)
    parser.add_argument("--max-chunks-per-example", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--min-train-chunks", type=int, default=2)
    parser.add_argument("--length-bias-power", type=float, default=2.0)
    parser.add_argument("--eval-lengths", type=str, default="2,4,6,8,12")
    parser.add_argument("--early-stop-length", type=int, default=12)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--min-epochs-before-stop", type=int, default=4)
    parser.add_argument("--cache-path", type=str, default="outputs/longbench_embeddings_120_long12.pt")
    parser.add_argument("--generator-checkpoint", type=str, default="outputs/state_conditioned_generator_mixed_lengths_120_longer.pt")
    parser.add_argument("--output-path", type=str, default="outputs/frozen_generator_executor_120.json")
    parser.add_argument("--checkpoint-path", type=str, default="outputs/frozen_generator_executor_120.pt")
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


def build_frozen_generator(checkpoint_path: str, device: torch.device) -> tuple[StateConditionedGrowingSlotTextReasoner, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = StateConditionedGrowingSlotTextReasoner(
        input_dim=checkpoint["input_dim"],
        memory_dim=config["memory_dim"],
        max_slots=config["num_slots"],
        allocation_threshold=config["allocation_threshold"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def run_generator(
    generator: StateConditionedGrowingSlotTextReasoner,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        outputs = generator(
            chunk_embeddings=batch["chunk_embeddings"],
            chunk_mask=batch["chunk_mask"],
            query_embeddings=batch["query_embeddings"],
            choice_embeddings=batch["choice_embeddings"],
        )
    return outputs


def evaluate_examples(
    generator: StateConditionedGrowingSlotTextReasoner,
    executor: FrozenBrainExecutor,
    examples: list[EmbeddedMCExample],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    executor.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
    total_active_slots = 0.0
    total_allocations = 0.0
    total_write_steps = 0.0

    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            batch = collate_mc_examples(batch_examples, device=device)
            generator_outputs = run_generator(generator, batch)
            executor_outputs = executor(
                memory=generator_outputs["final_memory"],
                active_mask=generator_outputs["active_mask"],
                query_embeddings=batch["query_embeddings"],
                choice_embeddings=batch["choice_embeddings"],
            )
            loss = F.cross_entropy(executor_outputs["choice_logits"], batch["answer_indices"])
            predictions = executor_outputs["choice_logits"].argmax(dim=-1)
            batch_size_actual = len(batch_examples)
            total_loss += float(loss.item()) * batch_size_actual
            total_correct += float((predictions == batch["answer_indices"]).float().sum().item())
            total_count += batch_size_actual
            total_active_slots += float(generator_outputs["active_counts"].float().sum().item())
            total_allocations += float(generator_outputs["allocation_mask"].sum().item())
            total_write_steps += float(batch["chunk_mask"].sum().item())

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
        "mean_active_slots": total_active_slots / max(total_count, 1),
        "allocation_rate": total_allocations / max(total_write_steps, 1.0),
        "mean_allocations_per_example": total_allocations / max(total_count, 1),
    }


def evaluate_by_length(
    generator: StateConditionedGrowingSlotTextReasoner,
    executor: FrozenBrainExecutor,
    eval_examples: list[EmbeddedMCExample],
    eval_lengths: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for length in eval_lengths:
        bucket_examples = [truncate_example(example, length) for example in eval_examples if example.chunk_embeddings.size(0) >= length]
        if not bucket_examples:
            continue
        results[str(length)] = evaluate_examples(
            generator,
            executor,
            bucket_examples,
            batch_size=batch_size,
            device=device,
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

    generator, generator_checkpoint = build_frozen_generator(args.generator_checkpoint, device=device)
    executor = FrozenBrainExecutor(
        input_dim=generator_checkpoint["input_dim"],
        memory_dim=generator_checkpoint["config"]["memory_dim"],
    ).to(device)
    optimizer = torch.optim.Adam(executor.parameters(), lr=args.learning_rate)

    history = []
    best_epoch = 0
    best_score = float("-inf")
    best_state_dict = executor.state_dict()
    early_stop_length = str(args.early_stop_length)
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        executor.train()
        for batch_examples in iterate_batches(train_examples, batch_size=args.batch_size, seed=epoch):
            batch = build_training_batch(
                batch_examples,
                min_train_chunks=args.min_train_chunks,
                length_bias_power=args.length_bias_power,
                device=device,
            )
            generator_outputs = run_generator(generator, batch)
            executor_outputs = executor(
                memory=generator_outputs["final_memory"],
                active_mask=generator_outputs["active_mask"],
                query_embeddings=batch["query_embeddings"],
                choice_embeddings=batch["choice_embeddings"],
            )
            loss = F.cross_entropy(executor_outputs["choice_logits"], batch["answer_indices"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_examples(
            generator,
            executor,
            train_examples,
            batch_size=args.batch_size,
            device=device,
        )
        eval_by_length = evaluate_by_length(
            generator,
            executor,
            eval_examples,
            eval_lengths=eval_lengths,
            batch_size=args.batch_size,
            device=device,
        )
        history.append({"epoch": epoch, "train": train_metrics, "eval_by_length": eval_by_length})

        summary_length = early_stop_length if early_stop_length in eval_by_length else str(max(eval_lengths))
        summary = eval_by_length.get(summary_length) or next(iter(eval_by_length.values()))
        score = float(summary["accuracy"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state_dict = {key: value.detach().cpu().clone() for key, value in executor.state_dict().items()}
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

    executor.load_state_dict(best_state_dict)

    final_eval_by_length = evaluate_by_length(
        generator,
        executor,
        eval_examples,
        eval_lengths=eval_lengths,
        batch_size=args.batch_size,
        device=device,
    )

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "frozen_generator_executor",
            "executor_state_dict": executor.state_dict(),
            "generator_checkpoint": args.generator_checkpoint,
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
        "generator_checkpoint": args.generator_checkpoint,
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
