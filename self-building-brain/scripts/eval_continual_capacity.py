from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from self_building_brain.config import ModelConfig, SyntheticTaskConfig, TrainingConfig
from self_building_brain.data.synthetic import SyntheticFactDataset
from self_building_brain.models.system import SelfBuildingBrain
from self_building_brain.models.teacher import SyntheticTeacherTraceProvider
from self_building_brain.training.losses import compute_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark how far the slot-memory brain can sustain longer continual read streams.")
    parser.add_argument("--train-lengths", type=str, default="4,8,12,16")
    parser.add_argument("--eval-lengths", type=str, default="4,8,12,16,24,32")
    parser.add_argument("--steps-per-stage", type=int, default=80)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--accuracy-threshold", type=float, default=0.9)
    parser.add_argument("--unique-keys", action="store_true")
    parser.add_argument("--output-path", type=str, default="outputs/continual_capacity.json")
    return parser.parse_args()


def parse_lengths(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def evaluate_length(
    model: SelfBuildingBrain,
    teacher: SyntheticTeacherTraceProvider,
    dataset: SyntheticFactDataset,
    training_config: TrainingConfig,
    batch_size: int,
    num_read_steps: int,
    eval_batches: int,
    device: torch.device,
    unique_keys: bool,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "answer_accuracy": 0.0,
        "routing_accuracy": 0.0,
        "query_slot_accuracy": 0.0,
    }
    with torch.no_grad():
        for _ in range(eval_batches):
            batch = dataset.sample_batch(
                batch_size=batch_size,
                device=device,
                num_read_steps=num_read_steps,
                unique_keys=unique_keys,
            )
            teacher_targets = teacher.extract_targets(
                read_tokens=batch.read_tokens,
                read_key_ids=batch.read_key_ids,
                read_value_ids=batch.read_value_ids,
                query_key_ids=batch.query_key_ids,
            )
            outputs = model(read_tokens=batch.read_tokens, query_tokens=batch.query_tokens)
            metrics = compute_losses(outputs=outputs, batch=batch, teacher_targets=teacher_targets, config=training_config)
            totals["loss"] += float(metrics["loss"].item())
            totals["answer_accuracy"] += float(metrics["answer_accuracy"].item())
            totals["routing_accuracy"] += float(metrics["routing_accuracy"].item())
            totals["query_slot_accuracy"] += float(metrics["query_slot_accuracy"].item())

    return {key: value / eval_batches for key, value in totals.items()}


def max_supported_length(results: dict[int, dict[str, float]], threshold: float) -> int:
    supported = [length for length, metrics in results.items() if metrics["answer_accuracy"] >= threshold]
    return max(supported) if supported else 0


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    train_lengths = parse_lengths(args.train_lengths)
    eval_lengths = parse_lengths(args.eval_lengths)

    task_config = SyntheticTaskConfig()
    model_config = ModelConfig(num_slots=task_config.num_slots)
    training_config = TrainingConfig(batch_size=args.batch_size, steps=args.steps_per_stage, log_every=max(1, args.steps_per_stage // 4))

    dataset = SyntheticFactDataset(task_config)
    model = SelfBuildingBrain(
        vocab_size=task_config.vocab_size,
        hidden_dim=model_config.hidden_dim,
        slot_dim=model_config.slot_dim,
        num_slots=model_config.num_slots,
        num_values=task_config.num_values,
    ).to(device)
    teacher = SyntheticTeacherTraceProvider(
        vocab_size=task_config.vocab_size,
        num_slots=task_config.num_slots,
        num_values=task_config.num_values,
        slot_dim=model_config.slot_dim,
        hidden_dim=model_config.hidden_dim,
    ).to(device)
    teacher.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    stage_results: list[dict[str, object]] = []
    for stage_index, train_length in enumerate(train_lengths, start=1):
        model.train()
        for step in range(1, args.steps_per_stage + 1):
            batch = dataset.sample_batch(
                batch_size=args.batch_size,
                device=device,
                num_read_steps=train_length,
                unique_keys=args.unique_keys,
            )
            teacher_targets = teacher.extract_targets(
                read_tokens=batch.read_tokens,
                read_key_ids=batch.read_key_ids,
                read_value_ids=batch.read_value_ids,
                query_key_ids=batch.query_key_ids,
            )
            outputs = model(read_tokens=batch.read_tokens, query_tokens=batch.query_tokens)
            metrics = compute_losses(outputs=outputs, batch=batch, teacher_targets=teacher_targets, config=training_config)

            optimizer.zero_grad()
            metrics["loss"].backward()
            optimizer.step()

            if step == 1 or step % training_config.log_every == 0:
                print(
                    f"stage={stage_index:02d} train_length={train_length:02d} step={step:04d} "
                    f"loss={metrics['loss'].item():.4f} answer_acc={metrics['answer_accuracy'].item():.3f}"
                )

        evaluation = {
            length: evaluate_length(
                model=model,
                teacher=teacher,
                dataset=dataset,
                training_config=training_config,
                batch_size=args.batch_size,
                num_read_steps=length,
                eval_batches=args.eval_batches,
                device=device,
                unique_keys=args.unique_keys,
            )
            for length in eval_lengths
        }
        support = max_supported_length(evaluation, threshold=args.accuracy_threshold)
        print(f"stage={stage_index:02d} train_length={train_length:02d} max_supported_length={support}")
        stage_results.append(
            {
                "stage_index": stage_index,
                "train_length": train_length,
                "max_supported_length": support,
                "evaluation": evaluation,
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_config": asdict(task_config),
        "model_config": asdict(model_config),
        "training_config": asdict(training_config),
        "train_lengths": train_lengths,
        "eval_lengths": eval_lengths,
        "accuracy_threshold": args.accuracy_threshold,
        "unique_keys": args.unique_keys,
        "results": stage_results,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"saved continual-capacity report to {output_path}")


if __name__ == "__main__":
    main()
