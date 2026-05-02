from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from self_building_brain.config import ModelConfig, QwenTeacherConfig, SyntheticTaskConfig, TrainingConfig
from self_building_brain.data.text_facts import TextFactDataset
from self_building_brain.models.system import SelfBuildingBrain
from self_building_brain.models.teacher import QwenTeacherTraceProvider
from self_building_brain.training.losses import compute_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill a persistent brain state from Qwen hidden-state traces.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--teacher-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--teacher-layer", type=int, default=-1)
    parser.add_argument("--teacher-max-length", type=int, default=64)
    parser.add_argument("--checkpoint-path", type=str, default="outputs/qwen_distill_brain.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    task_config = SyntheticTaskConfig()
    model_config = ModelConfig(num_slots=task_config.num_slots)
    training_config = TrainingConfig(steps=args.steps, batch_size=args.batch_size, log_every=max(1, args.steps // 5))
    teacher_config = QwenTeacherConfig(
        model_name=args.teacher_model,
        max_length=args.teacher_max_length,
        layer_index=args.teacher_layer,
        slot_projection_dim=model_config.slot_dim,
    )

    dataset = TextFactDataset(task_config)
    model = SelfBuildingBrain(
        vocab_size=task_config.vocab_size,
        hidden_dim=model_config.hidden_dim,
        slot_dim=model_config.slot_dim,
        num_slots=model_config.num_slots,
        num_values=task_config.num_values,
    ).to(device)
    teacher = QwenTeacherTraceProvider(
        model_name=teacher_config.model_name,
        num_slots=model_config.num_slots,
        slot_dim=model_config.slot_dim,
        max_length=teacher_config.max_length,
        layer_index=teacher_config.layer_index,
        local_files_only=teacher_config.local_files_only,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    for step in range(1, training_config.steps + 1):
        batch = dataset.sample_batch(training_config.batch_size, device=device)
        teacher_targets = teacher.extract_targets(
            read_texts=batch.read_texts,
            read_key_ids=batch.read_key_ids.to(device),
            query_key_ids=batch.query_key_ids.to(device),
        )

        outputs = model(read_tokens=batch.read_tokens, query_tokens=batch.query_tokens)
        metrics = compute_losses(outputs=outputs, batch=batch, teacher_targets=teacher_targets, config=training_config)

        optimizer.zero_grad()
        metrics["loss"].backward()
        optimizer.step()

        if step == 1 or step % training_config.log_every == 0:
            print(
                f"step={step:04d} "
                f"loss={metrics['loss'].item():.4f} "
                f"answer_acc={metrics['answer_accuracy'].item():.3f} "
                f"routing_acc={metrics['routing_accuracy'].item():.3f} "
                f"query_slot_acc={metrics['query_slot_accuracy'].item():.3f}"
            )

    eval_batch = dataset.sample_batch(training_config.batch_size, device=device)
    eval_targets = teacher.extract_targets(
        read_texts=eval_batch.read_texts,
        read_key_ids=eval_batch.read_key_ids.to(device),
        query_key_ids=eval_batch.query_key_ids.to(device),
    )
    model.eval()
    with torch.no_grad():
        eval_outputs = model(read_tokens=eval_batch.read_tokens, query_tokens=eval_batch.query_tokens)
        eval_metrics = compute_losses(
            outputs=eval_outputs,
            batch=eval_batch,
            teacher_targets=eval_targets,
            config=training_config,
        )
    print(
        f"eval loss={eval_metrics['loss'].item():.4f} "
        f"answer_acc={eval_metrics['answer_accuracy'].item():.3f} "
        f"routing_acc={eval_metrics['routing_accuracy'].item():.3f} "
        f"query_slot_acc={eval_metrics['query_slot_accuracy'].item():.3f}"
    )

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "task_config": asdict(task_config),
            "model_config": asdict(model_config),
            "training_config": asdict(training_config),
            "teacher_config": asdict(teacher_config),
            "teacher": "qwen_trace",
        },
        checkpoint_path,
    )
    print(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
