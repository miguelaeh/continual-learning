from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from self_building_brain.config import GraphModelConfig, SyntheticTaskConfig, TrainingConfig
from self_building_brain.data.synthetic import SyntheticFactDataset
from self_building_brain.models.graph_brain import ExecutableGraphBrain
from self_building_brain.training.graph_losses import compute_executable_graph_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the executable graph brain with fixed nonlinear runtime.")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-read-steps", type=int, default=6)
    parser.add_argument("--message-passing-steps", type=int, default=2)
    parser.add_argument("--unique-keys", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--construction-ratio", type=float, default=0.35)
    parser.add_argument("--alignment-ratio", type=float, default=0.35)
    parser.add_argument("--construction-answer-weight", type=float, default=0.1)
    parser.add_argument("--alignment-answer-weight", type=float, default=0.5)
    parser.add_argument("--execution-answer-weight", type=float, default=1.0)
    parser.add_argument("--construction-structure-scale", type=float, default=2.0)
    parser.add_argument("--alignment-structure-scale", type=float, default=1.0)
    parser.add_argument("--execution-structure-scale", type=float, default=0.35)
    parser.add_argument("--checkpoint-path", type=str, default="outputs/executable_graph_brain.pt")
    return parser.parse_args()


def phase_name_and_weights(args: argparse.Namespace, step: int, total_steps: int) -> tuple[str, float, float]:
    construction_end = max(1, int(total_steps * args.construction_ratio))
    alignment_end = max(construction_end + 1, int(total_steps * (args.construction_ratio + args.alignment_ratio)))
    if step <= construction_end:
        return "construction", args.construction_answer_weight, args.construction_structure_scale
    if step <= alignment_end:
        return "alignment", args.alignment_answer_weight, args.alignment_structure_scale
    return "execution", args.execution_answer_weight, args.execution_structure_scale


def build_staged_loss(
    metrics: dict[str, torch.Tensor],
    config: TrainingConfig,
    answer_weight: float,
    structure_scale: float,
) -> torch.Tensor:
    structure_loss = (
        metrics["target_node_loss"]
        + metrics["source_node_loss"]
        + metrics["query_node_loss"]
        + config.read_value_loss_weight * metrics["read_value_loss"]
        + config.edge_loss_weight * metrics["edge_loss"]
        + config.node_sparsity_loss_weight * metrics["active_fraction_loss"]
        + config.edge_sparsity_loss_weight * metrics["edge_sparsity_loss"]
    )
    return answer_weight * metrics["answer_loss"] + structure_scale * structure_loss


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if args.construction_ratio + args.alignment_ratio >= 1.0:
        raise ValueError("construction-ratio + alignment-ratio must be less than 1.0 to leave room for execution.")

    task_config = SyntheticTaskConfig()
    graph_config = GraphModelConfig(num_nodes=task_config.num_slots, message_passing_steps=args.message_passing_steps)
    training_config = TrainingConfig(steps=args.steps, batch_size=args.batch_size, log_every=max(1, args.steps // 6))

    dataset = SyntheticFactDataset(task_config)
    model = ExecutableGraphBrain(
        vocab_size=task_config.vocab_size,
        hidden_dim=graph_config.hidden_dim,
        node_dim=graph_config.node_dim,
        num_nodes=graph_config.num_nodes,
        num_operators=graph_config.num_operators,
        message_passing_steps=graph_config.message_passing_steps,
        num_values=task_config.num_values,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)

    for step in range(1, training_config.steps + 1):
        batch = dataset.sample_batch(
            batch_size=training_config.batch_size,
            device=device,
            num_read_steps=args.num_read_steps,
            unique_keys=args.unique_keys,
        )
        outputs = model(read_tokens=batch.read_tokens, query_tokens=batch.query_tokens)
        metrics = compute_executable_graph_losses(outputs=outputs, batch=batch, config=training_config)
        phase, answer_weight, structure_scale = phase_name_and_weights(args, step, training_config.steps)
        loss = build_staged_loss(
            metrics=metrics,
            config=training_config,
            answer_weight=answer_weight,
            structure_scale=structure_scale,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % training_config.log_every == 0:
            print(
                f"step={step:04d} "
                f"phase={phase} "
                f"loss={loss.item():.4f} "
                f"answer_loss={metrics['answer_loss'].item():.4f} "
                f"answer_acc={metrics['answer_accuracy'].item():.3f} "
                f"target_acc={metrics['target_node_accuracy'].item():.3f} "
                f"source_acc={metrics['source_node_accuracy'].item():.3f} "
                f"query_acc={metrics['query_node_accuracy'].item():.3f}"
            )

    eval_batch = dataset.sample_batch(
        batch_size=training_config.batch_size,
        device=device,
        num_read_steps=args.num_read_steps,
        unique_keys=args.unique_keys,
    )
    model.eval()
    with torch.no_grad():
        eval_outputs = model(read_tokens=eval_batch.read_tokens, query_tokens=eval_batch.query_tokens)
        eval_metrics = compute_executable_graph_losses(outputs=eval_outputs, batch=eval_batch, config=training_config)
    print(
        f"eval loss={eval_metrics['loss'].item():.4f} "
        f"answer_acc={eval_metrics['answer_accuracy'].item():.3f} "
        f"target_acc={eval_metrics['target_node_accuracy'].item():.3f} "
        f"source_acc={eval_metrics['source_node_accuracy'].item():.3f} "
        f"query_acc={eval_metrics['query_node_accuracy'].item():.3f}"
    )

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "task_config": asdict(task_config),
            "graph_config": asdict(graph_config),
            "training_config": asdict(training_config),
            "model_type": "executable_graph_brain",
        },
        checkpoint_path,
    )
    print(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
