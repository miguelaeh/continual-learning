from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from self_building_brain.config import GraphModelConfig, SyntheticTaskConfig, TrainingConfig
from self_building_brain.data.synthetic import SyntheticFactDataset
from self_building_brain.models.graph_brain import GraphStructuredBrain
from self_building_brain.training.graph_losses import compute_graph_losses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the sparse operator-graph brain prototype.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-read-steps", type=int, default=6)
    parser.add_argument("--unique-keys", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-path", type=str, default="outputs/graph_brain.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    task_config = SyntheticTaskConfig()
    graph_config = GraphModelConfig(num_nodes=task_config.num_slots)
    training_config = TrainingConfig(steps=args.steps, batch_size=args.batch_size, log_every=max(1, args.steps // 5))

    dataset = SyntheticFactDataset(task_config)
    model = GraphStructuredBrain(
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
        metrics = compute_graph_losses(outputs=outputs, batch=batch, config=training_config)

        optimizer.zero_grad()
        metrics["loss"].backward()
        optimizer.step()

        if step == 1 or step % training_config.log_every == 0:
            print(
                f"step={step:04d} "
                f"loss={metrics['loss'].item():.4f} "
                f"answer_acc={metrics['answer_accuracy'].item():.3f} "
                f"node_acc={metrics['node_accuracy'].item():.3f} "
                f"query_node_acc={metrics['query_node_accuracy'].item():.3f}"
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
        eval_metrics = compute_graph_losses(outputs=eval_outputs, batch=eval_batch, config=training_config)
    print(
        f"eval loss={eval_metrics['loss'].item():.4f} "
        f"answer_acc={eval_metrics['answer_accuracy'].item():.3f} "
        f"node_acc={eval_metrics['node_accuracy'].item():.3f} "
        f"query_node_acc={eval_metrics['query_node_accuracy'].item():.3f}"
    )

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "task_config": asdict(task_config),
            "graph_config": asdict(graph_config),
            "training_config": asdict(training_config),
            "model_type": "graph_brain",
        },
        checkpoint_path,
    )
    print(f"saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
