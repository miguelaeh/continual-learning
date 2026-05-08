from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from self_building_brain.benchmarks.realistic import QwenTextBackend
from self_building_brain.data.realistic_mc import collate_mc_examples, load_longbench_mc_examples, train_eval_split
from self_building_brain.models.text_memory_reasoners import SlotTextReasoner, StrictGraphTextReasoner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train slot vs strict graph-dependent memory on LongBench v2 MC.")
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
    parser.add_argument("--cache-path", type=str, default="outputs/longbench_embeddings_48_coarse.pt")
    parser.add_argument("--output-path", type=str, default="outputs/strict_graph_training_results.json")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/strict_graph_checkpoints")
    return parser.parse_args()


def iterate_batches(examples, batch_size: int, seed: int):
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [examples[index] for index in indices[start : start + batch_size]]


def evaluate_model(model, examples, batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_count = 0
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
            loss = F.cross_entropy(outputs["choice_logits"], batch["answer_indices"])
            predictions = outputs["choice_logits"].argmax(dim=-1)
            total_loss += float(loss.item()) * len(batch_examples)
            total_correct += float((predictions == batch["answer_indices"]).float().sum().item())
            total_count += len(batch_examples)
    return {"loss": total_loss / total_count, "accuracy": total_correct / total_count}


def train_model(model, train_examples, eval_examples, batch_size: int, epochs: int, learning_rate: float, device: torch.device):
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
            loss = F.cross_entropy(outputs["choice_logits"], batch["answer_indices"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics = evaluate_model(model, train_examples, batch_size=batch_size, device=device)
        eval_metrics = evaluate_model(model, eval_examples, batch_size=batch_size, device=device)
        history.append({"epoch": epoch, "train": train_metrics, "eval": eval_metrics})
        print(
            f"epoch={epoch:02d} "
            f"train_acc={train_metrics['accuracy']:.3f} train_loss={train_metrics['loss']:.4f} "
            f"eval_acc={eval_metrics['accuracy']:.3f} eval_loss={eval_metrics['loss']:.4f}"
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
        "strict_graph": StrictGraphTextReasoner(input_dim=input_dim, memory_dim=args.memory_dim, num_nodes=args.num_slots).to(device),
    }

    results = {"config": vars(args), "models": {}}
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
        )
        final_eval = evaluate_model(model, eval_examples, batch_size=args.batch_size, device=device)
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
