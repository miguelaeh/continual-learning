from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from self_building_brain.benchmarks.realistic import QwenTextBackend
from self_building_brain.data.realistic_mc import collate_mc_examples, load_longbench_mc_examples, train_eval_split
from self_building_brain.models.text_program_graph_brain import SelfContainedTextProgramGraphBrain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the self-contained text program graph on LongBench multiple-choice QA.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--chunk-chars", type=int, default=400)
    parser.add_argument("--max-chunks-per-example", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-nodes", type=int, default=24)
    parser.add_argument("--num-operators", type=int, default=6)
    parser.add_argument("--propagation-steps", type=int, default=3)
    parser.add_argument("--choice-temperature", type=float, default=12.0)
    parser.add_argument("--routing-temperature", type=float, default=0.35)
    parser.add_argument("--routing-topk", type=int, default=2)
    parser.add_argument("--seed-weight", type=float, default=0.6)
    parser.add_argument("--message-activation-weight", type=float, default=2.0)
    parser.add_argument("--message-value-weight", type=float, default=2.0)
    parser.add_argument("--edge-sparsity-weight", type=float, default=0.01)
    parser.add_argument("--active-node-weight", type=float, default=0.005)
    parser.add_argument("--write-alignment-weight", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--min-epochs-before-stop", type=int, default=12)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cache-path", type=str, default="outputs/longbench_embeddings_120_long12.pt")
    parser.add_argument("--output-path", type=str, default="outputs/text_program_graph_longbench_results.json")
    parser.add_argument("--checkpoint-path", type=str, default="outputs/text_program_graph_longbench.pt")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iterate_batches(examples, batch_size: int, seed: int):
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [examples[index] for index in indices[start : start + batch_size]]


def compute_alignment_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    candidate_values = outputs["step_candidate_values"]
    target_chunks = batch["chunk_embeddings"]
    alignment = 1.0 - F.cosine_similarity(candidate_values, target_chunks, dim=-1)
    return (alignment * batch["chunk_mask"]).sum() / batch["chunk_mask"].sum().clamp_min(1.0)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    label_smoothing: float,
    edge_sparsity_weight: float,
    active_node_weight: float,
    write_alignment_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    answer_loss = F.cross_entropy(outputs["choice_logits"], batch["answer_indices"], label_smoothing=label_smoothing)
    edge_sparsity_loss = outputs["final_edge_weights"].mean()
    active_node_loss = outputs["final_node_active_mass"].mean()
    alignment_loss = compute_alignment_loss(outputs, batch)
    total_loss = (
        answer_loss
        + edge_sparsity_weight * edge_sparsity_loss
        + active_node_weight * active_node_loss
        + write_alignment_weight * alignment_loss
    )
    return total_loss, {
        "answer_loss": answer_loss,
        "edge_sparsity_loss": edge_sparsity_loss,
        "active_node_loss": active_node_loss,
        "alignment_loss": alignment_loss,
    }


def compute_batch_diagnostics(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    active_mask = outputs["final_node_active"]
    active_values = outputs["final_node_value_vectors"]
    pairwise = F.cosine_similarity(
        active_values.unsqueeze(2),
        active_values.unsqueeze(1),
        dim=-1,
    )
    active_pairs = active_mask.unsqueeze(2) & active_mask.unsqueeze(1)
    off_diag = ~torch.eye(active_mask.size(1), device=active_mask.device, dtype=torch.bool).unsqueeze(0)
    valid_pairs = active_pairs & off_diag
    if valid_pairs.any():
        node_redundancy = pairwise.masked_select(valid_pairs).mean()
    else:
        node_redundancy = torch.tensor(0.0, device=active_mask.device)

    target_probs = outputs["step_target_probs"].clamp_min(1e-8)
    source_probs = outputs["step_source_probs"].clamp_min(1e-8)
    target_entropy = -(target_probs * target_probs.log()).sum(dim=-1).mean()
    source_entropy = -(source_probs * source_probs.log()).sum(dim=-1).mean()
    return {
        "node_redundancy": node_redundancy,
        "target_routing_entropy": target_entropy,
        "source_routing_entropy": source_entropy,
    }


def evaluate_model(model, examples, batch_size: int, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_answer_loss = 0.0
    total_correct = 0.0
    total_count = 0
    total_active = 0.0
    total_edge = 0.0
    total_redundancy = 0.0
    total_target_entropy = 0.0
    total_source_entropy = 0.0
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
            loss, pieces = compute_loss(
                outputs=outputs,
                batch=batch,
                label_smoothing=args.label_smoothing,
                edge_sparsity_weight=args.edge_sparsity_weight,
                active_node_weight=args.active_node_weight,
                write_alignment_weight=args.write_alignment_weight,
            )
            predictions = outputs["choice_logits"].argmax(dim=-1)
            batch_size_actual = len(batch_examples)
            total_loss += float(loss.item()) * batch_size_actual
            total_answer_loss += float(pieces["answer_loss"].item()) * batch_size_actual
            total_correct += float((predictions == batch["answer_indices"]).float().sum().item())
            total_count += batch_size_actual
            total_active += float(outputs["final_node_active"].float().sum().item())
            total_edge += float(outputs["final_edge_weights"].mean().item()) * batch_size_actual
            diagnostics = compute_batch_diagnostics(outputs)
            total_redundancy += float(diagnostics["node_redundancy"].item()) * batch_size_actual
            total_target_entropy += float(diagnostics["target_routing_entropy"].item()) * batch_size_actual
            total_source_entropy += float(diagnostics["source_routing_entropy"].item()) * batch_size_actual
    return {
        "loss": total_loss / max(total_count, 1),
        "answer_loss": total_answer_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
        "mean_active_nodes": total_active / max(total_count, 1),
        "mean_edge_weight": total_edge / max(total_count, 1),
        "node_redundancy": total_redundancy / max(total_count, 1),
        "target_routing_entropy": total_target_entropy / max(total_count, 1),
        "source_routing_entropy": total_source_entropy / max(total_count, 1),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    backend = QwenTextBackend(model_name=args.model_name, device=device)
    examples = load_longbench_mc_examples(
        backend=backend,
        limit=args.limit,
        chunk_chars=args.chunk_chars,
        max_chunks_per_example=args.max_chunks_per_example,
        seed=args.seed,
        cache_path=args.cache_path,
    )
    train_examples, eval_examples = train_eval_split(examples, eval_fraction=0.2, seed=args.seed)
    input_dim = examples[0].chunk_embeddings.size(-1)

    model = SelfContainedTextProgramGraphBrain(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_nodes=args.num_nodes,
        num_operators=args.num_operators,
        propagation_steps=args.propagation_steps,
        choice_temperature=args.choice_temperature,
        routing_temperature=args.routing_temperature,
        routing_topk=args.routing_topk,
        seed_weight=args.seed_weight,
        message_activation_weight=args.message_activation_weight,
        message_value_weight=args.message_value_weight,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = []
    best_eval_accuracy = float("-inf")
    best_epoch = 0
    best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_examples in iterate_batches(train_examples, batch_size=args.batch_size, seed=args.seed + epoch):
            batch = collate_mc_examples(batch_examples, device=device)
            outputs = model(
                chunk_embeddings=batch["chunk_embeddings"],
                chunk_mask=batch["chunk_mask"],
                query_embeddings=batch["query_embeddings"],
                choice_embeddings=batch["choice_embeddings"],
            )
            loss, _ = compute_loss(
                outputs=outputs,
                batch=batch,
                label_smoothing=args.label_smoothing,
                edge_sparsity_weight=args.edge_sparsity_weight,
                active_node_weight=args.active_node_weight,
                write_alignment_weight=args.write_alignment_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()

        train_metrics = evaluate_model(model, train_examples, batch_size=args.batch_size, device=device, args=args)
        eval_metrics = evaluate_model(model, eval_examples, batch_size=args.batch_size, device=device, args=args)
        history.append({"epoch": epoch, "train": train_metrics, "eval": eval_metrics})
        if eval_metrics["accuracy"] > best_eval_accuracy:
            best_eval_accuracy = eval_metrics["accuracy"]
            best_epoch = epoch
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"epoch={epoch:02d} "
            f"train_acc={train_metrics['accuracy']:.3f} train_answer_loss={train_metrics['answer_loss']:.4f} "
            f"eval_acc={eval_metrics['accuracy']:.3f} eval_answer_loss={eval_metrics['answer_loss']:.4f} "
            f"active={eval_metrics['mean_active_nodes']:.3f} edge={eval_metrics['mean_edge_weight']:.4f} "
            f"redundancy={eval_metrics['node_redundancy']:.3f} "
            f"trent={eval_metrics['target_routing_entropy']:.3f} "
            f"srent={eval_metrics['source_routing_entropy']:.3f}"
        )
        if epoch >= args.min_epochs_before_stop and epochs_without_improvement >= args.early_stop_patience:
            print(f"early stopping at epoch={epoch:02d} best_epoch={best_epoch:02d} best_eval_acc={best_eval_accuracy:.3f}")
            break

    model.load_state_dict(best_state_dict)
    final_eval = evaluate_model(model, eval_examples, batch_size=args.batch_size, device=device, args=args)

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "text_program_graph_brain",
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "input_dim": input_dim,
        },
        checkpoint_path,
    )

    results = {
        "config": vars(args),
        "dataset": {
            "num_examples": len(examples),
            "num_train": len(train_examples),
            "num_eval": len(eval_examples),
        },
        "history": history,
        "best_epoch": best_epoch,
        "best_eval_accuracy": best_eval_accuracy,
        "final_eval": final_eval,
        "checkpoint_path": str(checkpoint_path),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"saved results to {output_path}")


if __name__ == "__main__":
    main()
