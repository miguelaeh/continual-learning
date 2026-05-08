from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from self_building_brain.benchmarks.realistic import QwenTextBackend
from self_building_brain.data.realistic_continuation import (
    collate_continuation_examples,
    load_longbench_continuation_examples,
    train_eval_split,
)
from self_building_brain.models.text_program_graph_brain import SelfContainedTextProgramGraphBrain
from train_text_program_graph_longbench import compute_batch_diagnostics, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the self-contained text program graph on next-chunk continuation retrieval.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit-documents", type=int, default=120)
    parser.add_argument("--max-examples", type=int, default=600)
    parser.add_argument("--chunk-chars", type=int, default=260)
    parser.add_argument("--max-chunks-per-document", type=int, default=12)
    parser.add_argument("--min-prefix-chunks", type=int, default=1)
    parser.add_argument("--max-pairs-per-document", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-nodes", type=int, default=48)
    parser.add_argument("--num-operators", type=int, default=6)
    parser.add_argument("--propagation-steps", type=int, default=3)
    parser.add_argument("--routing-temperature", type=float, default=0.35)
    parser.add_argument("--routing-topk", type=int, default=2)
    parser.add_argument("--seed-weight", type=float, default=0.6)
    parser.add_argument("--message-activation-weight", type=float, default=2.0)
    parser.add_argument("--message-value-weight", type=float, default=2.0)
    parser.add_argument("--contrastive-temperature", type=float, default=16.0)
    parser.add_argument("--positive-cosine-weight", type=float, default=0.1)
    parser.add_argument("--edge-sparsity-weight", type=float, default=0.0005)
    parser.add_argument("--active-node-weight", type=float, default=0.0)
    parser.add_argument("--write-alignment-weight", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--min-epochs-before-stop", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cache-path", type=str, default="outputs/longbench_continuation_embeddings_120_docs_600_examples.pt")
    parser.add_argument("--output-path", type=str, default="outputs/text_program_graph_continuation_results.json")
    parser.add_argument("--checkpoint-path", type=str, default="outputs/text_program_graph_continuation.pt")
    return parser.parse_args()


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


def forward_context(model, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    dummy_choices = batch["target_embeddings"].unsqueeze(1)
    return model(
        chunk_embeddings=batch["chunk_embeddings"],
        chunk_mask=batch["chunk_mask"],
        query_embeddings=batch["query_embeddings"],
        choice_embeddings=dummy_choices,
    )


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    contrastive_temperature: float,
    positive_cosine_weight: float,
    edge_sparsity_weight: float,
    active_node_weight: float,
    write_alignment_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    context = F.normalize(outputs["context"], dim=-1)
    targets = F.normalize(batch["target_embeddings"], dim=-1)
    logits = contrastive_temperature * torch.matmul(context, targets.T)
    labels = torch.arange(context.size(0), device=context.device)
    retrieval_loss = F.cross_entropy(logits, labels)
    positive_cosine_loss = 1.0 - F.cosine_similarity(context, targets, dim=-1).mean()
    edge_sparsity_loss = outputs["final_edge_weights"].mean()
    active_node_loss = outputs["final_node_active_mass"].mean()
    alignment_loss = compute_alignment_loss(outputs, batch)
    total_loss = (
        retrieval_loss
        + positive_cosine_weight * positive_cosine_loss
        + edge_sparsity_weight * edge_sparsity_loss
        + active_node_weight * active_node_loss
        + write_alignment_weight * alignment_loss
    )
    retrieval_predictions = logits.argmax(dim=-1)
    retrieval_accuracy = (retrieval_predictions == labels).float().mean()
    return total_loss, {
        "retrieval_loss": retrieval_loss,
        "positive_cosine_loss": positive_cosine_loss,
        "edge_sparsity_loss": edge_sparsity_loss,
        "active_node_loss": active_node_loss,
        "alignment_loss": alignment_loss,
        "retrieval_accuracy": retrieval_accuracy,
    }


def evaluate_model(model, examples, batch_size: int, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_retrieval_loss = 0.0
    total_in_batch_accuracy = 0.0
    total_cosine = 0.0
    total_active = 0.0
    total_edge = 0.0
    total_redundancy = 0.0
    total_target_entropy = 0.0
    total_source_entropy = 0.0
    total_count = 0

    all_contexts = []
    all_targets = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            batch = collate_continuation_examples(batch_examples, device=device)
            outputs = forward_context(model, batch)
            loss, pieces = compute_loss(
                outputs=outputs,
                batch=batch,
                contrastive_temperature=args.contrastive_temperature,
                positive_cosine_weight=args.positive_cosine_weight,
                edge_sparsity_weight=args.edge_sparsity_weight,
                active_node_weight=args.active_node_weight,
                write_alignment_weight=args.write_alignment_weight,
            )
            batch_size_actual = len(batch_examples)
            diagnostics = compute_batch_diagnostics(outputs)
            context = F.normalize(outputs["context"], dim=-1)
            targets = F.normalize(batch["target_embeddings"], dim=-1)

            total_loss += float(loss.item()) * batch_size_actual
            total_retrieval_loss += float(pieces["retrieval_loss"].item()) * batch_size_actual
            total_in_batch_accuracy += float(pieces["retrieval_accuracy"].item()) * batch_size_actual
            total_cosine += float(F.cosine_similarity(context, targets, dim=-1).mean().item()) * batch_size_actual
            total_active += float(outputs["final_node_active"].float().sum().item())
            total_edge += float(outputs["final_edge_weights"].mean().item()) * batch_size_actual
            total_redundancy += float(diagnostics["node_redundancy"].item()) * batch_size_actual
            total_target_entropy += float(diagnostics["target_routing_entropy"].item()) * batch_size_actual
            total_source_entropy += float(diagnostics["source_routing_entropy"].item()) * batch_size_actual
            total_count += batch_size_actual

            all_contexts.append(context.cpu())
            all_targets.append(targets.cpu())

    context_bank = torch.cat(all_contexts, dim=0)
    target_bank = torch.cat(all_targets, dim=0)
    full_scores = args.contrastive_temperature * torch.matmul(context_bank, target_bank.T)
    labels = torch.arange(full_scores.size(0))
    top1 = full_scores.argmax(dim=-1)
    full_bank_accuracy = float((top1 == labels).float().mean().item())
    top5 = full_scores.topk(k=min(5, full_scores.size(1)), dim=-1).indices
    top5_accuracy = float((top5 == labels.unsqueeze(-1)).any(dim=-1).float().mean().item())

    return {
        "loss": total_loss / max(total_count, 1),
        "retrieval_loss": total_retrieval_loss / max(total_count, 1),
        "in_batch_accuracy": total_in_batch_accuracy / max(total_count, 1),
        "full_bank_accuracy": full_bank_accuracy,
        "top5_accuracy": top5_accuracy,
        "mean_positive_cosine": total_cosine / max(total_count, 1),
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
    examples = load_longbench_continuation_examples(
        backend=backend,
        limit_documents=args.limit_documents,
        max_examples=args.max_examples,
        chunk_chars=args.chunk_chars,
        max_chunks_per_document=args.max_chunks_per_document,
        min_prefix_chunks=args.min_prefix_chunks,
        max_pairs_per_document=args.max_pairs_per_document,
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
        choice_temperature=1.0,
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
            batch = collate_continuation_examples(batch_examples, device=device)
            outputs = forward_context(model, batch)
            loss, _ = compute_loss(
                outputs=outputs,
                batch=batch,
                contrastive_temperature=args.contrastive_temperature,
                positive_cosine_weight=args.positive_cosine_weight,
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
        if eval_metrics["full_bank_accuracy"] > best_eval_accuracy:
            best_eval_accuracy = eval_metrics["full_bank_accuracy"]
            best_epoch = epoch
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        print(
            f"epoch={epoch:02d} "
            f"train_full={train_metrics['full_bank_accuracy']:.3f} train_inbatch={train_metrics['in_batch_accuracy']:.3f} "
            f"eval_full={eval_metrics['full_bank_accuracy']:.3f} eval_top5={eval_metrics['top5_accuracy']:.3f} "
            f"cos={eval_metrics['mean_positive_cosine']:.3f} active={eval_metrics['mean_active_nodes']:.3f} "
            f"edge={eval_metrics['mean_edge_weight']:.4f} redundancy={eval_metrics['node_redundancy']:.3f}"
        )
        if epoch >= args.min_epochs_before_stop and epochs_without_improvement >= args.early_stop_patience:
            print(f"early stopping at epoch={epoch:02d} best_epoch={best_epoch:02d} best_eval_full={best_eval_accuracy:.3f}")
            break

    model.load_state_dict(best_state_dict)
    final_eval = evaluate_model(model, eval_examples, batch_size=args.batch_size, device=device, args=args)

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": "text_program_graph_continuation",
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
        "best_eval_full_bank_accuracy": best_eval_accuracy,
        "final_eval": final_eval,
        "checkpoint_path": str(checkpoint_path),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"saved results to {output_path}")


if __name__ == "__main__":
    main()
