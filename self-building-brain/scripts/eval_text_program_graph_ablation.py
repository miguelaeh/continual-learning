from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from self_building_brain.benchmarks.realistic import QwenTextBackend
from self_building_brain.data.realistic_mc import collate_mc_examples, load_longbench_mc_examples, train_eval_split
from self_building_brain.models.text_program_graph_brain import (
    SelfContainedTextProgramGraphBrain,
    TextProgramGraphState,
)
from train_text_program_graph_longbench import compute_batch_diagnostics, compute_loss, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate text program graph checkpoints under graph ablations.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-path", type=str, required=True)
    return parser.parse_args()


def build_state(
    model: SelfContainedTextProgramGraphBrain,
    chunk_embeddings: torch.Tensor,
    chunk_mask: torch.Tensor,
) -> TextProgramGraphState:
    batch_size, num_chunks, _ = chunk_embeddings.shape
    state = model.initial_state(batch_size=batch_size, device=chunk_embeddings.device)
    for chunk_index in range(num_chunks):
        valid_mask = chunk_mask[:, chunk_index] > 0
        next_state, _ = model.generator.forward_step(state=state, chunk_embeddings=chunk_embeddings[:, chunk_index])
        valid_nodes = valid_mask.view(batch_size, 1, 1).float()
        valid_edges = valid_mask.view(batch_size, 1, 1, 1).float()
        valid_mass = valid_mask.view(batch_size, 1).float()
        state = TextProgramGraphState(
            node_key_vectors=valid_nodes * next_state.node_key_vectors + (1.0 - valid_nodes) * state.node_key_vectors,
            node_value_vectors=valid_nodes * next_state.node_value_vectors + (1.0 - valid_nodes) * state.node_value_vectors,
            node_operator_logits=valid_nodes * next_state.node_operator_logits + (1.0 - valid_nodes) * state.node_operator_logits,
            edge_weights=valid_nodes * next_state.edge_weights + (1.0 - valid_nodes) * state.edge_weights,
            edge_operator_logits=valid_edges * next_state.edge_operator_logits + (1.0 - valid_edges) * state.edge_operator_logits,
            node_active_mass=valid_mass * next_state.node_active_mass + (1.0 - valid_mass) * state.node_active_mass,
            last_target_nodes=torch.where(valid_mask, next_state.last_target_nodes, state.last_target_nodes),
        )
    return state


def select_top1_state(state: TextProgramGraphState) -> TextProgramGraphState:
    top1 = state.node_active_mass.argmax(dim=-1)
    keep_mask = torch.zeros_like(state.node_active_mass)
    keep_mask.scatter_(dim=-1, index=top1.unsqueeze(-1), value=1.0)
    keep_nodes = keep_mask.unsqueeze(-1)
    keep_edges = keep_mask.unsqueeze(1) * keep_mask.unsqueeze(2)
    return TextProgramGraphState(
        node_key_vectors=state.node_key_vectors * keep_nodes,
        node_value_vectors=state.node_value_vectors * keep_nodes,
        node_operator_logits=state.node_operator_logits * keep_nodes,
        edge_weights=state.edge_weights * keep_edges,
        edge_operator_logits=state.edge_operator_logits * keep_edges.unsqueeze(-1),
        node_active_mass=state.node_active_mass * keep_mask,
        last_target_nodes=top1,
    )


def shuffle_node_contents(state: TextProgramGraphState, seed: int) -> TextProgramGraphState:
    generator = torch.Generator(device=state.node_active_mass.device)
    generator.manual_seed(seed)
    permutations = torch.stack(
        [torch.randperm(state.node_active_mass.size(1), generator=generator, device=state.node_active_mass.device) for _ in range(state.node_active_mass.size(0))],
        dim=0,
    )
    gather_nodes = permutations.unsqueeze(-1).expand(-1, -1, state.node_key_vectors.size(-1))
    gather_ops = permutations.unsqueeze(-1).expand(-1, -1, state.node_operator_logits.size(-1))
    return TextProgramGraphState(
        node_key_vectors=torch.gather(state.node_key_vectors, dim=1, index=gather_nodes),
        node_value_vectors=torch.gather(state.node_value_vectors, dim=1, index=gather_nodes),
        node_operator_logits=torch.gather(state.node_operator_logits, dim=1, index=gather_ops),
        edge_weights=state.edge_weights,
        edge_operator_logits=state.edge_operator_logits,
        node_active_mass=torch.gather(state.node_active_mass, dim=1, index=permutations),
        last_target_nodes=state.last_target_nodes,
    )


def run_vm(
    model: SelfContainedTextProgramGraphBrain,
    state: TextProgramGraphState,
    query_embeddings: torch.Tensor,
    choice_embeddings: torch.Tensor,
    propagation_steps: int | None = None,
) -> dict[str, torch.Tensor]:
    original_steps = model.vm.propagation_steps
    if propagation_steps is not None:
        model.vm.propagation_steps = propagation_steps
    outputs = model.vm(state=state, query_embeddings=query_embeddings, choice_embeddings=choice_embeddings)
    if propagation_steps is not None:
        model.vm.propagation_steps = original_steps
    return outputs


def evaluate_checkpoint(
    model: SelfContainedTextProgramGraphBrain,
    examples,
    batch_size: int,
    device: torch.device,
    config: dict[str, object],
) -> dict[str, dict[str, float]]:
    model.eval()
    modes = {
        "normal": lambda state, _: state,
        "zero_edges": lambda state, _: TextProgramGraphState(
            node_key_vectors=state.node_key_vectors,
            node_value_vectors=state.node_value_vectors,
            node_operator_logits=state.node_operator_logits,
            edge_weights=torch.zeros_like(state.edge_weights),
            edge_operator_logits=state.edge_operator_logits,
            node_active_mass=state.node_active_mass,
            last_target_nodes=state.last_target_nodes,
        ),
        "top1_node": lambda state, _: select_top1_state(state),
        "shuffle_nodes": lambda state, batch_index: shuffle_node_contents(state, seed=int(config.get("seed", 13)) + batch_index),
    }
    propagation_modes = {
        "normal": None,
        "zero_edges": None,
        "top1_node": None,
        "shuffle_nodes": None,
        "no_propagation": 0,
    }
    accumulators = {
        name: {
            "loss": 0.0,
            "answer_loss": 0.0,
            "accuracy": 0.0,
            "mean_active_nodes": 0.0,
            "mean_edge_weight": 0.0,
            "node_redundancy": 0.0,
            "count": 0,
        }
        for name in [*modes.keys(), *[name for name in propagation_modes if name not in modes]]
    }
    with torch.no_grad():
        for batch_start in range(0, len(examples), batch_size):
            batch_examples = examples[batch_start : batch_start + batch_size]
            batch = collate_mc_examples(batch_examples, device=device)
            state = build_state(model, batch["chunk_embeddings"], batch["chunk_mask"])
            for mode_name in accumulators:
                if mode_name in modes:
                    eval_state = modes[mode_name](state, batch_start)
                    vm_outputs = run_vm(model, eval_state, batch["query_embeddings"], batch["choice_embeddings"])
                else:
                    eval_state = state
                    vm_outputs = run_vm(
                        model,
                        eval_state,
                        batch["query_embeddings"],
                        batch["choice_embeddings"],
                        propagation_steps=propagation_modes[mode_name],
                    )
                outputs = {
                    "choice_logits": vm_outputs["choice_logits"],
                    "final_edge_weights": eval_state.edge_weights,
                    "final_node_active_mass": eval_state.node_active_mass,
                    "final_node_active": eval_state.node_active_mass >= 0.1,
                    "final_node_value_vectors": eval_state.node_value_vectors,
                    "step_candidate_values": batch["chunk_embeddings"],
                    "step_target_probs": torch.zeros(
                        batch["chunk_embeddings"].size(0),
                        batch["chunk_embeddings"].size(1),
                        eval_state.node_active_mass.size(1),
                        device=device,
                    ),
                    "step_source_probs": torch.zeros(
                        batch["chunk_embeddings"].size(0),
                        batch["chunk_embeddings"].size(1),
                        eval_state.node_active_mass.size(1),
                        device=device,
                    ),
                }
                loss, pieces = compute_loss(
                    outputs=outputs,
                    batch=batch,
                    label_smoothing=float(config.get("label_smoothing", 0.0)),
                    edge_sparsity_weight=float(config.get("edge_sparsity_weight", 0.01)),
                    active_node_weight=float(config.get("active_node_weight", 0.005)),
                    write_alignment_weight=0.0,
                )
                predictions = vm_outputs["choice_logits"].argmax(dim=-1)
                batch_size_actual = len(batch_examples)
                diagnostics = compute_batch_diagnostics(outputs)
                acc = accumulators[mode_name]
                acc["loss"] += float(loss.item()) * batch_size_actual
                acc["answer_loss"] += float(pieces["answer_loss"].item()) * batch_size_actual
                acc["accuracy"] += float((predictions == batch["answer_indices"]).float().sum().item())
                acc["mean_active_nodes"] += float((eval_state.node_active_mass >= 0.1).float().sum().item())
                acc["mean_edge_weight"] += float(eval_state.edge_weights.mean().item()) * batch_size_actual
                acc["node_redundancy"] += float(diagnostics["node_redundancy"].item()) * batch_size_actual
                acc["count"] += batch_size_actual
    results = {}
    for mode_name, metrics in accumulators.items():
        count = max(metrics.pop("count"), 1)
        results[mode_name] = {key: value / count for key, value in metrics.items()}
    return results


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    set_seed(int(config.get("seed", 13)))
    device = torch.device(args.device)

    backend = QwenTextBackend(model_name=config["model_name"], device=device)
    examples = load_longbench_mc_examples(
        backend=backend,
        limit=int(config["limit"]),
        chunk_chars=int(config["chunk_chars"]),
        max_chunks_per_example=int(config["max_chunks_per_example"]),
        seed=int(config.get("seed", 13)),
        cache_path=config["cache_path"],
    )
    _, eval_examples = train_eval_split(examples, eval_fraction=0.2, seed=int(config.get("seed", 13)))

    model = SelfContainedTextProgramGraphBrain(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(config.get("hidden_dim", 256)),
        num_nodes=int(config.get("num_nodes", 24)),
        num_operators=int(config.get("num_operators", 6)),
        propagation_steps=int(config.get("propagation_steps", 3)),
        choice_temperature=float(config.get("choice_temperature", 12.0)),
        routing_temperature=float(config.get("routing_temperature", 0.35)),
        routing_topk=int(config.get("routing_topk", 2)),
        seed_weight=float(config.get("seed_weight", 0.6)),
        message_activation_weight=float(config.get("message_activation_weight", 2.0)),
        message_value_weight=float(config.get("message_value_weight", 2.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    results = {
        "checkpoint_path": args.checkpoint_path,
        "config": config,
        "eval_examples": len(eval_examples),
        "ablations": evaluate_checkpoint(
            model=model,
            examples=eval_examples,
            batch_size=args.batch_size,
            device=device,
            config=config,
        ),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"saved ablation results to {output_path}")


if __name__ == "__main__":
    main()
