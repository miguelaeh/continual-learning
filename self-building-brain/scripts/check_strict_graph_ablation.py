from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from self_building_brain.benchmarks.realistic import QwenTextBackend
from self_building_brain.data.realistic_mc import collate_mc_examples, load_longbench_mc_examples, train_eval_split
from self_building_brain.models.text_memory_reasoners import StrictGraphTextReasoner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate the strict graph-dependent model.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--chunk-chars", type=int, default=1500)
    parser.add_argument("--max-chunks-per-example", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-slots", type=int, default=24)
    parser.add_argument("--memory-dim", type=int, default=256)
    parser.add_argument("--cache-path", type=str, default="outputs/longbench_embeddings_48_coarse.pt")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="outputs/strict_graph_ablation_results.json")
    return parser.parse_args()


def evaluate_model(model, examples, batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
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
            predictions = outputs["choice_logits"].argmax(dim=-1)
            total_correct += float((predictions == batch["answer_indices"]).float().sum().item())
            total_count += len(batch_examples)
    return {"accuracy": total_correct / total_count}


def no_propagation(self, chunk_embeddings, chunk_mask, query_embeddings, choice_embeddings):
    node_states, edge_weights, routing_logits = self.build_graph(chunk_embeddings, chunk_mask)
    batch_size = chunk_embeddings.size(0)
    memory_context = torch.zeros(batch_size, self.memory_dim, device=chunk_embeddings.device)
    query_vector = self.query_projection(query_embeddings)
    choice_vectors = self.choice_projection(choice_embeddings)
    expanded_context = memory_context.unsqueeze(1).expand(-1, choice_vectors.size(1), -1)
    expanded_query = query_vector.unsqueeze(1).expand_as(expanded_context)
    choice_logits = self.output(torch.cat([expanded_context, expanded_query, choice_vectors], dim=-1)).squeeze(-1)
    seed_logits = self.query_seed_router(query_embeddings)
    return {
        "choice_logits": choice_logits,
        "memory_context": memory_context,
        "routing_logits": routing_logits,
        "seed_logits": seed_logits,
        "edge_weights": edge_weights,
    }


def zero_edges_forward(self, chunk_embeddings, chunk_mask, query_embeddings, choice_embeddings):
    node_states, edge_weights, routing_logits = self.build_graph(chunk_embeddings, chunk_mask)
    edge_weights.zero_()
    memory_context, seed_logits = self.propagate(node_states, edge_weights, query_embeddings)
    query_vector = self.query_projection(query_embeddings)
    choice_vectors = self.choice_projection(choice_embeddings)
    expanded_context = memory_context.unsqueeze(1).expand(-1, choice_vectors.size(1), -1)
    expanded_query = query_vector.unsqueeze(1).expand_as(expanded_context)
    choice_logits = self.output(torch.cat([expanded_context, expanded_query, choice_vectors], dim=-1)).squeeze(-1)
    return {
        "choice_logits": choice_logits,
        "memory_context": memory_context,
        "routing_logits": routing_logits,
        "seed_logits": seed_logits,
        "edge_weights": edge_weights,
    }


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
    _, eval_examples = train_eval_split(examples, eval_fraction=0.2)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model = StrictGraphTextReasoner(
        input_dim=checkpoint["input_dim"],
        memory_dim=args.memory_dim,
        num_nodes=args.num_slots,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    normal = evaluate_model(model, eval_examples, batch_size=args.batch_size, device=device)
    no_message = copy.deepcopy(model)
    no_message.forward = no_propagation.__get__(no_message, StrictGraphTextReasoner)
    no_message_result = evaluate_model(no_message, eval_examples, batch_size=args.batch_size, device=device)
    zero_edges = copy.deepcopy(model)
    zero_edges.forward = zero_edges_forward.__get__(zero_edges, StrictGraphTextReasoner)
    zero_edges_result = evaluate_model(zero_edges, eval_examples, batch_size=args.batch_size, device=device)

    results = {
        "normal": normal,
        "no_message_passing": no_message_result,
        "zero_edges": zero_edges_result,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
