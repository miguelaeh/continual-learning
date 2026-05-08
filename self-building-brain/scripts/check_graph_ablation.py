from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from self_building_brain.benchmarks.realistic import QwenTextBackend
from self_building_brain.data.realistic_mc import collate_mc_examples, load_longbench_mc_examples, train_eval_split
from self_building_brain.models.text_memory_reasoners import GraphTextReasoner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether the trained graph model actually uses graph edges and message passing.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--chunk-chars", type=int, default=1500)
    parser.add_argument("--max-chunks-per-example", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-slots", type=int, default=24)
    parser.add_argument("--memory-dim", type=int, default=256)
    parser.add_argument("--num-operators", type=int, default=4)
    parser.add_argument("--cache-path", type=str, default="outputs/longbench_embeddings_48_coarse.pt")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="outputs/graph_ablation_results.json")
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


def zero_edge_forward(self, chunk_embeddings, chunk_mask, query_embeddings, choice_embeddings):
    batch_size, num_chunks, _ = chunk_embeddings.shape
    node_states = torch.zeros(batch_size, self.num_nodes, self.memory_dim, device=chunk_embeddings.device)
    node_logits_history = []

    for chunk_index in range(num_chunks):
        valid = chunk_mask[:, chunk_index].unsqueeze(-1)
        chunk_embedding = chunk_embeddings[:, chunk_index]
        node_logits = self.node_router(chunk_embedding)
        node_weights = torch.softmax(node_logits, dim=-1)
        write_vector = self.write_projection(chunk_embedding).unsqueeze(1)
        update = node_weights.unsqueeze(-1) * write_vector * valid.unsqueeze(-1)
        retain = 1.0 - node_weights.unsqueeze(-1) * valid.unsqueeze(-1)
        node_states = node_states * retain + update
        node_logits_history.append(node_logits)

    propagated = node_states
    query_vector = self.query_projection(query_embeddings)
    attention_logits = torch.matmul(propagated, query_vector.unsqueeze(-1)).squeeze(-1)
    node_attention = torch.softmax(attention_logits, dim=-1)
    memory_context = torch.sum(node_attention.unsqueeze(-1) * propagated, dim=1)
    choice_vectors = self.choice_projection(choice_embeddings)
    expanded_context = memory_context.unsqueeze(1).expand(-1, choice_vectors.size(1), -1)
    expanded_query = query_vector.unsqueeze(1).expand_as(expanded_context)
    choice_logits = self.output(torch.cat([expanded_context, expanded_query, choice_vectors], dim=-1)).squeeze(-1)
    return {
        "choice_logits": choice_logits,
        "memory_context": memory_context,
        "routing_logits": torch.stack(node_logits_history, dim=1),
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
    model = GraphTextReasoner(
        input_dim=checkpoint["input_dim"],
        memory_dim=args.memory_dim,
        num_nodes=args.num_slots,
        num_operators=args.num_operators,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    normal = evaluate_model(model, eval_examples, batch_size=args.batch_size, device=device)

    no_message = copy.deepcopy(model)
    no_message.message_passing_steps = 0
    no_message_result = evaluate_model(no_message, eval_examples, batch_size=args.batch_size, device=device)

    zero_edges = copy.deepcopy(model)
    zero_edges.forward = zero_edge_forward.__get__(zero_edges, GraphTextReasoner)
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
