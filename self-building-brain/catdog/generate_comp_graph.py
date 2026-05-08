from __future__ import annotations

import argparse
from pathlib import Path

import torch

from catdog.comp_model import (
    ACTION_CREATE_NODE,
    INPUT_NODE_INDEX,
    WORK_NODE_INDEX,
    build_token_embeddings,
    ComputationalGraphEnvironment,
    ComputationalPolicy,
    FixedComputationalExecutor,
    project_to_vocab,
    token_embedding,
)
from catdog.vocab import ID_TO_TOKEN, VOCAB, decode_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a trained catdog computational graph policy and generate tokens.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="<BOS>")
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def parse_prompt_tokens(prompt: str) -> list[str]:
    tokens = [token for token in prompt.strip().split() if token]
    if not tokens:
        tokens = ["<BOS>"]
    if tokens[0] != "<BOS>":
        tokens = ["<BOS>", *tokens]
    return tokens


def load_policy(checkpoint_path: Path, device: torch.device) -> tuple[ComputationalPolicy, dict]:
    payload = torch.load(checkpoint_path, map_location=device)
    config = payload["config"]
    policy = ComputationalPolicy(
        num_nodes=config["num_nodes"],
        hidden_dim=config["hidden_dim"],
        state_dim=config.get("state_dim", 16),
    ).to(device)
    policy.load_state_dict(payload["policy_state_dict"])
    policy.eval()
    return policy, payload


def masked_generation_logits(logits: torch.Tensor, *, allow_eos: bool) -> torch.Tensor:
    masked = logits.clone()
    masked[VOCAB["<PAD>"]] = -1e9
    masked[VOCAB["<UNK>"]] = -1e9
    masked[VOCAB["<BOS>"]] = -1e9
    if not allow_eos:
        masked[VOCAB["<EOS>"]] = -1e9
    return masked


def greedy_action(
    policy: ComputationalPolicy,
    token_id: int,
    graph,
    token_embeddings: torch.Tensor,
) -> tuple[int, int, int, int, torch.Tensor, int]:
    token_vector = token_embedding(token_id, token_embeddings)
    action_logits, source_logits, target_logits, op_logits, write_mean = policy.forward(token_vector, graph)
    write_vector = torch.tanh(write_mean)
    write_probe = int(project_to_vocab(write_vector, token_embeddings).argmax().item())
    return (
        int(action_logits.argmax().item()),
        int(source_logits.argmax().item()),
        int(target_logits.argmax().item()),
        int(op_logits.argmax().item()),
        write_vector,
        write_probe,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint_path)
    policy, payload = load_policy(checkpoint_path, device)
    config = payload["config"]

    state_dim = config.get("state_dim", 16)
    token_embeddings = build_token_embeddings(state_dim, device)
    env = ComputationalGraphEnvironment(
        num_nodes=config["num_nodes"],
        state_dim=state_dim,
        token_embeddings=token_embeddings,
        device=device,
    )
    executor = FixedComputationalExecutor()
    graph = env.initial_graph()

    prompt_tokens = parse_prompt_tokens(args.prompt)
    prompt_ids = [VOCAB[token] for token in prompt_tokens]

    produced_ids = list(prompt_ids)
    action_trace: list[dict[str, object]] = []

    current_id = produced_ids[0]
    for next_prompt_id in produced_ids[1:]:
        action_type, source, target, op, write_vector, write_probe = greedy_action(policy, current_id, graph, token_embeddings)
        if len(action_trace) == 0 and len(prompt_ids) == 1:
            action_type = ACTION_CREATE_NODE
            source = INPUT_NODE_INDEX
            target = WORK_NODE_INDEX
        graph = env.apply_action(
            graph,
            token_id=current_id,
            write_vector=write_vector,
            action_type=action_type,
            source=source,
            target=target,
            op=op,
        )
        action_trace.append(
            {
                "input_token": ID_TO_TOKEN[current_id],
                "action_type": action_type,
                "source": source,
                "target": target,
                "op": op,
                "write_probe": ID_TO_TOKEN[write_probe],
                "forced_next_token": ID_TO_TOKEN[next_prompt_id],
            }
        )
        current_id = next_prompt_id

    generated_ids: list[int] = []
    for _ in range(args.max_new_tokens):
        action_type, source, target, op, write_vector, write_probe = greedy_action(policy, current_id, graph, token_embeddings)
        graph = env.apply_action(
            graph,
            token_id=current_id,
            write_vector=write_vector,
            action_type=action_type,
            source=source,
            target=target,
            op=op,
        )
        output_state = executor.execute(graph, token_embedding(current_id, token_embeddings))
        logits = project_to_vocab(output_state, token_embeddings)
        logits = masked_generation_logits(logits, allow_eos=len(generated_ids) >= 1)
        next_id = int(logits.argmax().item())
        generated_ids.append(next_id)
        action_trace.append(
            {
                "input_token": ID_TO_TOKEN[current_id],
                "action_type": action_type,
                "source": source,
                "target": target,
                "op": op,
                "write_probe": ID_TO_TOKEN[write_probe],
                "predicted_next_token": ID_TO_TOKEN[next_id],
            }
        )
        current_id = next_id
        if next_id == VOCAB["<EOS>"]:
            break

    active_nodes = [index for index, value in enumerate(graph.active_mask.tolist()) if value > 0.5]
    print("prompt_tokens:", prompt_tokens)
    print("generated_tokens:", decode_tokens(generated_ids))
    print("full_tokens:", decode_tokens(prompt_ids + generated_ids))
    print("active_nodes:", active_nodes)
    print("pointer:", graph.pointer)
    print("action_trace:")
    for step, item in enumerate(action_trace):
        print(f"  step={step} {item}")


if __name__ == "__main__":
    main()
