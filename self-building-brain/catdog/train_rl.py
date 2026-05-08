from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from catdog.data import Episode, sample_batch, train_eval_split, train_eval_split_by_stage
from catdog.model import FixedGraphExecutor, GraphEditEnvironment, PolicyGenerator, one_hot
from catdog.vocab import ID_TO_TOKEN, VOCAB_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal catdog graph generator with REINFORCE.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-nodes", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--output-path", type=str, default="catdog/outputs/train_rl_results.json")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def run_episode(
    episode: Episode,
    generator: PolicyGenerator,
    env: GraphEditEnvironment,
    executor: FixedGraphExecutor,
    greedy: bool = False,
) -> dict[str, object]:
    graph = env.initial_graph()
    log_probs = []
    entropies = []
    actions = []
    total_reward = torch.tensor(0.0, device=graph.node_states.device)
    total_correct = 0.0
    total_predictions = 0

    for position in range(len(episode.token_ids) - 1):
        token_id = episode.token_ids[position]
        target_id = episode.token_ids[position + 1]
        token_vector = one_hot(token_id, graph.node_states.device)
        if greedy:
            target_logits = generator.forward(token_vector, graph)
            target = int(target_logits.argmax().item())
        else:
            sampled = generator.sample_action(token_vector, graph)
            target = int(sampled["target"])
            log_probs.append(sampled["log_prob"])
            entropies.append(sampled["entropy"])
        actions.append(target)
        graph = env.apply_action(graph, token_id=token_id, target=target)

        query_vector = one_hot(token_id, graph.node_states.device)
        logits = executor.execute(graph, query_vector)
        log_probs_vocab = torch.log_softmax(logits, dim=0)
        reward = log_probs_vocab[target_id]
        prediction = int(logits.argmax().item())
        total_reward = total_reward + reward
        total_correct += float(prediction == target_id)
        total_predictions += 1

    return {
        "reward": total_reward / max(total_predictions, 1),
        "correct": total_correct / max(total_predictions, 1),
        "prediction": prediction,
        "target": target_id,
        "log_prob_sum": torch.stack(log_probs).sum() if log_probs else torch.tensor(0.0, device=graph.node_states.device),
        "entropy_sum": torch.stack(entropies).sum() if entropies else torch.tensor(0.0, device=graph.node_states.device),
        "actions": actions,
    }


def evaluate(
    episodes: list[Episode],
    generator: PolicyGenerator,
    env: GraphEditEnvironment,
    executor: FixedGraphExecutor,
    max_examples: int = 512,
) -> dict[str, float]:
    if not episodes:
        return {
            "mean_reward": 0.0,
            "accuracy": 0.0,
        }
    total_reward = 0.0
    total_correct = 0.0
    total = min(len(episodes), max_examples)
    for episode in episodes[:total]:
        result = run_episode(episode, generator=generator, env=env, executor=executor, greedy=True)
        total_reward += float(result["reward"].item())
        total_correct += float(result["correct"])
    return {
        "mean_reward": total_reward / max(total, 1),
        "accuracy": total_correct / max(total, 1),
    }


def training_pool_for_step(stage_episodes: list[list[Episode]], step: int, total_steps: int) -> list[Episode]:
    progress = step / max(total_steps, 1)
    num_stages = len(stage_episodes)
    unlocked = min(num_stages, max(1, int(progress * num_stages) + 1))
    pool: list[Episode] = []
    for stage in stage_episodes[:unlocked]:
        pool.extend(stage)
    return pool


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.curriculum:
        train_stage_episodes, eval_stage_episodes = train_eval_split_by_stage(seed=args.seed)
        train_episodes = []
        for stage in train_stage_episodes:
            train_episodes.extend(stage)
        eval_episodes = []
        for stage in eval_stage_episodes:
            eval_episodes.extend(stage)
    else:
        train_episodes, eval_episodes = train_eval_split(seed=args.seed)
        train_stage_episodes = []
        eval_stage_episodes = []
    generator = PolicyGenerator(num_nodes=args.num_nodes, hidden_dim=args.hidden_dim).to(device)
    env = GraphEditEnvironment(num_nodes=args.num_nodes, device=device)
    executor = FixedGraphExecutor()
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.learning_rate)
    rng = random.Random(args.seed)

    history = []
    baseline = 0.0
    baseline_momentum = 0.95

    for step in range(1, args.steps + 1):
        if args.curriculum:
            current_pool = training_pool_for_step(train_stage_episodes, step=step, total_steps=args.steps)
        else:
            current_pool = train_episodes
        batch = sample_batch(current_pool, batch_size=args.batch_size, rng=rng)
        rewards = []
        log_prob_sums = []
        entropy_sums = []

        for episode in batch:
            result = run_episode(episode, generator=generator, env=env, executor=executor, greedy=False)
            rewards.append(result["reward"])
            log_prob_sums.append(result["log_prob_sum"])
            entropy_sums.append(result["entropy_sum"])

        reward_tensor = torch.stack(rewards)
        baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * float(reward_tensor.mean().item())
        advantage = (reward_tensor - baseline).detach()
        log_prob_tensor = torch.stack(log_prob_sums)
        entropy_tensor = torch.stack(entropy_sums)

        loss = -(advantage * log_prob_tensor).mean() - args.entropy_weight * entropy_tensor.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            train_metrics = evaluate(train_episodes, generator=generator, env=env, executor=executor)
            eval_metrics = evaluate(eval_episodes, generator=generator, env=env, executor=executor)
            current_pool_metrics = evaluate(current_pool, generator=generator, env=env, executor=executor)
            stage_metrics = {
                f"stage_{index}": evaluate(stage, generator=generator, env=env, executor=executor)
                for index, stage in enumerate(eval_stage_episodes)
            } if args.curriculum else {}
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "batch_reward": float(reward_tensor.mean().item()),
                    "curriculum_pool_size": len(current_pool),
                    "current_pool": current_pool_metrics,
                    "train": train_metrics,
                    "eval": eval_metrics,
                    "eval_by_stage": stage_metrics,
                }
            )
            print(
                f"step={step:04d} "
                f"loss={loss.item():.4f} "
                f"batch_reward={reward_tensor.mean().item():.4f} "
                f"pool={len(current_pool)} "
                f"pool_acc={current_pool_metrics['accuracy']:.3f} "
                f"train_acc={train_metrics['accuracy']:.3f} "
                f"eval_acc={eval_metrics['accuracy']:.3f}"
            )

    sample = eval_episodes[0]
    sample_result = run_episode(sample, generator=generator, env=env, executor=executor, greedy=True)
    results = {
        "config": vars(args),
        "dataset": {
            "num_train_episodes": len(train_episodes),
            "num_eval_episodes": len(eval_episodes),
            "vocab_size": VOCAB_SIZE,
        },
        "history": history,
        "sample_eval": {
            "token_ids": list(sample.token_ids),
            "prediction_id": sample_result["prediction"],
            "target_id": sample_result["target"],
            "target_token": ID_TO_TOKEN[sample_result["target"]],
            "prediction_token": ID_TO_TOKEN[sample_result["prediction"]],
            "actions": sample_result["actions"],
        },
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"saved results to {output_path}")


if __name__ == "__main__":
    main()
