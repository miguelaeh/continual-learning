from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import torch
from torch.distributions import Categorical, Normal

from catdog.comp_model import (
    CompGraphState,
    ComputationalGraphEnvironment,
    ComputationalPolicy,
    FixedComputationalExecutor,
    INPUT_NODE_INDEX,
    WORK_NODE_INDEX,
    build_token_embeddings,
    clone_graph,
    detach_graph,
    graph_from_payload,
    graph_to_payload,
    project_to_vocab,
    token_embedding,
)
from catdog.data import Episode, train_eval_split_by_stage
from catdog.vocab import ID_TO_TOKEN, VOCAB, VOCAB_SIZE, decode_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a persistent executable catdog graph brain.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-nodes", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--op-entropy-weight", type=float, default=0.03)
    parser.add_argument("--write-pathwise-weight", type=float, default=1.0)
    parser.add_argument("--rollout-reward-weight", type=float, default=1.0)
    parser.add_argument("--exact-reward-weight", type=float, default=1.0)
    parser.add_argument("--commit-margin", type=float, default=0.0)
    parser.add_argument("--commit-exploration-rate", type=float, default=0.01)
    parser.add_argument("--commit-score-scope", choices=["episode", "pool"], default="pool")
    parser.add_argument("--commit-score-max-examples", type=int, default=16)
    parser.add_argument("--cover-small-pools", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--edge-decay", type=float, default=0.98)
    parser.add_argument("--edge-prune-threshold", type=float, default=0.25)
    parser.add_argument("--max-edge-weight", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--curriculum-pool-sizes", type=str, default="1,3,5,10,20,40,80")
    parser.add_argument("--curriculum-unlock-mode", choices=["schedule", "performance"], default="performance")
    parser.add_argument("--curriculum-phase-fracs", type=str, default="0.02,0.08,0.18,0.32,0.50,0.72,0.90")
    parser.add_argument("--unlock-rollout-accuracy", type=float, default=0.6)
    parser.add_argument("--unlock-patience", type=int, default=2)
    parser.add_argument("--resume-checkpoint", type=str, default="")
    parser.add_argument("--resume-best-checkpoint", type=str, default="")
    parser.add_argument("--output-path", type=str, default="catdog/outputs/train_persistent_graph_rl.json")
    parser.add_argument("--checkpoint-path", type=str, default="catdog/outputs/train_persistent_graph_rl.pt")
    parser.add_argument("--best-checkpoint-path", type=str, default="")
    parser.add_argument("--checkpoint-dir", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def parse_schedule_list(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def masked_logits_for_step(logits: torch.Tensor, *, allow_eos: bool) -> torch.Tensor:
    masked = logits.clone()
    masked[VOCAB["<PAD>"]] = -1e9
    masked[VOCAB["<UNK>"]] = -1e9
    masked[VOCAB["<BOS>"]] = -1e9
    if not allow_eos:
        masked[VOCAB["<EOS>"]] = -1e9
    return masked


def training_pool_for_index(ordered_episodes: list[Episode], pool_sizes: list[int], pool_index: int) -> list[Episode]:
    if pool_index >= len(pool_sizes):
        return ordered_episodes
    return ordered_episodes[: min(pool_sizes[pool_index], len(ordered_episodes))]


def training_pool_for_step(
    ordered_episodes: list[Episode],
    step: int,
    total_steps: int,
    pool_sizes: list[int],
    phase_fracs: list[float],
) -> list[Episode]:
    progress = step / max(total_steps, 1)
    for pool_size, phase_frac in zip(pool_sizes, phase_fracs):
        if progress <= phase_frac:
            return ordered_episodes[: min(pool_size, len(ordered_episodes))]
    return ordered_episodes


def sample_rehearsal_batch(episodes: list[Episode], batch_size: int, rng: random.Random, cover_small_pools: bool) -> list[Episode]:
    if cover_small_pools and len(episodes) <= batch_size:
        batch = list(episodes)
        while len(batch) < batch_size:
            batch.append(episodes[rng.randrange(len(episodes))])
        rng.shuffle(batch)
        return batch
    return [episodes[rng.randrange(len(episodes))] for _ in range(batch_size)]


def step_checkpoint_path(args: argparse.Namespace, step: int) -> Path:
    latest_path = Path(args.checkpoint_path)
    directory = Path(args.checkpoint_dir) if args.checkpoint_dir else latest_path.parent
    stem = latest_path.stem
    width = max(6, len(str(args.steps)))
    return directory / f"{stem}_step{step:0{width}d}.pt"


def best_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.best_checkpoint_path:
        return Path(args.best_checkpoint_path)
    latest_path = Path(args.checkpoint_path)
    return latest_path.with_name(f"{latest_path.stem}_best.pt")


def choose_action(
    policy: ComputationalPolicy,
    graph: CompGraphState,
    token_id: int,
    token_embeddings: torch.Tensor,
    greedy: bool,
) -> dict[str, object]:
    token_vector = token_embedding(token_id, token_embeddings)
    action_logits, source_logits, target_logits, op_logits, write_mean = policy.forward(token_vector, graph)
    if greedy:
        write_vector = torch.tanh(write_mean)
        return {
            "action_type": int(action_logits.argmax().item()),
            "source": int(source_logits.argmax().item()),
            "target": int(target_logits.argmax().item()),
            "op": int(op_logits.argmax().item()),
            "write_vector": write_vector,
            "log_prob": torch.tensor(0.0, device=graph.node_states.device),
            "entropy": torch.tensor(0.0, device=graph.node_states.device),
            "op_entropy": torch.tensor(0.0, device=graph.node_states.device),
        }

    action_dist = Categorical(logits=action_logits)
    source_dist = Categorical(logits=source_logits)
    target_dist = Categorical(logits=target_logits)
    op_dist = Categorical(logits=op_logits)
    write_std = policy.write_log_std.exp().clamp_min(1e-4)
    write_dist = Normal(write_mean, write_std)
    action_type = action_dist.sample()
    source = source_dist.sample()
    target = target_dist.sample()
    op = op_dist.sample()
    raw_write_vector = write_dist.rsample()
    write_vector = torch.tanh(raw_write_vector)
    log_prob = (
        action_dist.log_prob(action_type)
        + source_dist.log_prob(source)
        + target_dist.log_prob(target)
        + op_dist.log_prob(op)
        + write_dist.log_prob(raw_write_vector).sum()
    )
    entropy = (
        action_dist.entropy()
        + source_dist.entropy()
        + target_dist.entropy()
        + op_dist.entropy()
        + write_dist.entropy().sum()
    )
    return {
        "action_type": int(action_type.item()),
        "source": int(source.item()),
        "target": int(target.item()),
        "op": int(op.item()),
        "write_vector": write_vector,
        "log_prob": log_prob,
        "entropy": entropy,
        "op_entropy": op_dist.entropy(),
    }


def graph_teacher_forced_metrics(
    episode: Episode,
    graph: CompGraphState,
    executor: FixedComputationalExecutor,
    token_embeddings: torch.Tensor,
) -> dict[str, object]:
    rewards = []
    correct = 0.0
    predictions = []
    device = graph.node_states.device
    runtime_graph = clone_graph(graph)
    for token_id, target_id in zip(episode.token_ids, episode.token_ids[1:]):
        output_state, runtime_graph = executor.execute_state(runtime_graph, token_embedding(token_id, token_embeddings))
        logits = project_to_vocab(output_state, token_embeddings)
        logits = masked_logits_for_step(logits, allow_eos=target_id == VOCAB["<EOS>"])
        log_probs_vocab = torch.log_softmax(logits, dim=0)
        prediction = int(logits.argmax().item())
        predictions.append(prediction)
        correct += float(prediction == target_id)
        rewards.append(log_probs_vocab[target_id])
    reward = torch.stack(rewards).mean() if rewards else torch.tensor(0.0, device=device)
    return {
        "reward": reward,
        "accuracy": correct / max(len(predictions), 1),
        "predictions": predictions,
    }


def graph_rollout_metrics(
    episode: Episode,
    graph: CompGraphState,
    executor: FixedComputationalExecutor,
    token_embeddings: torch.Tensor,
) -> dict[str, object]:
    target_ids = list(episode.token_ids[1:])
    current_id = int(episode.token_ids[0])
    generated_ids = []
    runtime_graph = clone_graph(graph)
    for index in range(len(target_ids)):
        output_state, runtime_graph = executor.execute_state(runtime_graph, token_embedding(current_id, token_embeddings))
        logits = project_to_vocab(output_state, token_embeddings)
        logits = masked_logits_for_step(logits, allow_eos=index > 0 or target_ids[index] == VOCAB["<EOS>"])
        next_id = int(logits.argmax().item())
        generated_ids.append(next_id)
        current_id = next_id
        if next_id == VOCAB["<EOS>"]:
            break
    compare_len = max(len(target_ids), 1)
    padded_generated = generated_ids + [VOCAB["<PAD>"]] * max(0, len(target_ids) - len(generated_ids))
    correct = sum(float(pred == target) for pred, target in zip(padded_generated[: len(target_ids)], target_ids))
    exact = float(generated_ids == target_ids)
    return {
        "rollout_accuracy": correct / compare_len,
        "exact": exact,
        "generated_ids": generated_ids,
    }


def graph_commit_score(
    episode: Episode,
    graph: CompGraphState,
    executor: FixedComputationalExecutor,
    token_embeddings: torch.Tensor,
) -> float:
    with torch.no_grad():
        teacher = graph_teacher_forced_metrics(episode, graph, executor, token_embeddings)
        rollout = graph_rollout_metrics(episode, graph, executor, token_embeddings)
    return float(rollout["rollout_accuracy"]) + float(rollout["exact"]) + 0.1 * float(teacher["accuracy"])


def graph_rehearsal_score(
    episodes: list[Episode],
    graph: CompGraphState,
    executor: FixedComputationalExecutor,
    token_embeddings: torch.Tensor,
    max_examples: int,
) -> float:
    selected = episodes[:max_examples]
    if not selected:
        return 0.0
    return sum(graph_commit_score(episode, graph, executor, token_embeddings) for episode in selected) / len(selected)


def stabilize_graph_edges(
    graph: CompGraphState,
    *,
    edge_decay: float,
    edge_prune_threshold: float,
    max_edge_weight: float,
) -> CompGraphState:
    baseline = torch.eye(graph.edge_weights.size(0), dtype=graph.edge_weights.dtype, device=graph.edge_weights.device)
    baseline[INPUT_NODE_INDEX, WORK_NODE_INDEX] += 0.5
    excess = (graph.edge_weights - baseline).clamp_min(0.0) * edge_decay
    excess = torch.where(excess >= edge_prune_threshold, excess, torch.zeros_like(excess))
    edge_weights = (baseline + excess).clamp_max(max_edge_weight)
    edge_ops = torch.where(excess > 0.0, graph.edge_ops, torch.zeros_like(graph.edge_ops))
    return CompGraphState(
        node_states=graph.node_states,
        node_ops=graph.node_ops,
        edge_weights=edge_weights,
        edge_ops=edge_ops,
        active_mask=graph.active_mask,
        pointer=graph.pointer,
    )


def train_episode(
    episode: Episode,
    graph: CompGraphState,
    policy: ComputationalPolicy,
    env: ComputationalGraphEnvironment,
    executor: FixedComputationalExecutor,
    token_embeddings: torch.Tensor,
    rollout_reward_weight: float,
    exact_reward_weight: float,
) -> dict[str, object]:
    log_probs = []
    entropies = []
    op_entropies = []
    actions = []
    step_rewards = []

    for token_id, target_id in zip(episode.token_ids, episode.token_ids[1:]):
        action = choose_action(policy, graph, int(token_id), token_embeddings, greedy=False)
        write_vector = action["write_vector"]
        write_probe = int(project_to_vocab(write_vector, token_embeddings).argmax().item())
        graph = env.apply_action(
            graph,
            token_id=int(token_id),
            write_vector=write_vector,
            action_type=int(action["action_type"]),
            source=int(action["source"]),
            target=int(action["target"]),
            op=int(action["op"]),
        )
        actions.append((int(action["action_type"]), int(action["source"]), int(action["target"]), int(action["op"]), write_probe))
        log_probs.append(action["log_prob"])
        entropies.append(action["entropy"])
        op_entropies.append(action["op_entropy"])

        output_state = executor.execute(graph, token_embedding(int(token_id), token_embeddings))
        logits = project_to_vocab(output_state, token_embeddings)
        logits = masked_logits_for_step(logits, allow_eos=target_id == VOCAB["<EOS>"])
        step_rewards.append(torch.log_softmax(logits, dim=0)[int(target_id)])

    teacher_metrics = graph_teacher_forced_metrics(episode, graph, executor, token_embeddings)
    rollout_metrics = graph_rollout_metrics(episode, graph, executor, token_embeddings)
    device = graph.node_states.device
    step_reward = torch.stack(step_rewards).mean() if step_rewards else torch.tensor(0.0, device=device)
    rollout_reward = torch.tensor(float(rollout_metrics["rollout_accuracy"]), device=device)
    exact_reward = torch.tensor(float(rollout_metrics["exact"]), device=device)
    reward = step_reward + rollout_reward_weight * rollout_reward + exact_reward_weight * exact_reward
    return {
        "graph": graph,
        "reward": reward,
        "teacher_accuracy": teacher_metrics["accuracy"],
        "rollout_accuracy": rollout_metrics["rollout_accuracy"],
        "exact": rollout_metrics["exact"],
        "generated_ids": rollout_metrics["generated_ids"],
        "log_prob_sum": torch.stack(log_probs).sum() if log_probs else torch.tensor(0.0, device=device),
        "entropy_sum": torch.stack(entropies).sum() if entropies else torch.tensor(0.0, device=device),
        "op_entropy_sum": torch.stack(op_entropies).sum() if op_entropies else torch.tensor(0.0, device=device),
        "actions": actions,
    }


def evaluate_graph(
    episodes: list[Episode],
    graph: CompGraphState,
    executor: FixedComputationalExecutor,
    token_embeddings: torch.Tensor,
    max_examples: int = 512,
) -> dict[str, object]:
    if not episodes:
        return {
            "mean_reward": 0.0,
            "accuracy": 0.0,
            "rollout_accuracy": 0.0,
            "exact": 0.0,
            "samples": [],
        }
    total = min(len(episodes), max_examples)
    reward = 0.0
    accuracy = 0.0
    rollout_accuracy = 0.0
    exact = 0.0
    samples = []
    for episode in episodes[:total]:
        teacher = graph_teacher_forced_metrics(episode, graph, executor, token_embeddings)
        rollout = graph_rollout_metrics(episode, graph, executor, token_embeddings)
        reward += float(teacher["reward"].item())
        accuracy += float(teacher["accuracy"])
        rollout_accuracy += float(rollout["rollout_accuracy"])
        exact += float(rollout["exact"])
        if len(samples) < 5:
            samples.append(
                {
                    "target": decode_tokens(episode.token_ids[1:]),
                    "generated": decode_tokens(rollout["generated_ids"]),
                }
            )
    return {
        "mean_reward": reward / total,
        "accuracy": accuracy / total,
        "rollout_accuracy": rollout_accuracy / total,
        "exact": exact / total,
        "samples": samples,
    }


def save_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    policy: ComputationalPolicy,
    optimizer: torch.optim.Optimizer,
    graph: CompGraphState,
    step: int,
    baseline: float,
    history: list[dict[str, object]],
    current_pool_index: int,
    unlock_streak: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": vars(args),
            "step": step,
            "baseline": baseline,
            "current_pool_index": current_pool_index,
            "unlock_streak": unlock_streak,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "graph": graph_to_payload(graph),
            "history": history,
        },
        checkpoint_path,
    )


def best_eval_from_history(history: list[dict[str, object]]) -> tuple[int, float]:
    best_step = 0
    best_score = -1.0
    for item in history:
        current_pool_metrics = item.get("current_pool")
        pool_size = int(item.get("curriculum_pool_size", 0))
        if not isinstance(current_pool_metrics, dict):
            continue
        score = pool_size + float(current_pool_metrics.get("rollout_accuracy", -1.0))
        if score > best_score:
            best_score = score
            best_step = int(item.get("step", 0))
    return best_step, best_score


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    curriculum_pool_sizes = parse_schedule_list(args.curriculum_pool_sizes, int)
    curriculum_phase_fracs = parse_schedule_list(args.curriculum_phase_fracs, float)
    if args.curriculum_unlock_mode == "schedule" and len(curriculum_pool_sizes) != len(curriculum_phase_fracs):
        raise ValueError("curriculum-pool-sizes and curriculum-phase-fracs must have the same length")

    train_stage_episodes, eval_stage_episodes = train_eval_split_by_stage(
        seed=args.seed,
        curriculum_mode="sentences",
    )
    train_episodes = [episode for stage in train_stage_episodes for episode in stage]
    eval_episodes = [episode for stage in eval_stage_episodes for episode in stage]

    token_embeddings = build_token_embeddings(args.state_dim, device)
    policy = ComputationalPolicy(num_nodes=args.num_nodes, hidden_dim=args.hidden_dim, state_dim=args.state_dim).to(device)
    env = ComputationalGraphEnvironment(
        num_nodes=args.num_nodes,
        state_dim=args.state_dim,
        token_embeddings=token_embeddings,
        device=device,
    )
    executor = FixedComputationalExecutor()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    rng = random.Random(args.seed)

    brain_graph = env.initial_graph()
    history: list[dict[str, object]] = []
    baseline = 0.0
    baseline_momentum = 0.95
    current_pool_index = 0
    unlock_streak = 0
    start_step = 0
    checkpoint_path = Path(args.checkpoint_path)
    best_path = best_checkpoint_path(args)
    best_eval_step = 0
    best_eval_score = -1.0
    best_eval_selection_metric = ""
    best_eval_checkpoint_path = str(best_path)

    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        checkpoint = torch.load(resume_path, map_location=device)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "graph" not in checkpoint:
            raise ValueError("resume checkpoint does not contain a persistent graph")
        brain_graph = graph_from_payload(checkpoint["graph"], device)
        history = list(checkpoint.get("history", []))
        start_step = int(checkpoint.get("step", 0))
        baseline = float(checkpoint.get("baseline", 0.0))
        current_pool_index = int(checkpoint.get("current_pool_index", 0))
        unlock_streak = int(checkpoint.get("unlock_streak", 0))
        best_eval_step, best_eval_score = best_eval_from_history(history)
        best_eval_selection_metric = "restored_from_history"
        best_eval_checkpoint_path = args.resume_best_checkpoint or args.resume_checkpoint
        if start_step >= args.steps:
            raise ValueError(f"checkpoint is already at step {start_step}; --steps must be larger")
        print(
            f"resumed checkpoint={resume_path} start_step={start_step} "
            f"best_rollout_acc={best_eval_score:.3f} pool_index={current_pool_index}"
        )

    for step in range(start_step + 1, args.steps + 1):
        if args.curriculum_unlock_mode == "performance":
            current_pool = training_pool_for_index(train_episodes, curriculum_pool_sizes, current_pool_index)
        else:
            current_pool = training_pool_for_step(
                train_episodes,
                step=step,
                total_steps=args.steps,
                pool_sizes=curriculum_pool_sizes,
                phase_fracs=curriculum_phase_fracs,
            )

        batch = sample_rehearsal_batch(
            current_pool,
            batch_size=args.batch_size,
            rng=rng,
            cover_small_pools=args.cover_small_pools,
        )
        rewards = []
        log_prob_sums = []
        entropy_sums = []
        op_entropy_sums = []
        batch_rollout_acc = 0.0
        batch_exact = 0.0
        batch_commits = 0
        commit_scope_episodes = current_pool[: args.commit_score_max_examples]

        for episode in batch:
            if args.commit_score_scope == "pool":
                before_commit_score = graph_rehearsal_score(
                    commit_scope_episodes,
                    brain_graph,
                    executor,
                    token_embeddings,
                    max_examples=args.commit_score_max_examples,
                )
            else:
                before_commit_score = graph_commit_score(episode, brain_graph, executor, token_embeddings)
            result = train_episode(
                episode=episode,
                graph=brain_graph,
                policy=policy,
                env=env,
                executor=executor,
                token_embeddings=token_embeddings,
                rollout_reward_weight=args.rollout_reward_weight,
                exact_reward_weight=args.exact_reward_weight,
            )
            candidate_graph = result["graph"]
            if args.commit_score_scope == "pool":
                after_commit_score = graph_rehearsal_score(
                    commit_scope_episodes,
                    candidate_graph,
                    executor,
                    token_embeddings,
                    max_examples=args.commit_score_max_examples,
                )
            else:
                after_commit_score = graph_commit_score(episode, candidate_graph, executor, token_embeddings)
            should_commit = after_commit_score > before_commit_score + args.commit_margin
            should_commit = should_commit or rng.random() < args.commit_exploration_rate
            if should_commit:
                brain_graph = stabilize_graph_edges(
                    candidate_graph,
                    edge_decay=args.edge_decay,
                    edge_prune_threshold=args.edge_prune_threshold,
                    max_edge_weight=args.max_edge_weight,
                )
                batch_commits += 1
            rewards.append(result["reward"])
            log_prob_sums.append(result["log_prob_sum"])
            entropy_sums.append(result["entropy_sum"])
            op_entropy_sums.append(result["op_entropy_sum"])
            batch_rollout_acc += float(result["rollout_accuracy"])
            batch_exact += float(result["exact"])

        reward_tensor = torch.stack(rewards)
        baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * float(reward_tensor.mean().item())
        advantage = (reward_tensor - baseline).detach()
        log_prob_tensor = torch.stack(log_prob_sums)
        entropy_tensor = torch.stack(entropy_sums)
        op_entropy_tensor = torch.stack(op_entropy_sums)

        reinforce_loss = -(advantage * log_prob_tensor).mean()
        entropy_loss = -args.entropy_weight * entropy_tensor.mean()
        op_entropy_loss = -args.op_entropy_weight * op_entropy_tensor.mean()
        pathwise_write_loss = -args.write_pathwise_weight * reward_tensor.mean()
        loss = reinforce_loss + entropy_loss + op_entropy_loss + pathwise_write_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        brain_graph = detach_graph(brain_graph)

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            train_metrics = evaluate_graph(train_episodes, brain_graph, executor, token_embeddings)
            eval_metrics = evaluate_graph(eval_episodes, brain_graph, executor, token_embeddings)
            current_pool_metrics = evaluate_graph(current_pool, brain_graph, executor, token_embeddings)
            stage_metrics = {
                f"stage_{index}": evaluate_graph(stage, brain_graph, executor, token_embeddings)
                for index, stage in enumerate(eval_stage_episodes)
            }
            active_nodes = int((brain_graph.active_mask > 0.5).sum().item())
            active_edges = int((brain_graph.edge_weights > 1.0).sum().item())
            history_item = {
                "step": step,
                "loss": float(loss.item()),
                "batch_reward": float(reward_tensor.mean().item()),
                "batch_rollout_accuracy": batch_rollout_acc / max(len(batch), 1),
                "batch_exact": batch_exact / max(len(batch), 1),
                "batch_commit_rate": batch_commits / max(len(batch), 1),
                "curriculum_pool_size": len(current_pool),
                "current_pool": current_pool_metrics,
                "train": train_metrics,
                "eval": eval_metrics,
                "eval_by_stage": stage_metrics,
                "graph": {
                    "active_nodes": active_nodes,
                    "active_edges": active_edges,
                    "pointer": brain_graph.pointer,
                },
            }
            history.append(history_item)
            print(
                f"step={step:04d} loss={loss.item():.4f} "
                f"batch_reward={reward_tensor.mean().item():.4f} "
                f"pool={len(current_pool)} "
                f"commit={batch_commits / max(len(batch), 1):.2f} "
                f"pool_rollout={current_pool_metrics['rollout_accuracy']:.3f} "
                f"train_rollout={train_metrics['rollout_accuracy']:.3f} "
                f"eval_rollout={eval_metrics['rollout_accuracy']:.3f} "
                f"eval_exact={eval_metrics['exact']:.3f} "
                f"tf_acc={eval_metrics['accuracy']:.3f} "
                f"active={active_nodes} edges={active_edges}"
            )

            if len(current_pool) < len(train_episodes):
                selection_score = len(current_pool) + float(current_pool_metrics["rollout_accuracy"])
                selection_metric = "curriculum_pool_size_plus_current_pool_rollout"
            else:
                selection_score = len(train_episodes) + float(eval_metrics["rollout_accuracy"])
                selection_metric = "full_pool_size_plus_eval_rollout"
            if selection_score > best_eval_score:
                best_eval_score = selection_score
                best_eval_selection_metric = selection_metric
                best_eval_step = step
                best_eval_checkpoint_path = str(best_path)
                save_checkpoint(
                    best_path,
                    args,
                    policy,
                    optimizer,
                    brain_graph,
                    step,
                    baseline,
                    history,
                    current_pool_index,
                    unlock_streak,
                )

            if args.curriculum_unlock_mode == "performance" and len(current_pool) < len(train_episodes):
                if float(current_pool_metrics["rollout_accuracy"]) >= args.unlock_rollout_accuracy:
                    unlock_streak += 1
                else:
                    unlock_streak = 0
                if unlock_streak >= args.unlock_patience:
                    current_pool_index += 1
                    unlock_streak = 0

        if args.save_every > 0 and (step % args.save_every == 0 or step == args.steps):
            save_checkpoint(
                step_checkpoint_path(args, step),
                args,
                policy,
                optimizer,
                brain_graph,
                step,
                baseline,
                history,
                current_pool_index,
                unlock_streak,
            )
            save_checkpoint(
                checkpoint_path,
                args,
                policy,
                optimizer,
                brain_graph,
                step,
                baseline,
                history,
                current_pool_index,
                unlock_streak,
            )

    sample = eval_episodes[0]
    sample_teacher = graph_teacher_forced_metrics(sample, brain_graph, executor, token_embeddings)
    sample_rollout = graph_rollout_metrics(sample, brain_graph, executor, token_embeddings)
    results = {
        "config": vars(args),
        "dataset": {
            "num_train_episodes": len(train_episodes),
            "num_eval_episodes": len(eval_episodes),
            "vocab_size": VOCAB_SIZE,
        },
        "best_eval": {
            "step": best_eval_step,
            "selection_score": best_eval_score,
            "selection_metric": best_eval_selection_metric,
            "selection_note": "larger curriculum pool wins first; rollout accuracy breaks ties within that pool",
            "checkpoint_path": best_eval_checkpoint_path,
        },
        "history": history,
        "sample_eval": {
            "target_tokens": decode_tokens(sample.token_ids[1:]),
            "teacher_predictions": [ID_TO_TOKEN[token_id] for token_id in sample_teacher["predictions"]],
            "generated_tokens": decode_tokens(sample_rollout["generated_ids"]),
        },
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    save_checkpoint(
        checkpoint_path,
        args,
        policy,
        optimizer,
        brain_graph,
        args.steps,
        baseline,
        history,
        current_pool_index,
        unlock_streak,
    )
    if Path(best_eval_checkpoint_path) != best_path and Path(best_eval_checkpoint_path).exists() and not best_path.exists():
        best_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_eval_checkpoint_path, best_path)
        best_eval_checkpoint_path = str(best_path)
        results["best_eval"]["checkpoint_path"] = best_eval_checkpoint_path
        output_path.write_text(json.dumps(results, indent=2))
    print(f"saved results to {output_path}")
    print(f"saved checkpoint to {checkpoint_path}")
    print(f"best checkpoint at {best_eval_checkpoint_path}")


if __name__ == "__main__":
    main()
