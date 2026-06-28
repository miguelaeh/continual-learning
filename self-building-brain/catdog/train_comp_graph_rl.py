from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import torch
from torch.distributions import Categorical

from catdog.comp_model import (
    ACTION_CREATE_NODE,
    INPUT_NODE_INDEX,
    OUTPUT_NODE_INDEX,
    NUM_ACTION_TYPES,
    NUM_NODE_OPS,
    build_token_embeddings,
    ComputationalGraphEnvironment,
    ComputationalPolicy,
    FixedComputationalExecutor,
    project_to_vocab,
    token_embedding,
)
from catdog.data import Episode, sample_batch, train_eval_split_by_stage
from catdog.vocab import ID_TO_TOKEN, VOCAB, VOCAB_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the computational catdog graph generator with REINFORCE.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-nodes", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--op-entropy-weight", type=float, default=0.0)
    parser.add_argument("--op-entropy-final-weight", type=float, default=None)
    parser.add_argument("--write-pathwise-weight", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--curriculum-pool-sizes", type=str, default="1,3,5,7,10")
    parser.add_argument("--curriculum-phase-fracs", type=str, default="0.01,0.08,0.20,0.40,0.70")
    parser.add_argument("--curriculum-mode", choices=["sentences", "transitions"], default="transitions")
    parser.add_argument("--curriculum-unlock-mode", choices=["schedule", "performance"], default="performance")
    parser.add_argument("--unlock-accuracy", type=float, default=0.8)
    parser.add_argument("--unlock-patience", type=int, default=1)
    parser.add_argument("--output-path", type=str, default="catdog/outputs/train_comp_graph_rl.json")
    parser.add_argument("--checkpoint-path", type=str, default="catdog/outputs/train_comp_graph_rl.pt")
    parser.add_argument("--best-checkpoint-path", type=str, default="")
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--resume-checkpoint", type=str, default="")
    parser.add_argument("--resume-best-checkpoint", type=str, default="")
    parser.add_argument("--save-every", type=int, default=0)
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
    capped = min(pool_sizes[pool_index], len(ordered_episodes))
    return ordered_episodes[:capped]


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
            capped = min(pool_size, len(ordered_episodes))
            return ordered_episodes[:capped]
    return ordered_episodes


def run_episode(
    episode: Episode,
    policy: ComputationalPolicy,
    env: ComputationalGraphEnvironment,
    executor: FixedComputationalExecutor,
    token_embeddings: torch.Tensor,
    greedy: bool = False,
) -> dict[str, object]:
    graph = env.initial_graph()
    log_probs = []
    entropies = []
    op_entropies = []
    actions = []
    total_reward = torch.tensor(0.0, device=graph.node_states.device)
    total_correct = 0.0
    total_predictions = 0

    for position in range(len(episode.token_ids) - 1):
        token_id = episode.token_ids[position]
        target_id = episode.token_ids[position + 1]
        token_vector = token_embedding(token_id, token_embeddings)
        allow_eos = target_id == VOCAB["<EOS>"]
        is_stage0 = episode.stage_index == 0

        if greedy:
            action_logits, source_logits, target_logits, op_logits, write_mean = policy.forward(token_vector, graph)
            action_type = int(action_logits.argmax().item())
            source = int(source_logits.argmax().item())
            target = int(target_logits.argmax().item())
            op = int(op_logits.argmax().item())
            write_vector = torch.tanh(write_mean)
            if is_stage0:
                action_type = ACTION_CREATE_NODE
                source = INPUT_NODE_INDEX
                target = OUTPUT_NODE_INDEX
                op = 0
        else:
            if is_stage0:
                _, _, _, _op_logits, write_mean = policy.forward(token_vector, graph)
                write_std = policy.write_log_std.exp().clamp_min(1e-4)
                write_dist = torch.distributions.Normal(write_mean, write_std)
                raw_write_vector = write_dist.rsample()
                write_vector = torch.tanh(raw_write_vector)
                action_type = ACTION_CREATE_NODE
                source = INPUT_NODE_INDEX
                target = OUTPUT_NODE_INDEX
                op = 0
                log_probs.append(write_dist.log_prob(raw_write_vector).sum())
                entropies.append(write_dist.entropy().sum())
                op_entropies.append(torch.tensor(0.0, device=graph.node_states.device))
            else:
                sampled = policy.sample_action(token_vector, graph)
                action_type = int(sampled["action_type"])
                source = int(sampled["source"])
                target = int(sampled["target"])
                op = int(sampled["op"])
                write_vector = sampled["write_vector"]
                log_probs.append(sampled["log_prob"])
                entropies.append(sampled["entropy"])
                op_entropies.append(sampled["op_entropy"])

        write_probe = int(project_to_vocab(write_vector, token_embeddings).argmax().item())
        actions.append((action_type, source, target, op, write_probe))
        graph = env.apply_action(
            graph,
            token_id=token_id,
            write_vector=write_vector,
            action_type=action_type,
            source=source,
            target=target,
            op=op,
        )

        query_vector = token_embedding(token_id, token_embeddings)
        output_state = executor.execute(graph, query_vector)
        logits = project_to_vocab(output_state, token_embeddings)
        logits = masked_logits_for_step(logits, allow_eos=allow_eos)
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
        "op_entropy_sum": torch.stack(op_entropies).sum() if op_entropies else torch.tensor(0.0, device=graph.node_states.device),
        "actions": actions,
    }


def evaluate(
    episodes: list[Episode],
    policy: ComputationalPolicy,
    env: ComputationalGraphEnvironment,
    executor: FixedComputationalExecutor,
    token_embeddings: torch.Tensor,
    max_examples: int = 512,
) -> dict[str, float]:
    if not episodes:
        return {
            "mean_reward": 0.0,
            "accuracy": 0.0,
        }
    total_reward = 0.0
    total_correct = 0.0
    action_usage = [0 for _ in range(NUM_ACTION_TYPES)]
    op_usage = [0 for _ in range(NUM_NODE_OPS)]
    action_count = 0
    total = min(len(episodes), max_examples)
    for episode in episodes[:total]:
        result = run_episode(
            episode,
            policy=policy,
            env=env,
            executor=executor,
            token_embeddings=token_embeddings,
            greedy=True,
        )
        total_reward += float(result["reward"].item())
        total_correct += float(result["correct"])
        for action_type, _source, _target, op, _write_probe in result["actions"]:
            action_usage[int(action_type)] += 1
            op_usage[int(op)] += 1
            action_count += 1
    return {
        "mean_reward": total_reward / max(total, 1),
        "accuracy": total_correct / max(total, 1),
        "action_usage": [count / max(action_count, 1) for count in action_usage],
        "op_usage": [count / max(action_count, 1) for count in op_usage],
    }


def save_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    policy: ComputationalPolicy,
    optimizer: torch.optim.Optimizer,
    step: int,
    baseline: float,
    history: list[dict[str, object]],
    current_pool_index: int = 0,
    unlock_streak: int = 0,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "step": step,
        "baseline": baseline,
        "current_pool_index": current_pool_index,
        "unlock_streak": unlock_streak,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }
    torch.save(payload, checkpoint_path)


def step_checkpoint_path(args: argparse.Namespace, step: int) -> Path:
    if args.checkpoint_dir:
        directory = Path(args.checkpoint_dir)
        stem = Path(args.checkpoint_path).stem
    else:
        latest_path = Path(args.checkpoint_path)
        directory = latest_path.parent
        stem = latest_path.stem
    width = max(6, len(str(args.steps)))
    return directory / f"{stem}_step{step:0{width}d}.pt"


def best_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.best_checkpoint_path:
        return Path(args.best_checkpoint_path)
    latest_path = Path(args.checkpoint_path)
    return latest_path.with_name(f"{latest_path.stem}_best.pt")


def infer_pool_index_from_history(
    history: list[dict[str, object]],
    train_episode_count: int,
    pool_sizes: list[int],
) -> int:
    if not history:
        return 0
    last_pool_size = int(history[-1].get("curriculum_pool_size", 0))
    if last_pool_size >= train_episode_count:
        return len(pool_sizes)
    for index, pool_size in enumerate(pool_sizes):
        if min(pool_size, train_episode_count) >= last_pool_size:
            return index
    return len(pool_sizes)


def best_eval_from_history(history: list[dict[str, object]]) -> tuple[int, float]:
    best_step = 0
    best_accuracy = -1.0
    for item in history:
        eval_metrics = item.get("eval")
        if not isinstance(eval_metrics, dict):
            continue
        accuracy = float(eval_metrics.get("accuracy", -1.0))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_step = int(item.get("step", 0))
    return best_step, best_accuracy


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    curriculum_pool_sizes = parse_schedule_list(args.curriculum_pool_sizes, int)
    curriculum_phase_fracs = parse_schedule_list(args.curriculum_phase_fracs, float)
    if args.curriculum_unlock_mode == "schedule":
        if len(curriculum_pool_sizes) != len(curriculum_phase_fracs):
            raise ValueError("curriculum-pool-sizes and curriculum-phase-fracs must have the same length")
        if any(not (0.0 < frac <= 1.0) for frac in curriculum_phase_fracs):
            raise ValueError("curriculum-phase-fracs must be in (0, 1]")
        if sorted(curriculum_phase_fracs) != curriculum_phase_fracs:
            raise ValueError("curriculum-phase-fracs must be sorted in ascending order")

    train_stage_episodes, eval_stage_episodes = train_eval_split_by_stage(
        seed=args.seed,
        curriculum_mode=args.curriculum_mode,
    )
    train_episodes: list[Episode] = []
    for stage in train_stage_episodes:
        train_episodes.extend(stage)
    eval_episodes: list[Episode] = []
    for stage in eval_stage_episodes:
        eval_episodes.extend(stage)

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

    history = []
    baseline = 0.0
    baseline_momentum = 0.95
    checkpoint_path = Path(args.checkpoint_path)
    best_path = best_checkpoint_path(args)
    best_eval_accuracy = -1.0
    best_eval_step = 0
    best_eval_checkpoint_path = str(best_path)
    current_pool_index = 0
    unlock_streak = 0
    start_step = 0

    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        checkpoint = torch.load(resume_path, map_location=device)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = list(checkpoint.get("history", []))
        start_step = int(checkpoint.get("step", history[-1]["step"] if history else 0))
        baseline = float(checkpoint.get("baseline", 0.0))
        current_pool_index = int(
            checkpoint.get(
                "current_pool_index",
                infer_pool_index_from_history(
                    history=history,
                    train_episode_count=len(train_episodes),
                    pool_sizes=curriculum_pool_sizes,
                ),
            )
        )
        unlock_streak = int(checkpoint.get("unlock_streak", 0))
        best_eval_step, best_eval_accuracy = best_eval_from_history(history)
        if args.resume_best_checkpoint:
            best_eval_checkpoint_path = args.resume_best_checkpoint
        else:
            best_eval_checkpoint_path = args.resume_checkpoint
        if start_step >= args.steps:
            raise ValueError(
                f"resume checkpoint is already at step {start_step}; "
                f"--steps must be larger than the checkpoint step"
            )
        print(
            f"resumed checkpoint={resume_path} "
            f"start_step={start_step} "
            f"best_eval_acc={best_eval_accuracy:.3f} "
            f"pool_index={current_pool_index}"
        )

    for step in range(start_step + 1, args.steps + 1):
        if args.curriculum_unlock_mode == "performance":
            current_pool = training_pool_for_index(
                ordered_episodes=train_episodes,
                pool_sizes=curriculum_pool_sizes,
                pool_index=current_pool_index,
            )
        else:
            current_pool = training_pool_for_step(
                ordered_episodes=train_episodes,
                step=step,
                total_steps=args.steps,
                pool_sizes=curriculum_pool_sizes,
                phase_fracs=curriculum_phase_fracs,
            )
        batch = sample_batch(current_pool, batch_size=args.batch_size, rng=rng)
        rewards = []
        log_prob_sums = []
        entropy_sums = []
        op_entropy_sums = []

        for episode in batch:
            result = run_episode(
                episode,
                policy=policy,
                env=env,
                executor=executor,
                token_embeddings=token_embeddings,
                greedy=False,
            )
            rewards.append(result["reward"])
            log_prob_sums.append(result["log_prob_sum"])
            entropy_sums.append(result["entropy_sum"])
            op_entropy_sums.append(result["op_entropy_sum"])

        reward_tensor = torch.stack(rewards)
        baseline = baseline_momentum * baseline + (1.0 - baseline_momentum) * float(reward_tensor.mean().item())
        advantage = (reward_tensor - baseline).detach()
        log_prob_tensor = torch.stack(log_prob_sums)
        entropy_tensor = torch.stack(entropy_sums)
        op_entropy_tensor = torch.stack(op_entropy_sums)

        reinforce_loss = -(advantage * log_prob_tensor).mean()
        entropy_loss = -args.entropy_weight * entropy_tensor.mean()
        if args.op_entropy_final_weight is None:
            op_entropy_weight = args.op_entropy_weight
        else:
            progress = step / max(args.steps, 1)
            op_entropy_weight = args.op_entropy_weight + progress * (args.op_entropy_final_weight - args.op_entropy_weight)
        op_entropy_loss = -op_entropy_weight * op_entropy_tensor.mean()
        # Discrete graph edits still use REINFORCE. The latent write value is a
        # differentiable action, so also let reward gradients shape it directly.
        pathwise_write_loss = -args.write_pathwise_weight * reward_tensor.mean()
        loss = reinforce_loss + entropy_loss + op_entropy_loss + pathwise_write_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            train_metrics = evaluate(train_episodes, policy=policy, env=env, executor=executor, token_embeddings=token_embeddings)
            eval_metrics = evaluate(eval_episodes, policy=policy, env=env, executor=executor, token_embeddings=token_embeddings)
            current_pool_metrics = evaluate(current_pool, policy=policy, env=env, executor=executor, token_embeddings=token_embeddings)
            stage_metrics = {
                f"stage_{index}": evaluate(stage, policy=policy, env=env, executor=executor, token_embeddings=token_embeddings)
                for index, stage in enumerate(eval_stage_episodes)
            }
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
                f"eval_acc={eval_metrics['accuracy']:.3f} "
                f"stage0={stage_metrics['stage_0']['accuracy']:.3f} "
                f"top_op={max(range(len(train_metrics['op_usage'])), key=lambda i: train_metrics['op_usage'][i])}"
            )
            if eval_metrics["accuracy"] > best_eval_accuracy:
                best_eval_accuracy = eval_metrics["accuracy"]
                best_eval_step = step
                best_eval_checkpoint_path = str(best_path)
                save_checkpoint(
                    checkpoint_path=best_path,
                    args=args,
                    policy=policy,
                    optimizer=optimizer,
                    step=step,
                    baseline=baseline,
                    history=history,
                    current_pool_index=current_pool_index,
                    unlock_streak=unlock_streak,
                )
            if args.curriculum_unlock_mode == "performance" and len(current_pool) < len(train_episodes):
                if current_pool_metrics["accuracy"] >= args.unlock_accuracy:
                    unlock_streak += 1
                else:
                    unlock_streak = 0
                if unlock_streak >= args.unlock_patience:
                    current_pool_index += 1
                    unlock_streak = 0
        if args.save_every > 0 and (step % args.save_every == 0 or step == args.steps):
            save_checkpoint(
                checkpoint_path=step_checkpoint_path(args, step),
                args=args,
                policy=policy,
                optimizer=optimizer,
                step=step,
                baseline=baseline,
                history=history,
                current_pool_index=current_pool_index,
                unlock_streak=unlock_streak,
            )
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                args=args,
                policy=policy,
                optimizer=optimizer,
                step=step,
                baseline=baseline,
                history=history,
                current_pool_index=current_pool_index,
                unlock_streak=unlock_streak,
            )

    sample = eval_episodes[0]
    sample_result = run_episode(
        sample,
        policy=policy,
        env=env,
        executor=executor,
        token_embeddings=token_embeddings,
        greedy=True,
    )
    results = {
        "config": vars(args),
        "dataset": {
            "num_train_episodes": len(train_episodes),
            "num_eval_episodes": len(eval_episodes),
            "vocab_size": VOCAB_SIZE,
        },
        "best_eval": {
            "step": best_eval_step,
            "accuracy": best_eval_accuracy,
            "checkpoint_path": best_eval_checkpoint_path,
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
    save_checkpoint(
        checkpoint_path=checkpoint_path,
        args=args,
        policy=policy,
        optimizer=optimizer,
        step=args.steps,
        baseline=baseline,
        history=history,
        current_pool_index=current_pool_index,
        unlock_streak=unlock_streak,
    )
    print(f"saved results to {output_path}")
    print(f"saved checkpoint to {checkpoint_path}")
    if Path(best_eval_checkpoint_path) != best_path and Path(best_eval_checkpoint_path).exists() and not best_path.exists():
        best_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_eval_checkpoint_path, best_path)
        best_eval_checkpoint_path = str(best_path)
        results["best_eval"]["checkpoint_path"] = best_eval_checkpoint_path
        output_path.write_text(json.dumps(results, indent=2))
    print(f"best checkpoint at {best_eval_checkpoint_path}")


if __name__ == "__main__":
    main()
