from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = {
    "baseline_240_long12": {
        "limit": 240,
        "chunk_chars": 400,
        "max_chunks_per_example": 12,
        "epochs": 40,
        "batch_size": 8,
        "num_nodes": 48,
        "routing_temperature": 0.35,
        "routing_topk": 2,
        "active_node_weight": 0.0,
        "edge_sparsity_weight": 0.0005,
        "write_alignment_weight": 0.01,
        "label_smoothing": 0.05,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed_weight": 0.6,
        "message_activation_weight": 2.0,
        "message_value_weight": 2.0,
        "early_stop_patience": 12,
        "min_epochs_before_stop": 12,
        "cache_path": "outputs/longbench_embeddings_240_long12.pt",
    },
    "longer_stream_240_long16": {
        "limit": 240,
        "chunk_chars": 300,
        "max_chunks_per_example": 16,
        "epochs": 40,
        "batch_size": 8,
        "num_nodes": 48,
        "routing_temperature": 0.35,
        "routing_topk": 2,
        "active_node_weight": 0.0,
        "edge_sparsity_weight": 0.0005,
        "write_alignment_weight": 0.01,
        "label_smoothing": 0.05,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "seed_weight": 0.6,
        "message_activation_weight": 2.0,
        "message_value_weight": 2.0,
        "early_stop_patience": 12,
        "min_epochs_before_stop": 12,
        "cache_path": "outputs/longbench_embeddings_240_long16_c300.pt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the text program graph LongBench sweep.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["baseline_240_long12", "longer_stream_240_long16"],
        choices=sorted(EXPERIMENTS.keys()),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 11, 19, 29, 41])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--summary-path", type=str, default="outputs/text_program_graph_sweep_summary.json")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def train_output_paths(name: str, seed: int) -> tuple[str, str]:
    base = f"outputs/{name}_seed{seed}"
    return f"{base}.json", f"{base}.pt"


def ablation_output_path(name: str, seed: int) -> str:
    return f"outputs/{name}_seed{seed}_ablation.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    cwd = repo_root()
    env = os.environ.copy()
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not pythonpath else f"src:{pythonpath}"

    summary: dict[str, dict[str, object]] = {}
    for experiment_name in args.experiments:
        config = EXPERIMENTS[experiment_name]
        experiment_results = []
        for seed in args.seeds:
            output_path, checkpoint_path = train_output_paths(experiment_name, seed)
            train_cmd = [
                sys.executable,
                "scripts/train_text_program_graph_longbench.py",
                "--device",
                args.device,
                "--seed",
                str(seed),
                "--limit",
                str(config["limit"]),
                "--chunk-chars",
                str(config["chunk_chars"]),
                "--max-chunks-per-example",
                str(config["max_chunks_per_example"]),
                "--epochs",
                str(config["epochs"]),
                "--batch-size",
                str(config["batch_size"]),
                "--num-nodes",
                str(config["num_nodes"]),
                "--routing-temperature",
                str(config["routing_temperature"]),
                "--routing-topk",
                str(config["routing_topk"]),
                "--active-node-weight",
                str(config["active_node_weight"]),
                "--edge-sparsity-weight",
                str(config["edge_sparsity_weight"]),
                "--write-alignment-weight",
                str(config["write_alignment_weight"]),
                "--label-smoothing",
                str(config["label_smoothing"]),
                "--weight-decay",
                str(config["weight_decay"]),
                "--max-grad-norm",
                str(config["max_grad_norm"]),
                "--seed-weight",
                str(config["seed_weight"]),
                "--message-activation-weight",
                str(config["message_activation_weight"]),
                "--message-value-weight",
                str(config["message_value_weight"]),
                "--early-stop-patience",
                str(config["early_stop_patience"]),
                "--min-epochs-before-stop",
                str(config["min_epochs_before_stop"]),
                "--cache-path",
                str(config["cache_path"]),
                "--output-path",
                output_path,
                "--checkpoint-path",
                checkpoint_path,
            ]
            run_command(train_cmd, cwd=cwd, env=env)

            ablation_path = ablation_output_path(experiment_name, seed)
            ablate_cmd = [
                sys.executable,
                "scripts/eval_text_program_graph_ablation.py",
                "--device",
                args.device,
                "--checkpoint-path",
                checkpoint_path,
                "--output-path",
                ablation_path,
            ]
            run_command(ablate_cmd, cwd=cwd, env=env)

            result = load_json(cwd / output_path)
            ablation = load_json(cwd / ablation_path)
            experiment_results.append(
                {
                    "seed": seed,
                    "output_path": output_path,
                    "checkpoint_path": checkpoint_path,
                    "ablation_path": ablation_path,
                    "best_eval_accuracy": result.get("best_eval_accuracy"),
                    "best_epoch": result.get("best_epoch"),
                    "final_eval": result.get("final_eval"),
                    "ablations": ablation.get("ablations"),
                }
            )

        summary[experiment_name] = {
            "config": config,
            "runs": experiment_results,
        }

    summary_path = cwd / args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"saved sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
