from __future__ import annotations

import argparse
import json
from pathlib import Path

from self_building_brain.benchmarks.realistic import (
    QwenTextBackend,
    evaluate_examples,
    load_locomo_examples,
    load_longbench_v2_examples,
    load_musique_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate slot and graph memory methods on realistic text benchmarks.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--limit-per-benchmark", type=int, default=3)
    parser.add_argument("--num-slots", type=int, default=24)
    parser.add_argument("--slot-capacity", type=int, default=3)
    parser.add_argument("--top-k-slots", type=int, default=4)
    parser.add_argument("--top-k-chunks", type=int, default=6)
    parser.add_argument("--chunk-chars", type=int, default=900)
    parser.add_argument("--max-chunks-per-example", type=int, default=24)
    parser.add_argument("--output-path", type=str, default="outputs/realistic_benchmark_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = QwenTextBackend(model_name=args.model_name, device=args.device)

    benchmark_loaders = {
        "locomo": load_locomo_examples,
        "longbench_v2": load_longbench_v2_examples,
        "musique": load_musique_examples,
    }

    results: dict[str, object] = {}
    for benchmark_name, loader in benchmark_loaders.items():
        examples = loader(limit=args.limit_per_benchmark)
        print(f"evaluating {benchmark_name} on {len(examples)} examples")
        benchmark_result = {}
        for memory_type in ["slot", "graph"]:
            outcome = evaluate_examples(
                backend=backend,
                memory_type=memory_type,
                examples=examples,
                num_slots=args.num_slots,
                slot_capacity=args.slot_capacity,
                top_k_slots=args.top_k_slots,
                top_k_chunks=args.top_k_chunks,
                chunk_chars=args.chunk_chars,
                max_chunks_per_example=args.max_chunks_per_example,
            )
            benchmark_result[memory_type] = outcome
            print(f"  {memory_type}: {outcome['summary']}")
        results[benchmark_name] = benchmark_result

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"saved results to {output_path}")


if __name__ == "__main__":
    main()
