#!/usr/bin/env python3
"""Register a candidate checkpoint in the local registry."""

from __future__ import annotations

import argparse
import json

from agent_learning_system.registry import CheckpointRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--metric", action="append", default=[])
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    metrics = {}
    for item in args.metric:
        name, value = item.split("=", 1)
        metrics[name] = float(value)

    registry = CheckpointRegistry(args.registry)
    record = registry.register_candidate(
        checkpoint_id=args.candidate_id,
        path=args.path,
        metrics=metrics,
        notes=args.notes,
    )
    print(json.dumps(record.to_dict(), indent=2))


if __name__ == "__main__":
    main()
