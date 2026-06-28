#!/usr/bin/env python3
"""Promote a checkpoint in the local registry."""

from __future__ import annotations

import argparse
import json

from agent_learning_system.registry import CheckpointRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--candidate-id", required=True)
    args = parser.parse_args()

    registry = CheckpointRegistry(args.registry)
    promoted = registry.promote(args.candidate_id)
    print(json.dumps(promoted, indent=2))


if __name__ == "__main__":
    main()
