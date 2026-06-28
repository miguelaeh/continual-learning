#!/usr/bin/env python3
"""Show recent traces from the episodic store."""

from __future__ import annotations

import argparse
import json

from agent_learning_system.store import EpisodicStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    store = EpisodicStore(args.store_dir)
    for item in store.recent(limit=args.limit):
        print(json.dumps(item, indent=2))


if __name__ == "__main__":
    main()
