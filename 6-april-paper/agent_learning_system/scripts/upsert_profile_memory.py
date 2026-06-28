#!/usr/bin/env python3
"""Upsert a structured profile memory item."""

from __future__ import annotations

import argparse
import json

from agent_learning_system.profile import ProfileMemoryStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--memory-type", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--value", required=True)
    parser.add_argument("--source", default="")
    parser.add_argument("--tag", action="append", default=[])
    args = parser.parse_args()

    store = ProfileMemoryStore(args.store_dir)
    item = store.upsert(
        memory_type=args.memory_type,
        key=args.key,
        value=args.value,
        source=args.source,
        tags=args.tag,
    )
    print(json.dumps(item, indent=2))


if __name__ == "__main__":
    main()
