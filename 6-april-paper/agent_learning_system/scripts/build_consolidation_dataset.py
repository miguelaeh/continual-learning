#!/usr/bin/env python3
"""Build a consolidation dataset from episodic traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_learning_system.consolidation import build_consolidation_examples
from agent_learning_system.store import EpisodicStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-outcome", action="append", default=["success", "corrected"])
    args = parser.parse_args()

    store = EpisodicStore(args.store_dir)
    traces = store.load_all()
    examples = build_consolidation_examples(traces, include_outcomes=set(args.include_outcome))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=True) + "\n")

    print(f"wrote {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
