#!/usr/bin/env python3
"""Run a regression gate against provided outputs."""

from __future__ import annotations

import argparse
import json

from agent_learning_system.regression import evaluate_outputs, load_regression_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()

    spec = load_regression_spec(args.spec)
    with open(args.candidate, "r", encoding="utf-8") as handle:
        outputs = json.load(handle)
    result = evaluate_outputs(spec, outputs)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
