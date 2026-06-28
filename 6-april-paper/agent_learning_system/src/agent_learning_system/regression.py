"""Regression gate helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class RegressionCheck:
    id: str
    prompt: str
    expected_contains_any: list[str]


def load_regression_spec(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def evaluate_outputs(spec: dict, outputs: dict[str, str]) -> dict:
    results = []
    passed = True
    for item in spec.get("checks", []):
        expected = [str(x) for x in item.get("expected_contains_any", [])]
        output = str(outputs.get(item["id"], ""))
        check_passed = any(token.lower() in output.lower() for token in expected)
        results.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "output": output,
                "expected_contains_any": expected,
                "passed": check_passed,
            }
        )
        passed = passed and check_passed
    return {
        "gate_name": spec.get("gate_name", "unnamed-gate"),
        "passed": passed,
        "results": results,
    }
