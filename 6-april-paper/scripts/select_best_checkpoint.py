#!/usr/bin/env python3
"""Score saved checkpoints against a prompt suite and choose the best one."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from smf_retrofit.eval import build_model, generate_text
from smf_retrofit.config import load_experiment_config
from smf_retrofit.modeling.qwen import load_memory_checkpoint


def _step_key(path: Path) -> tuple[int, str]:
    match = re.search(r"memory_step_(\d+)\.pt$", path.name)
    if match:
        return (int(match.group(1)), path.name)
    if path.name == "memory.pt":
        return (10**9, path.name)
    return (-1, path.name)


def _load_specs(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError("Prompt spec file must be a JSON list.")
    specs: list[dict] = []
    for item in raw:
        if not isinstance(item, dict) or "prompt" not in item:
            raise ValueError(f"Invalid prompt spec: {item!r}")
        specs.append(
            {
                "prompt": str(item["prompt"]),
                "expected_any": [str(x) for x in item.get("expected_any", [])],
                "forbidden_any": [str(x) for x in item.get("forbidden_any", [])],
                "weight": float(item.get("weight", 1.0)),
            }
        )
    return specs


def _score_output(output: str, spec: dict) -> tuple[float, bool]:
    lowered = output.lower()
    expected = [item.lower() for item in spec["expected_any"]]
    forbidden = [item.lower() for item in spec["forbidden_any"]]

    expected_ok = any(token in lowered for token in expected) if expected else True
    forbidden_ok = not any(token in lowered for token in forbidden)
    passed = expected_ok and forbidden_ok
    return (spec["weight"] if passed else 0.0), passed


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best checkpoint from a saved run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    specs = _load_specs(args.prompts_file)
    checkpoint_dir = Path(args.checkpoints_dir)
    checkpoints = sorted(checkpoint_dir.glob("memory*.pt"), key=_step_key)
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    model, tokenizer = build_model(cfg, checkpoint=str(checkpoints[0]))

    results = []
    best = None
    for checkpoint in checkpoints:
        load_memory_checkpoint(model, str(checkpoint))
        prompt_results = []
        total_score = 0.0
        passed_count = 0
        for spec in specs:
            output = generate_text(
                model,
                tokenizer,
                spec["prompt"],
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            score, passed = _score_output(output, spec)
            total_score += score
            passed_count += int(passed)
            prompt_results.append(
                {
                    "prompt": spec["prompt"],
                    "output": output,
                    "passed": passed,
                    "score": score,
                    "expected_any": spec["expected_any"],
                    "forbidden_any": spec["forbidden_any"],
                }
            )
        row = {
            "checkpoint": str(checkpoint),
            "score": total_score,
            "passed": passed_count,
            "total_prompts": len(specs),
            "prompt_results": prompt_results,
        }
        results.append(row)
        if best is None or (row["score"], row["passed"]) > (best["score"], best["passed"]):
            best = row

    assert best is not None
    print(f"best checkpoint: {best['checkpoint']}")
    print(f"score: {best['score']:.2f} | passed: {best['passed']}/{best['total_prompts']}")
    for item in best["prompt_results"]:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"[{status}] {item['prompt']}")
        print(item["output"].replace("\n", "\\n"))

    if args.output_json:
        payload = {"best": best, "results": results}
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
