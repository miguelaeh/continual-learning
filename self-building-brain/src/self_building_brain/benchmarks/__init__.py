"""Realistic benchmark adapters for text-based memory evaluation."""

from .realistic import (
    BenchmarkExample,
    GraphTextMemory,
    QwenTextBackend,
    SlotTextMemory,
    evaluate_examples,
    load_locomo_examples,
    load_longbench_v2_examples,
    load_musique_examples,
)

__all__ = [
    "BenchmarkExample",
    "GraphTextMemory",
    "QwenTextBackend",
    "SlotTextMemory",
    "evaluate_examples",
    "load_locomo_examples",
    "load_longbench_v2_examples",
    "load_musique_examples",
]
