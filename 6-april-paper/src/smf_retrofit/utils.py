"""Shared utilities."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_torch_dtype(name: str) -> torch.dtype:
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def detect_device(device_map: str | None) -> str | None:
    if device_map and device_map != "auto":
        return device_map
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
