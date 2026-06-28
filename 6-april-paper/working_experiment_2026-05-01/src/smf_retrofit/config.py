"""Configuration for the sparse memory retrofit prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    device_map: str | None = "auto"
    local_files_only: bool = False


@dataclass
class MemoryConfig:
    num_heads: int = 4
    top_k: int = 8
    n_keys: int = 128
    k_dim_per_head: int = 256
    v_dim: int = 128
    num_memory_layers: int = 3
    memory_layers: list[int] = field(default_factory=list)
    use_silu_gating: bool = True

    @property
    def num_entries(self) -> int:
        return self.n_keys**2

    @property
    def k_dim_half(self) -> int:
        return self.k_dim_per_head // 2


@dataclass
class TextDataConfig:
    path: str | None = None
    dataset_name: str | None = None
    dataset_config: str | None = None
    split: str = "train"
    text_field: str = "text"
    streaming: bool = False
    shuffle: bool = True
    shuffle_buffer: int = 10000
    max_samples: int | None = None
    seq_length: int = 512
    batch_size: int = 2
    pack_samples: bool = True
    num_workers: int = 0
    seed: int = 42


@dataclass
class RecoveryConfig:
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    total_steps: int = 200
    warmup_steps: int = 20
    gradient_accumulation_steps: int = 1
    log_every_steps: int = 10
    save_every_steps: int = 100
    gradient_clip: float = 1.0
    output_dir: str = "checkpoints/recovery"
    eval_data_path: str | None = None
    eval_max_batches: int = 20
    sanity_prompts_path: str | None = None
    max_loss_delta_vs_base: float = 1.0
    max_loss_ratio_vs_base: float = 1.25
    require_prompt_checks: bool = True


@dataclass
class BackgroundConfig:
    num_batches: int = 100
    output_path: str = "checkpoints/background_stats.pt"


@dataclass
class ContinualConfig:
    learning_rate: float = 1.0
    optimizer: str = "sgd"
    momentum: float = 0.0
    total_steps: int = 100
    gradient_accumulation_steps: int = 1
    log_every_steps: int = 10
    save_every_steps: int = 50
    gradient_clip: float = 1.0
    output_dir: str = "checkpoints/continual"
    train_full_memory: bool = False
    top_t: int = 64
    selector: str = "kl"
    background_stats_path: str = "checkpoints/background_stats.pt"
    recovery_checkpoint: str = "checkpoints/recovery/memory.pt"
    require_recovery_pass: bool = True
    recovery_eval_data_path: str | None = None
    recovery_eval_max_batches: int = 20
    recovery_max_loss_delta_vs_base: float = 1.0
    recovery_max_loss_ratio_vs_base: float = 1.25
    require_recovery_prompt_checks: bool = True


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    data: TextDataConfig = field(default_factory=TextDataConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    continual: ContinualConfig = field(default_factory=ContinualConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load YAML config into dataclasses."""
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return ExperimentConfig(
        model=ModelConfig(**raw.get("model", {})),
        memory=MemoryConfig(**raw.get("memory", {})),
        data=TextDataConfig(**raw.get("data", {})),
        recovery=RecoveryConfig(**raw.get("recovery", {})),
        background=BackgroundConfig(**raw.get("background", {})),
        continual=ContinualConfig(**raw.get("continual", {})),
    )
