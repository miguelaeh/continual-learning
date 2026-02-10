"""Configuration dataclasses for sparse memory finetuning."""

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import OmegaConf


@dataclass
class MemoryConfig:
    """Configuration for memory layers."""

    num_heads: int = 4
    top_k: int = 32
    n_keys: int = 1024  # sub-keys per half -> total entries = n_keys^2
    k_dim_per_head: int = 512  # key dim per head (split into 2 x 256 for product keys)
    v_dim: int = 1024  # value dimension in EmbeddingBag
    memory_layers: list[int] = field(default_factory=lambda: [9, 17, 25])
    use_silu_gating: bool = True  # Memory+ variant

    @property
    def num_entries(self) -> int:
        return self.n_keys**2

    @property
    def k_dim_half(self) -> int:
        return self.k_dim_per_head // 2

    @property
    def total_query_dim(self) -> int:
        return self.k_dim_per_head * self.num_heads


@dataclass
class PretrainConfig:
    """Configuration for Phase 1: memory layer pretraining."""

    base_model: str = "google/gemma-3-4b-it"
    learning_rate: float = 1e-4
    value_learning_rate: float = 1e-3
    warmup_steps: int = 4000
    total_steps: int = 128000
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    seq_length: int = 4096
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    dtype: str = "bfloat16"
    dataset: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "default"
    checkpoint_dir: str = "checkpoints/pretrain"
    save_every_steps: int = 5000
    log_every_steps: int = 100
    seed: int = 42


@dataclass
class IDFCollectionConfig:
    """Configuration for Phase 2: IDF background statistics collection."""

    base_model: str = "google/gemma-3-4b-it"
    memory_checkpoint: str = "checkpoints/pretrain/memory_layers.pt"
    num_background_batches: int = 1000
    batch_size: int = 8
    seq_length: int = 4096
    dtype: str = "bfloat16"
    dataset: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "default"
    output_path: str = "checkpoints/idf_statistics.pt"
    seed: int = 42


@dataclass
class ContinualLearningConfig:
    """Configuration for Phase 3: sparse memory finetuning."""

    base_model: str = "google/gemma-3-4b-it"
    memory_checkpoint: str = "checkpoints/pretrain/memory_layers.pt"
    idf_statistics_path: str = "checkpoints/idf_statistics.pt"
    top_t: int = 500  # number of trainable memory slots
    optimizer: str = "sgd"  # SGD is critical (11% vs 30% forgetting with AdamW)
    learning_rate: float = 2.0  # high LR works well with SGD for sparse updates
    momentum: float = 0.0
    batch_size: int = 64
    seq_length: int = 512
    dtype: str = "bfloat16"
    dataset: str = ""
    checkpoint_dir: str = "checkpoints/continual"
    save_every_steps: int = 500
    log_every_steps: int = 10
    eval_every_steps: int = 100
    total_steps: int = 5000
    seed: int = 42


@dataclass
class DistillConfig:
    """Configuration for distillation-based memory layer initialization."""

    base_model: str = "google/gemma-3-4b-it"
    learning_rate: float = 1e-3  # 10x pretrain (direct signal allows higher LR)
    value_learning_rate: float = 1e-2  # 10x projection LR
    warmup_steps: int = 100
    total_steps: int = 2000  # ~64x shorter than pretrain
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    seq_length: int = 2048
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    dtype: str = "bfloat16"
    dataset: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "default"
    checkpoint_dir: str = "checkpoints/distill"
    save_every_steps: int = 500
    log_every_steps: int = 50
    seed: int = 42


def load_config(config_path: str) -> OmegaConf:
    """Load a YAML config file and return an OmegaConf DictConfig."""
    return OmegaConf.load(config_path)


def make_memory_config(cfg: OmegaConf) -> MemoryConfig:
    """Extract MemoryConfig from a loaded OmegaConf config."""
    return MemoryConfig(**OmegaConf.to_container(cfg.memory, resolve=True))


def make_pretrain_config(cfg: OmegaConf) -> tuple[MemoryConfig, PretrainConfig]:
    """Extract MemoryConfig and PretrainConfig from a loaded config."""
    mem = make_memory_config(cfg)
    train = PretrainConfig(**OmegaConf.to_container(cfg.pretrain, resolve=True))
    return mem, train


def make_collection_config(
    cfg: OmegaConf,
) -> tuple[MemoryConfig, IDFCollectionConfig]:
    """Extract MemoryConfig and IDFCollectionConfig from a loaded config."""
    mem = make_memory_config(cfg)
    coll = IDFCollectionConfig(**OmegaConf.to_container(cfg.collection, resolve=True))
    return mem, coll


def make_continual_config(
    cfg: OmegaConf,
) -> tuple[MemoryConfig, ContinualLearningConfig]:
    """Extract MemoryConfig and ContinualLearningConfig from a loaded config."""
    mem = make_memory_config(cfg)
    cl = ContinualLearningConfig(
        **OmegaConf.to_container(cfg.continual, resolve=True)
    )
    return mem, cl


def make_distill_config(
    cfg: OmegaConf,
) -> tuple[MemoryConfig, DistillConfig]:
    """Extract MemoryConfig and DistillConfig from a loaded config."""
    mem = make_memory_config(cfg)
    distill = DistillConfig(**OmegaConf.to_container(cfg.distill, resolve=True))
    return mem, distill
