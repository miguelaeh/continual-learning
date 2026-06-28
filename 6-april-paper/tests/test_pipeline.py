from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

from smf_retrofit.config import ContinualConfig, MemoryConfig, RecoveryConfig
from smf_retrofit.modeling.qwen import (
    freeze_for_recovery,
    freeze_for_sparse_finetuning,
    inject_memory_layers,
)
from smf_retrofit.training.background import collect_background_statistics
from smf_retrofit.training.recovery import run_recovery
from smf_retrofit.training.sparse_finetune import run_sparse_finetuning


def _tiny_dataloader(vocab_size: int) -> DataLoader:
    batch = []
    for _ in range(4):
        input_ids = torch.randint(0, vocab_size, (8,), dtype=torch.long)
        labels = torch.randint(0, vocab_size, (8,), dtype=torch.long)
        batch.append({"input_ids": input_ids, "labels": labels})
    return DataLoader(batch, batch_size=2)


def test_end_to_end_sparse_memory_pipeline(tmp_path: Path):
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
    )
    model, _, layer_indices = inject_memory_layers(
        model,
        MemoryConfig(
            num_heads=2,
            top_k=4,
            n_keys=8,
            k_dim_per_head=16,
            v_dim=4,
            num_memory_layers=2,
        ),
    )

    dataloader = _tiny_dataloader(vocab_size=128)

    freeze_for_recovery(model, layer_indices)
    recovery_path = run_recovery(
        model=model,
        dataloader=dataloader,
        config=RecoveryConfig(
            learning_rate=1e-3,
            total_steps=2,
            log_every_steps=1,
            save_every_steps=2,
            output_dir=str(tmp_path / "recovery"),
        ),
        layer_indices=layer_indices,
        device="cpu",
    )
    assert Path(recovery_path).exists()

    background_stats = collect_background_statistics(
        model=model,
        dataloader=dataloader,
        layer_indices=layer_indices,
        num_entries=64,
        num_batches=2,
        output_path=str(tmp_path / "background.pt"),
        device="cpu",
    )
    assert int(background_stats["total_batches"]) == 2

    freeze_for_sparse_finetuning(model, layer_indices)
    continual_path = run_sparse_finetuning(
        model=model,
        dataloader=dataloader,
        config=ContinualConfig(
            learning_rate=0.1,
            total_steps=2,
            log_every_steps=1,
            save_every_steps=2,
            output_dir=str(tmp_path / "continual"),
            top_t=8,
            selector="kl",
        ),
        background_stats=background_stats,
        layer_indices=layer_indices,
        num_entries=64,
        device="cpu",
    )
    assert Path(continual_path).exists()
