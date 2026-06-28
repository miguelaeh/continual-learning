"""Background slot-usage statistics collection."""

from __future__ import annotations

import logging

import torch

from smf_retrofit.continual.selection import compute_access_counts
from smf_retrofit.modeling.qwen import get_memory_layers
from smf_retrofit.utils import ensure_parent_dir


logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_background_statistics(
    model: torch.nn.Module,
    dataloader,
    layer_indices: list[int],
    num_entries: int,
    num_batches: int,
    log_every_batches: int,
    output_path: str,
    device: str,
) -> dict:
    model.eval()
    memory_layers = get_memory_layers(model, layer_indices)
    document_frequency = torch.zeros(num_entries, dtype=torch.long)
    observed_batches = 0

    for batch in dataloader:
        if observed_batches >= num_batches:
            break

        for layer in memory_layers:
            layer.enable_index_tracking()

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        indices_list = []
        for layer in memory_layers:
            indices = layer.get_last_accessed_indices()
            if indices is not None:
                indices_list.append(indices)
            layer.disable_index_tracking()

        access_counts = compute_access_counts(indices_list, num_entries)
        document_frequency += (access_counts > 0).long()
        observed_batches += 1
        if log_every_batches > 0 and observed_batches % log_every_batches == 0:
            logger.info(
                "Background stats progress %s/%s batches",
                observed_batches,
                num_batches,
            )

    if observed_batches == 0:
        raise ValueError(
            "Background dataloader produced zero batches. "
            "Reduce data.seq_length or provide more background text."
        )

    stats = {
        "slot_document_frequency": document_frequency,
        "total_batches": observed_batches,
    }
    ensure_parent_dir(output_path)
    torch.save(stats, output_path)
    logger.info("Saved background statistics to %s", output_path)
    return stats
