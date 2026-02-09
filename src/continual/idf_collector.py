"""Phase 2: Collect background memory access statistics for IDF computation.

Runs inference on background data (e.g., FineWeb-Edu) and tracks which
memory slots are accessed. The document frequency (number of batches where
each slot appears) is used as the IDF denominator in TF-IDF slot selection
during Phase 3.
"""

import logging

import torch
from torch.utils.data import DataLoader

from src.config import MemoryConfig
from src.memory.memory_layer import MemoryPlusLayer
from src.model.memory_gemma import get_memory_layers

logger = logging.getLogger(__name__)


class IDFCollector:
    """Collects memory slot access statistics on background data.

    For each batch:
    1. Forward pass through model (no gradient)
    2. Record which memory slots were accessed across all memory layers
    3. Increment document_frequency[slot] if slot was accessed in this batch

    The document frequency counts how many batches each slot appears in,
    NOT how many times it appears within a batch.

    Args:
        model: Model with injected memory layers.
        memory_config: Memory layer configuration.
        num_batches: Number of background batches to process.
        device: Device for inference.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        memory_config: MemoryConfig,
        num_batches: int = 1000,
        device: torch.device | str = "cuda",
    ):
        self.model = model
        self.memory_config = memory_config
        self.num_batches = num_batches
        self.device = device
        self.memory_layers = get_memory_layers(model, memory_config)

    @torch.no_grad()
    def collect(self, dataloader: DataLoader) -> dict:
        """Collect IDF statistics from background data.

        Args:
            dataloader: Background data dataloader.

        Returns:
            Dictionary with:
            - 'slot_document_frequency': Tensor (num_entries,) - batches where each slot appears
            - 'total_batches': int - number of batches processed
        """
        self.model.eval()
        num_entries = self.memory_config.num_entries

        # Enable index tracking on all memory layers
        for layer in self.memory_layers:
            layer.enable_index_tracking()

        slot_df = torch.zeros(num_entries, dtype=torch.long)

        batch_count = 0
        for batch in dataloader:
            if batch_count >= self.num_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = torch.ones_like(input_ids)

            # Forward pass to trigger memory access
            self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Collect accessed indices from all memory layers
            all_indices = []
            for layer in self.memory_layers:
                indices = layer.get_last_accessed_indices()
                if indices is not None:
                    # indices: (B*T, H, top_k) -> flatten to get unique slots
                    all_indices.append(indices.reshape(-1).cpu())

            if all_indices:
                all_indices = torch.cat(all_indices)
                unique_indices = torch.unique(all_indices)
                slot_df[unique_indices] += 1

            batch_count += 1
            if batch_count % 100 == 0:
                logger.info(
                    f"Collected {batch_count}/{self.num_batches} batches, "
                    f"active slots: {(slot_df > 0).sum().item()}/{num_entries}"
                )

        # Disable index tracking
        for layer in self.memory_layers:
            layer.disable_index_tracking()

        logger.info(
            f"IDF collection complete: {batch_count} batches, "
            f"{(slot_df > 0).sum().item()} slots accessed at least once"
        )

        return {
            "slot_document_frequency": slot_df,
            "total_batches": batch_count,
        }
