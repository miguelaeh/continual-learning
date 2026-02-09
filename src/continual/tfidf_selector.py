"""TF-IDF based memory slot selection for continual learning.

Selects which memory slots to update based on TF-IDF ranking:
- TF (term frequency): how often a slot is accessed on the current batch
- IDF (inverse document frequency): how rare a slot is across background data

Slots that are highly accessed on new data but rare in pretraining data
are likely task-specific and should be updated.

Reference: "Continual Learning via Sparse Memory Finetuning" (Lin et al., 2025)
"""

import torch


class TFIDFSelector:
    """Selects top-t memory slots for updating based on TF-IDF ranking.

    TF-IDF formula:
        score(i) = TF(i) * IDF(i)
        TF(i) = c(i) / sum_j(c(j))
        IDF(i) = log((|B| + 1) / (df(i) + 1))

    Where:
    - c(i): access count for slot i on current batch
    - df(i): number of background batches where slot i was accessed
    - |B|: total background batches

    Args:
        idf_statistics: Dict from IDFCollector.collect() with
            'slot_document_frequency' and 'total_batches'.
        top_t: Number of slots to select for updating.
    """

    def __init__(self, idf_statistics: dict, top_t: int = 500):
        self.document_frequency = idf_statistics["slot_document_frequency"].float()
        self.total_batches = idf_statistics["total_batches"]
        self.top_t = top_t

        # Precompute IDF: log((|B| + 1) / (df + 1))
        self.idf = torch.log(
            (self.total_batches + 1.0) / (self.document_frequency + 1.0)
        )

    def select(self, access_counts: torch.Tensor) -> torch.Tensor:
        """Select top-t slots based on TF-IDF ranking.

        Args:
            access_counts: Tensor of shape (num_entries,) - access counts per
                slot on the current batch.

        Returns:
            trainable_indices: Tensor of shape (top_t,) - indices of selected slots.
        """
        # TF: normalized frequency in current batch
        total = access_counts.sum().clamp(min=1)
        tf = access_counts.float() / total

        # TF-IDF score
        idf = self.idf.to(tf.device)
        tfidf_scores = tf * idf

        # Select top-t
        _, trainable_indices = tfidf_scores.topk(self.top_t)
        return trainable_indices

    def compute_access_counts(
        self,
        indices_list: list[torch.Tensor],
        num_entries: int,
    ) -> torch.Tensor:
        """Compute per-slot access counts from a list of accessed index tensors.

        Args:
            indices_list: List of index tensors from memory layers
                (each of shape (B*T, H, top_k) or flattened).
            num_entries: Total number of memory entries.

        Returns:
            access_counts: Tensor of shape (num_entries,) - counts per slot.
        """
        counts = torch.zeros(num_entries, dtype=torch.long)

        for indices in indices_list:
            flat = indices.reshape(-1).cpu()
            counts.scatter_add_(
                0, flat.long(), torch.ones_like(flat, dtype=torch.long)
            )

        return counts

    def create_trainable_mask(
        self,
        trainable_indices: torch.Tensor,
        num_entries: int,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Create a boolean mask from selected indices.

        Args:
            trainable_indices: Tensor of shape (top_t,) - selected slot indices.
            num_entries: Total number of memory entries.
            device: Target device.

        Returns:
            mask: Boolean tensor of shape (num_entries,). True = trainable.
        """
        mask = torch.zeros(num_entries, dtype=torch.bool, device=device)
        mask[trainable_indices.to(device)] = True
        return mask
