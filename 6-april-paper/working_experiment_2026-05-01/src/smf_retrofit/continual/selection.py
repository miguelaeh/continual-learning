"""Slot selection for sparse memory updates."""

from __future__ import annotations

import torch


def compute_access_counts(
    indices_list: list[torch.Tensor],
    num_entries: int,
) -> torch.Tensor:
    counts = torch.zeros(num_entries, dtype=torch.long)
    for indices in indices_list:
        flat = indices.reshape(-1).cpu()
        counts.scatter_add_(0, flat.long(), torch.ones_like(flat, dtype=torch.long))
    return counts


def create_trainable_mask(
    trainable_indices: torch.Tensor,
    num_entries: int,
    device: torch.device | str,
) -> torch.Tensor:
    mask = torch.zeros(num_entries, dtype=torch.bool, device=device)
    mask[trainable_indices.to(device)] = True
    return mask


class TFIDFSlotSelector:
    """TF-IDF baseline for sparse slot selection."""

    def __init__(self, document_frequency: torch.Tensor, total_batches: int, top_t: int):
        self.document_frequency = document_frequency.float()
        self.total_batches = total_batches
        self.top_t = top_t
        self.idf = torch.log((self.total_batches + 1.0) / (self.document_frequency + 1.0))

    def select(self, access_counts: torch.Tensor) -> torch.Tensor:
        total = access_counts.sum().clamp(min=1)
        tf = access_counts.float() / total
        scores = tf * self.idf.to(tf.device)
        if access_counts.any():
            scores = scores.masked_fill(access_counts == 0, float("-inf"))
        return _safe_topk(scores, self.top_t)


class KLSlotSelector:
    """KL-based slot scoring for surprising memory usage."""

    def __init__(
        self,
        document_frequency: torch.Tensor,
        total_batches: int,
        top_t: int,
        smoothing: float = 1.0,
    ):
        self.document_frequency = document_frequency.float()
        self.total_batches = total_batches
        self.top_t = top_t
        self.smoothing = smoothing

    def select(self, access_counts: torch.Tensor) -> torch.Tensor:
        total_access = access_counts.sum().clamp(min=1).float()
        p_batch = access_counts.float() / total_access

        background = self.document_frequency + self.smoothing
        q_background = background / background.sum()
        scores = p_batch * torch.log(p_batch.clamp(min=1e-12) / q_background.to(p_batch.device))
        scores = scores.masked_fill(access_counts == 0, float("-inf"))
        return _safe_topk(scores, self.top_t)


def _safe_topk(scores: torch.Tensor, top_t: int) -> torch.Tensor:
    valid_mask = torch.isfinite(scores)
    if not valid_mask.any():
        return torch.zeros(min(top_t, scores.shape[0]), dtype=torch.long)

    valid_count = int(valid_mask.sum().item())
    k = min(top_t, valid_count)
    _, indices = scores.topk(k)
    return indices
