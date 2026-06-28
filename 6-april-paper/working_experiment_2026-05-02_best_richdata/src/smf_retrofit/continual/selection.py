"""Slot selection for sparse memory updates."""

from __future__ import annotations

import torch


def compute_access_counts(
    indices_list: list[torch.Tensor],
    num_entries: int,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    counts = torch.zeros(num_entries, dtype=torch.long)
    flat_token_mask = None
    if token_mask is not None:
        flat_token_mask = token_mask.reshape(-1).to(dtype=torch.bool, device="cpu")
    for indices in indices_list:
        selected = indices
        if flat_token_mask is not None:
            if indices.shape[0] != flat_token_mask.shape[0]:
                raise ValueError(
                    "Token mask length does not match tracked index rows. "
                    f"Got mask={flat_token_mask.shape[0]} rows={indices.shape[0]}."
                )
            selected = indices[flat_token_mask.to(indices.device)]
        flat = selected.reshape(-1).cpu()
        if flat.numel() == 0:
            continue
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


def compute_contrastive_access_counts(
    positive_access_counts: torch.Tensor,
    negative_access_counts: torch.Tensor | None,
    negative_scale: float = 1.0,
) -> torch.Tensor:
    if negative_access_counts is None:
        return positive_access_counts
    return torch.clamp(
        positive_access_counts.float() - (negative_scale * negative_access_counts.float()),
        min=0,
    )


def access_count_similarity(
    target_access_counts: torch.Tensor,
    candidate_access_counts: torch.Tensor,
    metric: str = "jaccard",
) -> float:
    target = target_access_counts.float()
    candidate = candidate_access_counts.float()

    if metric == "cosine":
        denom = target.norm() * candidate.norm()
        if denom.item() == 0:
            return 0.0
        return float(torch.dot(target, candidate) / denom)

    target_mask = target > 0
    candidate_mask = candidate > 0
    if metric == "jaccard":
        union = (target_mask | candidate_mask).sum().item()
        if union == 0:
            return 0.0
        intersection = (target_mask & candidate_mask).sum().item()
        return float(intersection / union)

    if metric == "overlap":
        denom = target_mask.sum().item()
        if denom == 0:
            return 0.0
        intersection = (target_mask & candidate_mask).sum().item()
        return float(intersection / denom)

    raise ValueError(f"Unsupported access similarity metric: {metric}")


def select_top_neighbor_indices(
    target_access_counts: torch.Tensor,
    candidate_access_counts: list[torch.Tensor],
    top_k: int,
    metric: str = "jaccard",
) -> list[int]:
    if not candidate_access_counts:
        return []
    scored = [
        (access_count_similarity(target_access_counts, candidate, metric=metric), idx)
        for idx, candidate in enumerate(candidate_access_counts)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    k = min(top_k, len(scored))
    return [idx for _, idx in scored[:k]]


def aggregate_access_counts(
    candidate_access_counts: list[torch.Tensor],
    indices: list[int],
) -> torch.Tensor | None:
    if not indices:
        return None
    total = torch.zeros_like(candidate_access_counts[indices[0]], dtype=torch.float32)
    for idx in indices:
        total += candidate_access_counts[idx].float()
    return total


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
