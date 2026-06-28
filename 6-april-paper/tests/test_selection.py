import pytest
import torch

from smf_retrofit.continual.selection import (
    KLSlotSelector,
    TFIDFSlotSelector,
    access_count_similarity,
    aggregate_access_counts,
    compute_contrastive_access_counts,
    compute_access_counts,
    select_top_neighbor_indices,
)


def test_tfidf_prefers_rare_slots():
    df = torch.zeros(64, dtype=torch.long)
    df[:32] = 100
    df[40] = 1
    selector = TFIDFSlotSelector(df, total_batches=200, top_t=4)

    access = torch.zeros(64, dtype=torch.long)
    access[5] = 10
    access[40] = 10
    chosen = selector.select(access)

    assert 40 in chosen.tolist()


def test_kl_prefers_surprising_slots():
    df = torch.ones(64, dtype=torch.long) * 50
    df[50] = 1
    selector = KLSlotSelector(df, total_batches=100, top_t=4)

    access = torch.zeros(64, dtype=torch.long)
    access[10] = 10
    access[50] = 10
    chosen = selector.select(access)

    assert 50 in chosen.tolist()


def test_compute_access_counts_can_focus_on_supervised_positions():
    indices = torch.tensor(
        [
            [[1, 2]],
            [[3, 4]],
            [[5, 6]],
        ],
        dtype=torch.long,
    )
    token_mask = torch.tensor([False, True, False], dtype=torch.bool)

    counts = compute_access_counts([indices], num_entries=8, token_mask=token_mask)

    assert counts[3].item() == 1
    assert counts[4].item() == 1
    assert counts.sum().item() == 2


def test_compute_access_counts_raises_on_bad_mask_length():
    indices = torch.tensor([[[1, 2]], [[3, 4]]], dtype=torch.long)
    token_mask = torch.tensor([True], dtype=torch.bool)

    with pytest.raises(ValueError):
        compute_access_counts([indices], num_entries=8, token_mask=token_mask)


def test_compute_contrastive_access_counts_suppresses_shared_slots():
    positive = torch.tensor([0, 5, 4, 1, 0], dtype=torch.long)
    negative = torch.tensor([0, 1, 4, 0, 0], dtype=torch.long)

    effective = compute_contrastive_access_counts(
        positive,
        negative,
        negative_scale=1.0,
    )

    assert effective.tolist() == [0.0, 4.0, 0.0, 1.0, 0.0]


def test_access_count_similarity_jaccard_prefers_more_overlap():
    target = torch.tensor([0, 1, 1, 0, 1], dtype=torch.long)
    close = torch.tensor([0, 1, 1, 0, 0], dtype=torch.long)
    far = torch.tensor([1, 0, 0, 1, 0], dtype=torch.long)

    assert access_count_similarity(target, close, metric="jaccard") > access_count_similarity(
        target,
        far,
        metric="jaccard",
    )


def test_select_top_neighbor_indices_returns_most_similar_candidates():
    target = torch.tensor([0, 1, 1, 0, 1], dtype=torch.long)
    candidates = [
        torch.tensor([1, 0, 0, 1, 0], dtype=torch.long),
        torch.tensor([0, 1, 1, 0, 0], dtype=torch.long),
        torch.tensor([0, 1, 0, 0, 1], dtype=torch.long),
    ]

    chosen = select_top_neighbor_indices(target, candidates, top_k=2, metric="jaccard")

    assert chosen == [1, 2]


def test_aggregate_access_counts_sums_selected_neighbors():
    candidates = [
        torch.tensor([1, 0, 2], dtype=torch.long),
        torch.tensor([0, 3, 0], dtype=torch.long),
        torch.tensor([1, 1, 1], dtype=torch.long),
    ]

    total = aggregate_access_counts(candidates, [0, 2])

    assert total.tolist() == [2.0, 1.0, 3.0]
