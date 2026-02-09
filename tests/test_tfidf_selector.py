"""Tests for TF-IDF slot selection."""

import pytest
import torch

from src.continual.tfidf_selector import TFIDFSelector


@pytest.fixture
def idf_statistics():
    num_entries = 4096
    # Simulate: some slots accessed in many batches (common), some in few (rare)
    df = torch.zeros(num_entries, dtype=torch.long)
    df[:1000] = 500  # common slots
    df[1000:2000] = 50  # medium slots
    df[2000:3000] = 5  # rare slots
    # df[3000:4096] = 0  # never accessed

    return {
        "slot_document_frequency": df,
        "total_batches": 1000,
    }


@pytest.fixture
def selector(idf_statistics):
    return TFIDFSelector(idf_statistics, top_t=100)


class TestTFIDFSelector:
    def test_select_shape(self, selector):
        # Simulate access counts where rare slots are accessed more
        access_counts = torch.zeros(4096, dtype=torch.long)
        access_counts[2500:2600] = 10  # access rare slots
        access_counts[500:550] = 5  # also access some common slots

        selected = selector.select(access_counts)
        assert selected.shape == (100,)

    def test_prefers_rare_slots(self, selector):
        """Slots that are rare in background but frequent in current batch should rank higher."""
        access_counts = torch.zeros(4096, dtype=torch.long)
        # Access rare slots (df=5) and common slots (df=500) equally
        access_counts[2500] = 10  # rare slot, high IDF
        access_counts[500] = 10  # common slot, low IDF

        selected = selector.select(access_counts)
        selected_set = set(selected.tolist())

        # Rare slot should be selected (high TF-IDF)
        assert 2500 in selected_set

    def test_prefers_never_seen_slots(self, selector):
        """Slots never seen in background should have highest IDF."""
        access_counts = torch.zeros(4096, dtype=torch.long)
        access_counts[3500] = 10  # never seen in background (df=0)
        access_counts[500] = 10  # very common (df=500)

        selected = selector.select(access_counts)
        selected_set = set(selected.tolist())

        # Never-seen slot should be selected
        assert 3500 in selected_set

    def test_compute_access_counts(self, selector):
        indices = [
            torch.tensor([0, 1, 2, 0, 1, 0]),
            torch.tensor([3, 4, 5]),
        ]
        counts = selector.compute_access_counts(indices, num_entries=10)

        assert counts[0] == 3
        assert counts[1] == 2
        assert counts[2] == 1
        assert counts[3] == 1
        assert counts[4] == 1
        assert counts[5] == 1
        assert counts[6:].sum() == 0

    def test_create_trainable_mask(self, selector):
        indices = torch.tensor([10, 20, 30])
        mask = selector.create_trainable_mask(indices, num_entries=100)

        assert mask.shape == (100,)
        assert mask.dtype == torch.bool
        assert mask[10] is True or mask[10].item()
        assert mask[20] is True or mask[20].item()
        assert mask[30] is True or mask[30].item()
        assert mask.sum() == 3

    def test_zero_access_counts(self, selector):
        """No accesses should still return top_t indices (from IDF alone)."""
        access_counts = torch.zeros(4096, dtype=torch.long)
        selected = selector.select(access_counts)
        assert selected.shape == (100,)
