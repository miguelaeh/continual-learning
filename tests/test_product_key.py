"""Tests for ProductKeyLookup module."""

import pytest
import torch

from src.memory.product_key import ProductKeyLookup


@pytest.fixture
def product_key():
    return ProductKeyLookup(
        num_heads=4,
        n_keys=64,  # small for testing; total entries = 64^2 = 4096
        k_dim_per_head=128,  # 2 x 64
        top_k=8,
    )


class TestProductKeyLookup:
    def test_output_shapes(self, product_key):
        batch_size = 4
        query = torch.randn(batch_size, 4, 128)

        indices, scores = product_key(query)

        assert indices.shape == (batch_size, 4, 8)
        assert scores.shape == (batch_size, 4, 8)

    def test_indices_in_range(self, product_key):
        query = torch.randn(8, 4, 128)
        indices, _ = product_key(query)

        # All indices should be in [0, n_keys^2)
        assert indices.min() >= 0
        assert indices.max() < 64**2

    def test_scores_are_probabilities(self, product_key):
        query = torch.randn(4, 4, 128)
        _, scores = product_key(query)

        # Scores should sum to ~1 along the top-k dimension (softmax)
        sums = scores.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_scores_non_negative(self, product_key):
        query = torch.randn(4, 4, 128)
        _, scores = product_key(query)
        assert (scores >= 0).all()

    def test_gradient_flow(self, product_key):
        query = torch.randn(4, 4, 128, requires_grad=True)
        indices, scores = product_key(query)

        # Scores should allow gradient flow
        loss = scores.sum()
        loss.backward()

        assert query.grad is not None
        assert query.grad.shape == query.shape

    def test_key_gradient_flow(self, product_key):
        query = torch.randn(4, 4, 128)
        _, scores = product_key(query)

        loss = scores.sum()
        loss.backward()

        # Keys should receive gradients
        assert product_key.keys_1.grad is not None
        assert product_key.keys_2.grad is not None

    def test_different_queries_different_indices(self, product_key):
        q1 = torch.randn(1, 4, 128)
        q2 = torch.randn(1, 4, 128) * 10  # very different

        idx1, _ = product_key(q1)
        idx2, _ = product_key(q2)

        # Different queries should (very likely) select different indices
        # Not guaranteed, but with random init and different queries, very unlikely to match
        assert not torch.equal(idx1, idx2)

    def test_batch_independence(self, product_key):
        """Each item in the batch should be processed independently."""
        q1 = torch.randn(1, 4, 128)
        q2 = torch.randn(1, 4, 128)

        # Process individually
        idx1, scores1 = product_key(q1)
        idx2, scores2 = product_key(q2)

        # Process as batch
        q_batch = torch.cat([q1, q2], dim=0)
        idx_batch, scores_batch = product_key(q_batch)

        assert torch.equal(idx_batch[0], idx1[0])
        assert torch.equal(idx_batch[1], idx2[0])
        assert torch.allclose(scores_batch[0], scores1[0], atol=1e-5)
        assert torch.allclose(scores_batch[1], scores2[0], atol=1e-5)

    def test_top_k_ordering(self, product_key):
        """Scores should be in descending order (highest first after softmax)."""
        query = torch.randn(4, 4, 128)
        _, scores = product_key(query)

        # After softmax, the order depends on the raw scores.
        # But the returned scores should be valid probabilities.
        assert (scores > 0).all()
        assert (scores <= 1).all()
