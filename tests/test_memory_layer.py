"""Tests for MemoryPlusLayer and SharedMemoryStore."""

import pytest
import torch

from src.memory.memory_layer import MemoryPlusLayer
from src.memory.shared_memory_store import SharedMemoryStore


@pytest.fixture
def shared_store():
    return SharedMemoryStore(
        num_heads=4,
        n_keys=64,  # 64^2 = 4096 entries
        k_dim_per_head=128,
        v_dim=64,
        top_k=8,
    )


@pytest.fixture
def memory_layer(shared_store):
    return MemoryPlusLayer(
        d_model=256,
        shared_store=shared_store,
        num_heads=4,
        k_dim_per_head=128,
        v_dim=64,
        top_k=8,
        use_silu_gating=True,
    )


@pytest.fixture
def memory_layer_no_gating(shared_store):
    return MemoryPlusLayer(
        d_model=256,
        shared_store=shared_store,
        num_heads=4,
        k_dim_per_head=128,
        v_dim=64,
        top_k=8,
        use_silu_gating=False,
    )


class TestSharedMemoryStore:
    def test_lookup_shapes(self, shared_store):
        query = torch.randn(8, 4, 128)
        indices, scores = shared_store.lookup(query)

        assert indices.shape == (8, 4, 8)
        assert scores.shape == (8, 4, 8)

    def test_retrieve_values_shape(self, shared_store):
        # Simulate flattened indices/scores: (batch, num_heads * top_k)
        indices = torch.randint(0, 64**2, (4, 32))
        scores = torch.softmax(torch.randn(4, 32), dim=-1)

        output = shared_store.retrieve_values(indices, scores)
        assert output.shape == (4, 64)  # (batch, v_dim)

    def test_num_entries(self, shared_store):
        assert shared_store.num_entries == 64**2


class TestMemoryPlusLayer:
    def test_forward_shape(self, memory_layer):
        x = torch.randn(2, 16, 256)  # (batch, seq, d_model)
        output = memory_layer(x)

        assert output.shape == (2, 16, 256)

    def test_forward_shape_no_gating(self, memory_layer_no_gating):
        x = torch.randn(2, 16, 256)
        output = memory_layer_no_gating(x)

        assert output.shape == (2, 16, 256)

    def test_gradient_flow_to_query(self, memory_layer):
        x = torch.randn(2, 8, 256, requires_grad=True)
        output = memory_layer(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_flow_to_params(self, memory_layer):
        x = torch.randn(2, 8, 256)
        output = memory_layer(x)

        loss = output.sum()
        loss.backward()

        # Check key memory layer parameters have gradients
        assert memory_layer.query_proj.weight.grad is not None
        assert memory_layer.silu_proj.weight.grad is not None
        assert memory_layer.value_proj.weight.grad is not None

    def test_index_tracking(self, memory_layer):
        memory_layer.enable_index_tracking()
        x = torch.randn(2, 8, 256)
        memory_layer(x)

        indices = memory_layer.get_last_accessed_indices()
        assert indices is not None
        # indices should be (B*T, H, top_k) = (16, 4, 8)
        assert indices.shape == (16, 4, 8)

        memory_layer.disable_index_tracking()
        assert memory_layer.get_last_accessed_indices() is None

    def test_gradient_masking(self, memory_layer, shared_store):
        # Create a mask that makes only first 100 slots trainable
        num_entries = shared_store.num_entries
        mask = torch.zeros(num_entries, dtype=torch.bool)
        mask[:100] = True

        memory_layer.set_trainable_mask(mask)

        x = torch.randn(2, 8, 256, requires_grad=True)
        output = memory_layer(x)
        loss = output.sum()
        loss.backward()

        # The forward should still work
        assert output.shape == (2, 8, 256)

        # Clean up
        memory_layer.set_trainable_mask(None)

    def test_single_token(self, memory_layer):
        """Test with a single token (seq_len=1)."""
        x = torch.randn(1, 1, 256)
        output = memory_layer(x)
        assert output.shape == (1, 1, 256)

    def test_deterministic(self, memory_layer):
        """Same input should produce same output."""
        x = torch.randn(2, 8, 256)
        out1 = memory_layer(x)
        out2 = memory_layer(x)
        assert torch.allclose(out1, out2, atol=1e-6)
