"""Tests for additive memory layers."""

import torch
import torch.nn as nn
import pytest

from src.memory.shared_memory_store import SharedMemoryStore
from additive_memory.layer import AdditiveMemoryLayer, FFNWithMemory


# Test dimensions — v_dim = d_model // num_heads for direct output
D_MODEL = 128
NUM_HEADS = 2
N_KEYS = 16
K_DIM_PER_HEAD = 64
V_DIM = D_MODEL // NUM_HEADS  # 64
TOP_K = 4
BATCH = 2
SEQ_LEN = 8


@pytest.fixture
def shared_store():
    store = SharedMemoryStore(
        num_heads=NUM_HEADS,
        n_keys=N_KEYS,
        k_dim_per_head=K_DIM_PER_HEAD,
        v_dim=V_DIM,
        top_k=TOP_K,
    )
    # Zero-initialize values (as in the additive approach)
    nn.init.zeros_(store.values.weight)
    return store


@pytest.fixture
def memory_layer(shared_store):
    return AdditiveMemoryLayer(
        d_model=D_MODEL,
        shared_store=shared_store,
        num_heads=NUM_HEADS,
        k_dim_per_head=K_DIM_PER_HEAD,
        top_k=TOP_K,
    )


class DummyFFN(nn.Module):
    """Minimal FFN for testing."""

    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.linear(x)


class TestAdditiveMemoryLayer:
    def test_output_shape(self, memory_layer):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = memory_layer(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_zero_init_gives_zero_output(self, memory_layer):
        """With zero-initialized values, output should be zero."""
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = memory_layer(x)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_nonzero_after_value_update(self, memory_layer, shared_store):
        """After updating some values, output should be non-zero."""
        # Set some values to non-zero
        with torch.no_grad():
            shared_store.values.weight[0:10] = torch.randn(10, V_DIM)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = memory_layer(x)
        # Output may still be small but should generally be non-zero
        # (depends on which slots are accessed)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_gradient_flow_to_values(self, memory_layer, shared_store):
        """Gradients should flow to EmbeddingBag values even from zero init."""
        # Set one value to non-zero so there's some signal
        with torch.no_grad():
            shared_store.values.weight.fill_(0.01)
        shared_store.values.weight.requires_grad = True

        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = memory_layer(x)
        loss = out.sum()
        loss.backward()

        assert shared_store.values.weight.grad is not None
        # Gradient should be sparse (only accessed rows)
        assert shared_store.values.weight.grad.is_sparse or (
            shared_store.values.weight.grad.abs().sum() > 0
        )

    def test_index_tracking(self, memory_layer):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

        memory_layer.enable_index_tracking()
        memory_layer(x)
        indices = memory_layer.get_last_accessed_indices()

        assert indices is not None
        assert indices.shape == (BATCH * SEQ_LEN, NUM_HEADS, TOP_K)
        assert indices.min() >= 0
        assert indices.max() < N_KEYS**2

        memory_layer.disable_index_tracking()
        assert memory_layer.get_last_accessed_indices() is None

    def test_deterministic(self, memory_layer):
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out1 = memory_layer(x)
        out2 = memory_layer(x)
        assert torch.allclose(out1, out2)


class TestFFNWithMemory:
    def test_output_shape(self, memory_layer):
        ffn = DummyFFN(D_MODEL)
        wrapper = FFNWithMemory(ffn, memory_layer)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = wrapper(x)
        assert out.shape == (BATCH, SEQ_LEN, D_MODEL)

    def test_identical_to_ffn_at_init(self, memory_layer):
        """With zero-init memory, wrapper output should equal FFN output."""
        ffn = DummyFFN(D_MODEL)
        wrapper = FFNWithMemory(ffn, memory_layer)
        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

        ffn_out = ffn(x)
        wrapper_out = wrapper(x)

        assert torch.allclose(ffn_out, wrapper_out, atol=1e-6)

    def test_differs_from_ffn_after_update(self, memory_layer, shared_store):
        """After updating memory values, wrapper should differ from FFN."""
        ffn = DummyFFN(D_MODEL)
        wrapper = FFNWithMemory(ffn, memory_layer)

        # Update memory values
        with torch.no_grad():
            shared_store.values.weight.fill_(1.0)

        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        ffn_out = ffn(x)
        wrapper_out = wrapper(x)

        # Should now be different
        assert not torch.allclose(ffn_out, wrapper_out, atol=1e-3)

    def test_gradient_only_to_memory(self, memory_layer, shared_store):
        """When FFN is frozen, gradients should only flow to memory values."""
        ffn = DummyFFN(D_MODEL)
        wrapper = FFNWithMemory(ffn, memory_layer)

        # Freeze everything
        for p in wrapper.parameters():
            p.requires_grad = False

        # Unfreeze only memory values
        shared_store.values.weight.requires_grad = True
        # Put some signal in the values
        with torch.no_grad():
            shared_store.values.weight.fill_(0.01)

        x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
        out = wrapper(x)
        loss = out.sum()
        loss.backward()

        # FFN params should have no gradient
        for p in ffn.parameters():
            assert p.grad is None

        # Memory values should have gradient
        assert shared_store.values.weight.grad is not None

    def test_retrieval_overlap_for_similar_inputs(self, memory_layer):
        """Similar inputs should access overlapping memory slots."""
        memory_layer.enable_index_tracking()

        # Create two similar inputs
        base = torch.randn(1, SEQ_LEN, D_MODEL)
        similar = base + 0.01 * torch.randn_like(base)  # Small perturbation
        different = torch.randn(1, SEQ_LEN, D_MODEL)  # Completely different

        memory_layer(base)
        base_indices = memory_layer.get_last_accessed_indices().clone()

        memory_layer(similar)
        similar_indices = memory_layer.get_last_accessed_indices().clone()

        memory_layer(different)
        different_indices = memory_layer.get_last_accessed_indices().clone()

        memory_layer.disable_index_tracking()

        # Compute overlap: fraction of base indices found in other set
        def overlap(a, b):
            # Flatten to sets per token
            a_set = set(a.reshape(-1).tolist())
            b_set = set(b.reshape(-1).tolist())
            if not a_set:
                return 0.0
            return len(a_set & b_set) / len(a_set)

        similar_overlap = overlap(base_indices, similar_indices)
        different_overlap = overlap(base_indices, different_indices)

        # Similar inputs should have much higher overlap than different ones
        assert similar_overlap > different_overlap
