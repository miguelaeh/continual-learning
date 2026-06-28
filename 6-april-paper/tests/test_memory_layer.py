import torch

from smf_retrofit.memory.layer import SparseMemoryLayer
from smf_retrofit.memory.shared_store import SharedMemoryStore


def test_sparse_memory_layer_forward_shape():
    store = SharedMemoryStore(
        num_heads=2,
        n_keys=8,
        k_dim_per_head=16,
        v_dim=4,
        top_k=4,
    )
    layer = SparseMemoryLayer(
        d_model=12,
        shared_store=store,
        num_heads=2,
        k_dim_per_head=16,
        v_dim=4,
        top_k=4,
    )
    x = torch.randn(3, 7, 12)
    y = layer(x)
    assert y.shape == x.shape


def test_dual_bank_layer_is_initially_equivalent_when_delta_is_zero():
    base_store = SharedMemoryStore(
        num_heads=2,
        n_keys=8,
        k_dim_per_head=16,
        v_dim=4,
        top_k=4,
    )
    base_layer = SparseMemoryLayer(
        d_model=12,
        shared_store=base_store,
        num_heads=2,
        k_dim_per_head=16,
        v_dim=4,
        top_k=4,
    )

    dual_store = SharedMemoryStore(
        num_heads=2,
        n_keys=8,
        k_dim_per_head=16,
        v_dim=4,
        top_k=4,
        use_delta_bank=True,
    )
    dual_store.load_state_dict(base_store.state_dict(), strict=False)
    dual_store.zero_delta_values_()
    dual_layer = SparseMemoryLayer(
        d_model=12,
        shared_store=dual_store,
        num_heads=2,
        k_dim_per_head=16,
        v_dim=4,
        top_k=4,
    )
    dual_layer.load_state_dict(base_layer.state_dict(), strict=False)

    x = torch.randn(2, 5, 12)
    y_base = base_layer(x)
    y_dual = dual_layer(x)
    assert torch.allclose(y_base, y_dual, atol=1e-6)


def test_dual_bank_residual_layer_is_initially_equivalent_when_delta_is_zero():
    base_store = SharedMemoryStore(
        num_heads=2,
        n_keys=8,
        k_dim_per_head=16,
        v_dim=4,
        top_k=4,
    )
    base_layer = SparseMemoryLayer(
        d_model=12,
        shared_store=base_store,
        num_heads=2,
        k_dim_per_head=16,
        v_dim=4,
        top_k=4,
    )

    dual_store = SharedMemoryStore(
        num_heads=2,
        n_keys=8,
        k_dim_per_head=16,
        v_dim=4,
        top_k=4,
        use_delta_bank=True,
    )
    dual_store.load_state_dict(base_store.state_dict(), strict=False)
    dual_store.zero_delta_values_()
    dual_layer = SparseMemoryLayer(
        d_model=12,
        shared_store=dual_store,
        num_heads=2,
        k_dim_per_head=16,
        v_dim=4,
        top_k=4,
        use_delta_residual=True,
        use_delta_value_proj=True,
    )
    dual_layer.load_state_dict(base_layer.state_dict(), strict=False)

    x = torch.randn(2, 5, 12)
    y_base = base_layer(x)
    y_dual = dual_layer(x)
    assert torch.allclose(y_base, y_dual, atol=1e-6)
