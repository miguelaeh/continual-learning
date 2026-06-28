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
