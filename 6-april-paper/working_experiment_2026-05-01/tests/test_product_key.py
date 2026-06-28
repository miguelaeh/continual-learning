import torch

from smf_retrofit.memory.product_key import ProductKeyLookup


def test_product_key_lookup_shapes():
    lookup = ProductKeyLookup(num_heads=2, n_keys=8, k_dim_per_head=16, top_k=4)
    query = torch.randn(5, 2, 16)
    indices, scores = lookup(query)

    assert indices.shape == (5, 2, 4)
    assert scores.shape == (5, 2, 4)
    assert torch.allclose(scores.sum(dim=-1), torch.ones(5, 2), atol=1e-5)
