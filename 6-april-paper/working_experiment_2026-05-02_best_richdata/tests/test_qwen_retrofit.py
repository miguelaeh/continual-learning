from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

from smf_retrofit.config import MemoryConfig
from smf_retrofit.memory.layer import SparseMemoryLayer
from smf_retrofit.modeling.qwen import freeze_for_sparse_finetuning, inject_memory_layers


def test_qwen_injection_replaces_mlp_and_keeps_forward_working():
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    model = Qwen2ForCausalLM(cfg)
    memory_cfg = MemoryConfig(
        num_heads=2,
        top_k=4,
        n_keys=8,
        k_dim_per_head=16,
        v_dim=4,
        num_memory_layers=2,
    )

    model, _, layer_indices = inject_memory_layers(model, memory_cfg)
    assert len(layer_indices) == 2
    for idx in layer_indices:
        assert isinstance(model.model.layers[idx].mlp, SparseMemoryLayer)

    batch = model(
        input_ids=__import__("torch").randint(0, 128, (2, 8)),
        labels=__import__("torch").randint(0, 128, (2, 8)),
    )
    assert batch.loss is not None


def test_sparse_finetune_freeze_modes():
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    model = Qwen2ForCausalLM(cfg)
    memory_cfg = MemoryConfig(
        num_heads=2,
        top_k=4,
        n_keys=8,
        k_dim_per_head=16,
        v_dim=4,
        num_memory_layers=1,
        memory_layers=[1],
    )

    model, _, layer_indices = inject_memory_layers(model, memory_cfg)
    freeze_for_sparse_finetuning(model, layer_indices, train_memory_mode="values_only")
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert trainable_names == {"model.layers.1.mlp.shared_store.values.weight"}

    freeze_for_sparse_finetuning(model, layer_indices, train_memory_mode="values_and_output")
    value_and_output_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert value_and_output_names == {
        "model.layers.1.mlp.shared_store.values.weight",
        "model.layers.1.mlp.value_proj.weight",
    }

    freeze_for_sparse_finetuning(model, layer_indices, train_memory_mode="values_gate_and_output")
    value_gate_output_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert value_gate_output_names == {
        "model.layers.1.mlp.shared_store.values.weight",
        "model.layers.1.mlp.silu_proj.weight",
        "model.layers.1.mlp.value_proj.weight",
    }

    freeze_for_sparse_finetuning(model, layer_indices, train_memory_mode="full_memory")
    full_memory_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert "model.layers.1.mlp.query_proj.weight" in full_memory_names
    assert "model.layers.1.mlp.value_proj.weight" in full_memory_names
    assert "model.layers.1.mlp.shared_store.values.weight" in full_memory_names


def test_dual_bank_sparse_finetune_freeze_modes():
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    model = Qwen2ForCausalLM(cfg)
    memory_cfg = MemoryConfig(
        num_heads=2,
        top_k=4,
        n_keys=8,
        k_dim_per_head=16,
        v_dim=4,
        num_memory_layers=1,
        memory_layers=[1],
        use_delta_bank=True,
        use_delta_residual=True,
        use_delta_value_proj=True,
    )

    model, _, layer_indices = inject_memory_layers(model, memory_cfg)
    freeze_for_sparse_finetuning(model, layer_indices, train_memory_mode="delta_values_only")
    delta_only_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert delta_only_names == {"model.layers.1.mlp.shared_store.delta_values.weight"}

    freeze_for_sparse_finetuning(
        model,
        layer_indices,
        train_memory_mode="delta_values_and_scale",
    )
    delta_scale_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert delta_scale_names == {
        "model.layers.1.mlp.delta_scale",
        "model.layers.1.mlp.shared_store.delta_values.weight",
    }

    freeze_for_sparse_finetuning(
        model,
        layer_indices,
        train_memory_mode="delta_values_scale_and_output",
    )
    delta_scale_output_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert delta_scale_output_names == {
        "model.layers.1.mlp.delta_scale",
        "model.layers.1.mlp.delta_value_proj.weight",
        "model.layers.1.mlp.shared_store.delta_values.weight",
    }

    freeze_for_sparse_finetuning(
        model,
        layer_indices,
        train_memory_mode="delta_values_and_gate",
    )
    delta_gate_names = {name for name, param in model.named_parameters() if param.requires_grad}
    assert delta_gate_names == {
        "model.layers.1.mlp.shared_store.delta_values.weight",
        "model.layers.1.mlp.silu_proj.weight",
    }
