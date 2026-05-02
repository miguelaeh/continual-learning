from dataclasses import dataclass


@dataclass
class SyntheticTaskConfig:
    num_entities: int = 12
    num_attributes: int = 6
    num_values: int = 18
    num_read_steps: int = 4
    pad_token_id: int = 0
    read_token_id: int = 1
    ask_token_id: int = 2
    sep_token_id: int = 3

    @property
    def num_keys(self) -> int:
        return self.num_entities * self.num_attributes

    @property
    def num_slots(self) -> int:
        return self.num_keys

    @property
    def entity_offset(self) -> int:
        return 4

    @property
    def attribute_offset(self) -> int:
        return self.entity_offset + self.num_entities

    @property
    def value_offset(self) -> int:
        return self.attribute_offset + self.num_attributes

    @property
    def vocab_size(self) -> int:
        return self.value_offset + self.num_values


@dataclass
class ModelConfig:
    hidden_dim: int = 96
    slot_dim: int = 64
    num_slots: int = 72


@dataclass
class TrainingConfig:
    batch_size: int = 64
    steps: int = 200
    learning_rate: float = 3e-3
    state_loss_weight: float = 0.5
    trace_loss_weight: float = 0.5
    read_value_loss_weight: float = 0.5
    routing_loss_weight: float = 0.2
    query_slot_loss_weight: float = 0.2
    sparsity_loss_weight: float = 0.01
    log_every: int = 20


@dataclass
class QwenTeacherConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_length: int = 64
    slot_projection_dim: int = 64
    layer_index: int = -1
    local_files_only: bool = True
