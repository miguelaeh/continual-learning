from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BrainState:
    slots: torch.Tensor
    usage: torch.Tensor

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        num_slots: int,
        slot_dim: int,
        device: torch.device | str,
    ) -> "BrainState":
        slots = torch.zeros(batch_size, num_slots, slot_dim, device=device)
        usage = torch.zeros(batch_size, num_slots, device=device)
        return cls(slots=slots, usage=usage)
