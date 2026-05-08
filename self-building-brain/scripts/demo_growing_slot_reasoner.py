from __future__ import annotations

import torch

from self_building_brain.models.text_memory_reasoners import GrowingSlotTextReasoner


def main() -> None:
    torch.manual_seed(0)

    model = GrowingSlotTextReasoner(
        input_dim=4,
        memory_dim=4,
        max_slots=6,
        allocation_threshold=0.92,
        update_momentum=0.6,
    )
    model.eval()

    with torch.no_grad():
        write_layer = model.write_projection[0]
        write_layer.weight.zero_()
        write_layer.bias.zero_()
        write_layer.weight.copy_(torch.eye(4))

        model.query_projection.weight.zero_()
        model.query_projection.bias.zero_()
        model.query_projection.weight.copy_(torch.eye(4))

        model.choice_projection.weight.zero_()
        model.choice_projection.bias.zero_()
        model.choice_projection.weight.copy_(torch.eye(4))

    chunk_embeddings = torch.tensor(
        [
            [
                [0.20, 0.00, 0.00, 0.00],
                [0.21, 0.01, 0.00, 0.00],
                [0.00, 0.20, 0.00, 0.00],
                [0.00, 0.22, 0.01, 0.00],
                [0.00, 0.00, 0.25, 0.00],
            ]
        ],
        dtype=torch.float32,
    )
    chunk_mask = torch.ones(1, chunk_embeddings.size(1), dtype=torch.float32)
    query_embeddings = torch.tensor([[0.00, 0.22, 0.00, 0.00]], dtype=torch.float32)
    choice_embeddings = torch.zeros(1, 4, 4, dtype=torch.float32)

    outputs = model(
        chunk_embeddings=chunk_embeddings,
        chunk_mask=chunk_mask,
        query_embeddings=query_embeddings,
        choice_embeddings=choice_embeddings,
    )

    print("selected slots per write:", outputs["selected_slots"][0].tolist())
    print("allocated new slot flags:", outputs["allocation_mask"][0].tolist())
    print("active slot count:", int(outputs["active_counts"][0].item()))
    print("active slots:", outputs["active_mask"][0].tolist())
    print("usage:", [round(value, 2) for value in outputs["usage"][0].tolist()])


if __name__ == "__main__":
    main()
