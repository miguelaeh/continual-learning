from __future__ import annotations

import argparse
from pathlib import Path

import torch

from self_building_brain.config import ModelConfig, SyntheticTaskConfig
from self_building_brain.data.text_facts import TextFactCodec
from self_building_brain.models.system import SelfBuildingBrain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a trained brain and run a read/query session.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument(
        "--read",
        action="append",
        nargs=3,
        metavar=("ENTITY", "ATTRIBUTE", "VALUE"),
        help="Read a fact into the brain, e.g. --read falcon color amber",
    )
    parser.add_argument(
        "--query",
        nargs=2,
        metavar=("ENTITY", "ATTRIBUTE"),
        required=True,
        help="Query a fact from the brain, e.g. --query falcon color",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(Path(args.checkpoint_path), map_location=device)

    task_config = SyntheticTaskConfig(**checkpoint["task_config"])
    model_config = ModelConfig(**checkpoint["model_config"])
    codec = TextFactCodec(task_config)

    model = SelfBuildingBrain(
        vocab_size=task_config.vocab_size,
        hidden_dim=model_config.hidden_dim,
        slot_dim=model_config.slot_dim,
        num_slots=model_config.num_slots,
        num_values=task_config.num_values,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    read_facts = args.read or []
    brain_state = model.initial_state(batch_size=1, device=device)
    with torch.no_grad():
        for entity, attribute, value in read_facts:
            read_tokens = codec.encode_read_triplet(entity, attribute, value, device=device).unsqueeze(0)
            brain_state, aux = model.updater(brain_state, read_tokens)
            slot_id = int(aux["slot_logits"].argmax(dim=-1).item())
            print(f"read {entity}/{attribute}={value} -> slot {slot_id}")

        query_entity, query_attribute = args.query
        query_tokens = codec.encode_query_pair(query_entity, query_attribute, device=device).unsqueeze(0)
        outputs = model.executor(brain_state, query_tokens)
        predicted_value_id = int(outputs["answer_logits"].argmax(dim=-1).item())
        query_slot = int(outputs["query_slot_logits"].argmax(dim=-1).item())
        print(f"query {query_entity}/{query_attribute} -> predicted slot {query_slot}")
        print(f"answer: {codec.value_word(predicted_value_id)}")


if __name__ == "__main__":
    main()
