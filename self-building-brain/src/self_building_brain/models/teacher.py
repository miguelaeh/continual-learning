from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SyntheticTeacherTargets:
    teacher_states: torch.Tensor
    step_slot_ids: torch.Tensor
    query_slot_ids: torch.Tensor
    trace_embeddings: torch.Tensor


class SyntheticTeacherTraceProvider(nn.Module):
    """
    Frozen teacher-side structure generator.

    This stands in for "activation-derived structure" until a real LLM
    trace extractor is attached.
    """

    def __init__(self, vocab_size: int, num_slots: int, num_values: int, slot_dim: int, hidden_dim: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.key_embedding = nn.Embedding(num_slots, slot_dim)
        self.value_embedding = nn.Embedding(num_values, slot_dim)
        self.trace_projection = nn.Linear(hidden_dim, slot_dim)
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        for parameter in self.parameters():
            parameter.requires_grad = False

    def extract_targets(
        self,
        read_tokens: torch.Tensor,
        read_key_ids: torch.Tensor,
        read_value_ids: torch.Tensor,
        query_key_ids: torch.Tensor,
    ) -> SyntheticTeacherTargets:
        batch_size, num_steps, _ = read_tokens.shape
        device = read_tokens.device

        slots = torch.zeros(batch_size, self.num_slots, self.slot_dim, device=device)
        teacher_states = []

        value_vectors = self.value_embedding(read_value_ids)
        write_vectors = F.normalize(value_vectors, dim=-1)

        for step in range(num_steps):
            slot_index = read_key_ids[:, step]
            one_hot = F.one_hot(slot_index, num_classes=self.num_slots).float().unsqueeze(-1)
            slots = slots * (1.0 - one_hot) + one_hot * write_vectors[:, step].unsqueeze(1)
            teacher_states.append(slots.clone())

        trace_embeddings = self.trace_projection(self.token_embedding(read_tokens)).mean(dim=2)
        teacher_states = torch.stack(teacher_states, dim=1)

        return SyntheticTeacherTargets(
            teacher_states=teacher_states,
            step_slot_ids=read_key_ids,
            query_slot_ids=query_key_ids,
            trace_embeddings=trace_embeddings,
        )


class QwenTeacherTraceProvider(nn.Module):
    """
    Frozen Hugging Face teacher that compresses real hidden-state traces into
    teacher slot states.
    """

    def __init__(
        self,
        model_name: str,
        num_slots: int,
        slot_dim: int,
        max_length: int = 64,
        layer_index: int = -1,
        local_files_only: bool = True,
    ):
        super().__init__()
        self.trace_cache: dict[str, torch.Tensor] = {}
        resolved_source = self._resolve_model_source(model_name=model_name, local_files_only=local_files_only)
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_source, local_files_only=local_files_only)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved_source,
            local_files_only=local_files_only,
            dtype=torch.float32,
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        hidden_size = self.model.config.hidden_size
        self.trace_projection = nn.Linear(hidden_size, slot_dim)
        nn.init.normal_(self.trace_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.trace_projection.bias)
        for parameter in self.trace_projection.parameters():
            parameter.requires_grad = False

        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.max_length = max_length
        self.layer_index = layer_index

    def _resolve_model_source(self, model_name: str, local_files_only: bool) -> str:
        candidate_path = Path(model_name).expanduser()
        if candidate_path.exists():
            return str(candidate_path)

        try:
            snapshot_path = snapshot_download(repo_id=model_name, local_files_only=local_files_only)
            if self._has_model_weights(Path(snapshot_path)):
                return snapshot_path
        except Exception:
            pass

        cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
        ref_path = cache_root / "refs" / "main"
        if ref_path.exists():
            snapshot_id = ref_path.read_text().strip()
            snapshot_path = cache_root / "snapshots" / snapshot_id
            if self._has_model_weights(snapshot_path):
                return str(snapshot_path)

        raise FileNotFoundError(
            f"Could not resolve a complete local model snapshot for {model_name}. "
            "Pass a local snapshot path or cache the full model weights first."
        )

    def _has_model_weights(self, snapshot_path: Path) -> bool:
        weight_names = [
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ]
        return any((snapshot_path / weight_name).exists() for weight_name in weight_names)

    def _encode_texts(self, texts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.to(device) for key, value in encoded.items()}

    def _pool_hidden_states(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_lengths = attention_mask.sum(dim=-1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, token_lengths]

    @torch.no_grad()
    def _encode_trace_texts(self, texts: list[str], device: torch.device) -> torch.Tensor:
        missing_texts = [text for text in texts if text not in self.trace_cache]
        if missing_texts:
            encoded_reads = self._encode_texts(missing_texts, device=device)
            read_outputs = self.model(**encoded_reads, output_hidden_states=True)
            read_hidden = read_outputs.hidden_states[self.layer_index]
            pooled_reads = self._pool_hidden_states(read_hidden, encoded_reads["attention_mask"])
            projected_reads = torch.tanh(self.trace_projection(pooled_reads))
            for text, projected in zip(missing_texts, projected_reads):
                self.trace_cache[text] = projected.detach().cpu()

        stacked = torch.stack([self.trace_cache[text].to(device) for text in texts], dim=0)
        return stacked

    @torch.no_grad()
    def extract_targets(
        self,
        read_texts: list[list[str]],
        read_key_ids: torch.Tensor,
        query_key_ids: torch.Tensor,
    ) -> SyntheticTeacherTargets:
        batch_size = len(read_texts)
        num_steps = len(read_texts[0])
        flat_reads = [text for episode in read_texts for text in episode]
        device = next(self.model.parameters()).device

        projected_reads = self._encode_trace_texts(flat_reads, device=device).view(batch_size, num_steps, self.slot_dim)
        trace_embeddings = projected_reads.clone()

        slots = torch.zeros(batch_size, self.num_slots, self.slot_dim, device=device)
        teacher_states = []
        for step in range(num_steps):
            slot_index = read_key_ids[:, step]
            one_hot = F.one_hot(slot_index, num_classes=self.num_slots).float().unsqueeze(-1)
            slots = slots * (1.0 - one_hot) + one_hot * projected_reads[:, step].unsqueeze(1)
            teacher_states.append(slots.clone())
        teacher_states = torch.stack(teacher_states, dim=1)

        return SyntheticTeacherTargets(
            teacher_states=teacher_states,
            step_slot_ids=read_key_ids,
            query_slot_ids=query_key_ids,
            trace_embeddings=trace_embeddings,
        )
