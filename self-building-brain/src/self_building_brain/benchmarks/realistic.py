from __future__ import annotations

import json
import random
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BenchmarkExample:
    benchmark: str
    example_id: str
    context: str
    question: str
    answer: str
    choices: list[tuple[str, str]] | None = None
    metadata: dict[str, object] | None = None


@dataclass
class StoredChunk:
    text: str
    embedding: torch.Tensor
    chunk_index: int


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(char for char in value if char not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def token_f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(answer).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    pred_counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    overlap = sum(min(pred_counts.get(token, 0), gold_counts.get(token, 0)) for token in pred_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, answer: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(answer))


def split_text_into_chunks(text: str, max_chars: int = 900) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        piece = paragraph.strip()
        if not piece:
            continue
        if len(piece) > max_chars:
            sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", piece) if sentence.strip()]
            for sentence in sentences:
                if current and current_len + len(sentence) + 1 > max_chars:
                    chunks.append(" ".join(current))
                    current = [sentence]
                    current_len = len(sentence)
                else:
                    current.append(sentence)
                    current_len += len(sentence) + 1
            continue

        if current and current_len + len(piece) + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current = [piece]
            current_len = len(piece)
        else:
            current.append(piece)
            current_len += len(piece) + 2

    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text[:max_chars]]


class QwenTextBackend:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: torch.device | str = "cpu",
        local_files_only: bool = True,
        max_length: int = 256,
    ):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            dtype=torch.float32,
        ).to(self.device)
        self.model.eval()
        self.max_length = max_length

    @property
    def vocab_size(self) -> int:
        return int(self.model.get_input_embeddings().weight.size(0))

    @property
    def hidden_size(self) -> int:
        return int(self.model.get_input_embeddings().weight.size(1))

    def tokenize(self, text: str, max_length: int | None = None) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=bool(max_length is not None),
            max_length=max_length,
        )
        return list(encoded["input_ids"])

    @torch.no_grad()
    def embed_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(self.device)
        embeddings = self.model.get_input_embeddings()(token_ids)
        return F.normalize(embeddings, dim=-1).cpu()

    @torch.no_grad()
    def token_embedding_matrix(self, token_ids: list[int] | torch.Tensor | None = None) -> torch.Tensor:
        weights = self.model.get_input_embeddings().weight
        if token_ids is None:
            selected = weights
        else:
            indices = torch.as_tensor(token_ids, dtype=torch.long, device=weights.device)
            selected = weights.index_select(0, indices)
        return F.normalize(selected, dim=-1).cpu()

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        outputs = self.model(**encoded, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        token_lengths = encoded["attention_mask"].sum(dim=-1) - 1
        batch_indices = torch.arange(hidden.size(0), device=self.device)
        pooled = hidden[batch_indices, token_lengths]
        return F.normalize(pooled, dim=-1).cpu()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 24) -> str:
        messages = [
            {
                "role": "system",
                "content": "You answer only from the provided memory excerpts. Be concise.",
            },
            {"role": "user", "content": prompt},
        ]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = outputs[0][encoded["input_ids"].size(1) :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class SlotTextMemory:
    def __init__(self, num_slots: int = 24, slot_capacity: int = 3, momentum: float = 0.6):
        self.num_slots = num_slots
        self.slot_capacity = slot_capacity
        self.momentum = momentum
        self.centroids: list[torch.Tensor | None] = [None for _ in range(num_slots)]
        self.items: list[list[StoredChunk]] = [[] for _ in range(num_slots)]

    def ingest(self, chunk_texts: list[str], chunk_embeddings: torch.Tensor) -> None:
        for chunk_index, (text, embedding) in enumerate(zip(chunk_texts, chunk_embeddings)):
            slot_index = self._select_slot(embedding)
            centroid = self.centroids[slot_index]
            self.centroids[slot_index] = embedding if centroid is None else F.normalize(
                self.momentum * centroid + (1 - self.momentum) * embedding,
                dim=0,
            )
            self.items[slot_index].append(StoredChunk(text=text, embedding=embedding, chunk_index=chunk_index))
            self._trim_slot(slot_index)

    def _select_slot(self, embedding: torch.Tensor) -> int:
        for index, centroid in enumerate(self.centroids):
            if centroid is None:
                return index
        similarities = [float(torch.dot(embedding, centroid)) for centroid in self.centroids if centroid is not None]
        return max(range(len(similarities)), key=similarities.__getitem__)

    def _trim_slot(self, slot_index: int) -> None:
        centroid = self.centroids[slot_index]
        scored = sorted(
            self.items[slot_index],
            key=lambda item: float(torch.dot(item.embedding, centroid)),
            reverse=True,
        )
        self.items[slot_index] = scored[: self.slot_capacity]

    def retrieve(self, query_embedding: torch.Tensor, top_k_slots: int = 4, top_k_chunks: int = 6) -> list[str]:
        scored_slots = []
        for index, centroid in enumerate(self.centroids):
            if centroid is None:
                continue
            score = float(torch.dot(query_embedding, centroid))
            scored_slots.append((score, index))
        scored_slots.sort(reverse=True)
        candidate_chunks: list[tuple[float, StoredChunk]] = []
        for _, slot_index in scored_slots[:top_k_slots]:
            for item in self.items[slot_index]:
                candidate_chunks.append((float(torch.dot(query_embedding, item.embedding)), item))
        candidate_chunks.sort(key=lambda pair: pair[0], reverse=True)
        return [item.text for _, item in candidate_chunks[:top_k_chunks]]


class GraphTextMemory(SlotTextMemory):
    def __init__(
        self,
        num_slots: int = 24,
        slot_capacity: int = 3,
        momentum: float = 0.6,
        propagation_steps: int = 2,
        propagation_alpha: float = 0.6,
    ):
        super().__init__(num_slots=num_slots, slot_capacity=slot_capacity, momentum=momentum)
        self.edge_weights = torch.zeros(num_slots, num_slots)
        self.previous_slot: int | None = None
        self.propagation_steps = propagation_steps
        self.propagation_alpha = propagation_alpha

    def ingest(self, chunk_texts: list[str], chunk_embeddings: torch.Tensor) -> None:
        for chunk_index, (text, embedding) in enumerate(zip(chunk_texts, chunk_embeddings)):
            slot_index = self._select_slot(embedding)
            if self.previous_slot is not None:
                self.edge_weights[self.previous_slot, slot_index] += 1.0
            self.previous_slot = slot_index
            centroid = self.centroids[slot_index]
            self.centroids[slot_index] = embedding if centroid is None else F.normalize(
                self.momentum * centroid + (1 - self.momentum) * embedding,
                dim=0,
            )
            self.items[slot_index].append(StoredChunk(text=text, embedding=embedding, chunk_index=chunk_index))
            self._trim_slot(slot_index)

    def retrieve(self, query_embedding: torch.Tensor, top_k_slots: int = 4, top_k_chunks: int = 6) -> list[str]:
        base_scores = torch.zeros(self.num_slots)
        for index, centroid in enumerate(self.centroids):
            if centroid is not None:
                base_scores[index] = float(torch.dot(query_embedding, centroid))

        if torch.any(self.edge_weights > 0):
            normalized_edges = self.edge_weights / self.edge_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            propagated = base_scores.clone()
            for _ in range(self.propagation_steps):
                propagated = self.propagation_alpha * base_scores + (1 - self.propagation_alpha) * normalized_edges.T @ propagated
        else:
            propagated = base_scores

        scored_slots = [(float(propagated[index]), index) for index in range(self.num_slots) if self.centroids[index] is not None]
        scored_slots.sort(reverse=True)

        candidate_chunks: list[tuple[float, StoredChunk]] = []
        for _, slot_index in scored_slots[:top_k_slots]:
            for item in self.items[slot_index]:
                candidate_chunks.append((float(torch.dot(query_embedding, item.embedding)), item))
        candidate_chunks.sort(key=lambda pair: pair[0], reverse=True)
        return [item.text for _, item in candidate_chunks[:top_k_chunks]]


def build_open_ended_prompt(retrieved_chunks: list[str], question: str) -> str:
    memory = "\n\n".join(f"[Memory {index + 1}]\n{chunk}" for index, chunk in enumerate(retrieved_chunks))
    return (
        f"Memory excerpts:\n{memory}\n\n"
        f"Question: {question}\n"
        "Answer with a short span or sentence only."
    )


def build_multiple_choice_prompt(retrieved_chunks: list[str], question: str, choices: list[tuple[str, str]]) -> str:
    memory = "\n\n".join(f"[Memory {index + 1}]\n{chunk}" for index, chunk in enumerate(retrieved_chunks))
    choices_text = "\n".join(f"{label}. {choice}" for label, choice in choices)
    return (
        f"Memory excerpts:\n{memory}\n\n"
        f"Question: {question}\n"
        f"Choices:\n{choices_text}\n\n"
        "Answer with only one letter: A, B, C, or D."
    )


def parse_choice_letter(generation: str) -> str:
    match = re.search(r"\b([ABCD])\b", generation.upper())
    return match.group(1) if match else ""


def score_prediction(example: BenchmarkExample, prediction: str) -> dict[str, float]:
    if example.choices is not None:
        return {"accuracy": float(prediction.strip().upper() == example.answer.strip().upper())}
    answers = [example.answer]
    if example.metadata and "answer_aliases" in example.metadata:
        answers.extend([alias for alias in example.metadata["answer_aliases"] if alias])
    return {
        "exact_match": max(exact_match(prediction, answer) for answer in answers),
        "f1": max(token_f1(prediction, answer) for answer in answers),
    }


def evaluate_examples(
    backend: QwenTextBackend,
    memory_type: str,
    examples: Iterable[BenchmarkExample],
    num_slots: int = 24,
    slot_capacity: int = 3,
    top_k_slots: int = 4,
    top_k_chunks: int = 6,
    chunk_chars: int = 900,
    max_chunks_per_example: int = 24,
) -> dict[str, object]:
    all_rows: list[dict[str, object]] = []
    aggregate: dict[str, float] = {}

    for example in examples:
        chunk_texts = split_text_into_chunks(example.context, max_chars=chunk_chars)[:max_chunks_per_example]
        chunk_embeddings = backend.encode_texts(chunk_texts)
        query_embedding = backend.encode_texts([example.question])[0]
        memory = SlotTextMemory(num_slots=num_slots, slot_capacity=slot_capacity) if memory_type == "slot" else GraphTextMemory(
            num_slots=num_slots,
            slot_capacity=slot_capacity,
        )
        memory.ingest(chunk_texts=chunk_texts, chunk_embeddings=chunk_embeddings)
        retrieved = memory.retrieve(query_embedding=query_embedding, top_k_slots=top_k_slots, top_k_chunks=top_k_chunks)

        if example.choices is not None:
            prompt = build_multiple_choice_prompt(retrieved, example.question, example.choices)
            prediction = parse_choice_letter(backend.generate(prompt, max_new_tokens=6))
        else:
            prompt = build_open_ended_prompt(retrieved, example.question)
            prediction = backend.generate(prompt, max_new_tokens=20)

        metrics = score_prediction(example, prediction)
        row = {
            "benchmark": example.benchmark,
            "example_id": example.example_id,
            "prediction": prediction,
            "answer": example.answer,
            "retrieved_chunks": retrieved,
            **metrics,
        }
        all_rows.append(row)
        for key, value in metrics.items():
            aggregate[key] = aggregate.get(key, 0.0) + value

    count = max(1, len(all_rows))
    summary = {key: value / count for key, value in aggregate.items()}
    return {"summary": summary, "rows": all_rows}


def load_locomo_examples(limit: int = 3) -> list[BenchmarkExample]:
    data = json.loads(Path(".benchmarks/locomo/data/locomo10.json").read_text())
    examples: list[BenchmarkExample] = []
    for conversation in data:
        context_parts = []
        for key, value in conversation["conversation"].items():
            if key.startswith("session_") and isinstance(value, list):
                context_parts.append(f"{key}:")
                for turn in value:
                    context_parts.append(f"{turn['speaker']}: {turn['text']}")
        context = "\n".join(context_parts)
        for qa_index, qa in enumerate(conversation["qa"]):
            examples.append(
                BenchmarkExample(
                    benchmark="locomo",
                    example_id=f"{conversation['sample_id']}_{qa_index}",
                    context=context,
                    question=qa["question"],
                    answer=str(qa["answer"]),
                    metadata={"category": qa.get("category"), "evidence": qa.get("evidence", [])},
                )
            )
            if len(examples) >= limit:
                return examples
    return examples


def load_longbench_v2_examples(limit: int = 3, seed: int = 13) -> list[BenchmarkExample]:
    random.seed(seed)
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    chosen = indices[:limit]
    examples: list[BenchmarkExample] = []
    for index in chosen:
        item = dataset[index]
        choices = [
            ("A", item["choice_A"]),
            ("B", item["choice_B"]),
            ("C", item["choice_C"]),
            ("D", item["choice_D"]),
        ]
        examples.append(
            BenchmarkExample(
                benchmark="longbench_v2",
                example_id=item["_id"],
                context=item["context"],
                question=item["question"],
                answer=str(item["answer"]).strip(),
                choices=choices,
                metadata={
                    "domain": item["domain"],
                    "sub_domain": item["sub_domain"],
                    "difficulty": item["difficulty"],
                    "length": item["length"],
                },
            )
        )
    return examples


def load_musique_examples(limit: int = 3) -> list[BenchmarkExample]:
    dev_path = Path(".benchmarks/musique/data/musique_ans_v1.0_dev.jsonl")
    if not dev_path.exists():
        raise FileNotFoundError("MuSiQue dev file not found. Download the benchmark data first.")

    examples: list[BenchmarkExample] = []
    with dev_path.open() as handle:
        for line in handle:
            item = json.loads(line)
            paragraphs = []
            for paragraph in item["paragraphs"]:
                title = paragraph.get("title", "")
                text = paragraph.get("paragraph_text", "")
                paragraphs.append(f"{title}\n{text}".strip())
            examples.append(
                BenchmarkExample(
                    benchmark="musique",
                    example_id=item["id"],
                    context="\n\n".join(paragraphs),
                    question=item["question"],
                    answer=item["answer"],
                    metadata={"answer_aliases": item.get("answer_aliases", [])},
                )
            )
            if len(examples) >= limit:
                break
    return examples
