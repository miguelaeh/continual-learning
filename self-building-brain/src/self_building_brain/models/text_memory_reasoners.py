from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotTextReasoner(nn.Module):
    def __init__(self, input_dim: int, memory_dim: int, num_slots: int):
        super().__init__()
        self.num_slots = num_slots
        self.memory_dim = memory_dim
        self.router = nn.Linear(input_dim, num_slots)
        self.write_projection = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.Tanh(),
        )
        self.query_projection = nn.Linear(input_dim, memory_dim)
        self.choice_projection = nn.Linear(input_dim, memory_dim)
        self.output = nn.Sequential(
            nn.Linear(memory_dim * 3, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
        )

    def forward(
        self,
        chunk_embeddings: torch.Tensor,
        chunk_mask: torch.Tensor,
        query_embeddings: torch.Tensor,
        choice_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_chunks, _ = chunk_embeddings.shape
        memory = torch.zeros(batch_size, self.num_slots, self.memory_dim, device=chunk_embeddings.device)
        routing_logits = []

        for chunk_index in range(num_chunks):
            valid = chunk_mask[:, chunk_index].unsqueeze(-1)
            chunk_embedding = chunk_embeddings[:, chunk_index]
            slot_logits = self.router(chunk_embedding)
            slot_weights = F.softmax(slot_logits, dim=-1)
            write_vector = self.write_projection(chunk_embedding).unsqueeze(1)
            update = slot_weights.unsqueeze(-1) * write_vector * valid.unsqueeze(-1)
            retain = 1.0 - slot_weights.unsqueeze(-1) * valid.unsqueeze(-1)
            memory = memory * retain + update
            routing_logits.append(slot_logits)

        query_vector = self.query_projection(query_embeddings)
        attention_logits = torch.matmul(memory, query_vector.unsqueeze(-1)).squeeze(-1)
        memory_attention = F.softmax(attention_logits, dim=-1)
        memory_context = torch.sum(memory_attention.unsqueeze(-1) * memory, dim=1)

        choice_vectors = self.choice_projection(choice_embeddings)
        expanded_context = memory_context.unsqueeze(1).expand(-1, choice_vectors.size(1), -1)
        expanded_query = query_vector.unsqueeze(1).expand_as(expanded_context)
        choice_logits = self.output(torch.cat([expanded_context, expanded_query, choice_vectors], dim=-1)).squeeze(-1)

        return {
            "choice_logits": choice_logits,
            "memory_context": memory_context,
            "routing_logits": torch.stack(routing_logits, dim=1),
        }


class GrowingSlotTextReasoner(nn.Module):
    """
    Dynamic slot memory with novelty-based allocation.

    The model starts with zero active slots. Each new write either:
    - reuses the most similar active slot and blends into it
    - or allocates the next free slot if the write is novel enough

    Memory growth is therefore dynamic up to `max_slots`.
    """

    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        max_slots: int,
        allocation_threshold: float = 0.55,
        update_momentum: float = 0.7,
    ):
        super().__init__()
        self.max_slots = max_slots
        self.memory_dim = memory_dim
        self.allocation_threshold = allocation_threshold
        self.update_momentum = update_momentum

        self.key_projection = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.Tanh(),
        )
        self.write_projection = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.Tanh(),
        )
        self.novelty_head = nn.Sequential(
            nn.Linear(memory_dim * 2 + 2, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
        )
        self.query_projection = nn.Linear(input_dim, memory_dim)
        self.choice_projection = nn.Linear(input_dim, memory_dim)
        self.output = nn.Sequential(
            nn.Linear(memory_dim * 3, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
        )

    def _update_memory(
        self,
        memory: torch.Tensor,
        slot_keys: torch.Tensor,
        active_mask: torch.Tensor,
        usage: torch.Tensor,
        key_vectors: torch.Tensor,
        write_vectors: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = memory.size(0)
        next_memory = memory.clone()
        next_slot_keys = slot_keys.clone()
        next_active_mask = active_mask.clone()
        next_usage = usage.clone()
        assignment_logits = torch.full(
            (batch_size, self.max_slots),
            fill_value=-1e4,
            device=memory.device,
        )
        allocated = torch.zeros(batch_size, dtype=torch.bool, device=memory.device)
        selected_slots = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=memory.device)
        novelty_logits = torch.zeros(batch_size, device=memory.device)
        novelty_targets = torch.zeros(batch_size, device=memory.device)
        max_similarities = torch.zeros(batch_size, device=memory.device)

        normalized_keys = F.normalize(slot_keys, dim=-1, eps=1e-8)
        normalized_inputs = F.normalize(key_vectors, dim=-1, eps=1e-8)
        similarity = torch.matmul(normalized_keys, normalized_inputs.unsqueeze(-1)).squeeze(-1)
        similarity = similarity.masked_fill(~active_mask, -1e4)

        for batch_index in range(batch_size):
            if not bool(valid_mask[batch_index].item()):
                continue

            has_active = bool(active_mask[batch_index].any().item())
            best_slot = torch.tensor(0, device=memory.device, dtype=torch.long)
            best_score = torch.tensor(-1.0, device=memory.device)
            if has_active:
                best_score, best_slot = similarity[batch_index].max(dim=0)
                assignment_logits[batch_index] = similarity[batch_index]
                max_similarities[batch_index] = best_score
            else:
                max_similarities[batch_index] = -1.0

            best_key = slot_keys[batch_index, best_slot] if has_active else torch.zeros_like(key_vectors[batch_index])
            novelty_features = torch.cat(
                [
                    key_vectors[batch_index],
                    best_key,
                    max_similarities[batch_index].view(1),
                    active_mask[batch_index].float().mean().view(1),
                ],
                dim=0,
            )
            novelty_logit = self.novelty_head(novelty_features).squeeze(-1)
            novelty_prob = torch.sigmoid(novelty_logit)
            novelty_logits[batch_index] = novelty_logit

            target_allocate = (not has_active) or float(max_similarities[batch_index].item()) < self.allocation_threshold
            novelty_targets[batch_index] = float(target_allocate)

            if has_active and not bool((novelty_prob >= 0.5).item()):
                if float(best_score.item()) >= self.allocation_threshold:
                    current = memory[batch_index, best_slot]
                    current_key = slot_keys[batch_index, best_slot]
                    incoming = write_vectors[batch_index]
                    incoming_key = key_vectors[batch_index]
                    next_memory[batch_index, best_slot] = (
                        self.update_momentum * current + (1.0 - self.update_momentum) * incoming
                    )
                    next_slot_keys[batch_index, best_slot] = (
                        self.update_momentum * current_key + (1.0 - self.update_momentum) * incoming_key
                    )
                    next_usage[batch_index, best_slot] += 1.0
                    selected_slots[batch_index] = best_slot
                    continue

            free_slots = (~active_mask[batch_index]).nonzero(as_tuple=False).squeeze(-1)
            if free_slots.numel() > 0:
                slot_index = int(free_slots[0].item())
            elif has_active:
                _, best_slot = similarity[batch_index].max(dim=0)
                slot_index = int(best_slot.item())
            else:
                slot_index = 0

            next_memory[batch_index, slot_index] = write_vectors[batch_index]
            next_slot_keys[batch_index, slot_index] = key_vectors[batch_index]
            next_active_mask[batch_index, slot_index] = True
            next_usage[batch_index, slot_index] += 1.0
            allocated[batch_index] = True
            assignment_logits[batch_index, slot_index] = 1.0
            selected_slots[batch_index] = slot_index

        return {
            "memory": next_memory,
            "slot_keys": next_slot_keys,
            "active_mask": next_active_mask,
            "usage": next_usage,
            "assignment_logits": assignment_logits,
            "allocated": allocated,
            "selected_slots": selected_slots,
            "novelty_logits": novelty_logits,
            "novelty_targets": novelty_targets,
            "max_similarities": max_similarities,
        }

    def _slot_separation_loss(self, memory: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        normalized_memory = F.normalize(memory, dim=-1, eps=1e-8)
        similarity = torch.matmul(normalized_memory, normalized_memory.transpose(1, 2))
        pair_mask = active_mask.unsqueeze(1) & active_mask.unsqueeze(2)
        diagonal = torch.eye(self.max_slots, dtype=torch.bool, device=memory.device).unsqueeze(0)
        pair_mask = pair_mask & ~diagonal
        if not bool(pair_mask.any().item()):
            return torch.zeros((), device=memory.device)
        return similarity.masked_select(pair_mask).pow(2).mean()

    def forward(
        self,
        chunk_embeddings: torch.Tensor,
        chunk_mask: torch.Tensor,
        query_embeddings: torch.Tensor,
        choice_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_chunks, _ = chunk_embeddings.shape
        memory = torch.zeros(batch_size, self.max_slots, self.memory_dim, device=chunk_embeddings.device)
        slot_keys = torch.zeros(batch_size, self.max_slots, self.memory_dim, device=chunk_embeddings.device)
        active_mask = torch.zeros(batch_size, self.max_slots, dtype=torch.bool, device=chunk_embeddings.device)
        usage = torch.zeros(batch_size, self.max_slots, device=chunk_embeddings.device)

        assignment_logits_history = []
        allocation_history = []
        selected_slot_history = []
        novelty_logits_history = []
        novelty_target_history = []
        max_similarity_history = []
        key_vector_history = []

        for chunk_index in range(num_chunks):
            valid = chunk_mask[:, chunk_index] > 0
            key_vectors = self.key_projection(chunk_embeddings[:, chunk_index])
            write_vectors = self.write_projection(chunk_embeddings[:, chunk_index])
            update_outputs = self._update_memory(
                memory=memory,
                slot_keys=slot_keys,
                active_mask=active_mask,
                usage=usage,
                key_vectors=key_vectors,
                write_vectors=write_vectors,
                valid_mask=valid,
            )
            memory = update_outputs["memory"]
            slot_keys = update_outputs["slot_keys"]
            active_mask = update_outputs["active_mask"]
            usage = update_outputs["usage"]
            assignment_logits_history.append(update_outputs["assignment_logits"])
            allocation_history.append(update_outputs["allocated"].float())
            selected_slot_history.append(update_outputs["selected_slots"])
            novelty_logits_history.append(update_outputs["novelty_logits"])
            novelty_target_history.append(update_outputs["novelty_targets"])
            max_similarity_history.append(update_outputs["max_similarities"])
            key_vector_history.append(key_vectors)

        query_vector = self.query_projection(query_embeddings)
        attention_logits = torch.matmul(memory, query_vector.unsqueeze(-1)).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~active_mask, -1e4)
        has_any_active = active_mask.any(dim=-1, keepdim=True)
        safe_logits = torch.where(has_any_active, attention_logits, torch.zeros_like(attention_logits))
        memory_attention = F.softmax(safe_logits, dim=-1)
        memory_attention = memory_attention * active_mask.float()
        attention_norm = memory_attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        memory_attention = memory_attention / attention_norm
        memory_context = torch.sum(memory_attention.unsqueeze(-1) * memory, dim=1)

        choice_vectors = self.choice_projection(choice_embeddings)
        expanded_context = memory_context.unsqueeze(1).expand(-1, choice_vectors.size(1), -1)
        expanded_query = query_vector.unsqueeze(1).expand_as(expanded_context)
        choice_logits = self.output(torch.cat([expanded_context, expanded_query, choice_vectors], dim=-1)).squeeze(-1)
        slot_separation_loss = self._slot_separation_loss(memory=memory, active_mask=active_mask)

        return {
            "choice_logits": choice_logits,
            "memory_context": memory_context,
            "routing_logits": torch.stack(assignment_logits_history, dim=1),
            "allocation_mask": torch.stack(allocation_history, dim=1),
            "selected_slots": torch.stack(selected_slot_history, dim=1),
            "novelty_logits": torch.stack(novelty_logits_history, dim=1),
            "novelty_targets": torch.stack(novelty_target_history, dim=1),
            "max_similarities": torch.stack(max_similarity_history, dim=1),
            "key_vectors": torch.stack(key_vector_history, dim=1),
            "active_mask": active_mask,
            "active_counts": active_mask.sum(dim=-1),
            "usage": usage,
            "slot_separation_loss": slot_separation_loss,
        }


class StateConditionedGrowingSlotTextReasoner(nn.Module):
    """
    Dynamic slot memory whose update decisions are conditioned on the current brain.

    Each write step:
    - reads the current active memory
    - scores merge targets using both slot state and the incoming chunk
    - predicts whether to allocate a new slot or update an existing one
    - applies a learned retain/write gate when merging
    """

    def __init__(
        self,
        input_dim: int,
        memory_dim: int,
        max_slots: int,
        allocation_threshold: float = 0.55,
    ):
        super().__init__()
        self.max_slots = max_slots
        self.memory_dim = memory_dim
        self.allocation_threshold = allocation_threshold

        self.key_projection = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.Tanh(),
        )
        self.write_projection = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.Tanh(),
        )
        self.read_query_projection = nn.Linear(input_dim, memory_dim)
        self.state_summary = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Tanh(),
        )
        self.slot_scorer = nn.Sequential(
            nn.Linear(memory_dim * 4, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
        )
        self.allocation_head = nn.Sequential(
            nn.Linear(memory_dim * 4 + 2, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
        )
        self.retain_head = nn.Sequential(
            nn.Linear(memory_dim * 4, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
        )
        self.query_projection = nn.Linear(input_dim, memory_dim)
        self.choice_projection = nn.Linear(input_dim, memory_dim)
        self.output = nn.Sequential(
            nn.Linear(memory_dim * 3, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
        )

    def _slot_separation_loss(self, memory: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
        normalized_memory = F.normalize(memory, dim=-1, eps=1e-8)
        similarity = torch.matmul(normalized_memory, normalized_memory.transpose(1, 2))
        pair_mask = active_mask.unsqueeze(1) & active_mask.unsqueeze(2)
        diagonal = torch.eye(self.max_slots, dtype=torch.bool, device=memory.device).unsqueeze(0)
        pair_mask = pair_mask & ~diagonal
        if not bool(pair_mask.any().item()):
            return torch.zeros((), device=memory.device)
        return similarity.masked_select(pair_mask).pow(2).mean()

    def _update_memory(
        self,
        memory: torch.Tensor,
        slot_keys: torch.Tensor,
        active_mask: torch.Tensor,
        usage: torch.Tensor,
        key_vectors: torch.Tensor,
        write_vectors: torch.Tensor,
        read_queries: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = memory.size(0)
        next_memory = memory.clone()
        next_slot_keys = slot_keys.clone()
        next_active_mask = active_mask.clone()
        next_usage = usage.clone()

        merge_logits = torch.full((batch_size, self.max_slots), fill_value=-1e4, device=memory.device)
        allocation_mask = torch.zeros(batch_size, dtype=torch.bool, device=memory.device)
        selected_slots = torch.full((batch_size,), fill_value=-1, dtype=torch.long, device=memory.device)
        allocation_logits = torch.zeros(batch_size, device=memory.device)
        allocation_targets = torch.zeros(batch_size, device=memory.device)
        max_similarities = torch.zeros(batch_size, device=memory.device)
        read_contexts = torch.zeros(batch_size, self.memory_dim, device=memory.device)

        normalized_keys = F.normalize(slot_keys, dim=-1, eps=1e-8)
        normalized_inputs = F.normalize(key_vectors, dim=-1, eps=1e-8)
        cosine_similarity = torch.matmul(normalized_keys, normalized_inputs.unsqueeze(-1)).squeeze(-1)
        cosine_similarity = cosine_similarity.masked_fill(~active_mask, -1e4)

        for batch_index in range(batch_size):
            if not bool(valid_mask[batch_index].item()):
                continue

            active = active_mask[batch_index]
            has_active = bool(active.any().item())
            key_vector = key_vectors[batch_index]
            write_vector = write_vectors[batch_index]

            if has_active:
                query = read_queries[batch_index]
                active_memory = memory[batch_index]
                read_logits = torch.matmul(active_memory, query.unsqueeze(-1)).squeeze(-1)
                read_logits = read_logits.masked_fill(~active, -1e4)
                read_weights = F.softmax(read_logits, dim=-1) * active.float()
                read_weights = read_weights / read_weights.sum().clamp_min(1e-8)
                read_context = torch.sum(read_weights.unsqueeze(-1) * active_memory, dim=0)
                read_contexts[batch_index] = read_context

                expanded_key = key_vector.unsqueeze(0).expand(self.max_slots, -1)
                expanded_context = read_context.unsqueeze(0).expand(self.max_slots, -1)
                slot_features = torch.cat(
                    [
                        slot_keys[batch_index],
                        memory[batch_index],
                        expanded_key,
                        expanded_context,
                    ],
                    dim=-1,
                )
                slot_merge_logits = self.slot_scorer(slot_features).squeeze(-1)
                slot_merge_logits = slot_merge_logits.masked_fill(~active, -1e4)
                merge_logits[batch_index] = slot_merge_logits
                best_score, best_slot = cosine_similarity[batch_index].max(dim=0)
                max_similarities[batch_index] = best_score
                summary = self.state_summary(torch.cat([key_vector, read_context], dim=-1))
            else:
                read_context = torch.zeros(self.memory_dim, device=memory.device)
                read_contexts[batch_index] = read_context
                best_score = torch.tensor(-1.0, device=memory.device)
                best_slot = torch.tensor(0, device=memory.device, dtype=torch.long)
                max_similarities[batch_index] = -1.0
                summary = self.state_summary(torch.cat([key_vector, read_context], dim=-1))

            allocation_features = torch.cat(
                [
                    key_vector,
                    write_vector,
                    read_context,
                    summary,
                    max_similarities[batch_index].view(1),
                    active.float().mean().view(1),
                ],
                dim=0,
            )
            allocation_logit = self.allocation_head(allocation_features).squeeze(-1)
            allocation_prob = torch.sigmoid(allocation_logit)
            allocation_logits[batch_index] = allocation_logit

            target_allocate = (not has_active) or float(max_similarities[batch_index].item()) < self.allocation_threshold
            allocation_targets[batch_index] = float(target_allocate)

            if has_active and not bool((allocation_prob >= 0.5).item()):
                target_slot = int(merge_logits[batch_index].argmax().item())
                retain_features = torch.cat(
                    [
                        slot_keys[batch_index, target_slot],
                        memory[batch_index, target_slot],
                        key_vector,
                        read_context,
                    ],
                    dim=0,
                )
                retain_gate = torch.sigmoid(self.retain_head(retain_features).squeeze(-1))
                next_memory[batch_index, target_slot] = (
                    retain_gate * memory[batch_index, target_slot] + (1.0 - retain_gate) * write_vector
                )
                next_slot_keys[batch_index, target_slot] = (
                    retain_gate * slot_keys[batch_index, target_slot] + (1.0 - retain_gate) * key_vector
                )
                next_usage[batch_index, target_slot] += 1.0
                selected_slots[batch_index] = target_slot
                continue

            free_slots = (~active).nonzero(as_tuple=False).squeeze(-1)
            if free_slots.numel() > 0:
                slot_index = int(free_slots[0].item())
            elif has_active:
                slot_index = int(merge_logits[batch_index].argmax().item())
            else:
                slot_index = 0

            next_memory[batch_index, slot_index] = write_vector
            next_slot_keys[batch_index, slot_index] = key_vector
            next_active_mask[batch_index, slot_index] = True
            next_usage[batch_index, slot_index] += 1.0
            allocation_mask[batch_index] = True
            selected_slots[batch_index] = slot_index

        return {
            "memory": next_memory,
            "slot_keys": next_slot_keys,
            "active_mask": next_active_mask,
            "usage": next_usage,
            "merge_logits": merge_logits,
            "allocation_mask": allocation_mask,
            "selected_slots": selected_slots,
            "allocation_logits": allocation_logits,
            "allocation_targets": allocation_targets,
            "max_similarities": max_similarities,
            "read_contexts": read_contexts,
        }

    def forward(
        self,
        chunk_embeddings: torch.Tensor,
        chunk_mask: torch.Tensor,
        query_embeddings: torch.Tensor,
        choice_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_chunks, _ = chunk_embeddings.shape
        memory = torch.zeros(batch_size, self.max_slots, self.memory_dim, device=chunk_embeddings.device)
        slot_keys = torch.zeros(batch_size, self.max_slots, self.memory_dim, device=chunk_embeddings.device)
        active_mask = torch.zeros(batch_size, self.max_slots, dtype=torch.bool, device=chunk_embeddings.device)
        usage = torch.zeros(batch_size, self.max_slots, device=chunk_embeddings.device)

        merge_logits_history = []
        allocation_history = []
        selected_slot_history = []
        allocation_logits_history = []
        allocation_target_history = []
        max_similarity_history = []
        key_vector_history = []
        read_context_history = []

        for chunk_index in range(num_chunks):
            valid = chunk_mask[:, chunk_index] > 0
            key_vectors = self.key_projection(chunk_embeddings[:, chunk_index])
            write_vectors = self.write_projection(chunk_embeddings[:, chunk_index])
            read_queries = self.read_query_projection(chunk_embeddings[:, chunk_index])
            update_outputs = self._update_memory(
                memory=memory,
                slot_keys=slot_keys,
                active_mask=active_mask,
                usage=usage,
                key_vectors=key_vectors,
                write_vectors=write_vectors,
                read_queries=read_queries,
                valid_mask=valid,
            )
            memory = update_outputs["memory"]
            slot_keys = update_outputs["slot_keys"]
            active_mask = update_outputs["active_mask"]
            usage = update_outputs["usage"]
            merge_logits_history.append(update_outputs["merge_logits"])
            allocation_history.append(update_outputs["allocation_mask"].float())
            selected_slot_history.append(update_outputs["selected_slots"])
            allocation_logits_history.append(update_outputs["allocation_logits"])
            allocation_target_history.append(update_outputs["allocation_targets"])
            max_similarity_history.append(update_outputs["max_similarities"])
            key_vector_history.append(key_vectors)
            read_context_history.append(update_outputs["read_contexts"])

        query_vector = self.query_projection(query_embeddings)
        attention_logits = torch.matmul(memory, query_vector.unsqueeze(-1)).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~active_mask, -1e4)
        has_any_active = active_mask.any(dim=-1, keepdim=True)
        safe_logits = torch.where(has_any_active, attention_logits, torch.zeros_like(attention_logits))
        memory_attention = F.softmax(safe_logits, dim=-1)
        memory_attention = memory_attention * active_mask.float()
        memory_attention = memory_attention / memory_attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        memory_context = torch.sum(memory_attention.unsqueeze(-1) * memory, dim=1)

        choice_vectors = self.choice_projection(choice_embeddings)
        expanded_context = memory_context.unsqueeze(1).expand(-1, choice_vectors.size(1), -1)
        expanded_query = query_vector.unsqueeze(1).expand_as(expanded_context)
        choice_logits = self.output(torch.cat([expanded_context, expanded_query, choice_vectors], dim=-1)).squeeze(-1)
        slot_separation_loss = self._slot_separation_loss(memory=memory, active_mask=active_mask)

        return {
            "choice_logits": choice_logits,
            "memory_context": memory_context,
            "final_memory": memory,
            "final_slot_keys": slot_keys,
            "routing_logits": torch.stack(merge_logits_history, dim=1),
            "allocation_mask": torch.stack(allocation_history, dim=1),
            "selected_slots": torch.stack(selected_slot_history, dim=1),
            "novelty_logits": torch.stack(allocation_logits_history, dim=1),
            "novelty_targets": torch.stack(allocation_target_history, dim=1),
            "max_similarities": torch.stack(max_similarity_history, dim=1),
            "key_vectors": torch.stack(key_vector_history, dim=1),
            "read_contexts": torch.stack(read_context_history, dim=1),
            "active_mask": active_mask,
            "active_counts": active_mask.sum(dim=-1),
            "usage": usage,
            "slot_separation_loss": slot_separation_loss,
        }


class FrozenBrainExecutor(nn.Module):
    """
    Stronger executor trained on top of a frozen brain generator.

    It scores each answer choice by attending over the generated slot memory with a
    joint query-choice representation.
    """

    def __init__(self, input_dim: int, memory_dim: int):
        super().__init__()
        self.memory_projection = nn.Linear(memory_dim, memory_dim)
        self.query_projection = nn.Linear(input_dim, memory_dim)
        self.choice_projection = nn.Linear(input_dim, memory_dim)
        self.joint_projection = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, memory_dim),
        )
        self.slot_scorer = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(memory_dim * 4 + 1, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
        )

    def forward(
        self,
        memory: torch.Tensor,
        active_mask: torch.Tensor,
        query_embeddings: torch.Tensor,
        choice_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_choices, _ = choice_embeddings.shape
        projected_memory = self.memory_projection(memory)
        query_vector = self.query_projection(query_embeddings)
        choice_vectors = self.choice_projection(choice_embeddings)
        active_counts = active_mask.sum(dim=-1, keepdim=True).float()

        choice_logits = []
        attention_maps = []
        for choice_index in range(num_choices):
            choice_vector = choice_vectors[:, choice_index]
            joint = self.joint_projection(torch.cat([query_vector, choice_vector], dim=-1))
            expanded_joint = joint.unsqueeze(1).expand(-1, projected_memory.size(1), -1)
            slot_features = torch.cat([projected_memory, expanded_joint], dim=-1)
            slot_logits = self.slot_scorer(slot_features).squeeze(-1)
            slot_logits = slot_logits.masked_fill(~active_mask, -1e4)
            has_active = active_mask.any(dim=-1, keepdim=True)
            safe_logits = torch.where(has_active, slot_logits, torch.zeros_like(slot_logits))
            slot_attention = F.softmax(safe_logits, dim=-1) * active_mask.float()
            slot_attention = slot_attention / slot_attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            context = torch.sum(slot_attention.unsqueeze(-1) * projected_memory, dim=1)
            score_input = torch.cat(
                [context, joint, query_vector, choice_vector, active_counts / projected_memory.size(1)],
                dim=-1,
            )
            choice_logits.append(self.output(score_input).squeeze(-1))
            attention_maps.append(slot_attention)

        return {
            "choice_logits": torch.stack(choice_logits, dim=1),
            "slot_attention": torch.stack(attention_maps, dim=1),
        }


class GraphTextReasoner(nn.Module):
    def __init__(self, input_dim: int, memory_dim: int, num_nodes: int, num_operators: int = 4, message_passing_steps: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.num_operators = num_operators
        self.message_passing_steps = message_passing_steps

        self.node_router = nn.Linear(input_dim, num_nodes)
        self.operator_selector = nn.Linear(input_dim, num_operators)
        self.write_projection = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.Tanh(),
        )
        self.query_projection = nn.Linear(input_dim, memory_dim)
        self.choice_projection = nn.Linear(input_dim, memory_dim)
        self.operator_modules = nn.ModuleList([nn.Linear(memory_dim, memory_dim) for _ in range(num_operators)])
        self.output = nn.Sequential(
            nn.Linear(memory_dim * 3, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
        )
        self.layer_norm = nn.LayerNorm(memory_dim)

    def forward(
        self,
        chunk_embeddings: torch.Tensor,
        chunk_mask: torch.Tensor,
        query_embeddings: torch.Tensor,
        choice_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_chunks, _ = chunk_embeddings.shape
        node_states = torch.zeros(batch_size, self.num_nodes, self.memory_dim, device=chunk_embeddings.device)
        edge_weights = torch.zeros(batch_size, self.num_nodes, self.num_nodes, device=chunk_embeddings.device)
        edge_operator_weights = torch.zeros(batch_size, self.num_nodes, self.num_nodes, self.num_operators, device=chunk_embeddings.device)

        previous_node_weights = torch.zeros(batch_size, self.num_nodes, device=chunk_embeddings.device)
        node_logits_history = []

        for chunk_index in range(num_chunks):
            valid = chunk_mask[:, chunk_index].unsqueeze(-1)
            chunk_embedding = chunk_embeddings[:, chunk_index]
            node_logits = self.node_router(chunk_embedding)
            node_weights = F.softmax(node_logits, dim=-1)
            operator_weights = F.softmax(self.operator_selector(chunk_embedding), dim=-1)
            write_vector = self.write_projection(chunk_embedding).unsqueeze(1)

            update = node_weights.unsqueeze(-1) * write_vector * valid.unsqueeze(-1)
            retain = 1.0 - node_weights.unsqueeze(-1) * valid.unsqueeze(-1)
            node_states = node_states * retain + update

            pair_update = previous_node_weights.unsqueeze(2) * node_weights.unsqueeze(1) * valid.unsqueeze(-1)
            edge_weights = edge_weights + pair_update
            edge_operator_weights = edge_operator_weights + pair_update.unsqueeze(-1) * operator_weights.unsqueeze(1).unsqueeze(1)
            previous_node_weights = node_weights
            node_logits_history.append(node_logits)

        if torch.any(edge_weights > 0):
            normalized_edges = edge_weights / edge_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            normalized_edges = edge_weights

        propagated = node_states
        for _ in range(self.message_passing_steps):
            operator_messages = torch.stack([torch.tanh(module(propagated)) for module in self.operator_modules], dim=2)
            operator_probs = edge_operator_weights / edge_operator_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            messages = (
                operator_probs.unsqueeze(-1) * operator_messages.unsqueeze(2)
            ).sum(dim=3)
            incoming = (normalized_edges.unsqueeze(-1) * messages).sum(dim=1)
            propagated = self.layer_norm(propagated + incoming)

        query_vector = self.query_projection(query_embeddings)
        attention_logits = torch.matmul(propagated, query_vector.unsqueeze(-1)).squeeze(-1)
        node_attention = F.softmax(attention_logits, dim=-1)
        memory_context = torch.sum(node_attention.unsqueeze(-1) * propagated, dim=1)

        choice_vectors = self.choice_projection(choice_embeddings)
        expanded_context = memory_context.unsqueeze(1).expand(-1, choice_vectors.size(1), -1)
        expanded_query = query_vector.unsqueeze(1).expand_as(expanded_context)
        choice_logits = self.output(torch.cat([expanded_context, expanded_query, choice_vectors], dim=-1)).squeeze(-1)

        return {
            "choice_logits": choice_logits,
            "memory_context": memory_context,
            "routing_logits": torch.stack(node_logits_history, dim=1),
        }


class StrictGraphTextReasoner(nn.Module):
    """
    Graph-dependent model: answer context is produced only through edge
    propagation from query-selected seed nodes. Without edges or message
    passing, the context collapses toward zero.
    """

    def __init__(self, input_dim: int, memory_dim: int, num_nodes: int, message_passing_steps: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.message_passing_steps = message_passing_steps

        self.node_router = nn.Linear(input_dim, num_nodes)
        self.write_projection = nn.Sequential(
            nn.Linear(input_dim, memory_dim),
            nn.Tanh(),
        )
        self.query_seed_router = nn.Linear(input_dim, num_nodes)
        self.query_projection = nn.Linear(input_dim, memory_dim)
        self.choice_projection = nn.Linear(input_dim, memory_dim)
        self.message_projection = nn.Linear(memory_dim, memory_dim)
        self.output = nn.Sequential(
            nn.Linear(memory_dim * 3, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
        )

    def build_graph(self, chunk_embeddings: torch.Tensor, chunk_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_chunks, _ = chunk_embeddings.shape
        node_states = torch.zeros(batch_size, self.num_nodes, self.memory_dim, device=chunk_embeddings.device)
        edge_weights = torch.zeros(batch_size, self.num_nodes, self.num_nodes, device=chunk_embeddings.device)
        previous_node_weights = torch.zeros(batch_size, self.num_nodes, device=chunk_embeddings.device)
        routing_logits = []

        for chunk_index in range(num_chunks):
            valid = chunk_mask[:, chunk_index].unsqueeze(-1)
            chunk_embedding = chunk_embeddings[:, chunk_index]
            node_logits = self.node_router(chunk_embedding)
            node_weights = F.softmax(node_logits, dim=-1)
            write_vector = self.write_projection(chunk_embedding).unsqueeze(1)
            update = node_weights.unsqueeze(-1) * write_vector * valid.unsqueeze(-1)
            retain = 1.0 - node_weights.unsqueeze(-1) * valid.unsqueeze(-1)
            node_states = node_states * retain + update

            pair_update = previous_node_weights.unsqueeze(2) * node_weights.unsqueeze(1) * valid.unsqueeze(-1)
            edge_weights = edge_weights + pair_update
            previous_node_weights = node_weights
            routing_logits.append(node_logits)

        edge_weights = edge_weights + edge_weights.transpose(1, 2)
        return node_states, edge_weights, torch.stack(routing_logits, dim=1)

    def propagate(self, node_states: torch.Tensor, edge_weights: torch.Tensor, query_embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seed_logits = self.query_seed_router(query_embeddings)
        activations = F.softmax(seed_logits, dim=-1)
        normalized_edges = edge_weights / edge_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        transformed_nodes = torch.tanh(self.message_projection(node_states))

        propagated_contexts = []
        current = activations
        for _ in range(self.message_passing_steps):
            current = torch.bmm(current.unsqueeze(1), normalized_edges).squeeze(1)
            propagated_contexts.append(torch.sum(current.unsqueeze(-1) * transformed_nodes, dim=1))

        if propagated_contexts:
            context = torch.stack(propagated_contexts, dim=1).mean(dim=1)
        else:
            context = torch.zeros_like(torch.sum(activations.unsqueeze(-1) * transformed_nodes, dim=1))
        return context, seed_logits

    def forward(
        self,
        chunk_embeddings: torch.Tensor,
        chunk_mask: torch.Tensor,
        query_embeddings: torch.Tensor,
        choice_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_states, edge_weights, routing_logits = self.build_graph(chunk_embeddings, chunk_mask)
        memory_context, seed_logits = self.propagate(node_states, edge_weights, query_embeddings)

        query_vector = self.query_projection(query_embeddings)
        choice_vectors = self.choice_projection(choice_embeddings)
        expanded_context = memory_context.unsqueeze(1).expand(-1, choice_vectors.size(1), -1)
        expanded_query = query_vector.unsqueeze(1).expand_as(expanded_context)
        choice_logits = self.output(torch.cat([expanded_context, expanded_query, choice_vectors], dim=-1)).squeeze(-1)

        return {
            "choice_logits": choice_logits,
            "memory_context": memory_context,
            "routing_logits": routing_logits,
            "seed_logits": seed_logits,
            "edge_weights": edge_weights,
        }
