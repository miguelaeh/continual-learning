"""Build consolidation datasets from interaction traces."""

from __future__ import annotations

from agent_learning_system.schema import ConsolidationExample


def _trace_to_messages(trace: dict) -> list[dict[str, str]]:
    user_message = str(trace.get("user_message", "")).strip()
    assistant_response = str(trace.get("assistant_response", "")).strip()
    correction = trace.get("correction")

    target = str(correction).strip() if correction else assistant_response
    messages = [{"role": "user", "content": user_message}]
    if target:
        messages.append({"role": "assistant", "content": target})
    return messages


def build_consolidation_examples(
    traces: list[dict],
    include_outcomes: set[str] | None = None,
) -> list[ConsolidationExample]:
    allowed = include_outcomes or {"success", "corrected"}
    examples: list[ConsolidationExample] = []

    for trace in traces:
        outcome = str(trace.get("outcome", "")).strip().lower()
        if outcome not in allowed:
            continue
        messages = _trace_to_messages(trace)
        if len(messages) < 2:
            continue
        examples.append(
            ConsolidationExample(
                source_trace_id=str(trace["trace_id"]),
                messages=messages,
                learned_type=str(trace.get("learned_type", "general")),
                tags=[str(tag) for tag in trace.get("tags", [])],
            )
        )
    return examples
