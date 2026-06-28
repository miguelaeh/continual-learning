"""Runtime prompt assembly using profile memory and episodic retrieval."""

from __future__ import annotations

from agent_learning_system.profile import ProfileMemoryStore
from agent_learning_system.retrieval import retrieve_relevant_traces
from agent_learning_system.store import EpisodicStore


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful autonomous assistant. Use the structured memory and retrieved "
    "interaction traces when they are relevant, but do not invent facts that are not supported."
)


def _format_profile_rows(rows: list[dict]) -> str:
    if not rows:
        return "None."
    lines = []
    for row in rows:
        memory_type = str(row.get("memory_type", "general")).strip()
        key = str(row.get("key", "")).strip()
        value = str(row.get("value", "")).strip()
        if key and value:
            lines.append(f"- [{memory_type}] {key}: {value}")
    return "\n".join(lines) if lines else "None."


def _format_trace_rows(rows: list[dict]) -> str:
    if not rows:
        return "None."
    formatted = []
    for row in rows:
        formatted.append(
            "\n".join(
                [
                    f"- trace_id: {row.get('trace_id', '')}",
                    f"  outcome: {row.get('outcome', '')}",
                    f"  learned_type: {row.get('learned_type', '')}",
                    f"  user: {str(row.get('user_message', '')).strip()}",
                    f"  assistant: {str(row.get('assistant_response', '')).strip()}",
                ]
            )
        )
    return "\n".join(formatted)


def build_runtime_prompt(
    store: EpisodicStore,
    profile_store: ProfileMemoryStore,
    user_message: str,
    session_id: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    retrieval_limit: int = 5,
    recent_limit: int = 3,
) -> dict:
    traces = store.load_all()
    retrieved = retrieve_relevant_traces(user_message, traces, limit=retrieval_limit)
    recent = []
    if session_id:
        session_rows = [row for row in traces if row.get("session_id") == session_id]
        recent = session_rows[-recent_limit:]

    profile_rows = profile_store.load_all()
    prompt = "\n\n".join(
        [
            f"System:\n{system_prompt}",
            f"Structured memory:\n{_format_profile_rows(profile_rows)}",
            f"Recent session traces:\n{_format_trace_rows(recent)}",
            f"Retrieved relevant traces:\n{_format_trace_rows(retrieved)}",
            f"User request:\n{user_message.strip()}",
        ]
    )

    return {
        "prompt": prompt,
        "structured_memory": profile_rows,
        "recent_traces": recent,
        "retrieved_traces": retrieved,
    }
