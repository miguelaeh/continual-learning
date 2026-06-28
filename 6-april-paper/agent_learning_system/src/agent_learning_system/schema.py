"""Data schema for interaction logging and promotion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ToolEvent:
    name: str
    input_text: str = ""
    output_text: str = ""
    success: bool = True


@dataclass
class InteractionTrace:
    trace_id: str
    session_id: str
    timestamp: str
    user_message: str
    assistant_response: str
    outcome: str
    summary: str = ""
    learned_type: str = "general"
    tags: list[str] = field(default_factory=list)
    tool_events: list[ToolEvent] = field(default_factory=list)
    correction: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["tool_events"] = [asdict(item) for item in self.tool_events]
        return payload


@dataclass
class ConsolidationExample:
    source_trace_id: str
    messages: list[dict[str, str]]
    learned_type: str = "general"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProfileMemory:
    memory_id: str
    created_at: str
    updated_at: str
    memory_type: str
    key: str
    value: str
    source: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CheckpointRecord:
    checkpoint_id: str
    created_at: str
    path: str
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
