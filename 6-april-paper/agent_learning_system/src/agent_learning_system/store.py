"""Append-only episodic store."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from agent_learning_system.schema import InteractionTrace, ToolEvent, utc_now_iso


class EpisodicStore:
    """Persist interaction traces as append-only JSONL."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.root_dir / "interactions.jsonl"

    def append(self, trace: InteractionTrace) -> None:
        with open(self.trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=True) + "\n")

    def append_interaction(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        outcome: str,
        summary: str = "",
        learned_type: str = "general",
        tags: list[str] | None = None,
        tool_events: list[ToolEvent] | None = None,
        correction: str | None = None,
    ) -> InteractionTrace:
        trace = InteractionTrace(
            trace_id=str(uuid4()),
            session_id=session_id,
            timestamp=utc_now_iso(),
            user_message=user_message,
            assistant_response=assistant_response,
            outcome=outcome,
            summary=summary,
            learned_type=learned_type,
            tags=list(tags or []),
            tool_events=list(tool_events or []),
            correction=correction,
        )
        self.append(trace)
        return trace

    def load_all(self) -> list[dict]:
        if not self.trace_path.exists():
            return []
        with open(self.trace_path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self.load_all()
        return rows[-limit:]
