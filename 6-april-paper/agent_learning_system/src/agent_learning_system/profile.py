"""Structured profile memory for durable user and agent context."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from agent_learning_system.schema import ProfileMemory, utc_now_iso


class ProfileMemoryStore:
    """Persist a small structured memory profile as JSON."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.profile_path = self.root_dir / "profile_memory.json"

    def load_all(self) -> list[dict]:
        if not self.profile_path.exists():
            return []
        return json.loads(self.profile_path.read_text(encoding="utf-8"))

    def save_all(self, rows: list[dict]) -> None:
        self.profile_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def upsert(
        self,
        memory_type: str,
        key: str,
        value: str,
        source: str = "",
        tags: list[str] | None = None,
    ) -> dict:
        rows = self.load_all()
        now = utc_now_iso()
        normalized_tags = list(tags or [])
        for row in rows:
            if row.get("memory_type") == memory_type and row.get("key") == key:
                row["value"] = value
                row["source"] = source
                row["tags"] = normalized_tags
                row["updated_at"] = now
                self.save_all(rows)
                return row

        item = ProfileMemory(
            memory_id=str(uuid4()),
            created_at=now,
            updated_at=now,
            memory_type=memory_type,
            key=key,
            value=value,
            source=source,
            tags=normalized_tags,
        ).to_dict()
        rows.append(item)
        self.save_all(rows)
        return item

    def list_by_type(self, memory_type: str | None = None) -> list[dict]:
        rows = self.load_all()
        if memory_type is None:
            return rows
        return [row for row in rows if row.get("memory_type") == memory_type]
