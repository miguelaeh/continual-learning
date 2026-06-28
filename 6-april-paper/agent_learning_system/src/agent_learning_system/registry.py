"""Checkpoint registry and promotion flow."""

from __future__ import annotations

import json
from pathlib import Path

from agent_learning_system.schema import CheckpointRecord, utc_now_iso


class CheckpointRegistry:
    """Track candidate and promoted checkpoints."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save(self, records: list[dict]) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)

    def register_candidate(
        self,
        checkpoint_id: str,
        path: str,
        metrics: dict[str, float] | None = None,
        notes: str = "",
    ) -> CheckpointRecord:
        record = CheckpointRecord(
            checkpoint_id=checkpoint_id,
            created_at=utc_now_iso(),
            path=path,
            status="candidate",
            metrics=metrics or {},
            notes=notes,
        )
        rows = self._load()
        rows.append(record.to_dict())
        self._save(rows)
        return record

    def promote(self, checkpoint_id: str) -> dict:
        rows = self._load()
        target = None
        for row in rows:
            if row["checkpoint_id"] == checkpoint_id:
                target = row
            elif row["status"] == "promoted":
                row["status"] = "archived"
        if target is None:
            raise KeyError(f"Checkpoint '{checkpoint_id}' not found.")
        target["status"] = "promoted"
        self._save(rows)
        return target

    def list_records(self) -> list[dict]:
        return self._load()
