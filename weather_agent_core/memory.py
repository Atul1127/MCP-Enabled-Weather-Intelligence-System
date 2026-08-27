"""Small durable checkpoint store for agent sessions."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from typing import Any

class AgentMemory:
    """Persist bounded graph state by thread id using SQLite."""
    def __init__(self, path: str = "observability/agent_memory.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS checkpoints (thread_id TEXT PRIMARY KEY, state_json TEXT NOT NULL, updated_at REAL NOT NULL)")
            db.commit()

    def save(self, thread_id: str, state: dict[str, Any]) -> None:
        if not thread_id.strip():
            raise ValueError("thread_id cannot be empty")
        payload = json.dumps(state, default=str, ensure_ascii=False)
        import time
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO checkpoints(thread_id,state_json,updated_at) VALUES(?,?,?) ON CONFLICT(thread_id) DO UPDATE SET state_json=excluded.state_json, updated_at=excluded.updated_at", (thread_id, payload, time.time()))
            db.commit()

    def load(self, thread_id: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT state_json FROM checkpoints WHERE thread_id=?", (thread_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def delete(self, thread_id: str) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM checkpoints WHERE thread_id=?", (thread_id,))
            db.commit()
