"""SQLite cache of validated LLM replies (SPEC §14).

The key is a SHA-256 of everything that can change a reply: provider, model, role, messages,
output schema, sampling parameters and seed. Re-running an experiment with the same settings
therefore makes no server calls. Writes are committed in batches; `close` commits the rest.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_COMMIT_EVERY = 200


def cache_key(**parts: Any) -> str:
    """Stable hash of keyword parts; key order and dict order do not matter."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


class ReplyCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._pending = 0
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS replies ("
            " key TEXT PRIMARY KEY, role TEXT, model TEXT, text TEXT,"
            " prompt_tokens INTEGER, completion_tokens INTEGER)"
        )

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT text FROM replies WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(
        self,
        key: str,
        *,
        role: str,
        model: str,
        text: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO replies VALUES (?, ?, ?, ?, ?, ?)",
                (key, role, model, text, prompt_tokens, completion_tokens),
            )
            self._pending += 1
            if self._pending >= _COMMIT_EVERY:
                self._db.commit()
                self._pending = 0

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM replies").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._db.commit()
            self._db.close()
