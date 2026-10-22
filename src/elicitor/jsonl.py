"""JSON Lines helpers. Files are UTF-8 with LF line endings on every platform."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def dumps_jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(dumps_jsonl(rows), encoding="utf-8", newline="\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
