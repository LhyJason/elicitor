"""Fixed synthetic catalog of items to recommend (SPEC §10, Decision Q10 B).

Every item has a concrete value on every slot (never `no_preference`), stratified like the
personas so that every value is offered. Ranking items against a belief is the recommender's
job (M3).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from elicitor.jsonl import read_jsonl, write_jsonl
from elicitor.personas import stratified
from elicitor.schema import Schema


@dataclass(frozen=True)
class Item:
    id: str
    attrs: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "attrs": self.attrs}


def generate_catalog(schema: Schema, size: int, seed: int) -> list[Item]:
    rng = random.Random(seed)
    columns = {s.name: stratified(s.concrete_values, size, rng) for s in schema.slots}
    return [
        Item(id=f"i{i + 1:03d}", attrs={name: col[i] for name, col in columns.items()})
        for i in range(size)
    ]


def load_catalog(path: str | Path) -> list[Item]:
    return [Item(id=row["id"], attrs=dict(row["attrs"])) for row in read_jsonl(path)]


def save_catalog(path: str | Path, items: list[Item]) -> None:
    write_jsonl(path, (item.to_dict() for item in items))
