"""Synthetic users: ground truth for every slot plus an answer style (SPEC §4, Decision Q4).

Ground truth is stratified: within each slot every value is used about equally often, and slots
are shuffled independently. Backstories live in a separate `backstories.jsonl` beside the persona
file (written once by an LLM in M5) and are merged at load time, so the persona file itself stays
reproducible from the schema alone.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path

from elicitor.jsonl import read_jsonl, write_jsonl
from elicitor.schema import Schema

STYLES = ("direct", "vague", "rambling", "indifferent")
DEV = "dev"
EVAL = "eval"
BACKSTORIES = "backstories.jsonl"


@dataclass(frozen=True)
class Persona:
    id: str
    split: str
    style: str
    truth: dict[str, str]
    backstory: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "split": self.split,
            "style": self.style,
            "truth": self.truth,
            "backstory": self.backstory,
        }

    @classmethod
    def from_dict(cls, row: dict[str, object]) -> Persona:
        return cls(
            id=str(row["id"]),
            split=str(row["split"]),
            style=str(row["style"]),
            truth=dict(row["truth"]),  # type: ignore[arg-type]
            backstory=str(row.get("backstory", "")),
        )


def stratified(values: tuple[str, ...], n: int, rng: random.Random) -> list[str]:
    """n draws that use every value equally often (up to rounding), in random order."""
    reps = -(-n // len(values))
    out = list(values) * reps
    rng.shuffle(out)
    # Shuffling before truncating keeps the leftover counts random rather than favouring the
    # first values.
    return out[:n]


def generate_personas(schema: Schema, n_dev: int, n_eval: int, seed: int) -> list[Persona]:
    """Dev personas are spread evenly over answer styles, and so are eval personas."""
    rng = random.Random(seed)
    n = n_dev + n_eval
    columns = {s.name: stratified(s.value_names, n, rng) for s in schema.slots}
    styles = stratified(STYLES, n, rng)

    # The first n_dev / len(STYLES) personas of each style go to dev.
    per_style = n_dev // len(STYLES)
    if per_style * len(STYLES) != n_dev:
        raise ValueError(f"n_dev must be a multiple of {len(STYLES)}")
    seen = dict.fromkeys(STYLES, 0)
    personas = []
    for i in range(n):
        style = styles[i]
        split = DEV if seen[style] < per_style else EVAL
        seen[style] += 1
        truth = {name: col[i] for name, col in columns.items()}
        personas.append(Persona(id=f"p{i + 1:03d}", split=split, style=style, truth=truth))
    return personas


def load_personas(path: str | Path, split: str | None = None) -> list[Persona]:
    """Personas of one split (or all), with backstories merged in when the file exists."""
    path = Path(path)
    personas = [Persona.from_dict(row) for row in read_jsonl(path)]
    backstories_path = path.with_name(BACKSTORIES)
    if backstories_path.exists():
        stories = {row["id"]: row["backstory"] for row in read_jsonl(backstories_path)}
        personas = [replace(p, backstory=stories.get(p.id, p.backstory)) for p in personas]
    return [p for p in personas if split is None or p.split == split]


def save_personas(path: str | Path, personas: list[Persona]) -> None:
    write_jsonl(path, (p.to_dict() for p in personas))
