"""Persona backstories, written once by the simulator model and then frozen (SPEC §4, M5).

The prompt sees only the persona's answer style and a seeded age, never the ground truth, and
lists the leakage keywords (SPEC §9) as words to avoid. A backstory in which the keywords still
find a slot value is drawn again with the next seed, up to `MAX_DRAWS` times.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from elicitor.jsonl import write_jsonl
from elicitor.llm import LLMClient, LLMParseError
from elicitor.personas import Persona
from elicitor.rng import derive_rng, derive_seed
from elicitor.schema import NO_PREFERENCE, Schema
from elicitor.updaters.rule import find_mentions

MAX_DRAWS = 5
AGE_RANGE = (22, 75)


@dataclass(frozen=True)
class Backstory:
    persona: str
    text: str
    draws: int
    flagged: tuple[str, ...]  # slots the keywords still find after the last draw


def flagged_slots(schema: Schema, text: str) -> list[str]:
    """Slots whose concrete values the text names, positively or negated."""
    return [
        slot.name
        for slot in schema.slots
        if any(v != NO_PREFERENCE for part in find_mentions(slot, text) for v in part)
    ]


def avoid_words(schema: Schema) -> str:
    words = [k for slot in schema.slots for v in slot.values if v.name != NO_PREFERENCE
             for k in v.keywords]  # fmt: skip
    return ", ".join(dict.fromkeys(words))


def write_backstory(
    client: LLMClient, schema: Schema, persona: Persona, styles: dict[str, str]
) -> Backstory:
    age = derive_rng("backstory", persona.id).randint(*AGE_RANGE)
    context = {"age": str(age), "style_guide": styles[persona.style],
               "avoid_words": avoid_words(schema)}  # fmt: skip
    text, flagged = "", ["(no valid reply)"]
    for draw in range(1, MAX_DRAWS + 1):
        seed = derive_seed("backstory", persona.id, draw) % 2**31
        try:
            reply = client.call("backstory", context, seed=seed)
        except LLMParseError:
            continue
        text = " ".join(reply.backstory.split())  # type: ignore[attr-defined]
        flagged = flagged_slots(schema, text)
        if not flagged:
            break
    return Backstory(persona.id, text, draw, tuple(flagged))


def write_backstories(
    client: LLMClient,
    schema: Schema,
    personas: list[Persona],
    styles: dict[str, str],
    path: str | Path,
    workers: int = 1,
    progress: Callable[[str], None] | None = None,
) -> list[Backstory]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        stories = list(pool.map(lambda p: write_backstory(client, schema, p, styles), personas))
    write_jsonl(path, ({"id": s.persona, "backstory": s.text} for s in stories))
    if progress:
        redrawn = sum(s.draws > 1 for s in stories)
        progress(f"wrote {len(stories)} backstories to {path} ({redrawn} drawn more than once)")
        for s in stories:
            if s.flagged:
                progress(f"  {s.persona} still flagged after {s.draws} draws: {list(s.flagged)}")
    return stories
