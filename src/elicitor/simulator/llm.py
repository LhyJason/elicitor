"""LLM simulator: role-play from the persona (SPEC §9, Decision Q11 A).

The prompt holds the full profile, so an answer can leak unasked slots; that is measured, not
prevented. The LLM seed comes from the answer generator of the episode, so arms that ask the
same question get the same answer (common random numbers). A reply that never validates becomes
an empty answer, which the updaters read as no information.
"""

from __future__ import annotations

import random
import tomllib
from pathlib import Path

from elicitor.llm import LLMClient, LLMParseError
from elicitor.personas import Persona
from elicitor.questions import Question
from elicitor.schema import Schema


def profile_lines(schema: Schema, persona: Persona) -> str:
    return "\n".join(
        f"- {slot.description.rstrip('.')}: {slot.value(persona.truth[slot.name]).label}"
        for slot in schema.slots
    )


def load_styles(path: str | Path) -> dict[str, str]:
    with open(path, "rb") as f:
        return {k: str(v) for k, v in tomllib.load(f).items()}


class LLMSimulator:
    name = "llm"

    def __init__(self, schema: Schema, client: LLMClient, styles: dict[str, str]):
        self.schema = schema
        self.client = client
        self.styles = styles

    def respond(self, question: Question, text: str, persona: Persona, rng: random.Random) -> str:
        slot = self.schema.slot(question.slot)
        context = {
            "slot": slot,
            "topic": slot.description.rstrip("."),
            "persona": persona,
            "profile": profile_lines(self.schema, persona),
            "backstory": f"Background: {persona.backstory}\n" if persona.backstory else "",
            "style_guide": self.styles[persona.style],
            "question": text,
        }
        try:
            reply = self.client.call("simulate", context, seed=rng.getrandbits(31))
        except LLMParseError:
            return ""
        return reply.answer.strip()  # type: ignore[attr-defined]
