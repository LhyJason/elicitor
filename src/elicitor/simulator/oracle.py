"""Rule oracle (SPEC Decision Q11 B): answers from ground truth with known noise.

With probability rho the answer names the persona's true value; otherwise it names a value drawn
uniformly from all of the slot's values (possibly the true one). This is exactly the answer model
the rule updater and the analytic EIG assume, so rho can be swept with a known ground truth.
The answer style is ignored here; styles matter only for the LLM simulator (M2).
"""

from __future__ import annotations

import random

from elicitor.personas import Persona
from elicitor.questions import Question
from elicitor.schema import NO_PREFERENCE, ORDINAL, Schema, Slot


def render_answer(slot: Slot, value: str) -> str:
    label = slot.value(value).label
    if value == NO_PREFERENCE:
        return f"I have {label} there."
    if slot.kind == ORDINAL:
        return f"Ideally {label}."
    return f"I'd like {label}."


class OracleSimulator:
    name = "oracle"

    def __init__(self, schema: Schema, rho: float = 1.0):
        if not 0 <= rho <= 1:
            raise ValueError("rho must be in [0, 1]")
        self.schema = schema
        self.rho = rho

    def respond(self, question: Question, text: str, persona: Persona, rng: random.Random) -> str:
        slot = self.schema.slot(question.slot)
        value = persona.truth[slot.name]
        if rng.random() >= self.rho:
            value = rng.choice(slot.value_names)
        return render_answer(slot, value)
