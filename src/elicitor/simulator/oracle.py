"""Rule oracle (SPEC Decision Q11 B): answers from ground truth with known noise.

The answer is an outcome drawn from the answer model of SPEC §7: with probability rho the
noise-free outcome for the persona's true value, otherwise one drawn uniformly from all of the
question's outcomes (for an open question: all of the slot's values). This is exactly the model
the rule updater and the analytic EIG assume, so rho can be swept with a known ground truth.
rho may differ by question type. The answer style is ignored here; styles matter only for the
LLM simulator (M2).
"""

from __future__ import annotations

import random
from collections.abc import Mapping

from elicitor.answer_model import AnswerModel
from elicitor.personas import Persona
from elicitor.questions import CHOICE, OPEN, OTHER, YES, Question
from elicitor.schema import NO_PREFERENCE, ORDINAL, Schema, Slot


def render_answer(slot: Slot, value: str) -> str:
    """The oracle's answer to an open question when it names `value`."""
    label = slot.value(value).label
    if value == NO_PREFERENCE:
        return f"I have {label} there."
    if slot.kind == ORDINAL:
        return f"Ideally {label}."
    return f"I'd like {label}."


def render_outcome(slot: Slot, question: Question, outcome: str) -> str:
    if question.kind == OPEN:
        return render_answer(slot, outcome)
    if question.kind == CHOICE:
        if outcome == OTHER:
            return "None of those, really."
        return f"I'd pick {slot.value(outcome).label}."
    return "Yes, that would suit me." if outcome == YES else "No, not really."


class OracleSimulator:
    name = "oracle"

    def __init__(self, schema: Schema, rho: float | Mapping[str, float] = 1.0):
        self.schema = schema
        self.model = AnswerModel(rho)

    def respond(self, question: Question, text: str, persona: Persona, rng: random.Random) -> str:
        slot = self.schema.slot(question.slot)
        outcome = self.model.sample(slot, question, persona.truth[slot.name], rng)
        return render_outcome(slot, question, outcome)
