"""Question phrasing (SPEC §8, Decision Q9): templates by default, LLM rewriting as an ablation."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Protocol

from elicitor.questions import CHOICE, OPEN, YESNO, Question
from elicitor.schema import ORDINAL, Schema, Slot

if TYPE_CHECKING:
    from elicitor.llm import LLMClient


# The LLM phrasing prompt's rule on answer options, per question type (SPEC §8). The open rule is
# the M2 wording, so open-question prompts are unchanged.
KIND_RULES = {
    OPEN: '- Do not list, suggest or contrast possible answers (no "this or that" questions).',
    CHOICE: "- Keep exactly the options it offers, in the same order, and add no others.",
    YESNO: "- Keep it a yes/no question about the option it names.",
}


def _join_options(labels: list[str]) -> str:
    """Options as `a or b` or `a, b, or c`; semicolons when a label has its own comma or "or"."""
    if any("," in label or " or " in label for label in labels):
        return "; ".join(labels[:-1]) + "; or " + labels[-1]
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + ", or " + labels[-1]


def template_question(slot: Slot, question: Question) -> str:
    """The template wording of any question type (SPEC §8)."""
    if question.kind == OPEN:
        return slot.question
    if question.kind == CHOICE:
        labels = [slot.value(v).label for v in question.values]
        return f"Would you prefer {_join_options(labels)}?"
    if slot.kind == ORDINAL:
        # Yes means one of the first len(values) bins, i.e. below that many edges in.
        edge = f"{slot.edges[len(question.values) - 1]:g}"
        threshold = (
            slot.threshold_question or f"Would you like it to be below {{edge}} {slot.unit}?"
        )
        return threshold.format(edge=edge)
    return f"Would {slot.value(question.values[0]).label} suit you?"


class Phraser(Protocol):
    name: str

    def render(self, question: Question, rng: random.Random) -> str: ...


class TemplatePhraser:
    name = "tpl"

    def __init__(self, schema: Schema):
        self.schema = schema

    def render(self, question: Question, rng: random.Random | None = None) -> str:
        return template_question(self.schema.slot(question.slot), question)


class LLMPhraser:
    """Rewrites the template; falls back to it when the reply never validates."""

    name = "llm"

    def __init__(self, schema: Schema, client: LLMClient):
        self.template = TemplatePhraser(schema)
        self.schema = schema
        self.client = client

    def render(self, question: Question, rng: random.Random) -> str:
        from elicitor.llm import LLMParseError

        template = self.template.render(question)
        slot = self.schema.slot(question.slot)
        context = {
            "slot": slot,
            "slot_description": slot.description,
            "template": template,
            "kind_rule": KIND_RULES[question.kind],
        }
        try:
            reply = self.client.call("phrase", context, seed=rng.getrandbits(31))
        except LLMParseError:
            return template
        return reply.question  # type: ignore[attr-defined]


def make_phraser(name: str, schema: Schema, client: LLMClient | None = None) -> Phraser:
    if name == "tpl":
        return TemplatePhraser(schema)
    if name == "llm":
        if client is None:
            raise ValueError("the llm phraser needs an LLM client")
        return LLMPhraser(schema, client)
    raise ValueError(f"unknown phraser {name!r}; choose from ['llm', 'tpl']")
