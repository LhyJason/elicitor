"""Question phrasing (SPEC §8, Decision Q9): templates by default, LLM rewriting as an ablation."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Protocol

from elicitor.questions import OPEN, Question
from elicitor.schema import Schema

if TYPE_CHECKING:
    from elicitor.llm import LLMClient


class Phraser(Protocol):
    name: str

    def render(self, question: Question, rng: random.Random) -> str: ...


class TemplatePhraser:
    name = "tpl"

    def __init__(self, schema: Schema):
        self.schema = schema

    def render(self, question: Question, rng: random.Random | None = None) -> str:
        if question.kind != OPEN:
            raise ValueError(f"no template for question kind {question.kind!r}")
        return self.schema.slot(question.slot).question


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
        context = {"slot": slot, "slot_description": slot.description, "template": template}
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
