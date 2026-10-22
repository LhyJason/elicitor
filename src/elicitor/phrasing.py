"""Question phrasing (SPEC §8). Templates from the schema; the LLM rewriter arrives in M2."""

from elicitor.questions import OPEN, Question
from elicitor.schema import Schema


class TemplatePhraser:
    name = "tpl"

    def __init__(self, schema: Schema):
        self.schema = schema

    def render(self, question: Question) -> str:
        if question.kind != OPEN:
            raise ValueError(f"no template for question kind {question.kind!r}")
        return self.schema.slot(question.slot).question
