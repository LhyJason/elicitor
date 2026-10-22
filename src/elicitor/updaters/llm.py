"""LLM updater: the LLM scores the answer, Bayes rule stays in code (SPEC Decision Q6 B).

For the asked slot the LLM returns an integer score s_v in [0, 10] per value. The likelihood is

    l_v = ((s_v + delta) / (10 + delta)) ** beta,

with smoothing delta (a score of 0 never rules a value out) and temper beta. Equal scores give a
flat likelihood, so an answer that says nothing about the slot leaves the belief unchanged.
"""

from __future__ import annotations

from typing import Any

from elicitor.belief import Belief
from elicitor.llm import LLMClient, LLMParseError
from elicitor.llm.roles import SCORE_MAX, update_scores, value_lines
from elicitor.questions import Question
from elicitor.schema import CATEGORICAL

_NO_PREFERENCE_NOTE = (
    '\nScore "no_preference" high when the traveller says they do not mind or have no preference.'
)


def likelihood(scores: list[int], delta: float = 0.5, beta: float = 1.0) -> list[float]:
    return [((s + delta) / (SCORE_MAX + delta)) ** beta for s in scores]


class LLMUpdater:
    name = "llm"

    def __init__(self, client: LLMClient, delta: float = 0.5, beta: float = 1.0):
        if delta <= 0 or beta <= 0:
            raise ValueError("delta and beta must be positive")
        self.client = client
        self.delta = delta
        self.beta = beta

    def update(
        self, belief: Belief, question: Question, text: str, answer: str
    ) -> tuple[Belief, dict[str, Any]]:
        slot = belief.schema.slot(question.slot)
        record: dict[str, Any] = {"slot": slot.name}
        if not answer.strip():
            record["skipped"] = "empty answer"
            return belief, record
        context = {
            "slot": slot,
            "slot_description": slot.description,
            "values": value_lines(slot),
            "question": text,
            "answer": answer,
            "no_preference_note": _NO_PREFERENCE_NOTE if slot.kind == CATEGORICAL else "",
        }
        try:
            reply = self.client.call("update", context, seed=0)
        except LLMParseError as exc:
            record["failed"] = str(exc)
            return belief, record
        scores = update_scores(reply, slot)
        record["scores"] = dict(zip(slot.value_names, scores, strict=True))
        record["evidence"] = reply.evidence  # type: ignore[attr-defined]
        return belief.update(slot.name, likelihood(scores, self.delta, self.beta)), record
