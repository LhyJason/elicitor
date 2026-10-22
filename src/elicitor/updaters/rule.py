"""Keyword matcher: the no-LLM baseline updater (SPEC Decision Q6 A).

The answer is searched for the keywords of the asked slot's values. A keyword preceded by a
negation word in the same clause ("not the beach") counts against its value. For ordinal slots
with no keyword match, numbers in the answer are averaged and placed in a bin ("about 150 a day").

The likelihood follows the answer model of SPEC Q7 A: the answer names the true value with
probability rho, otherwise a uniformly random one. For a slot with n values, a value that was
named keeps weight 1 and every other value gets

    eps = (1 - rho) / (n * rho + 1 - rho),

which is P(answer names a | true value b != a) / P(answer names a | true value a). A negated
value is multiplied by eps. An answer with no match leaves the belief unchanged.
"""

from __future__ import annotations

import bisect
import functools
import re
from typing import Any

from elicitor.belief import Belief
from elicitor.questions import Question
from elicitor.schema import ORDINAL, Slot

NEGATIONS = frozenset(
    {"not", "no", "never", "don't", "dont", "doesn't", "isn't", "without", "hate", "avoid", "skip"}
)
NEGATION_WINDOW = 3  # words before the keyword, within the same clause
_CLAUSE_END = re.compile(r"[.,;:!?]|\bbut\b")
_WORD = re.compile(r"[\w']+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


@functools.cache
def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    # Lookarounds instead of \b so that keywords starting with "$" still match.
    return re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)")


def _is_negated(text: str, start: int) -> bool:
    before = text[:start]
    clause_breaks = list(_CLAUSE_END.finditer(before))
    if clause_breaks:
        before = before[clause_breaks[-1].end() :]
    words = _WORD.findall(before)[-NEGATION_WINDOW:]
    return any(w in NEGATIONS for w in words)


def find_mentions(slot: Slot, answer: str) -> tuple[list[str], list[str]]:
    """Values named in the answer, split into (positive, negated), in schema order."""
    text = answer.lower().replace("’", "'")
    positive, negated = set(), set()
    for value in slot.values:
        for keyword in value.keywords:
            for m in _keyword_pattern(keyword).finditer(text):
                (negated if _is_negated(text, m.start()) else positive).add(value.name)
    negated -= positive
    order = slot.value_names
    return [v for v in order if v in positive], [v for v in order if v in negated]


def number_bin(slot: Slot, answer: str) -> tuple[float, str] | None:
    """Mean of the numbers in the answer and the ordinal bin it falls in."""
    numbers = [float(x) for x in _NUMBER.findall(answer.replace(",", ""))]
    if not numbers:
        return None
    mean = sum(numbers) / len(numbers)
    return mean, slot.value_names[bisect.bisect_right(slot.edges, mean)]


class RuleUpdater:
    name = "rule"

    def __init__(self, rho: float = 0.8):
        if not 0 < rho < 1:
            raise ValueError("rho must be strictly between 0 and 1")
        self.rho = rho

    def eps(self, n_values: int) -> float:
        return (1 - self.rho) / (n_values * self.rho + 1 - self.rho)

    def update(
        self, belief: Belief, question: Question, answer: str
    ) -> tuple[Belief, dict[str, Any]]:
        slot = belief.schema.slot(question.slot)
        positive, negated = find_mentions(slot, answer)
        record: dict[str, Any] = {"slot": slot.name, "positive": positive, "negated": negated}
        if not positive and not negated and slot.kind == ORDINAL:
            found = number_bin(slot, answer)
            if found is not None:
                record["number"], value = found
                positive = [value]
                record["positive"] = positive
        if not positive and not negated:
            record["matched"] = False
            return belief, record

        eps = self.eps(len(slot.values))
        likelihood = []
        for value in slot.value_names:
            weight = 1.0 if not positive or value in positive else eps
            if value in negated:
                weight *= eps
            likelihood.append(weight)
        record["matched"] = True
        return belief.update(slot.name, likelihood), record
