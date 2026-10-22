"""Keyword matcher: the no-LLM baseline updater (SPEC Decision Q6 A).

The answer is searched for the keywords of the asked slot's values. A keyword preceded by a
negation word in the same clause ("not the beach") counts against its value. For ordinal slots
with no keyword match, numbers in the answer are averaged and placed in a bin ("about 150 a day").

What was found is then read as outcomes of the question (SPEC §6, §7): for an open question the
named values themselves; for a choice question the offered values named, or OTHER; for a yes/no
question the first yes or no phrase, or else the outcome of the values named.

The likelihood follows the answer model of SPEC Q7 A: the answer gives the noise-free outcome
with probability rho, otherwise a uniformly random one. For a question with m outcomes, a value
whose noise-free outcome was named keeps weight 1 and every other value gets

    eps = (1 - rho) / (m * rho + 1 - rho),

which is P(answer is w | f(true value) != w) / P(answer is w | f(true value) == w). A negated
outcome is multiplied by eps. An answer with no match leaves the belief unchanged.
"""

from __future__ import annotations

import bisect
import functools
import re
from collections.abc import Mapping
from typing import Any

from elicitor.answer_model import AnswerModel
from elicitor.belief import Belief
from elicitor.questions import (
    CHOICE,
    NO,
    OPEN,
    OTHER,
    YES,
    YESNO,
    Question,
    noise_free_outcome,
    outcomes,
)
from elicitor.schema import ORDINAL, Slot

NEGATIONS = frozenset(
    {"not", "no", "never", "don't", "dont", "doesn't", "isn't", "without", "hate", "avoid", "skip"}
)
NEGATION_WINDOW = 3  # words before the keyword, within the same clause
_CLAUSE_END = re.compile(r"[.,;:!?]|\bbut\b")
_WORD = re.compile(r"[\w']+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

# Yes/no and "none of them" phrases for closed questions; the earliest one in the answer counts.
YES_PHRASES = (
    "yes", "yeah", "yep", "definitely", "absolutely", "of course", "sounds good", "sounds great",
    "sounds perfect", "that would suit", "i'd love", "i would love",
)  # fmt: skip
NO_PHRASES = (
    "no", "nope", "not really", "rather not", "wouldn't", "would not", "not for me",
    "don't think so", "not at all",
)  # fmt: skip
NONE_PHRASES = ("none of", "none", "neither", "something else", "something different")


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


def _first_phrase(text: str, phrases: dict[str, tuple[str, ...]]) -> str | None:
    """The key of the phrase that occurs first in `text` (lower case), or None."""
    best: tuple[int, str] | None = None
    for key, options in phrases.items():
        for phrase in options:
            m = _keyword_pattern(phrase).search(text)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), key)
    return best[1] if best else None


def parse_outcomes(
    slot: Slot, question: Question, answer: str
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Outcomes the answer gives and rules out, plus what was matched (for the log)."""
    positive, negated = find_mentions(slot, answer)
    extra: dict[str, Any] = {}
    if not positive and not negated and slot.kind == ORDINAL:
        found = number_bin(slot, answer)
        if found is not None:
            extra["number"], value = found
            positive = [value]
    if question.kind == OPEN:
        return positive, negated, extra

    text = answer.lower().replace("’", "'")
    if question.kind == YESNO:
        said = _first_phrase(text, {YES: YES_PHRASES, NO: NO_PHRASES})
        if said is not None:
            return [said], [], extra
    names = outcomes(slot, question)
    pos = {noise_free_outcome(question, v) for v in positive}
    # Ruling out a value rules out its outcome only when no other value has that outcome
    # ("not the beach" rules out "yes" to "the beach?", but not "no").
    sole = {
        w for w in names
        if sum(noise_free_outcome(question, v) == w for v in slot.value_names) == 1
    }  # fmt: skip
    neg = {noise_free_outcome(question, v) for v in negated} & sole - pos
    if question.kind == CHOICE and not pos and _first_phrase(text, {OTHER: NONE_PHRASES}):
        pos = {OTHER}
    return [w for w in names if w in pos], [w for w in names if w in neg], extra


class RuleUpdater:
    name = "rule"

    def __init__(self, rho: float | Mapping[str, float] = 0.8):
        self.model = AnswerModel(rho)
        if any(not 0 < r < 1 for r in self.model.rho.values()):
            raise ValueError("rho must be strictly between 0 and 1")

    def update(
        self, belief: Belief, question: Question, text: str, answer: str
    ) -> tuple[Belief, dict[str, Any]]:
        slot = belief.schema.slot(question.slot)
        positive, negated, extra = parse_outcomes(slot, question, answer)
        record: dict[str, Any] = {"slot": slot.name, "positive": positive, "negated": negated}
        record.update(extra)
        if not positive and not negated:
            record["matched"] = False
            return belief, record

        eps = self.model.eps(slot, question)
        likelihood = []
        for value in slot.value_names:
            outcome = noise_free_outcome(question, value)
            weight = 1.0 if not positive or outcome in positive else eps
            if outcome in negated:
                weight *= eps
            likelihood.append(weight)
        record["matched"] = True
        return belief.update(slot.name, likelihood), record
