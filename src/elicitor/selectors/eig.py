"""Ask the question with the largest expected information gain (SPEC §7, Decision Q7 A).

Candidates are every question type allowed on every slot (`questions.candidates`). EIG is exact
under the answer model of `answer_model`. The same question is never asked twice, since a real
user would repeat the same answer, but another question on a slot already asked is allowed: with
noisy answers a yes/no check after an open question can be worth more than a new slot.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from elicitor.answer_model import AnswerModel
from elicitor.belief import Belief
from elicitor.questions import KINDS, OPEN, Question, candidates

if TYPE_CHECKING:
    from elicitor.episode import Turn

# EIGs closer than this count as tied, so float rounding never decides between equal questions.
TIE = 1e-12


class EIGSelector:
    name = "eig"

    def __init__(
        self, kinds: Sequence[str] = KINDS, rho: float | Mapping[str, float] = 0.8
    ) -> None:
        unknown = set(kinds) - set(KINDS)
        if unknown or not kinds:
            raise ValueError(f"kinds must be a non-empty subset of {list(KINDS)}")
        self.kinds = tuple(k for k in KINDS if k in kinds)
        self.model = AnswerModel(rho)

    def scores(self, belief: Belief, history: Sequence[Turn]) -> list[tuple[Question, float]]:
        """EIG of every candidate not asked yet, in candidate order (schema order, then type)."""
        asked = {turn.question for turn in history}
        return [
            (q, self.model.eig(belief, q))
            for slot in belief.schema.slots
            for q in candidates(slot, self.kinds)
            if q not in asked
        ]

    def choose(
        self, belief: Belief, history: Sequence[Turn], rng: random.Random
    ) -> Question | None:
        best: tuple[Question, float] | None = None
        for q, gain in self.scores(belief, history):
            if best is None or gain > best[1] + TIE:
                best = (q, gain)
        return best[0] if best else None


class OpenEIGSelector(EIGSelector):
    """`eig` restricted to open questions: a diagnostic arm for H2 (SPEC §7)."""

    name = "eig_open"

    def __init__(self, rho: float | Mapping[str, float] = 0.8) -> None:
        super().__init__(kinds=(OPEN,), rho=rho)
