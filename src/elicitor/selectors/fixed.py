"""Ask slots in schema order."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

from elicitor.belief import Belief
from elicitor.questions import Question
from elicitor.selectors import unasked_slots

if TYPE_CHECKING:
    from elicitor.episode import Turn


class FixedSelector:
    name = "fixed"

    def choose(
        self, belief: Belief, history: Sequence[Turn], rng: random.Random
    ) -> Question | None:
        remaining = unasked_slots(belief, history)
        return Question(remaining[0]) if remaining else None
