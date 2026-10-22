"""Ask the unasked slot whose belief has the largest entropy."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

from elicitor.belief import Belief
from elicitor.questions import Question
from elicitor.selectors import unasked_slots

if TYPE_CHECKING:
    from elicitor.episode import Turn


class MaxEntropySelector:
    name = "max_entropy"

    def choose(
        self, belief: Belief, history: Sequence[Turn], rng: random.Random
    ) -> Question | None:
        remaining = unasked_slots(belief, history)
        if not remaining:
            return None
        # max() keeps the first of equal entropies, so ties go to schema order.
        return Question(max(remaining, key=belief.entropy))
