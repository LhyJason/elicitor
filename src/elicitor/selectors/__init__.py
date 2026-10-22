"""Question selection policies (SPEC §7).

A selector looks at the current belief and the turns so far and returns the next question, or
None when it has nothing left to ask. `fixed`, `random` and `max_entropy` ask open questions and
never ask a slot twice; re-asking only pays off with noisy answers, which the EIG selectors model
explicitly (they never repeat a question, but may ask another one on the same slot).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from elicitor.belief import Belief

if TYPE_CHECKING:
    from elicitor.episode import Turn
    from elicitor.questions import Question


class Selector(Protocol):
    name: str

    def choose(
        self, belief: Belief, history: Sequence[Turn], rng: random.Random
    ) -> Question | None: ...


def unasked_slots(belief: Belief, history: Sequence[Turn]) -> list[str]:
    """Slots not asked yet, in schema order."""
    asked = {turn.question.slot for turn in history}
    return [name for name in belief.schema.slot_names if name not in asked]


def make_selector(name: str, **params: Any) -> Selector:
    """`params` come from the experiment file's `[selector.<name>]` table."""
    from elicitor.selectors.eig import EIGSelector, OpenEIGSelector
    from elicitor.selectors.entropy import MaxEntropySelector
    from elicitor.selectors.fixed import FixedSelector
    from elicitor.selectors.random import RandomSelector

    registry: dict[str, type] = {
        "fixed": FixedSelector,
        "random": RandomSelector,
        "max_entropy": MaxEntropySelector,
        "eig": EIGSelector,
        "eig_open": OpenEIGSelector,
    }
    if name not in registry:
        raise ValueError(f"unknown selector {name!r}; choose from {sorted(registry)}")
    return registry[name](**params)
