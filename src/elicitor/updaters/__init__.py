"""Belief updaters (SPEC §6): turn a free-text answer into a new belief.

`update` returns the new belief and a JSON-ready record of what was read from the answer.
"""

from __future__ import annotations

from typing import Any, Protocol

from elicitor.belief import Belief
from elicitor.questions import Question


class Updater(Protocol):
    name: str

    def update(
        self, belief: Belief, question: Question, answer: str
    ) -> tuple[Belief, dict[str, Any]]: ...


def make_updater(name: str, **params: Any) -> Updater:
    from elicitor.updaters.rule import RuleUpdater

    registry: dict[str, type] = {"rule": RuleUpdater}
    if name not in registry:
        raise ValueError(f"unknown updater {name!r}; choose from {sorted(registry)}")
    return registry[name](**params)
