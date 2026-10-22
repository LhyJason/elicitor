"""Belief updaters (SPEC §6): turn a free-text answer into a new belief.

`update` gets the question as asked (its text may vary with LLM phrasing) and the answer, and
returns the new belief and a JSON-ready record of what was read from the answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from elicitor.belief import Belief
from elicitor.questions import Question

if TYPE_CHECKING:
    from elicitor.llm import LLMClient


class Updater(Protocol):
    name: str

    def update(
        self, belief: Belief, question: Question, text: str, answer: str
    ) -> tuple[Belief, dict[str, Any]]: ...


def make_updater(name: str, client: LLMClient | None = None, **params: Any) -> Updater:
    from elicitor.updaters.rule import RuleUpdater

    if name == "rule":
        return RuleUpdater(**params)
    if name == "llm":
        from elicitor.updaters.llm import LLMUpdater

        if client is None:
            raise ValueError("the llm updater needs an LLM client")
        return LLMUpdater(client, **params)
    raise ValueError(f"unknown updater {name!r}; choose from ['llm', 'rule']")
