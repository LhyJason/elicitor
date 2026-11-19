"""A dialogue driven from outside (SPEC Q21): the caller relays each question to a real user and
brings back the answer.

    session = Session.from_config()               # the travel domain that ships with the package
    while (question := session.next_question()) is not None:
        session.observe(input(question + "\\n> "))
    for rec in session.recommend(3):
        print(rec.item.id, round(rec.expected_utility, 2), rec.item.attrs)

The loop is the episode loop of SPEC §11 without a persona, a simulator or metrics: there is no
ground truth. `episode.run_episode` stays the experiments' loop and is not built on this class,
so the published runs do not depend on it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from elicitor import resources
from elicitor.belief import Belief
from elicitor.catalog import Item, load_catalog
from elicitor.episode import BUDGET, CONFIDENT, EXHAUSTED, Turn
from elicitor.experiments.config import ExperimentConfig, load_config
from elicitor.phrasing import Phraser, make_phraser
from elicitor.questions import Question
from elicitor.recommender import Recommender
from elicitor.rng import derive_rng
from elicitor.schema import Schema, load_schema
from elicitor.selectors import Selector, make_selector
from elicitor.updaters import Updater, make_updater

if TYPE_CHECKING:
    from elicitor.llm import LLMClient


@dataclass(frozen=True)
class Recommendation:
    item: Item
    expected_utility: float


def understood(parsed: Mapping[str, Any]) -> bool:
    """False when the updater read nothing from an answer: no keyword matched (rule), or no
    usable reply (LLM). `parsed` is a turn's record of what was read."""
    return parsed.get("matched", True) and "failed" not in parsed and "skipped" not in parsed


class Session:
    """One dialogue: `next_question()` gives the next question to put to the user, `observe()`
    takes their answer and updates the belief, `recommend()` ranks the catalog.

    `selector`, `updater` and `phraser` are names (as in an experiment file) or objects with the
    methods of the `Selector`, `Updater` and `Phraser` protocols. The LLM components (`llm`
    updater or phraser, `eig_mc` selector) need `client`. `budget` is the largest number of
    answered questions; None means no limit. `label` names the random streams, so two sessions
    with the same seed and label ask the same questions.
    """

    def __init__(
        self,
        schema: Schema,
        *,
        selector: str | Selector = "eig",
        updater: str | Updater = "rule",
        phraser: str | Phraser = "tpl",
        catalog: Sequence[Item] | None = None,
        budget: int | None = 5,
        tau: float = 0.9,
        seed: int = 0,
        client: LLMClient | None = None,
        selector_params: Mapping[str, Any] | None = None,
        updater_params: Mapping[str, Any] | None = None,
        label: str = "session",
    ) -> None:
        if budget is not None and budget < 1:
            raise ValueError("budget must be positive, or None for no limit")
        self.schema = schema
        self.selector: Selector = (
            make_selector(selector, client=client, seed=seed, **(selector_params or {}))
            if isinstance(selector, str)
            else selector
        )
        self.updater: Updater = (
            make_updater(updater, client=client, **(updater_params or {}))
            if isinstance(updater, str)
            else updater
        )
        self.phraser: Phraser = (
            make_phraser(phraser, schema, client) if isinstance(phraser, str) else phraser
        )
        self.budget = budget
        self.tau = tau
        self.seed = seed
        self.label = label
        self.turns: list[Turn] = []
        self.stop_reason = ""  # set when `next_question` returns None
        self._belief = Belief.prior(schema)
        self._recommender = Recommender(schema, catalog) if catalog is not None else None
        self._pending: tuple[Question, str] | None = None
        self._times_asked: dict[str, int] = {}
        self._selector_rng = derive_rng(seed, label, self.arm, "selector")

    @classmethod
    def from_config(
        cls,
        config: ExperimentConfig | str | Path | None = None,
        *,
        selector: str | None = None,
        updater: str | None = None,
        phraser: str | None = None,
        budget: int | None = None,
        client: LLMClient | None = None,
        label: str = "session",
    ) -> Session:
        """A session on an experiment file's domain, with its seed, tau and component settings.

        The arguments override the first entry of the file's `selectors`, `updaters`, `phrasers`
        and `budgets`. With no file, the demo file that ships with the package is read: the
        travel domain, `eig` with the rule updater, no LLM.
        """
        if config is None:
            config = resources.demo_config()
        if not isinstance(config, ExperimentConfig):
            config = load_config(config)
        selector = selector or config.selectors[0]
        updater = updater or config.updaters[0]
        return cls(
            load_schema(config.schema_path),
            selector=selector,
            updater=updater,
            phraser=phraser or config.phrasers[0],
            catalog=load_catalog(config.catalog_path),
            budget=budget or config.budgets[0],
            tau=config.tau,
            seed=config.seed,
            client=client,
            selector_params=config.selector_params.get(selector, {}),
            updater_params=config.updater_params.get(updater, {}),
            label=label,
        )

    @property
    def arm(self) -> str:
        """The components' names, written like an experiment arm."""
        return f"sel={self.selector.name},upd={self.updater.name},phr={self.phraser.name}"

    @property
    def belief(self) -> Belief:
        return self._belief

    @property
    def pending(self) -> Question | None:
        """The question waiting for an answer, if any."""
        return self._pending[0] if self._pending else None

    @property
    def done(self) -> bool:
        """True once `next_question` has returned None."""
        return bool(self.stop_reason)

    def next_question(self) -> str | None:
        """The wording of the next question, or None when the dialogue is over; `stop_reason`
        then says why (budget, confident or exhausted, as in an episode).

        Until the question is answered, calling again returns the same wording. The stopping
        rules are checked on every call, so a session that stopped on its budget goes on after
        `budget` is raised.
        """
        if self._pending is not None:
            return self._pending[1]
        self.stop_reason = ""
        if self.budget is not None and len(self.turns) >= self.budget:
            self.stop_reason = BUDGET
            return None
        if self._belief.is_confident(self.tau):
            self.stop_reason = CONFIDENT
            return None
        question = self.selector.choose(self._belief, self.turns, self._selector_rng)
        if question is None:
            self.stop_reason = EXHAUSTED
            return None
        n = self._times_asked.get(question.slot, 0)
        self._times_asked[question.slot] = n + 1
        rng = derive_rng(self.seed, self.label, "phrase", question.slot, n)
        text = self.phraser.render(question, rng)
        self._pending = (question, text)
        return text

    def observe(self, answer: str) -> Turn:
        """Update the belief from the user's answer to the pending question.

        The returned turn holds the question, the answer, what the updater read from it
        (`parsed`; see `understood`) and the new belief. Its `metrics` are empty: a session has
        no ground truth. If the updater raises, the question stays pending.
        """
        if self._pending is None:
            raise RuntimeError("no question is waiting for an answer; call next_question() first")
        question, text = self._pending
        self._belief, parsed = self.updater.update(self._belief, question, text, answer)
        self._pending = None
        turn = Turn(len(self.turns) + 1, question, text, answer, parsed, self._belief, {})
        self.turns.append(turn)
        return turn

    def recommend(self, k: int = 3) -> list[Recommendation]:
        """The `k` catalog items of highest expected utility under the current belief."""
        if self._recommender is None:
            raise ValueError("this session has no catalog to recommend from")
        utilities = self._recommender.expected_utilities(self._belief)
        ranking = self._recommender.rank(self._belief)[:k]
        return [Recommendation(self._recommender.catalog[i], utilities[i]) for i in ranking]
