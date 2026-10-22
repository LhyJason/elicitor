"""One dialogue between the agent and a simulated user (SPEC §11).

Recommendation and outcome scoring are added in M3; an M1 episode ends with the final belief.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from elicitor.belief import Belief
from elicitor.metrics import belief_metrics, leaked_slots
from elicitor.personas import Persona
from elicitor.phrasing import Phraser
from elicitor.questions import Question
from elicitor.rng import derive_rng
from elicitor.schema import Schema
from elicitor.selectors import Selector
from elicitor.simulator import Simulator
from elicitor.updaters import Updater

CONFIDENT = "confident"  # every slot reached tau before the budget ran out
BUDGET = "budget"  # asked K questions
EXHAUSTED = "exhausted"  # the selector had nothing left to ask


@dataclass(frozen=True)
class Turn:
    index: int
    question: Question
    text: str
    answer: str
    parsed: dict[str, Any]
    belief: Belief
    metrics: dict[str, float]
    leaks: tuple[str, ...] = ()  # unasked slots the answer mentions (SPEC §9)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.index,
            "question": self.question.to_dict(),
            "text": self.text,
            "answer": self.answer,
            "parsed": self.parsed,
            "leaks": list(self.leaks),
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "belief": self.belief.to_dict(),
        }


@dataclass
class Episode:
    persona_id: str
    arm: str
    replicate: int
    budget: int
    initial_metrics: dict[str, float]
    turns: list[Turn] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def final_metrics(self) -> dict[str, float]:
        return self.turns[-1].metrics if self.turns else self.initial_metrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona": self.persona_id,
            "arm": self.arm,
            "replicate": self.replicate,
            "budget": self.budget,
            "stop_reason": self.stop_reason,
            "n_turns": len(self.turns),
            "initial_metrics": {k: round(v, 6) for k, v in self.initial_metrics.items()},
            "final_metrics": {k: round(v, 6) for k, v in self.final_metrics.items()},
            "turns": [t.to_dict() for t in self.turns],
        }


def run_episode(
    *,
    schema: Schema,
    persona: Persona,
    arm: str,
    selector: Selector,
    updater: Updater,
    simulator: Simulator,
    phraser: Phraser,
    budget: int,
    tau: float,
    seed: int,
    replicate: int = 0,
) -> Episode:
    """`seed` is the replicate seed; all randomness is derived from it (SPEC §11)."""
    selector_rng = derive_rng(seed, persona.id, arm, "selector")
    belief = Belief.prior(schema)
    episode = Episode(
        persona_id=persona.id,
        arm=arm,
        replicate=replicate,
        budget=budget,
        initial_metrics=belief_metrics(belief, persona.truth),
    )
    times_asked: dict[str, int] = {}
    for index in range(1, budget + 1):
        if belief.is_confident(tau):
            episode.stop_reason = CONFIDENT
            break
        question = selector.choose(belief, episode.turns, selector_rng)
        if question is None:
            episode.stop_reason = EXHAUSTED
            break
        n = times_asked.get(question.slot, 0)
        times_asked[question.slot] = n + 1
        # Common random numbers: the same question gets the same noisy answer in every arm.
        # The wording (with LLM phrasing) follows the same rule.
        phrase_rng = derive_rng(seed, persona.id, "phrase", question.slot, n)
        answer_rng = derive_rng(seed, persona.id, "answer", question.slot, n)
        text = phraser.render(question, phrase_rng)
        answer = simulator.respond(question, text, persona, answer_rng)
        belief, parsed = updater.update(belief, question, text, answer)
        episode.turns.append(
            Turn(
                index=index,
                question=question,
                text=text,
                answer=answer,
                parsed=parsed,
                belief=belief,
                metrics=belief_metrics(belief, persona.truth),
                leaks=tuple(leaked_slots(schema, question.slot, answer)),
            )
        )
    else:
        episode.stop_reason = BUDGET
    return episode
