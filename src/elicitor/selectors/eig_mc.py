"""Monte Carlo EIG with the agent's own LLM (SPEC §7, Decision Q7 B; details in M5).

A shortlist of the best questions by analytic EIG is re-scored. For each shortlisted question
and each value v of its slot with belief mass, the agent imagines `samples` answers of a user
whose preference is v (role `imagine`) and reads each with the LLM updater; the estimate is

    EIG(q) = sum_v b(v) * mean_j [ H(b) - H(b updated with answer v_j) ].

The imagined answers are seeded by (replicate seed, question, value, j) only and neither prompt
sees the belief, so both calls hit the cache across turns and personas.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from elicitor.belief import Belief
from elicitor.llm import LLMClient, LLMParseError
from elicitor.phrasing import template_question
from elicitor.questions import Question
from elicitor.rng import derive_seed
from elicitor.selectors.eig import TIE, EIGSelector
from elicitor.updaters.llm import LLMUpdater

if TYPE_CHECKING:
    from elicitor.episode import Turn


class MCEIGSelector:
    name = "eig_mc"

    def __init__(
        self,
        client: LLMClient,
        seed: int,
        rho: float | Mapping[str, float] = 0.8,
        shortlist: int = 5,
        samples: int = 3,
    ) -> None:
        if shortlist < 1 or samples < 1:
            raise ValueError("shortlist and samples must be positive")
        self.client = client
        self.seed = seed
        self.analytic = EIGSelector(rho=rho)
        self.shortlist = shortlist
        self.samples = samples
        self.reader = LLMUpdater(client)

    def imagine(self, belief: Belief, question: Question, value: str, j: int) -> str:
        slot = belief.schema.slot(question.slot)
        key = json.dumps(question.to_dict(), sort_keys=True)
        context = {
            "slot": slot,
            "q": question,
            "value": value,
            "topic": slot.description.rstrip("."),
            "preference": slot.value(value).label,
            "question": template_question(slot, question),
        }
        seed = derive_seed(self.seed, "imagine", key, value, j) % 2**31
        try:
            reply = self.client.call("imagine", context, seed=seed)
        except LLMParseError:
            return ""
        return reply.answer.strip()  # type: ignore[attr-defined]

    def estimate(self, belief: Belief, question: Question) -> float:
        slot = belief.schema.slot(question.slot)
        text = template_question(slot, question)
        prior_h = belief.entropy(slot.name)
        total = 0.0
        for value, p in zip(slot.value_names, belief.probs(slot.name), strict=True):
            if p <= 0:
                continue
            gains = []
            for j in range(self.samples):
                answer = self.imagine(belief, question, value, j)
                posterior, _ = self.reader.update(belief, question, text, answer)
                gains.append(prior_h - posterior.entropy(slot.name))
            total += p * sum(gains) / len(gains)
        return total

    def choose(
        self, belief: Belief, history: Sequence[Turn], rng: random.Random
    ) -> Question | None:
        scored = self.analytic.scores(belief, history)
        # Stable sort: equal analytic EIGs keep candidate order.
        ranked = sorted(scored, key=lambda item: -round(item[1], 12))[: self.shortlist]
        best: tuple[Question, float] | None = None
        for q, _ in ranked:
            gain = self.estimate(belief, q)
            if best is None or gain > best[1] + TIE:
                best = (q, gain)
        return best[0] if best else None
