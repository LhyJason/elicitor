"""The answer model behind the oracle, the rule updater and the analytic EIG (SPEC §7, Q7 A).

A user with true value v gives the noise-free outcome f_q(v) with probability rho, otherwise an
outcome drawn uniformly from all of the question's outcomes (possibly f_q(v) again):

    P(w | v) = rho * [w == f_q(v)] + (1 - rho) / m,    m = number of outcomes.

rho may differ by question type. EIG(q) = H(A) - H(A | V), in bits, with P(A) the belief's
predictive distribution of the outcome.
"""

from __future__ import annotations

import random
from collections.abc import Mapping

from elicitor.belief import Belief, entropy
from elicitor.questions import KINDS, Question, noise_free_outcome, outcomes
from elicitor.schema import Slot


def parse_rho(rho: float | Mapping[str, float]) -> dict[str, float]:
    """One rho for every question type, or a table with exactly one entry per type."""
    if isinstance(rho, Mapping):
        if set(rho) != set(KINDS):
            raise ValueError(f"a rho table needs exactly the keys {list(KINDS)}")
        table = {k: float(v) for k, v in rho.items()}
    else:
        table = dict.fromkeys(KINDS, float(rho))
    if any(not 0 <= r <= 1 for r in table.values()):
        raise ValueError("rho must be in [0, 1]")
    return table


class AnswerModel:
    def __init__(self, rho: float | Mapping[str, float] = 0.8):
        self.rho = parse_rho(rho)

    def outcome_probs(self, slot: Slot, question: Question, value: str) -> list[float]:
        """P(outcome | true value), in the order of `outcomes(slot, question)`."""
        names = outcomes(slot, question)
        rho = self.rho[question.kind]
        base = (1 - rho) / len(names)
        truthful = noise_free_outcome(question, value)
        return [base + rho * (w == truthful) for w in names]

    def sample(self, slot: Slot, question: Question, value: str, rng: random.Random) -> str:
        # For open questions this draws exactly as the M1 oracle did, so M1 runs are unchanged.
        if rng.random() >= self.rho[question.kind]:
            return rng.choice(outcomes(slot, question))
        return noise_free_outcome(question, value)

    def eps(self, slot: Slot, question: Question) -> float:
        """P(outcome w | v with f(v) != w) / P(outcome w | v with f(v) == w); 1 when rho = 0."""
        rho = self.rho[question.kind]
        m = len(outcomes(slot, question))
        return (1 - rho) / (m * rho + 1 - rho)

    def predictive(self, belief: Belief, question: Question) -> list[float]:
        """P(outcome) under the belief, in the order of `outcomes(slot, question)`."""
        slot = belief.schema.slot(question.slot)
        names = outcomes(slot, question)
        rho = self.rho[question.kind]
        probs = [(1 - rho) / len(names)] * len(names)
        index = {w: i for i, w in enumerate(names)}
        for value, p in zip(slot.value_names, belief.probs(slot.name), strict=True):
            probs[index[noise_free_outcome(question, value)]] += rho * p
        return probs

    def eig(self, belief: Belief, question: Question) -> float:
        """Expected information gain in bits: H(A) - H(A | V)."""
        slot = belief.schema.slot(question.slot)
        conditional = sum(
            p * entropy(self.outcome_probs(slot, question, value))
            for value, p in zip(slot.value_names, belief.probs(slot.name), strict=True)
            if p > 0
        )
        # Clipped at 0: H(A) - H(A | V) can come out at -1e-16 from rounding.
        return max(0.0, entropy(self.predictive(belief, question)) - conditional)
