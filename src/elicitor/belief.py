"""Factorized belief: one categorical distribution per slot (SPEC §5, Decision Q5 A).

Entropies are in bits. A `Belief` is immutable; `update` returns a new one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from elicitor.schema import Schema


def entropy(probs: Sequence[float]) -> float:
    """Shannon entropy in bits; 0 log 0 is taken as 0."""
    return -sum(p * math.log2(p) for p in probs if p > 0)


def normalize(weights: Sequence[float]) -> tuple[float, ...]:
    total = sum(weights)
    if not total > 0:
        raise ValueError("weights must have a positive sum")
    return tuple(w / total for w in weights)


class Belief:
    def __init__(self, schema: Schema, probs: Mapping[str, Sequence[float]]):
        if set(probs) != set(schema.slot_names):
            raise ValueError("belief must cover exactly the schema's slots")
        self.schema = schema
        self._probs: dict[str, tuple[float, ...]] = {}
        for slot in schema.slots:
            p = probs[slot.name]
            if len(p) != len(slot.values) or any(x < 0 for x in p):
                raise ValueError(f"slot {slot.name}: bad probability vector")
            self._probs[slot.name] = normalize(p)

    @classmethod
    def prior(cls, schema: Schema) -> Belief:
        return cls(schema, {s.name: s.prior for s in schema.slots})

    def probs(self, slot: str) -> tuple[float, ...]:
        return self._probs[slot]

    def prob(self, slot: str, value: str) -> float:
        return self._probs[slot][self.schema.slot(slot).index(value)]

    def entropy(self, slot: str) -> float:
        return entropy(self._probs[slot])

    def max_prob(self, slot: str) -> float:
        return max(self._probs[slot])

    def map_value(self, slot: str) -> str:
        """Most probable value; ties go to the value listed first in the schema."""
        p = self._probs[slot]
        return self.schema.slot(slot).value_names[p.index(max(p))]

    def is_confident(self, tau: float) -> bool:
        """True when every slot puts at least `tau` on its most probable value (SPEC Q8)."""
        return all(max(p) >= tau for p in self._probs.values())

    def update(self, slot: str, likelihood: Sequence[float]) -> Belief:
        """Bayes rule on one slot: posterior is proportional to prior times likelihood."""
        prior = self._probs[slot]
        if len(likelihood) != len(prior) or any(x < 0 for x in likelihood):
            raise ValueError(f"slot {slot}: bad likelihood vector")
        posterior = [p * lik for p, lik in zip(prior, likelihood, strict=True)]
        if not sum(posterior) > 0:
            raise ValueError(f"slot {slot}: likelihood rules out every value with prior mass")
        probs = dict(self._probs)
        probs[slot] = posterior
        return Belief(self.schema, probs)

    def to_dict(self, digits: int = 6) -> dict[str, dict[str, float]]:
        return {
            s.name: {
                v: round(p, digits) for v, p in zip(s.value_names, self._probs[s.name], strict=True)
            }
            for s in self.schema.slots
        }
