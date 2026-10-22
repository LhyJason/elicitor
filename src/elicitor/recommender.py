"""Rank catalog items by expected utility under the belief (SPEC §10, Decision Q10 B).

True utility of an item for a user is the mean over slots of a match score in [0, 1]:
categorical slots score 1 when the item has the user's value or the user has no preference,
else 0; ordinal slots score 1 - |item bin - user bin| / (bins - 1). Because the belief is
factorized, the expected utility of an item is the same mean with each slot's match score
averaged over that slot's belief.
"""

from __future__ import annotations

from collections.abc import Sequence

from elicitor.belief import Belief
from elicitor.catalog import Item
from elicitor.schema import AT_MOST, NO_PREFERENCE, ORDINAL, Schema, Slot

_DIGITS = 12  # expected utilities are compared at this many decimals when ranking


def match(slot: Slot, item_value: str, true_value: str) -> float:
    if slot.kind == ORDINAL:
        distance = abs(slot.index(item_value) - slot.index(true_value))
        return 1 - distance / (len(slot.values) - 1)
    return float(item_value == true_value or true_value == NO_PREFERENCE)


def utility(schema: Schema, attrs: dict[str, str], truth: dict[str, str]) -> float:
    return sum(match(s, attrs[s.name], truth[s.name]) for s in schema.slots) / len(schema.slots)


def violations(schema: Schema, attrs: dict[str, str], truth: dict[str, str]) -> int:
    """Constraint slots (the ordinal ones) the item breaks (SPEC §3, §12)."""
    count = 0
    for slot in schema.slots:
        if slot.kind != ORDINAL:
            continue
        item, user = slot.index(attrs[slot.name]), slot.index(truth[slot.name])
        count += item > user if slot.constraint == AT_MOST else item != user
    return count


class Recommender:
    def __init__(self, schema: Schema, catalog: Sequence[Item]):
        if not catalog:
            raise ValueError("the catalog is empty")
        self.schema = schema
        self.catalog = list(catalog)
        # Per slot: one row per concrete item value, holding its match score against each
        # possible true value; and the row index of every item, in catalog order.
        self._match = {
            s.name: [[match(s, iv, tv) for tv in s.value_names] for iv in s.concrete_values]
            for s in schema.slots
        }
        self._rows = {
            s.name: [s.concrete_values.index(item.attrs[s.name]) for item in self.catalog]
            for s in schema.slots
        }
        self._true_utilities: dict[tuple[str, ...], list[float]] = {}

    def expected_utilities(self, belief: Belief) -> list[float]:
        """Expected utility of every item, in catalog order."""
        total = [0.0] * len(self.catalog)
        for slot in self.schema.slots:
            probs = belief.probs(slot.name)
            # Expected match of each item value under this slot's belief, then one lookup per item.
            expected = [
                sum(p * m for p, m in zip(probs, row, strict=True))
                for row in self._match[slot.name]
            ]
            total = [t + expected[r] for t, r in zip(total, self._rows[slot.name], strict=True)]
        n = len(self.schema.slots)
        return [t / n for t in total]

    def rank(self, belief: Belief) -> list[int]:
        """Item indices, best expected utility first; ties keep catalog order.

        Many items tie exactly in theory but differ by ~1e-16 in floating point, depending on
        summation order; rounding first lets catalog order, not rounding noise, break those ties.
        """
        eu = [round(x, _DIGITS) for x in self.expected_utilities(belief)]
        return sorted(range(len(self.catalog)), key=lambda i: -eu[i])

    def true_utilities(self, truth: dict[str, str]) -> list[float]:
        """True utility of every item for one user, in catalog order (cached per user)."""
        key = tuple(truth[name] for name in self.schema.slot_names)
        if key not in self._true_utilities:
            self._true_utilities[key] = [
                utility(self.schema, item.attrs, truth) for item in self.catalog
            ]
        return self._true_utilities[key]
