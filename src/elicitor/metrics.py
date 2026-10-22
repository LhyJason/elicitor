"""Belief and outcome metrics against a persona's ground truth (SPEC §12)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from elicitor.belief import Belief
from elicitor.schema import NO_PREFERENCE, Schema

if TYPE_CHECKING:
    from elicitor.recommender import Recommender

BELIEF_METRICS = ("mass_on_truth", "map_accuracy", "log_loss", "mean_entropy")
HIT_KS = (1, 5, 10)
OUTCOME_METRICS = ("regret", *(f"hit@{k}" for k in HIT_KS), "violations")

# Utilities closer than this count as equal when deciding which items are best.
_UTILITY_TIE = 1e-9

# Probability floor for log loss, so a value ruled out by mistake costs a finite amount.
_LOG_FLOOR = 1e-12


def belief_metrics(belief: Belief, truth: dict[str, str]) -> dict[str, float]:
    """Averages over slots. Log loss and entropy are in bits."""
    slots = belief.schema.slot_names
    mass = [belief.prob(s, truth[s]) for s in slots]
    n = len(slots)
    return {
        "mass_on_truth": sum(mass) / n,
        "map_accuracy": sum(belief.map_value(s) == truth[s] for s in slots) / n,
        "log_loss": sum(-math.log2(max(m, _LOG_FLOOR)) for m in mass) / n,
        "mean_entropy": sum(belief.entropy(s) for s in slots) / n,
    }


def outcome_metrics(
    recommender: Recommender, ranking: Sequence[int], truth: dict[str, str]
) -> dict[str, float]:
    """Regret and constraint violations of the top item, and hit@k (SPEC §12)."""
    from elicitor.recommender import violations

    utilities = recommender.true_utilities(truth)
    best = max(utilities)
    top = ranking[0]
    out = {"regret": best - utilities[top]}
    for k in HIT_KS:
        out[f"hit@{k}"] = float(any(utilities[i] >= best - _UTILITY_TIE for i in ranking[:k]))
    out["violations"] = float(violations(recommender.schema, recommender.catalog[top].attrs, truth))
    return out


def leaked_slots(schema: Schema, asked: str, answer: str) -> list[str]:
    """Unasked slots whose concrete values the answer names (SPEC §9 leakage check).

    Uses the rule matcher's keywords, so it is approximate. `no_preference` phrases are shared by
    every slot ("I don't mind") and are not counted.
    """
    from elicitor.updaters.rule import find_mentions

    leaks = []
    for slot in schema.slots:
        if slot.name == asked:
            continue
        positive, negated = find_mentions(slot, answer)
        if any(v != NO_PREFERENCE for v in positive + negated):
            leaks.append(slot.name)
    return leaks
