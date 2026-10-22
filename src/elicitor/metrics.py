"""Belief metrics against a persona's ground truth (SPEC §12). Outcome metrics come in M3."""

import math

from elicitor.belief import Belief

BELIEF_METRICS = ("mass_on_truth", "map_accuracy", "log_loss", "mean_entropy")

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
