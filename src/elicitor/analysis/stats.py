"""Paired statistics over personas (SPEC §13, statistics in detail).

Every function takes one value per persona (replicates already averaged), so n is the number of
personas. Differences are rounded to `DIFF_DECIMALS` before testing: a tie between two arms that
differ only by float noise must count as a tie, otherwise the Wilcoxon test sees a tiny nonzero
difference and scipy may switch between its exact and approximate methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

DIFF_DECIMALS = 10
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
CI_LEVEL = 0.95


@dataclass(frozen=True)
class PairedResult:
    n: int
    mean_a: float
    mean_b: float
    diff: float  # mean of A - B
    ci_low: float
    ci_high: float
    wins: int  # personas where A is better (by the metric's direction)
    ties: int
    losses: int
    statistic: float
    p: float


def bootstrap_mean_ci(
    values: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling personas with replacement.

    The same seed is used for every call, so a table does not depend on the order in which its
    rows were computed.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[idx].mean(axis=1)
    tail = (1 - CI_LEVEL) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def wilcoxon(diffs: np.ndarray) -> tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test; zero differences are dropped.

    Returns (statistic, p); p = 1 when every difference is zero.
    """
    nonzero = diffs[diffs != 0]
    if nonzero.size == 0:
        return 0.0, 1.0
    result = stats.wilcoxon(nonzero, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def paired_test(
    a: np.ndarray,
    b: np.ndarray,
    higher_is_better: bool = True,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> PairedResult:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired values must be two vectors of the same length")
    diffs = np.round(a - b, DIFF_DECIMALS)
    better = diffs if higher_is_better else -diffs
    low, high = bootstrap_mean_ci(diffs, resamples)
    statistic, p = wilcoxon(diffs)
    return PairedResult(
        n=int(a.size),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        diff=float(diffs.mean()),
        ci_low=low,
        ci_high=high,
        wins=int((better > 0).sum()),
        ties=int((better == 0).sum()),
        losses=int((better < 0).sum()),
        statistic=statistic,
        p=p,
    )


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjustment: the i-th smallest p (from 0) is multiplied by m - i, then the
    adjusted values are made non-decreasing in that order and capped at 1."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted
