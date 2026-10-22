"""Per-turn curve figures (SPEC §12, §16): one metric and budget per figure, one line per arm.

Colour follows the selector and line style the updater and phraser, so an arm keeps its look in
every figure of a run, and colour never has to tell more than eight arms apart. The band is
±1.96 standard errors over personas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from elicitor.experiments.config import Arm  # noqa: E402

# Categorical colours in a fixed order (validated for colour-vision deficiency as a set).
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
LINESTYLES = ("solid", (0, (5, 2)), (0, (1, 1.5)), (0, (5, 1.5, 1, 1.5)))
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"

LABELS = {
    "mass_on_truth": "probability on the true value",
    "map_accuracy": "MAP accuracy",
    "log_loss": "log loss (bits)",
    "mean_entropy": "entropy (bits)",
    "regret": "regret of the top item",
    "hit@1": "hit@1",
    "hit@5": "hit@5",
    "hit@10": "hit@10",
    "violations": "constraint violations",
}


def arm_styles(arms: list[str]) -> dict[str, dict[str, Any]]:
    """Colour by selector, line style by (updater, phraser), both in order of first appearance."""
    parsed = {a: Arm.parse(a) for a in arms}
    selectors = list(dict.fromkeys(p.selector for p in parsed.values()))
    variants = list(dict.fromkeys((p.updater, p.phraser) for p in parsed.values()))
    if len(selectors) > len(COLORS) or len(variants) > len(LINESTYLES):
        raise ValueError("too many selectors or updater/phraser variants for one figure")
    return {
        a: {
            "color": COLORS[selectors.index(p.selector)],
            "linestyle": LINESTYLES[variants.index((p.updater, p.phraser))],
            "label": f"{p.selector} · {p.updater} · {p.phraser}",
        }
        for a, p in parsed.items()
    }


def plot_curves(
    rows: list[dict[str, Any]],
    metric: str,
    budget: int,
    arms: list[str],
    path: str | Path,
    title: str = "",
) -> Path:
    """`rows` are the curve-table rows of one metric and budget."""
    styles = arm_styles(arms)
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    for arm in arms:
        points = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["turn"])
        if not points:
            continue
        turns = [r["turn"] for r in points]
        mean = [r["mean"] for r in points]
        low = [r["mean"] - 1.96 * r["se"] for r in points]
        high = [r["mean"] + 1.96 * r["se"] for r in points]
        style = styles[arm]
        ax.fill_between(turns, low, high, color=style["color"], alpha=0.12, linewidth=0)
        ax.plot(
            turns, mean, color=style["color"], linestyle=style["linestyle"], linewidth=2,
            marker="o", markersize=4.5, markeredgecolor="white", markeredgewidth=1,
            label=style["label"],
        )  # fmt: skip
    ax.set_xlabel("turn (questions asked)", color=INK_MUTED)
    ax.set_ylabel(LABELS.get(metric, metric), color=INK_MUTED)
    ax.set_xticks(range(budget + 1))
    ax.set_title(f"{title}: {LABELS.get(metric, metric)}, K = {budget}", color=INK, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)
    ax.legend(title="selector · updater · phrasing", fontsize=8, title_fontsize=8, frameon=False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # No software or date stamp, so the same data gives the same file.
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
    return path
