"""Descriptive tables for H3 and H4 of the frozen hypotheses (SPEC §13 M6, Decision Q13).

Neither table is a test: the style table splits the LLM − rule updater gap by answer style
(12 eval personas per style), and the correlation table relates final belief quality to regret
across episodes, which are not independent within a persona. So there are CIs but no p-values.
"""

from __future__ import annotations

import itertools
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from elicitor.analysis.stats import bootstrap_mean_ci
from elicitor.experiments.config import Arm
from elicitor.personas import STYLES, load_personas

if TYPE_CHECKING:
    from elicitor.analysis.report import Run

STYLE_METRICS = ("mass_on_truth", "map_accuracy")
CORRELATION_METRICS = ("mass_on_truth", "map_accuracy")  # each against final regret


def persona_styles(run: Run, root: str | Path) -> dict[str, str] | None:
    """Style of every persona of the run, from `data/<domain>/personas.jsonl` under `root`.

    None when the file is missing or does not cover the run's personas: the episodes do not
    store styles, so the style table is skipped rather than guessed.
    """
    domain = tomllib.loads(run.state.get("config", "")).get("domain")
    path = Path(root) / "data" / str(domain) / "personas.jsonl"
    if domain is None or not path.exists():
        return None
    styles = {p.id: p.style for p in load_personas(path)}
    if not set(run.state["personas"]) <= set(styles):
        return None
    return styles


def updater_pairs(arms: list[str]) -> list[tuple[str, str]]:
    """(rule arm, llm arm) for every selector and phraser that has both updaters."""
    names = set(arms)
    pairs = []
    for name in arms:
        arm = Arm.parse(name)
        if arm.updater != "llm":
            continue
        rule = Arm(arm.selector, "rule", arm.phraser).name
        if rule in names:
            pairs.append((rule, name))
    return pairs


def _persona_means(run: Run, metric: str) -> dict[tuple[str, int, str], float]:
    """Final metric averaged over replicates: {(arm, budget, persona): mean}."""
    groups: dict[tuple[str, int, str], list[float]] = {}
    for e in run.episodes:
        groups.setdefault((e["arm"], e["budget"], e["persona"]), []).append(
            float(e["final_metrics"][metric])
        )
    return {key: float(np.mean(v)) for key, v in groups.items()}


def style_table(run: Run, styles: dict[str, str], resamples: int) -> list[dict[str, Any]]:
    rows = []
    for metric in STYLE_METRICS:
        means = _persona_means(run, metric)
        for (rule, llm), budget, style in itertools.product(
            updater_pairs(run.arms), run.budgets, STYLES
        ):
            personas = sorted(
                p
                for p in run.state["personas"]
                if styles[p] == style and (rule, budget, p) in means and (llm, budget, p) in means
            )
            if not personas:
                continue
            a = np.array([means[(rule, budget, p)] for p in personas])
            b = np.array([means[(llm, budget, p)] for p in personas])
            low, high = bootstrap_mean_ci(b - a, resamples)
            arm = Arm.parse(llm)
            rows.append(
                {"metric": metric, "selector": arm.selector, "phraser": arm.phraser,
                 "budget": budget, "style": style, "personas": len(personas),
                 "rule": a.mean(), "llm": b.mean(), "gap": (b - a).mean(),
                 "ci_low": low, "ci_high": high}
            )  # fmt: skip
    return rows


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation; NaN when either side is constant (no ranking to compare)."""
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y).statistic)


def correlation_table(run: Run) -> list[dict[str, Any]]:
    rows = []
    for arm, budget in itertools.product(run.arms, run.budgets):
        episodes = [e for e in run.episodes if e["arm"] == arm and e["budget"] == budget]
        if not episodes or "regret" not in episodes[0]["final_metrics"]:
            continue
        regret = np.array([e["final_metrics"]["regret"] for e in episodes], dtype=float)
        row: dict[str, Any] = {
            "arm": arm,
            "budget": budget,
            "episodes": len(episodes),
            "zero_regret_share": float((regret == 0).mean()),
        }
        for metric in CORRELATION_METRICS:
            x = np.array([e["final_metrics"][metric] for e in episodes], dtype=float)
            row[f"rho_{metric}"] = spearman(x, regret)
        rows.append(row)
    return rows


def _rho(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{value:+.2f}"


def markdown(styles: list[dict[str, Any]] | None, correlations: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    if styles:
        lines += [
            "",
            "## Updater gap by answer style (H3, descriptive)",
            "",
            "LLM − rule updater, final value, persona means (replicates averaged first) with 95 %"
            " bootstrap CIs over the personas of each style. No test.",
        ]
        present = [s for s in STYLES if any(r["style"] == s for r in styles)]
        for metric in STYLE_METRICS:
            rows = [r for r in styles if r["metric"] == metric]
            if not rows:
                continue
            n = {s: next(r["personas"] for r in rows if r["style"] == s) for s in present}
            lines += ["", f"### {metric}", ""]
            header = " | ".join(f"{s} (n={n[s]})" for s in present)
            lines.append(f"| selector | phraser | K | {header} |")
            lines.append("|---|---|---|" + "---|" * len(present))
            cell = {(r["selector"], r["phraser"], r["budget"], r["style"]): r for r in rows}
            keys = dict.fromkeys((r["selector"], r["phraser"], r["budget"]) for r in rows)
            for sel, phr, k in keys:
                entries = []
                for s in present:
                    r = cell.get((sel, phr, k, s))
                    entries.append(
                        f"{r['gap']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]" if r else "—"
                    )
                lines.append(f"| `{sel}` | `{phr}` | {k} | " + " | ".join(entries) + " |")
    if correlations:
        lines += [
            "",
            "## Final belief vs regret (H4, descriptive)",
            "",
            "Spearman correlation over episodes (persona × replicate) between a final belief metric"
            " and final regret; negative means a better belief goes with lower regret. Episodes of"
            " one persona are not independent, so no p-value.",
            "",
            "| arm | K | episodes | zero regret | ρ(mass_on_truth, regret)"
            " | ρ(map_accuracy, regret) |",
            "|---|---|---|---|---|---|",
        ]
        for r in correlations:
            lines.append(
                f"| `{r['arm']}` | {r['budget']} | {r['episodes']} | "
                f"{r['zero_regret_share']:.2f} | {_rho(r['rho_mass_on_truth'])} | "
                f"{_rho(r['rho_map_accuracy'])} |"
            )
    return lines
