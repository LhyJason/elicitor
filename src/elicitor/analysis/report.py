"""Analysis of a run folder: arm tables, paired comparisons and per-turn curves (SPEC §13).

The unit is the persona. Every number below is computed in two steps: average an episode-level
value over the replicates of each persona, then summarize or test across personas. The analysis
reads the finished cells, so it also works on a run that is still going; committed outputs
(`results_summary/`, `docs/figures/`) are only written from complete, non-mock eval runs.
"""

from __future__ import annotations

import csv
import io
import itertools
import os
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from elicitor.analysis.stats import BOOTSTRAP_RESAMPLES, bootstrap_mean_ci, holm, paired_test
from elicitor.episode import CONFIDENT
from elicitor.experiments.config import Arm
from elicitor.experiments.store import RunStore, cell_id
from elicitor.metrics import BELIEF_METRICS, OUTCOME_METRICS

# Episode-level values besides the final metrics (SPEC §12, efficiency).
EXTRA_VALUES = ("turns", "confident_share", "leak_rate", "map_auc")
KNOWN_VALUES = (*BELIEF_METRICS, *OUTCOME_METRICS, *EXTRA_VALUES)
LOWER_IS_BETTER = {"log_loss", "mean_entropy", "regret", "violations", "turns"}
DEFAULT_METRICS = ("mass_on_truth", "map_accuracy", "log_loss", "regret", "hit@1", "violations",
                   "map_auc")  # fmt: skip
DEFAULT_CURVE_METRICS = ("mass_on_truth", "map_accuracy", "regret")
COMMITTED_DIRS = (("results_summary",), ("docs", "figures"))

Key = tuple[str, int]  # (arm, budget)


class RefusedToPublish(RuntimeError):
    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("refusing to write committed outputs: " + "; ".join(reasons))


@dataclass
class Run:
    root: Path
    state: dict[str, Any]
    episodes: list[dict[str, Any]]
    cells_done: int
    cells_total: int

    @property
    def complete(self) -> bool:
        return self.cells_done == self.cells_total

    @property
    def arms(self) -> list[str]:
        return list(self.state["arms"])

    @property
    def budgets(self) -> list[int]:
        return list(self.state["budgets"])


def load_run(run_dir: str | Path) -> Run:
    store = RunStore(run_dir)
    state = store.read_state()
    if state is None:
        raise ValueError(f"{run_dir} is not a run folder (no run_state.json); run it again")
    cells = [cell_id(arm, r) for arm in state["arms"] for r in range(state["replicates"])]
    done = [c for c in cells if store.is_done(c)]
    if not done:
        raise ValueError(f"{run_dir} has no finished cells yet")
    episodes = [e for c in done for e in store.read_cell(c)]
    return Run(Path(run_dir), state, episodes, len(done), len(cells))


# ---------------------------------------------------------------- episode-level values


def curve(episode: dict[str, Any], metric: str) -> list[float]:
    """The metric at turns 0..K; an episode that stopped early keeps its last value (SPEC §12)."""
    points = [episode["initial_metrics"][metric]]
    points += [t["metrics"][metric] for t in episode["turns"]]
    return points + [points[-1]] * (episode["budget"] + 1 - len(points))


def episode_value(episode: dict[str, Any], name: str) -> float:
    if name == "turns":
        return float(episode["n_turns"])
    if name == "confident_share":
        return float(episode["stop_reason"] == CONFIDENT)
    if name == "leak_rate":
        turns = episode["turns"]
        return sum(bool(t["leaks"]) for t in turns) / len(turns) if turns else 0.0
    if name == "map_auc":
        points = curve(episode, "map_accuracy")
        k = episode["budget"]
        return (sum(points) - (points[0] + points[-1]) / 2) / k
    return float(episode["final_metrics"][name])


def per_persona(
    episodes: Iterable[dict[str, Any]], fn: Callable[[dict[str, Any]], Any]
) -> dict[Key, dict[str, np.ndarray]]:
    """fn(episode) averaged over replicates: {(arm, budget): {persona: mean}}."""
    groups: dict[Key, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for e in episodes:
        groups[(e["arm"], e["budget"])][e["persona"]].append(fn(e))
    return {
        key: {p: np.mean(np.asarray(v, dtype=float), axis=0) for p, v in by_p.items()}
        for key, by_p in groups.items()
    }


def _vector(by_persona: dict[str, np.ndarray], personas: list[str]) -> np.ndarray:
    return np.array([by_persona[p] for p in personas], dtype=float)


# ---------------------------------------------------------------- tables


def default_comparisons(arms: list[str]) -> list[tuple[str, str]]:
    """Every pair of arms that differ in exactly one component; A is the arm listed later."""
    parsed = [Arm.parse(a) for a in arms]
    pairs = []
    for (i, x), (j, y) in itertools.combinations(enumerate(parsed), 2):
        differ = (x.selector != y.selector) + (x.updater != y.updater) + (x.phraser != y.phraser)
        if differ == 1:
            pairs.append((arms[j], arms[i]))
    return pairs


def arm_table(run: Run, metrics: list[str], resamples: int) -> list[dict[str, Any]]:
    names = list(dict.fromkeys([*metrics, "turns", "confident_share", "leak_rate"]))
    rows = []
    for name in names:
        values = per_persona(run.episodes, lambda e, n=name: episode_value(e, n))
        for arm, budget in itertools.product(run.arms, run.budgets):
            by_p = values.get((arm, budget))
            if not by_p:
                continue
            v = _vector(by_p, sorted(by_p))
            low, high = bootstrap_mean_ci(v, resamples)
            rows.append(
                {"arm": arm, "budget": budget, "metric": name, "personas": v.size,
                 "mean": v.mean(), "ci_low": low, "ci_high": high}
            )  # fmt: skip
    return rows


def comparison_table(
    run: Run, pairs: list[tuple[str, str]], metrics: list[str], resamples: int
) -> list[dict[str, Any]]:
    """Paired tests over the personas both arms have; one Holm family per metric."""
    rows = []
    for metric in metrics:
        values = per_persona(run.episodes, lambda e, m=metric: episode_value(e, m))
        family = []
        for (a, b), budget in itertools.product(pairs, run.budgets):
            va, vb = values.get((a, budget)), values.get((b, budget))
            if not va or not vb:
                continue
            personas = sorted(set(va) & set(vb))
            higher = metric not in LOWER_IS_BETTER
            r = paired_test(_vector(va, personas), _vector(vb, personas), higher, resamples)
            family.append(
                {"metric": metric, "budget": budget, "arm_a": a, "arm_b": b, "n": r.n,
                 "mean_a": r.mean_a, "mean_b": r.mean_b, "diff": r.diff, "ci_low": r.ci_low,
                 "ci_high": r.ci_high, "wins": r.wins, "ties": r.ties, "losses": r.losses,
                 "statistic": r.statistic, "p": r.p, "p_holm": 1.0,
                 "better": "higher" if higher else "lower"}
            )  # fmt: skip
        for row, p_holm in zip(family, holm([row["p"] for row in family]), strict=True):
            row["p_holm"] = p_holm
        rows += family
    return rows


def curve_table(run: Run, metrics: list[str]) -> list[dict[str, Any]]:
    rows = []
    for metric in metrics:
        values = per_persona(run.episodes, lambda e, m=metric: curve(e, m))
        for arm, budget in itertools.product(run.arms, run.budgets):
            by_p = values.get((arm, budget))
            if not by_p:
                continue
            matrix = _vector(by_p, sorted(by_p))  # personas x (K + 1)
            n = matrix.shape[0]
            se = matrix.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(matrix.shape[1])
            for turn in range(matrix.shape[1]):
                rows.append(
                    {"metric": metric, "arm": arm, "budget": budget, "turn": turn,
                     "personas": n, "mean": matrix[:, turn].mean(), "se": se[turn]}
                )  # fmt: skip
    return rows


# ---------------------------------------------------------------- output


def is_committed_path(path: str | Path) -> bool:
    parts = Path(path).resolve().parts
    return any(
        parts[i : i + len(d)] == d for d in COMMITTED_DIRS for i in range(len(parts) - len(d) + 1)
    )


def publish_problems(run: Run) -> list[str]:
    """Why this run may not feed committed tables (empty when it may)."""
    problems = []
    if not run.complete:
        problems.append(f"the run is incomplete ({run.cells_done} of {run.cells_total} cells)")
    if run.state.get("provider") == "mock":
        problems.append("the run used the mock LLM provider")
    if run.state.get("split") != "eval":
        problems.append(f"the run used the {run.state.get('split')!r} split, not 'eval'")
    if run.state.get("max_personas") is not None:
        problems.append(f"the run was cut to {run.state['max_personas']} personas")
    return problems


def _fmt(value: Any) -> Any:
    return f"{value:.6g}" if isinstance(value, float | np.floating) else value


def _csv(rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    if rows:
        writer = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({k: _fmt(v) for k, v in row.items()} for row in rows)
    return buf.getvalue()


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _markdown(
    run: Run,
    arms: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    metrics: list[str],
    figures: list[Path],
    out: Path,
) -> str:
    s = run.state
    lines = [
        f"# Results: {s['name']}",
        "",
        f"- Personas: {len(s['personas'])} ({s['split']} split), replicates: {s['replicates']}, "
        f"budgets: {', '.join(map(str, run.budgets))}",
        f"- Simulator: {s['simulator']}, LLM provider: {s['provider'] or 'none'}",
        f"- Cells finished: {run.cells_done} of {run.cells_total}"
        + ("" if run.complete else " (incomplete run)"),
        f"- Run fingerprint: `{s['fingerprint'][:16]}`, package version {s['version']}",
        "",
        "Means over personas (replicates averaged first) with 95 % bootstrap CIs.",
    ]
    cell = {(r["arm"], r["budget"], r["metric"]): r for r in arms}
    columns = list(dict.fromkeys([*metrics, "turns"]))
    for budget in run.budgets:
        lines += ["", f"## Budget K = {budget}", ""]
        lines.append("| arm | " + " | ".join(columns) + " |")
        lines.append("|---|" + "---|" * len(columns))
        for arm in run.arms:
            if (arm, budget, columns[0]) not in cell:
                continue
            entries = []
            for m in columns:
                r = cell[(arm, budget, m)]
                entries.append(f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
            lines.append(f"| `{arm}` | " + " | ".join(entries) + " |")
    if comparisons:
        lines += [
            "",
            "## Paired comparisons",
            "",
            "A − B per persona, Wilcoxon signed-rank (two-sided), Holm-adjusted within each metric."
            " W/T/L counts personas where A is better / tied / worse.",
        ]
        for metric in metrics:
            rows = [r for r in comparisons if r["metric"] == metric]
            if not rows:
                continue
            lines += ["", f"### {metric} ({rows[0]['better']} is better)", ""]
            lines.append("| A | B | K | A − B [95 % CI] | W/T/L | p | p (Holm) |")
            lines.append("|---|---|---|---|---|---|---|")
            for r in rows:
                lines.append(
                    f"| `{r['arm_a']}` | `{r['arm_b']}` | {r['budget']} | "
                    f"{r['diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
                    f"{r['wins']}/{r['ties']}/{r['losses']} | {_p(r['p'])} | {_p(r['p_holm'])} |"
                )
    if figures:
        lines += ["", "## Per-turn curves", ""]
        for path in figures:
            rel = Path(os.path.relpath(path, out)).as_posix()
            lines.append(f"![{path.stem}]({rel})")
    return "\n".join(lines) + "\n"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def analyze(
    run_dir: str | Path,
    out: str | Path | None = None,
    figures_dir: str | Path | None = None,
    publish: bool = False,
    root: str | Path = ".",
    make_figures: bool = True,
) -> list[Path]:
    """Write the analysis of a run; returns the files written.

    `publish` sends the output to `results_summary/<name>/` and the figures to
    `docs/figures/<name>/` under `root`. Writing anywhere inside those folders requires a run that
    may be published (`publish_problems`); otherwise nothing is written.
    """
    run = load_run(run_dir)
    name = run.state["name"]
    if publish:
        out = Path(root) / "results_summary" / name
        figures_dir = Path(root) / "docs" / "figures" / name
    out = Path(out) if out is not None else run.root / "analysis"
    figures_dir = Path(figures_dir) if figures_dir is not None else out
    if is_committed_path(out) or (make_figures and is_committed_path(figures_dir)):
        problems = publish_problems(run)
        if problems:
            raise RefusedToPublish(problems)

    settings = run.state.get("analysis", {})
    metrics = list(settings.get("metrics", DEFAULT_METRICS))
    curve_metrics = list(settings.get("curve_metrics", DEFAULT_CURVE_METRICS))
    unknown = sorted(set(metrics + curve_metrics) - set(KNOWN_VALUES))
    if unknown:
        raise ValueError(f"unknown metrics in [analysis]: {unknown}")
    resamples = int(settings.get("bootstrap", BOOTSTRAP_RESAMPLES))
    pairs = [
        (Arm.parse(a).name, Arm.parse(b).name) for a, b in settings.get("comparisons", [])
    ] or default_comparisons(run.arms)

    arms = arm_table(run, metrics, resamples)
    comparisons = comparison_table(run, pairs, metrics, resamples)
    curves = curve_table(run, curve_metrics)

    written = [
        _write(out / "arms.csv", _csv(arms)),
        _write(out / "comparisons.csv", _csv(comparisons)),
        _write(out / "curves.csv", _csv(curves)),
    ]
    figures: list[Path] = []
    if make_figures:
        from elicitor.analysis.figures import plot_curves

        for metric, budget in itertools.product(curve_metrics, run.budgets):
            rows = [r for r in curves if r["metric"] == metric and r["budget"] == budget]
            path = figures_dir / f"curve_{metric.replace('@', '_at')}_K{budget}.png"
            figures.append(plot_curves(rows, metric, budget, run.arms, path, title=name))
    md = _markdown(run, arms, comparisons, metrics, figures, out)
    written.append(_write(out / "results.md", md))
    return written + figures
