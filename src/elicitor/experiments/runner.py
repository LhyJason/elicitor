"""Run every (arm, budget, replicate, persona) episode of an experiment.

Writes `episodes.jsonl` (one episode per line) and `summary.csv` (means per arm and budget) to
the output directory. This is the minimal M1 runner: it always starts from scratch. Resume,
run health and statistics come in M4.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from elicitor.episode import CONFIDENT, Episode, run_episode
from elicitor.experiments.config import ExperimentConfig
from elicitor.jsonl import write_jsonl
from elicitor.metrics import BELIEF_METRICS
from elicitor.personas import load_personas
from elicitor.phrasing import TemplatePhraser
from elicitor.schema import load_schema
from elicitor.selectors import make_selector
from elicitor.simulator import make_simulator
from elicitor.updaters import make_updater

SUMMARY_FIELDS = ("arm", "budget", "episodes", *BELIEF_METRICS, "turns", "confident_share")


def run_experiment(
    config: ExperimentConfig,
    out_dir: str | Path,
    max_personas: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    schema = load_schema(config.schema_path)
    personas = load_personas(config.personas_path, split=config.split)[:max_personas]
    if not personas:
        raise ValueError(f"no personas in split {config.split!r}")
    sim_params = {k: v for k, v in config.simulator.items() if k != "kind"}
    simulator = make_simulator(schema, config.simulator.get("kind", "oracle"), **sim_params)
    phraser = TemplatePhraser(schema)

    episodes: list[Episode] = []
    for arm in config.arms:
        selector = make_selector(arm.selector)
        updater = make_updater(arm.updater, **config.updater_params.get(arm.updater, {}))
        for budget in config.budgets:
            for replicate in range(config.replicates):
                for persona in personas:
                    episodes.append(
                        run_episode(
                            schema=schema,
                            persona=persona,
                            arm=arm.name,
                            selector=selector,
                            updater=updater,
                            simulator=simulator,
                            phraser=phraser,
                            budget=budget,
                            tau=config.tau,
                            seed=config.seed + replicate,
                            replicate=replicate,
                        )
                    )
            if progress:
                progress(f"{arm.name} K={budget}: {config.replicates * len(personas)} episodes")

    out = Path(out_dir)
    write_jsonl(out / "episodes.jsonl", (e.to_dict() for e in episodes))
    rows = summarize(episodes)
    with open(out / "summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize(episodes: list[Episode]) -> list[dict[str, Any]]:
    """Plain means over all episodes of each (arm, budget). Paired statistics come in M4."""
    groups: dict[tuple[str, int], list[Episode]] = defaultdict(list)
    for e in episodes:
        groups[(e.arm, e.budget)].append(e)
    rows = []
    for (arm, budget), group in groups.items():
        n = len(group)
        row: dict[str, Any] = {"arm": arm, "budget": budget, "episodes": n}
        for metric in BELIEF_METRICS:
            row[metric] = round(sum(e.final_metrics[metric] for e in group) / n, 4)
        row["turns"] = round(sum(len(e.turns) for e in group) / n, 3)
        row["confident_share"] = round(sum(e.stop_reason == CONFIDENT for e in group) / n, 3)
        rows.append(row)
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    widths = {f: max(len(f), *(len(str(r[f])) for r in rows)) for f in SUMMARY_FIELDS}
    lines = ["  ".join(f.ljust(widths[f]) for f in SUMMARY_FIELDS)]
    lines += ["  ".join(str(r[f]).ljust(widths[f]) for f in SUMMARY_FIELDS) for r in rows]
    return "\n".join(lines)
