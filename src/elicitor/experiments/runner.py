"""Run every (arm, budget, replicate, persona) episode of an experiment.

Writes `episodes.jsonl` (one episode per line), `summary.csv` (means per arm and budget) and,
when an LLM was used, `llm_stats.json` (calls, cache hits and failures per role) to the output
directory. The LLM reply cache defaults to `llm_cache.sqlite` in the same directory, so a re-run
into the same directory makes no server calls. The runner always starts from scratch; resume,
run health and statistics come in M4.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from elicitor.catalog import load_catalog
from elicitor.episode import CONFIDENT, Episode, run_episode
from elicitor.experiments.config import ExperimentConfig
from elicitor.jsonl import write_jsonl
from elicitor.metrics import BELIEF_METRICS, OUTCOME_METRICS
from elicitor.personas import Persona, load_personas
from elicitor.phrasing import make_phraser
from elicitor.recommender import Recommender
from elicitor.schema import Schema, load_schema
from elicitor.selectors import make_selector
from elicitor.simulator import make_simulator
from elicitor.updaters import make_updater

if TYPE_CHECKING:
    from elicitor.llm import LLMClient

SUMMARY_FIELDS = (
    "arm", "budget", "episodes", *BELIEF_METRICS, *OUTCOME_METRICS, "turns", "confident_share",
    "leak_rate",
)  # fmt: skip


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
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    client = None
    if config.needs_llm:
        from elicitor.llm import build_client

        client = build_client(config.llm, config.prompts_dir, out / "llm_cache.sqlite")
    try:
        episodes = _run_all(config, schema, personas, client, progress)
    finally:
        # Stats are written even when a transport error stops the run, to show what happened.
        if client is not None:
            client.write_stats(out / "llm_stats.json")
            client.close()
            if progress:
                progress(client.stats_table())

    write_jsonl(out / "episodes.jsonl", (e.to_dict() for e in episodes))
    rows = summarize(episodes)
    with open(out / "summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _run_all(
    config: ExperimentConfig,
    schema: Schema,
    personas: list[Persona],
    client: LLMClient | None,
    progress: Callable[[str], None] | None,
) -> list[Episode]:
    sim_params = {k: v for k, v in config.simulator.items() if k != "kind"}
    simulator = make_simulator(
        schema,
        config.simulator.get("kind", "oracle"),
        client=client,
        prompts_dir=config.prompts_dir,
        **sim_params,
    )
    recommender = Recommender(schema, load_catalog(config.catalog_path))
    episodes: list[Episode] = []
    for arm in config.arms:
        selector = make_selector(arm.selector, **config.selector_params.get(arm.selector, {}))
        params = config.updater_params.get(arm.updater, {})
        updater = make_updater(arm.updater, client=client, **params)
        phraser = make_phraser(arm.phraser, schema, client)
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
                            recommender=recommender,
                        )
                    )
            if progress:
                progress(f"{arm.name} K={budget}: {config.replicates * len(personas)} episodes")
    return episodes


def summarize(episodes: list[Episode]) -> list[dict[str, Any]]:
    """Plain means over all episodes of each (arm, budget). Paired statistics come in M4.

    `leak_rate` is the share of answers that mention an unasked slot (SPEC §9).
    """
    groups: dict[tuple[str, int], list[Episode]] = defaultdict(list)
    for e in episodes:
        groups[(e.arm, e.budget)].append(e)
    rows = []
    for (arm, budget), group in groups.items():
        n = len(group)
        row: dict[str, Any] = {"arm": arm, "budget": budget, "episodes": n}
        for metric in (*BELIEF_METRICS, *OUTCOME_METRICS):
            row[metric] = round(sum(e.final_metrics[metric] for e in group) / n, 4)
        row["turns"] = round(sum(len(e.turns) for e in group) / n, 3)
        row["confident_share"] = round(sum(e.stop_reason == CONFIDENT for e in group) / n, 3)
        turns = [t for e in group for t in e.turns]
        row["leak_rate"] = round(sum(bool(t.leaks) for t in turns) / len(turns), 3) if turns else 0
        rows.append(row)
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    widths = {f: max(len(f), *(len(str(r[f])) for r in rows)) for f in SUMMARY_FIELDS}
    lines = ["  ".join(f.ljust(widths[f]) for f in SUMMARY_FIELDS)]
    lines += ["  ".join(str(r[f]).ljust(widths[f]) for f in SUMMARY_FIELDS) for r in rows]
    return "\n".join(lines)
