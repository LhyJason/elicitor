"""Run every (arm, budget, replicate, persona) episode of an experiment, resuming by default.

The work is split into cells, one per (arm, replicate), each covering every budget and persona.
A finished cell is stored in the run folder (see `store.py`) and skipped by later runs into the
same folder, so a cluster job that hits its time limit is resubmitted, not restarted (SPEC §15).
Once every cell is finished, `episodes.jsonl`, `summary.csv` (means per arm and budget) and, when
an LLM was used, `llm_stats.json` (calls, cache hits and failures per role, summed over cells)
are rebuilt from the cells. The LLM reply cache defaults to `llm_cache.sqlite` in the same folder.
"""

from __future__ import annotations

import csv
import io
import json
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from elicitor.catalog import load_catalog
from elicitor.episode import CONFIDENT, run_episode
from elicitor.experiments.config import Arm, ExperimentConfig
from elicitor.experiments.store import RunStore, cell_id, sum_stats, write_text_atomic
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
    fresh: bool = False,
) -> list[dict[str, Any]]:
    """Run the cells not finished yet, then merge all cells; returns the summary rows."""
    schema = load_schema(config.schema_path)
    personas = load_personas(config.personas_path, split=config.split)[:max_personas]
    if not personas:
        raise ValueError(f"no personas in split {config.split!r}")
    store = RunStore(out_dir)
    store.prepare(config, max_personas, [p.id for p in personas], fresh=fresh)

    cells = [(arm, r) for arm in config.arms for r in range(config.replicates)]
    todo = [(arm, r) for arm, r in cells if not store.is_done(cell_id(arm.name, r))]
    if progress and len(todo) < len(cells):
        progress(f"resuming: {len(cells) - len(todo)} of {len(cells)} cells already finished")

    client = None
    if todo and config.needs_llm:
        from elicitor.llm import build_client

        client = build_client(config.llm, config.prompts_dir, store.root / "llm_cache.sqlite")
    try:
        if todo:
            _run_cells(config, schema, personas, todo, store, client, progress)
    finally:
        if client is not None:
            client.close()

    rows = merge(store, [cell_id(arm.name, r) for arm, r in cells])
    store.clear_live()
    return rows


def _run_cells(
    config: ExperimentConfig,
    schema: Schema,
    personas: list[Persona],
    todo: list[tuple[Arm, int]],
    store: RunStore,
    client: LLMClient | None,
    progress: Callable[[str], None] | None,
) -> None:
    sim_params = {k: v for k, v in config.simulator.items() if k != "kind"}
    simulator = make_simulator(
        schema,
        config.simulator.get("kind", "oracle"),
        client=client,
        prompts_dir=config.prompts_dir,
        **sim_params,
    )
    recommender = Recommender(schema, load_catalog(config.catalog_path))
    per_cell = len(config.budgets) * len(personas)
    for n, (arm, replicate) in enumerate(todo, 1):
        cell = cell_id(arm.name, replicate)
        selector = make_selector(arm.selector, **config.selector_params.get(arm.selector, {}))
        params = config.updater_params.get(arm.updater, {})
        updater = make_updater(arm.updater, client=client, **params)
        phraser = make_phraser(arm.phraser, schema, client)
        if client is not None:
            client.reset_stats()  # stats are kept per cell and summed when merging
        start = time.perf_counter()
        episodes: list[dict[str, Any]] = []

        try:
            for budget in config.budgets:
                for persona in personas:
                    episode = run_episode(
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
                    episodes.append(episode.to_dict())
                    _write_live(store, cell, len(episodes), per_cell, start, client)
        except Exception as exc:
            # What the cell got through stays readable by `elicitor health`.
            error = f"{type(exc).__name__}: {exc}"
            _write_live(store, cell, len(episodes), per_cell, start, client, error)
            raise
        seconds = time.perf_counter() - start
        llm = client.stats_dict() if client is not None else None
        store.write_cell(cell, episodes, llm, seconds)
        if progress:
            progress(f"[{n}/{len(todo)}] {cell}: {len(episodes)} episodes in {seconds:.1f} s")


def _write_live(
    store: RunStore,
    cell: str,
    episodes: int,
    per_cell: int,
    start: float,
    client: LLMClient | None,
    error: str | None = None,
) -> None:
    store.write_live(
        {
            "cell": cell,
            "episodes": episodes,
            "episodes_per_cell": per_cell,
            "seconds": round(time.perf_counter() - start, 3),
            "llm": client.stats_dict() if client is not None else None,
            "error": error,
        }
    )


def merge(store: RunStore, cells: list[str]) -> list[dict[str, Any]]:
    """Rebuild the merged outputs from the cells, in grid order. Every cell must be finished."""
    missing = [c for c in cells if not store.is_done(c)]
    if missing:
        raise RuntimeError(f"{len(missing)} cells are not finished, e.g. {missing[0]}")
    texts = [store.read_cell_text(c) for c in cells]
    write_text_atomic(store.root / "episodes.jsonl", "".join(texts))
    episodes = [json.loads(line) for text in texts for line in text.splitlines() if line]
    rows = summarize(episodes)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(store.root / "summary.csv", buf.getvalue())
    llm = sum_stats((store.read_cell_stats(c) or {}).get("llm") for c in cells)
    llm_path = store.root / "llm_stats.json"
    if llm is not None:
        write_text_atomic(llm_path, json.dumps(llm, indent=2) + "\n")
    else:
        llm_path.unlink(missing_ok=True)
    return rows


def summarize(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Plain means over all episodes of each (arm, budget); paired statistics are in `analysis`.

    Takes episode records as stored (`Episode.to_dict()`). `leak_rate` is the share of answers
    that mention an unasked slot (SPEC §9).
    """
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for e in episodes:
        groups[(e["arm"], e["budget"])].append(e)
    rows = []
    for (arm, budget), group in groups.items():
        n = len(group)
        row: dict[str, Any] = {"arm": arm, "budget": budget, "episodes": n}
        for metric in (*BELIEF_METRICS, *OUTCOME_METRICS):
            row[metric] = round(sum(e["final_metrics"][metric] for e in group) / n, 4)
        row["turns"] = round(sum(e["n_turns"] for e in group) / n, 3)
        row["confident_share"] = round(sum(e["stop_reason"] == CONFIDENT for e in group) / n, 3)
        turns = [t for e in group for t in e["turns"]]
        leaks = sum(bool(t["leaks"]) for t in turns)
        row["leak_rate"] = round(leaks / len(turns), 3) if turns else 0
        rows.append(row)
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    widths = {f: max(len(f), *(len(str(r[f])) for r in rows)) for f in SUMMARY_FIELDS}
    lines = ["  ".join(f.ljust(widths[f]) for f in SUMMARY_FIELDS)]
    lines += ["  ".join(str(r[f]).ljust(widths[f]) for f in SUMMARY_FIELDS) for r in rows]
    return "\n".join(lines)
