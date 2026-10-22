"""Run every (arm, budget, replicate, persona) episode of an experiment, resuming by default.

The work is split into cells, one per (arm, replicate), each covering every budget and persona.
A finished cell is stored in the run folder (see `store.py`) and skipped by later runs into the
same folder, so a cluster job that hits its time limit is resubmitted, not restarted (SPEC §15).
Once every cell is finished, `episodes.jsonl`, `summary.csv` (means per arm and budget) and, when
an LLM was used, `llm_stats.json` (calls, cache hits and failures per role, summed over cells)
are rebuilt from the cells. The LLM reply cache defaults to `llm_cache.sqlite` in the same folder.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from elicitor.catalog import load_catalog
from elicitor.episode import CONFIDENT, run_episode
from elicitor.experiments.config import Arm, ExperimentConfig
from elicitor.experiments.store import RunStore, cell_id, sum_stats, write_text_atomic
from elicitor.metrics import BELIEF_METRICS, OUTCOME_METRICS
from elicitor.personas import Persona, load_personas
from elicitor.phrasing import Phraser, make_phraser
from elicitor.recommender import Recommender
from elicitor.schema import Schema, load_schema
from elicitor.selectors import Selector, make_selector
from elicitor.simulator import make_simulator
from elicitor.updaters import Updater, make_updater

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
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Run the cells not finished yet, then merge all cells; returns the summary rows.

    With `workers` > 1, (cell, persona) tasks run on that many threads (SPEC §15, M5); the cells
    written are the same as with one worker.
    """
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
            _run_cells(config, schema, personas, todo, store, client, progress, workers)
    finally:
        if client is not None:
            client.close()

    rows = merge(store, [cell_id(arm.name, r) for arm, r in cells])
    store.clear_live()
    return rows


@dataclass
class _Cell:
    """One (arm, replicate) being run: its components and the episodes of its finished tasks."""

    id: str
    arm: Arm
    replicate: int
    budgets: tuple[int, ...]  # in the order of the experiment file
    selector: Selector
    updater: Updater
    phraser: Phraser
    started: float = 0.0
    results: dict[int, list[dict[str, Any]]] = field(default_factory=dict)  # by persona index
    stats: list[dict[str, Any] | None] = field(default_factory=list)


def _run_cells(
    config: ExperimentConfig,
    schema: Schema,
    personas: list[Persona],
    todo: list[tuple[Arm, int]],
    store: RunStore,
    client: LLMClient | None,
    progress: Callable[[str], None] | None,
    workers: int = 1,
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

    def make_cell(arm: Arm, replicate: int) -> _Cell:
        params = config.selector_params.get(arm.selector, {})
        return _Cell(
            id=cell_id(arm.name, replicate),
            arm=arm,
            replicate=replicate,
            budgets=config.budgets_for(arm),
            selector=make_selector(
                arm.selector, client=client, seed=config.seed + replicate, **params
            ),
            updater=make_updater(
                arm.updater, client=client, **config.updater_params.get(arm.updater, {})
            ),
            phraser=make_phraser(arm.phraser, schema, client),
        )

    def run_task(cell: _Cell, persona: Persona) -> tuple[list[dict[str, Any]], Any]:
        """One persona's episodes at every budget of the cell. The smallest budget runs first,
        so the longer episodes find its turns in the LLM cache."""
        if not cell.started:
            cell.started = time.perf_counter()
        episodes: dict[int, dict[str, Any]] = {}
        tracking = client.track() if client is not None else contextlib.nullcontext()
        with tracking as tally:
            for budget in sorted(cell.budgets):
                episode = run_episode(
                    schema=schema,
                    persona=persona,
                    arm=cell.arm.name,
                    selector=cell.selector,
                    updater=cell.updater,
                    simulator=simulator,
                    phraser=cell.phraser,
                    budget=budget,
                    tau=config.tau,
                    seed=config.seed + cell.replicate,
                    replicate=cell.replicate,
                    recommender=recommender,
                )
                episodes[budget] = episode.to_dict()
        stats = client.stats_dict(tally) if client is not None else None
        return [episodes[k] for k in cell.budgets], stats

    live = _Live(store, len(personas), uses_llm=client is not None)
    cells = [make_cell(arm, r) for arm, r in todo]
    finished = 0

    def finish_task(cell: _Cell, index: int, episodes: list[dict[str, Any]], stats: Any) -> None:
        nonlocal finished
        cell.results[index] = episodes
        cell.stats.append(stats)
        if len(cell.results) < len(personas):
            live.update(cell)
            return
        # Budget order, then persona order: the cell one worker writes.
        ordered = [
            cell.results[i][b] for b in range(len(cell.budgets)) for i in range(len(personas))
        ]
        seconds = time.perf_counter() - cell.started
        llm = sum_stats(cell.stats) if client is not None else None
        store.write_cell(cell.id, ordered, llm, seconds)
        live.done(cell)
        finished += 1
        if progress:
            progress(
                f"[{finished}/{len(cells)}] {cell.id}: {len(ordered)} episodes in {seconds:.1f} s"
            )

    tasks = [(cell, i, persona) for cell in cells for i, persona in enumerate(personas)]
    if workers <= 1:
        for cell, i, persona in tasks:
            try:
                episodes, stats = run_task(cell, persona)
            except Exception as exc:
                live.error(cell, exc)
                raise
            finish_task(cell, i, episodes, stats)
        return

    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(run_task, cell, persona): (cell, i) for cell, i, persona in tasks}
    try:
        # Tasks finish in any order; a cell's content does not depend on it.
        for future in as_completed(futures):
            cell, i = futures[future]
            try:
                episodes, stats = future.result()
            except Exception as exc:
                live.error(cell, exc)
                raise
            finish_task(cell, i, episodes, stats)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


class _Live:
    """`live.json`: every cell in progress, its finished episodes and their LLM statistics."""

    def __init__(self, store: RunStore, personas: int, uses_llm: bool):
        self.store = store
        self.personas = personas
        self.uses_llm = uses_llm
        self.cells: dict[str, dict[str, Any]] = {}
        self.error_text: str | None = None
        self.start = time.perf_counter()

    def update(self, cell: _Cell) -> None:
        self.cells[cell.id] = {
            "episodes": len(cell.results) * len(cell.budgets),
            "episodes_per_cell": self.personas * len(cell.budgets),
            "llm": sum_stats(cell.stats) if self.uses_llm else None,
        }
        self.write()

    def done(self, cell: _Cell) -> None:
        self.cells.pop(cell.id, None)
        self.write()

    def error(self, cell: _Cell, exc: BaseException) -> None:
        self.error_text = f"{cell.id}: {type(exc).__name__}: {exc}"
        self.update(cell)

    def write(self) -> None:
        self.store.write_live(
            {
                "cells": self.cells,
                "seconds": round(time.perf_counter() - self.start, 3),
                "error": self.error_text,
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
