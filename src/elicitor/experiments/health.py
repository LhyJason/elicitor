"""Run-health report (SPEC §15, M4): progress and per-role LLM failure rates of a run folder.

Reads only the files the runner writes (`run_state.json`, the cell stats, `live.json`), so it
works while a job is running and on a copy of the folder. The verdict uses the M5 bar: no failed
call, no transport error and a parse-failure rate below 1 % in every role.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from elicitor.experiments.store import RunStore, cell_id, sum_stats

MAX_PARSE_RATE = 0.01

_COLUMNS = ("calls", "cache_hits", "attempts", "parse_failures", "parse_rate", "failed_calls",
            "transport_errors", "s/attempt")  # fmt: skip


@dataclass
class Health:
    lines: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def text(self) -> str:
        verdict = (
            ["health: OK"]
            if self.ok
            else ["health: PROBLEMS", *(f"  - {p}" for p in self.problems)]
        )
        return "\n".join([*self.lines, *verdict])


def role_rows(roles: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for name, s in roles.items():
        attempts = s.get("attempts", 0)
        rows.append(
            {
                "role": name,
                **{k: s.get(k, 0) for k in ("calls", "cache_hits", "attempts", "parse_failures")},
                "parse_rate": s.get("parse_failures", 0) / attempts if attempts else 0.0,
                "failed_calls": s.get("failed_calls", 0),
                "transport_errors": s.get("transport_errors", 0),
                "s/attempt": s.get("latency_s", 0.0) / attempts if attempts else 0.0,
            }
        )
    return rows


def format_roles(roles: dict[str, dict[str, Any]]) -> str:
    rows = role_rows(roles)
    if not rows:
        return "no LLM calls"
    lines = ["role".ljust(10) + "".join(c.rjust(17) for c in _COLUMNS)]
    for row in rows:
        cells = []
        for c in _COLUMNS:
            v = row[c]
            cells.append(
                f"{v:.2%}" if c == "parse_rate" else f"{v:.2f}" if isinstance(v, float) else str(v)
            )
        lines.append(row["role"].ljust(10) + "".join(x.rjust(17) for x in cells))
    return "\n".join(lines)


def check_health(run_dir: str | Path, now: float | None = None) -> Health:
    store = RunStore(run_dir)
    state = store.read_state()
    health = Health()
    if state is None:
        health.problems.append(f"{run_dir} is not a run folder (no run_state.json)")
        return health

    cells = [cell_id(arm, r) for arm in state["arms"] for r in range(state["replicates"])]
    done = [c for c in cells if store.is_done(c)]
    stats = {c: store.read_cell_stats(c) or {} for c in done}
    per_cell = len(state["budgets"]) * len(state["personas"])
    episodes = sum(s.get("episodes", 0) for s in stats.values())

    live = store.read_live()
    in_progress = live is not None and live.get("cell") in cells and live["cell"] not in done
    if in_progress:
        episodes += live.get("episodes", 0)

    health.lines.append(f"run {state['name']} in {run_dir}")
    health.lines.append(
        f"cells {len(done)}/{len(cells)}, episodes {episodes}/{len(cells) * per_cell}"
    )
    seconds = [s["seconds"] for s in stats.values() if "seconds" in s]
    if seconds and len(done) < len(cells):
        mean = sum(seconds) / len(seconds)
        eta = mean * (len(cells) - len(done))
        health.lines.append(f"mean {mean:.1f} s per finished cell, about {eta / 3600:.2f} h left")
    if live is not None:
        age = (now if now is not None else time.time()) - live.get("updated", 0)
        status = "in progress" if in_progress else "last cell"
        health.lines.append(
            f"live: {live.get('cell')} ({status}), {live.get('episodes')}/"
            f"{live.get('episodes_per_cell')} episodes, updated {age:.0f} s ago"
        )
        if live.get("error"):
            health.problems.append(f"cell {live['cell']} stopped: {live['error']}")

    parts = [s.get("llm") for s in stats.values()]
    if in_progress:
        parts.append(live.get("llm"))
    llm = sum_stats(parts)
    if llm is None:
        health.lines.append("no LLM calls")
        return health
    health.lines.append(f"provider {llm['provider']}, models {llm['models']}")
    health.lines.append(format_roles(llm["roles"]))
    for row in role_rows(llm["roles"]):
        if row["failed_calls"]:
            health.problems.append(f"{row['role']}: {row['failed_calls']} failed calls")
        if row["transport_errors"]:
            health.problems.append(f"{row['role']}: {row['transport_errors']} transport errors")
        if row["parse_rate"] >= MAX_PARSE_RATE:
            health.problems.append(
                f"{row['role']}: parse failure rate {row['parse_rate']:.2%}"
                f" (bar {MAX_PARSE_RATE:.0%})"
            )
    return health
