"""The run folder: finished cells, the run's fingerprint and live progress (SPEC §15, M4).

    run_state.json                what the folder holds: fingerprint, arms, replicates, personas
    cells/<arm>__r<k>.jsonl       episodes of one finished (arm, replicate) cell
    cells/<arm>__r<k>.stats.json  that cell's LLM call statistics and wall time
    live.json                     the cell in progress, rewritten after every episode
    episodes.jsonl, summary.csv, llm_stats.json   merged once every cell is finished
    llm_cache.sqlite              LLM reply cache (kept by --fresh)

A cell file is written to a temporary name and then renamed, so it is either complete or absent:
a job killed mid-cell leaves no cell file and the cell runs again on resume.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from elicitor import __version__
from elicitor.experiments.config import ExperimentConfig

STATE = "run_state.json"
LIVE = "live.json"
CELLS = "cells"
MERGED = ("episodes.jsonl", "summary.csv", "llm_stats.json")

# [llm] settings that do not change any reply, so they stay out of the fingerprint.
_LLM_TRANSPORT_KEYS = ("base_url", "timeout", "cache")


class FingerprintMismatch(RuntimeError):
    """The folder holds cells of a run with different settings."""


def cell_id(arm: str, replicate: int) -> str:
    return f"{arm}__r{replicate}"


def fingerprint(config: ExperimentConfig, max_personas: int | None) -> str:
    """SHA-256 over everything that changes a cell's content, but not the arm list or the
    replicate count, so both can grow on a finished run."""
    settings = {
        "version": __version__,
        "domain": config.domain,
        "split": config.split,
        "seed": config.seed,
        "budgets": list(config.budgets),
        "tau": config.tau,
        "simulator": config.simulator,
        "selector": config.selector_params,
        "updater": config.updater_params,
        "llm": {k: v for k, v in config.llm.items() if k not in _LLM_TRANSPORT_KEYS},
        "max_personas": max_personas,
    }
    h = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    # Every file of the domain (schema, personas, catalog, backstories, prompts). The repo checks
    # them out with LF endings, so the hash is the same on every platform.
    domain_dir = config.data_dir / config.domain
    for path in sorted(p for p in domain_dir.rglob("*") if p.is_file()):
        h.update(path.relative_to(domain_dir).as_posix().encode() + b"\0")
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def sum_stats(parts: Iterable[dict[str, Any] | None]) -> dict[str, Any] | None:
    """Add up `LLMClient.stats_dict()` outputs role by role; None when no part used an LLM."""
    parts = [p for p in parts if p]
    if not parts:
        return None
    roles: dict[str, dict[str, float]] = {}
    for part in parts:
        for role, counts in part["roles"].items():
            total = roles.setdefault(role, {})
            for key, value in counts.items():
                total[key] = total.get(key, 0) + value
    for counts in roles.values():
        for key, value in counts.items():
            if isinstance(value, float):
                counts[key] = round(value, 3)
    return {"provider": parts[0]["provider"], "models": parts[0]["models"], "roles": roles}


class RunStore:
    def __init__(self, out_dir: str | Path):
        self.root = Path(out_dir)
        self.cells_dir = self.root / CELLS

    # ------------------------------------------------------------ setup

    def prepare(
        self,
        config: ExperimentConfig,
        max_personas: int | None,
        persona_ids: list[str],
        fresh: bool = False,
    ) -> None:
        """Check the fingerprint against an earlier run (or discard it with `fresh`) and record
        what this invocation will run."""
        self.root.mkdir(parents=True, exist_ok=True)
        if fresh:
            self.discard()
        fp = fingerprint(config, max_personas)
        old = self.read_state()
        if old is not None and old["fingerprint"] != fp and self.finished_cells():
            raise FingerprintMismatch(
                f"{self.root} holds cells of a run with different settings; "
                "pass --fresh to discard them or choose another --out"
            )
        self.cells_dir.mkdir(exist_ok=True)
        state = {
            "name": config.name,
            "fingerprint": fp,
            "version": __version__,
            "split": config.split,
            "max_personas": max_personas,
            "seed": config.seed,
            "budgets": list(config.budgets),
            "arms": [arm.name for arm in config.arms],
            "replicates": config.replicates,
            "personas": persona_ids,
            "simulator": config.simulator.get("kind", "oracle"),
            "provider": config.llm.get("provider", "mock") if config.needs_llm else None,
            "analysis": config.analysis,
            "config": config.source,
        }
        write_text_atomic(self.root / STATE, json.dumps(state, indent=2) + "\n")

    def discard(self) -> None:
        """Remove cells, state and merged outputs; the LLM cache stays (its keys are exact)."""
        if self.cells_dir.exists():
            shutil.rmtree(self.cells_dir)
        for name in (STATE, LIVE, *MERGED):
            (self.root / name).unlink(missing_ok=True)

    def read_state(self) -> dict[str, Any] | None:
        path = self.root / STATE
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    # ------------------------------------------------------------ cells

    def cell_path(self, cell: str) -> Path:
        return self.cells_dir / f"{cell}.jsonl"

    def stats_path(self, cell: str) -> Path:
        return self.cells_dir / f"{cell}.stats.json"

    def is_done(self, cell: str) -> bool:
        return self.cell_path(cell).exists()

    def finished_cells(self) -> list[str]:
        if not self.cells_dir.exists():
            return []
        return sorted(p.name.removesuffix(".jsonl") for p in self.cells_dir.glob("*.jsonl"))

    def write_cell(
        self, cell: str, episodes: list[dict[str, Any]], llm: dict[str, Any] | None, seconds: float
    ) -> None:
        # Stats first: a cell file without its stats file never exists.
        stats = {"episodes": len(episodes), "seconds": round(seconds, 3), "llm": llm}
        write_text_atomic(self.stats_path(cell), json.dumps(stats, indent=2) + "\n")
        text = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in episodes)
        write_text_atomic(self.cell_path(cell), text)

    def read_cell_text(self, cell: str) -> str:
        return self.cell_path(cell).read_text(encoding="utf-8")

    def read_cell(self, cell: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.read_cell_text(cell).splitlines() if line]

    def read_cell_stats(self, cell: str) -> dict[str, Any] | None:
        path = self.stats_path(cell)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    # ------------------------------------------------------------ live progress

    def write_live(self, live: dict[str, Any]) -> None:
        """Best effort: on Windows the rename fails while a reader holds the file open."""
        live = {**live, "updated": time.time()}
        with contextlib.suppress(OSError):
            write_text_atomic(self.root / LIVE, json.dumps(live, indent=2) + "\n")

    def read_live(self) -> dict[str, Any] | None:
        path = self.root / LIVE
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def clear_live(self) -> None:
        (self.root / LIVE).unlink(missing_ok=True)
