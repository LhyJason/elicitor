"""Experiment files: TOML read into dataclasses. Unknown keys are errors, not silently ignored."""

from __future__ import annotations

import itertools
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PHRASER = "tpl"  # only templates until M2

_KEYS = {
    "name", "domain", "data_dir", "split", "seed", "replicates", "budgets", "tau",
    "selectors", "updaters", "simulator", "updater",
}  # fmt: skip


@dataclass(frozen=True)
class Arm:
    selector: str
    updater: str
    phraser: str = PHRASER

    @property
    def name(self) -> str:
        return f"sel={self.selector},upd={self.updater},phr={self.phraser}"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    domain: str
    data_dir: Path
    split: str
    seed: int
    replicates: int
    budgets: tuple[int, ...]
    tau: float
    selectors: tuple[str, ...]
    updaters: tuple[str, ...]
    simulator: dict[str, Any] = field(default_factory=lambda: {"kind": "oracle"})
    updater_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def arms(self) -> list[Arm]:
        return [Arm(s, u) for s, u in itertools.product(self.selectors, self.updaters)]

    @property
    def schema_path(self) -> Path:
        return self.data_dir / self.domain / "schema.toml"

    @property
    def personas_path(self) -> Path:
        return self.data_dir / self.domain / "personas.jsonl"


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    unknown = set(raw) - _KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    budgets = tuple(int(k) for k in raw["budgets"])
    if not budgets or min(budgets) < 1:
        raise ValueError(f"{path}: budgets must be positive")
    return ExperimentConfig(
        name=raw["name"],
        domain=raw["domain"],
        # Relative data paths are relative to the experiment file, not the working directory.
        data_dir=(path.parent / raw.get("data_dir", "../data")).resolve(),
        split=raw.get("split", "eval"),
        seed=int(raw["seed"]),
        replicates=int(raw.get("replicates", 1)),
        budgets=budgets,
        tau=float(raw.get("tau", 0.9)),
        selectors=tuple(raw["selectors"]),
        updaters=tuple(raw["updaters"]),
        simulator=dict(raw.get("simulator", {"kind": "oracle"})),
        updater_params={k: dict(v) for k, v in raw.get("updater", {}).items()},
    )
