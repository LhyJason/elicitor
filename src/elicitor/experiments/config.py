"""Experiment files: TOML read into dataclasses. Unknown keys are errors, not silently ignored."""

from __future__ import annotations

import itertools
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_KEYS = {
    "name", "domain", "data_dir", "split", "seed", "replicates", "budgets", "tau",
    "selectors", "updaters", "phrasers", "extra_arms", "simulator", "selector", "updater", "llm",
    "analysis",
}  # fmt: skip
_ANALYSIS_KEYS = {"comparisons", "metrics", "curve_metrics", "bootstrap"}
_ARM_PARTS = ("sel", "upd", "phr")


@dataclass(frozen=True)
class Arm:
    selector: str
    updater: str
    phraser: str = "tpl"

    @property
    def name(self) -> str:
        return f"sel={self.selector},upd={self.updater},phr={self.phraser}"

    @property
    def uses_llm(self) -> bool:
        return self.updater == "llm" or self.phraser == "llm"

    @classmethod
    def parse(cls, name: str) -> Arm:
        """Inverse of `name`: "sel=eig,upd=llm,phr=tpl" -> Arm("eig", "llm", "tpl")."""
        try:
            parts = dict(item.split("=", 1) for item in name.split(","))
        except ValueError:
            raise ValueError(f"bad arm name {name!r}") from None
        if set(parts) != set(_ARM_PARTS):
            raise ValueError(f"arm {name!r} needs exactly the parts {_ARM_PARTS}")
        return cls(parts["sel"], parts["upd"], parts["phr"])


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
    phrasers: tuple[str, ...] = ("tpl",)
    extra_arms: tuple[Arm, ...] = ()
    simulator: dict[str, Any] = field(default_factory=lambda: {"kind": "oracle"})
    selector_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    updater_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    llm: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)  # SPEC §13, statistics in detail
    source: str = field(default="", compare=False, repr=False)  # the TOML text, for provenance

    @property
    def arms(self) -> list[Arm]:
        """The selector x updater x phraser grid, then the extra arms not already in it."""
        grid = itertools.product(self.selectors, self.updaters, self.phrasers)
        arms = [Arm(s, u, p) for s, u, p in grid]
        return arms + [a for a in self.extra_arms if a not in arms]

    @property
    def needs_llm(self) -> bool:
        return self.simulator.get("kind") == "llm" or any(a.uses_llm for a in self.arms)

    @property
    def schema_path(self) -> Path:
        return self.data_dir / self.domain / "schema.toml"

    @property
    def personas_path(self) -> Path:
        return self.data_dir / self.domain / "personas.jsonl"

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / self.domain / "catalog.jsonl"

    @property
    def prompts_dir(self) -> Path:
        return self.data_dir / self.domain / "prompts"


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    raw = tomllib.loads(source)
    unknown = set(raw) - _KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    budgets = tuple(int(k) for k in raw["budgets"])
    if not budgets or min(budgets) < 1:
        raise ValueError(f"{path}: budgets must be positive")
    config = ExperimentConfig(
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
        phrasers=tuple(raw.get("phrasers", ["tpl"])),
        extra_arms=tuple(Arm.parse(name) for name in raw.get("extra_arms", [])),
        simulator=dict(raw.get("simulator", {"kind": "oracle"})),
        selector_params={k: dict(v) for k, v in raw.get("selector", {}).items()},
        updater_params={k: dict(v) for k, v in raw.get("updater", {}).items()},
        llm=dict(raw.get("llm", {})),
        analysis=dict(raw.get("analysis", {})),
        source=source,
    )
    _check_analysis(config, path)
    return config


def _check_analysis(config: ExperimentConfig, path: Path) -> None:
    unknown = set(config.analysis) - _ANALYSIS_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown [analysis] keys {sorted(unknown)}")
    names = {arm.name for arm in config.arms}
    for pair in config.analysis.get("comparisons", []):
        if len(pair) != 2:
            raise ValueError(f"{path}: a comparison is a pair of arm names, got {pair!r}")
        missing = [Arm.parse(name).name for name in pair if Arm.parse(name).name not in names]
        if missing:
            raise ValueError(f"{path}: comparison names arms not in the grid: {missing}")
