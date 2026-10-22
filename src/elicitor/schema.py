"""Domain schema: slots, their values and priors, loaded from a TOML file (SPEC §3)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

NO_PREFERENCE = "no_preference"
CATEGORICAL = "categorical"
ORDINAL = "ordinal"
AT_MOST = "at_most"  # constraint: the item's bin must not be above the user's (e.g. budget)
EXACT = "exact"  # constraint: the item's bin must be the user's (e.g. trip length)


@dataclass(frozen=True)
class Value:
    name: str
    label: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class Slot:
    name: str
    kind: str
    description: str
    question: str
    values: tuple[Value, ...]
    prior: tuple[float, ...]
    edges: tuple[float, ...] = ()
    unit: str = ""
    threshold_question: str = ""  # ordinal only: yes/no template with an {edge} placeholder
    constraint: str = ""  # ordinal only: AT_MOST or EXACT (SPEC §10)

    @property
    def value_names(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.values)

    @property
    def concrete_values(self) -> tuple[str, ...]:
        """Values an item in the catalog can have (everything except `no_preference`)."""
        return tuple(v.name for v in self.values if v.name != NO_PREFERENCE)

    def index(self, value: str) -> int:
        return self.value_names.index(value)

    def value(self, name: str) -> Value:
        return self.values[self.index(name)]


@dataclass(frozen=True)
class Schema:
    domain: str
    description: str
    slots: tuple[Slot, ...]
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def slot_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.slots)

    def slot(self, name: str) -> Slot:
        for s in self.slots:
            if s.name == name:
                return s
        raise KeyError(f"unknown slot {name!r}")


def load_schema(path: str | Path) -> Schema:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    no_pref_raw = raw.get("no_preference", {})
    no_pref = Value(
        name=NO_PREFERENCE,
        label=no_pref_raw.get("label", "no particular preference"),
        keywords=tuple(no_pref_raw.get("keywords", ["no preference"])),
    )
    slots = tuple(_parse_slot(name, body, no_pref) for name, body in raw["slot"].items())
    if not slots:
        raise ValueError("schema has no slots")
    return Schema(
        domain=raw["domain"],
        description=raw.get("description", ""),
        slots=slots,
        data=dict(raw.get("data", {})),
    )


def _parse_slot(name: str, body: dict[str, Any], no_pref: Value) -> Slot:
    kind = body["kind"]
    if kind not in (CATEGORICAL, ORDINAL):
        raise ValueError(f"slot {name}: unknown kind {kind!r}")
    values = [
        Value(v["name"], v["label"], tuple(k.lower() for k in v.get("keywords", ())))
        for v in body["values"]
    ]
    # The label is always a keyword, so a verbatim answer is always understood.
    values = [Value(v.name, v.label, (v.label.lower(), *v.keywords)) for v in values]
    if kind == CATEGORICAL:
        values.append(no_pref)
    names = [v.name for v in values]
    if len(set(names)) != len(names):
        raise ValueError(f"slot {name}: duplicate value names")
    if len(values) < 3:
        raise ValueError(f"slot {name}: needs at least 3 values")

    edges: tuple[float, ...] = ()
    constraint = ""
    if kind == ORDINAL:
        edges = tuple(float(e) for e in body["edges"])
        if len(edges) != len(values) - 1 or list(edges) != sorted(set(edges)):
            raise ValueError(f"slot {name}: need {len(values) - 1} strictly increasing edges")
        constraint = body.get("constraint", EXACT)
        if constraint not in (AT_MOST, EXACT):
            raise ValueError(f"slot {name}: unknown constraint {constraint!r}")
    elif "constraint" in body or "threshold_question" in body:
        raise ValueError(f"slot {name}: only ordinal slots take a constraint or threshold question")

    return Slot(
        name=name,
        kind=kind,
        description=body.get("description", ""),
        question=body["question"],
        values=tuple(values),
        prior=_parse_prior(name, body.get("prior", "uniform"), len(values)),
        edges=edges,
        unit=body.get("unit", ""),
        threshold_question=body.get("threshold_question", ""),
        constraint=constraint,
    )


def _parse_prior(slot: str, prior: str | list[float], n: int) -> tuple[float, ...]:
    if prior == "uniform":
        return tuple([1.0 / n] * n)
    weights = [float(w) for w in prior]
    if len(weights) != n or any(w <= 0 for w in weights):
        raise ValueError(f"slot {slot}: prior needs {n} positive weights (incl. no_preference)")
    total = sum(weights)
    return tuple(w / total for w in weights)
