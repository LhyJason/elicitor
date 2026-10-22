"""User simulators (SPEC §9): answer a question in character, from the persona only."""

from __future__ import annotations

import random
from typing import Any, Protocol

from elicitor.personas import Persona
from elicitor.questions import Question
from elicitor.schema import Schema


class Simulator(Protocol):
    name: str

    def respond(
        self, question: Question, text: str, persona: Persona, rng: random.Random
    ) -> str: ...


def make_simulator(schema: Schema, kind: str, **params: Any) -> Simulator:
    from elicitor.simulator.oracle import OracleSimulator

    registry: dict[str, type] = {"oracle": OracleSimulator}
    if kind not in registry:
        raise ValueError(f"unknown simulator {kind!r}; choose from {sorted(registry)}")
    return registry[kind](schema, **params)
