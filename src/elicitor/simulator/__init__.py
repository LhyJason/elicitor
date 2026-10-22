"""User simulators (SPEC §9): answer a question in character, from the persona only."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from elicitor.personas import Persona
from elicitor.questions import Question
from elicitor.schema import Schema

if TYPE_CHECKING:
    from elicitor.llm import LLMClient


class Simulator(Protocol):
    name: str

    def respond(
        self, question: Question, text: str, persona: Persona, rng: random.Random
    ) -> str: ...


def make_simulator(
    schema: Schema,
    kind: str,
    client: LLMClient | None = None,
    prompts_dir: str | Path | None = None,
    **params: Any,
) -> Simulator:
    from elicitor.simulator.oracle import OracleSimulator

    if kind == "oracle":
        return OracleSimulator(schema, **params)
    if kind == "llm":
        from elicitor.simulator.llm import LLMSimulator, load_styles

        if client is None or prompts_dir is None:
            raise ValueError("the llm simulator needs an LLM client and a prompts folder")
        return LLMSimulator(schema, client, load_styles(Path(prompts_dir) / "styles.toml"))
    raise ValueError(f"unknown simulator {kind!r}; choose from ['llm', 'oracle']")
