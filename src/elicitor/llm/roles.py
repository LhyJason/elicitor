"""Role registry (SPEC §14, Decision Q15).

A role is one kind of LLM call: a prompt template, a pydantic output model (which also gives the
JSON schema sent to the server), sampling settings, the model group that serves it and a mock
function for the mock provider. The judge study's `plan` and `judge` run in both groups; the
caller then names the group per call (SPEC §14, M7 details).

The caller passes a `context` dict. String entries fill the prompt's `$name` placeholders; other
entries (a `Slot`, a `Persona`) are there for the mock and for building the output model, and
never reach the server.
"""

from __future__ import annotations

import functools
import random
import string
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from elicitor.schema import Slot

AGENT = "agent"  # the agent's roles: update, phrase, imagine
USER = "user"  # the simulated user's roles: simulate, backstory; the judge study's pair
GROUPS = (AGENT, USER)

SCORE_MAX = 10


# ---------------------------------------------------------------- output models


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PhraseReply(Strict):
    question: str = Field(min_length=5, max_length=300)

    @field_validator("question")
    @classmethod
    def _is_question(cls, v: str) -> str:
        v = v.strip()
        if "?" not in v:
            raise ValueError("must be a question")
        return v


class SimulateReply(Strict):
    answer: str = Field(min_length=1, max_length=800)


class BackstoryReply(Strict):
    backstory: str = Field(min_length=20, max_length=600)


class PlanReply(Strict):
    plan: str = Field(min_length=40, max_length=1200)


# The reason comes first, so a model that follows the schema's order writes it before the
# verdict.
class JudgeReply(Strict):
    reason: str = Field(max_length=400)
    score: int = Field(ge=0, le=SCORE_MAX)


class PairReply(Strict):
    reason: str = Field(max_length=400)
    choice: Literal["first", "second", "equal"]


@functools.cache
def _update_model(slot_name: str, value_names: tuple[str, ...]) -> type[BaseModel]:
    # Python field names are v0, v1, ...; the JSON keys are the value names (aliases), so any
    # value name works, not only valid identifiers.
    fields: dict[str, Any] = {
        f"v{i}": (int, Field(alias=name, ge=0, le=SCORE_MAX)) for i, name in enumerate(value_names)
    }
    scores = create_model(f"Scores_{slot_name}", __base__=Strict, **fields)
    return create_model(
        f"Update_{slot_name}",
        __base__=Strict,
        evidence=(str, Field(max_length=300)),
        scores=(scores, ...),
    )


def update_model(slot: Slot) -> type[BaseModel]:
    """Output model of the `update` role for one slot: evidence plus a score for every value."""
    return _update_model(slot.name, slot.value_names)


def update_scores(reply: BaseModel, slot: Slot) -> list[int]:
    """Scores from an `update` reply, in the slot's value order."""
    return [getattr(reply.scores, f"v{i}") for i in range(len(slot.values))]  # type: ignore[attr-defined]


# ---------------------------------------------------------------- prompts


@dataclass(frozen=True)
class Prompt:
    system: string.Template
    user: string.Template

    def render(self, context: Mapping[str, Any]) -> list[dict[str, str]]:
        text = {k: v for k, v in context.items() if isinstance(v, str)}
        return [
            {"role": "system", "content": self.system.substitute(text).strip()},
            {"role": "user", "content": self.user.substitute(text).strip()},
        ]


def load_prompt(path: str | Path) -> Prompt:
    """A prompt file has a `### system` section and a `### user` section."""
    text = Path(path).read_text(encoding="utf-8")
    parts: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        header = line.strip().lower()
        if header in ("### system", "### user"):
            current = header.removeprefix("### ")
            parts[current] = []
        elif current is not None:
            parts[current].append(line)
    if set(parts) != {"system", "user"}:
        raise ValueError(f"{path}: needs a '### system' and a '### user' section")
    return Prompt(
        system=string.Template("\n".join(parts["system"])),
        user=string.Template("\n".join(parts["user"])),
    )


def value_lines(slot: Slot) -> str:
    """The slot's values as prompt lines: `- name: label`."""
    return "\n".join(f"- {v.name}: {v.label}" for v in slot.values)


# ---------------------------------------------------------------- registry


@dataclass(frozen=True)
class Role:
    name: str
    group: str
    prompt: Prompt
    output: Callable[[Mapping[str, Any]], type[BaseModel]]
    mock: Callable[[Mapping[str, Any], random.Random], BaseModel]
    temperature: float
    max_tokens: int


ROLE_NAMES = ("update", "phrase", "simulate", "imagine", "backstory", "plan", "judge", "pair")


def build_roles(prompts_dir: str | Path) -> dict[str, Role]:
    from elicitor.llm import mock

    prompts_dir = Path(prompts_dir)
    return {
        "update": Role(
            name="update",
            group=AGENT,
            prompt=load_prompt(prompts_dir / "update.txt"),
            output=lambda ctx: update_model(ctx["slot"]),
            mock=mock.mock_update,
            temperature=0.0,
            max_tokens=300,
        ),
        "phrase": Role(
            name="phrase",
            group=AGENT,
            prompt=load_prompt(prompts_dir / "phrase.txt"),
            output=lambda ctx: PhraseReply,
            mock=mock.mock_phrase,
            temperature=0.7,
            max_tokens=120,
        ),
        "simulate": Role(
            name="simulate",
            group=USER,
            prompt=load_prompt(prompts_dir / "simulate.txt"),
            output=lambda ctx: SimulateReply,
            mock=mock.mock_simulate,
            temperature=0.8,
            max_tokens=250,
        ),
        "imagine": Role(
            name="imagine",
            group=AGENT,
            prompt=load_prompt(prompts_dir / "imagine.txt"),
            output=lambda ctx: SimulateReply,
            mock=mock.mock_imagine,
            temperature=0.8,
            max_tokens=150,
        ),
        "backstory": Role(
            name="backstory",
            group=USER,
            prompt=load_prompt(prompts_dir / "backstory.txt"),
            output=lambda ctx: BackstoryReply,
            mock=mock.mock_backstory,
            temperature=0.8,
            max_tokens=250,
        ),
        # Judge study (SPEC Q10, M7). `plan` and `judge` default to the user group; the judge
        # study names the group per call.
        "plan": Role(
            name="plan",
            group=USER,
            prompt=load_prompt(prompts_dir / "plan.txt"),
            output=lambda ctx: PlanReply,
            mock=mock.mock_plan,
            temperature=0.7,
            max_tokens=300,
        ),
        "judge": Role(
            name="judge",
            group=USER,
            prompt=load_prompt(prompts_dir / "judge.txt"),
            output=lambda ctx: JudgeReply,
            mock=mock.mock_judge,
            temperature=0.0,
            max_tokens=150,
        ),
        "pair": Role(
            name="pair",
            group=USER,
            prompt=load_prompt(prompts_dir / "pair.txt"),
            output=lambda ctx: PairReply,
            mock=mock.mock_pair,
            temperature=0.0,
            max_tokens=150,
        ),
    }
