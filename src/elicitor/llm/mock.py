"""Mock replies, one deterministic function per role (SPEC §14, M2 details).

Each function reads the role's structured context (not the prompt) and returns a valid output
object; the mock provider serializes it to JSON so the client parses it like a server reply.
The mock is meant to behave plausibly, not well: it lets the full pipeline run in CI.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from elicitor.llm.roles import PhraseReply, SimulateReply, update_model
from elicitor.schema import ORDINAL
from elicitor.simulator.oracle import render_answer
from elicitor.updaters.rule import find_mentions, number_bin

MOCK_RHO = 0.8  # how often the mock simulator names the true value

_STYLE_WRAP = {
    "direct": "{answer}",
    "vague": "Hmm, hard to say. Maybe {answer_lc} I'm not sure yet.",
    "rambling": (
        "Funny you ask, I was talking about trips with a friend last week and we went back and "
        "forth for ages. In the end, {answer_lc} Though I can always change my mind."
    ),
    "indifferent": "Not something I think about much. {answer}",
}


def mock_update(ctx: Mapping[str, Any], rng: random.Random) -> BaseModel:
    """Keyword scores: named value 9, negated value 1, the rest 2 (or 5 when nothing matched)."""
    slot = ctx["slot"]
    answer = ctx["answer"]
    positive, negated = find_mentions(slot, answer)
    if not positive and not negated and slot.kind == ORDINAL:
        found = number_bin(slot, answer)
        if found is not None:
            positive = [found[1]]
    scores = {}
    for value in slot.value_names:
        if value in positive:
            scores[value] = 9
        elif value in negated:
            scores[value] = 1
        else:
            scores[value] = 2 if positive else 5
    evidence = answer[:80] if positive or negated else ""
    return update_model(slot).model_validate({"evidence": evidence, "scores": scores})


def mock_phrase(ctx: Mapping[str, Any], rng: random.Random) -> BaseModel:
    template = ctx["template"]
    return PhraseReply(question=f"Quick question: {template[0].lower()}{template[1:]}")


def mock_simulate(ctx: Mapping[str, Any], rng: random.Random) -> BaseModel:
    slot = ctx["slot"]
    persona = ctx["persona"]
    value = persona.truth[slot.name]
    if rng.random() >= MOCK_RHO:
        value = rng.choice(slot.value_names)
    answer = render_answer(slot, value)
    wrap = _STYLE_WRAP.get(persona.style, "{answer}")
    keep_case = answer.split()[0] in ("I", "I'd")
    answer_lc = answer if keep_case else answer[0].lower() + answer[1:]
    text = wrap.format(answer=answer, answer_lc=answer_lc)
    return SimulateReply(answer=text)
