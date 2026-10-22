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

from elicitor.answer_model import AnswerModel
from elicitor.llm.roles import PhraseReply, SimulateReply, update_model
from elicitor.questions import noise_free_outcome
from elicitor.simulator.oracle import render_outcome
from elicitor.updaters.rule import parse_outcomes

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
    """Keyword scores: a value whose outcome the answer gives 9, one it rules out 1, the rest 2
    (or 5 when nothing matched). Outcomes are read as the rule updater reads them."""
    slot = ctx["slot"]
    question = ctx["q"]
    answer = ctx["answer"]
    positive, negated, _ = parse_outcomes(slot, question, answer)
    scores = {}
    for value in slot.value_names:
        outcome = noise_free_outcome(question, value)
        if outcome in positive:
            scores[value] = 9
        elif outcome in negated:
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
    question = ctx["q"]
    persona = ctx["persona"]
    outcome = AnswerModel(MOCK_RHO).sample(slot, question, persona.truth[slot.name], rng)
    answer = render_outcome(slot, question, outcome)
    wrap = _STYLE_WRAP.get(persona.style, "{answer}")
    keep_case = answer.split()[0] in ("I", "I'd")
    answer_lc = answer if keep_case else answer[0].lower() + answer[1:]
    text = wrap.format(answer=answer, answer_lc=answer_lc)
    return SimulateReply(answer=text)
