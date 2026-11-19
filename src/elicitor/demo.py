"""Terminal demo (SPEC §18, Decision Q20): a person answers the questions in place of a persona.

The dialogue is a `Session` (SPEC Q21): choose a question, phrase it, read the answer, update the
belief, show it. At the end the catalog is ranked under the final belief.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from elicitor.belief import Belief
from elicitor.episode import CONFIDENT, EXHAUSTED
from elicitor.experiments.config import Arm, ExperimentConfig
from elicitor.schema import Schema
from elicitor.session import Session, understood

DEFAULT_CONFIG = Path("experiments") / "demo.toml"
CACHE = Path("results") / "demo_cache.sqlite"
STOP_WORDS = {"", "quit", "exit", "q"}
TOP = 3
BAR_WIDTH = 20


def bar(p: float, width: int = BAR_WIDTH) -> str:
    """ASCII so it prints on any console encoding."""
    filled = round(p * width)
    return "#" * filled + "." * (width - filled)


def belief_lines(schema: Schema, belief: Belief, asked: str | None = None) -> list[str]:
    """One bar per slot (its most probable value), then every value of the slot just asked."""
    width = max(len(name) for name in schema.slot_names)
    lines = []
    for slot in schema.slots:
        mark = ">" if slot.name == asked else " "
        p, value = belief.max_prob(slot.name), belief.map_value(slot.name)
        if belief.probs(slot.name).count(p) > 1:
            value = "(undecided)"
        lines.append(f"{mark} {slot.name:<{width}}  {bar(p)} {p:4.2f}  {value}")
    if asked is not None:
        slot = schema.slot(asked)
        lines.append("")
        lines.append(f"  {asked}:")
        value_width = max(len(v) for v in slot.value_names)
        for value, p in zip(slot.value_names, belief.probs(asked), strict=True):
            lines.append(f"    {value:<{value_width}}  {bar(p)} {p:4.2f}")
    return lines


def demo_arm(
    config: ExperimentConfig,
    selector: str | None = None,
    updater: str | None = None,
    phraser: str | None = None,
) -> Arm:
    arm = Arm(
        selector or config.selectors[0],
        updater or config.updaters[0],
        phraser or config.phrasers[0],
    )
    if arm.selector == "eig_mc":
        raise ValueError("eig_mc needs a simulated user and is not offered in the demo")
    return arm


def run_demo(
    config: ExperimentConfig,
    arm: Arm,
    budget: int | None = None,
    read: Callable[[str], str] | None = None,  # default input / print, looked up per call
    write: Callable[[str], None] | None = None,
    client=None,
) -> Belief:
    """Run one dialogue; returns the final belief. `client` is needed when the arm uses an LLM."""
    if arm.uses_llm and client is None:
        raise ValueError(f"{arm.name} uses an LLM; pass a client")
    read = read or input
    write = write or print
    session = Session.from_config(
        config,
        selector=arm.selector,
        updater=arm.updater,
        phraser=arm.phraser,
        budget=budget,
        client=client,
        label="demo",
    )
    schema, budget = session.schema, session.budget

    write(f"{schema.description} Agent: {arm.name}, up to {budget} questions.")
    write("Answer in your own words; an empty line or 'quit' stops early.")
    write("")
    write("\n".join(belief_lines(schema, session.belief)))
    while (text := session.next_question()) is not None:
        write("")
        answer = read(f"[{len(session.turns) + 1}/{budget}] {text}\n> ").strip()
        if answer.lower() in STOP_WORDS:
            write("Stopped.")
            break
        turn = session.observe(answer)
        write("")
        if not understood(turn.parsed):
            write("(answer not understood; the belief is unchanged)")
        write("\n".join(belief_lines(schema, session.belief, asked=turn.question.slot)))
    if session.stop_reason == CONFIDENT:
        write(f"\nEvery slot has reached {config.tau:g}; no more questions needed.")
    elif session.stop_reason == EXHAUSTED:
        write("\nNothing left to ask.")

    write("\nTop recommendations (expected utility under the final belief):")
    for rank, rec in enumerate(session.recommend(TOP), start=1):
        attrs = ", ".join(f"{slot}={value}" for slot, value in rec.item.attrs.items())
        write(f"  {rank}. {rec.item.id}  EU {rec.expected_utility:.2f}  {attrs}")
    return session.belief
