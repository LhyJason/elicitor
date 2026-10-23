"""Terminal demo (SPEC §18, Decision Q20): a person answers the questions in place of a persona.

The loop is the episode loop of §11 without ground truth: choose a question, phrase it, read the
answer, update the belief, show it. At the end the catalog is ranked under the final belief.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from elicitor.belief import Belief
from elicitor.catalog import load_catalog
from elicitor.episode import Turn
from elicitor.experiments.config import Arm, ExperimentConfig
from elicitor.phrasing import make_phraser
from elicitor.recommender import Recommender
from elicitor.rng import derive_rng
from elicitor.schema import Schema, load_schema
from elicitor.selectors import make_selector
from elicitor.updaters import make_updater

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


def understood(parsed: dict) -> bool:
    """False when the updater read nothing: no keyword matched (rule), or no usable reply (LLM)."""
    return parsed.get("matched", True) and "failed" not in parsed and "skipped" not in parsed


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
    schema = load_schema(config.schema_path)
    budget = budget or config.budgets[0]
    params = config.selector_params.get(arm.selector, {})
    selector = make_selector(arm.selector, client=client, seed=config.seed, **params)
    updater = make_updater(arm.updater, client=client, **config.updater_params.get(arm.updater, {}))
    phraser = make_phraser(arm.phraser, schema, client)
    recommender = Recommender(schema, load_catalog(config.catalog_path))

    write(f"{schema.description} Agent: {arm.name}, up to {budget} questions.")
    write("Answer in your own words; an empty line or 'quit' stops early.")
    write("")
    belief = Belief.prior(schema)
    write("\n".join(belief_lines(schema, belief)))
    selector_rng = derive_rng(config.seed, "demo", arm.name, "selector")
    history: list[Turn] = []
    times_asked: dict[str, int] = {}
    for index in range(1, budget + 1):
        if belief.is_confident(config.tau):
            write(f"\nEvery slot has reached {config.tau:g}; no more questions needed.")
            break
        question = selector.choose(belief, history, selector_rng)
        if question is None:
            write("\nNothing left to ask.")
            break
        n = times_asked.get(question.slot, 0)
        times_asked[question.slot] = n + 1
        text = phraser.render(question, derive_rng(config.seed, "demo", "phrase", question.slot, n))
        write("")
        answer = read(f"[{index}/{budget}] {text}\n> ").strip()
        if answer.lower() in STOP_WORDS:
            write("Stopped.")
            break
        belief, parsed = updater.update(belief, question, text, answer)
        history.append(Turn(index, question, text, answer, parsed, belief, {}))
        write("")
        if not understood(parsed):
            write("(answer not understood; the belief is unchanged)")
        write("\n".join(belief_lines(schema, belief, asked=question.slot)))

    utilities = recommender.expected_utilities(belief)
    write("\nTop recommendations (expected utility under the final belief):")
    for rank, i in enumerate(recommender.rank(belief)[:TOP], start=1):
        item = recommender.catalog[i]
        attrs = ", ".join(f"{slot}={value}" for slot, value in item.attrs.items())
        write(f"  {rank}. {item.id}  EU {utilities[i]:.2f}  {attrs}")
    return belief
