import subprocess
import sys
import tomllib

import pytest

import elicitor
from elicitor import Session, resources, understood
from elicitor.catalog import load_catalog
from elicitor.episode import BUDGET, CONFIDENT, EXHAUSTED, run_episode
from elicitor.experiments.config import load_config
from elicitor.llm import build_client
from elicitor.personas import load_personas
from elicitor.phrasing import TemplatePhraser
from elicitor.questions import Question
from elicitor.rng import derive_rng
from elicitor.selectors import make_selector
from elicitor.simulator.oracle import OracleSimulator
from elicitor.updaters.rule import RuleUpdater
from tests.conftest import ROOT, TRAVEL

DEMO = ROOT / "experiments" / "demo.toml"


def test_ask_observe_recommend(schema):
    session = Session(schema, selector="fixed", catalog=load_catalog(TRAVEL / "catalog.jsonl"))
    assert session.arm == "sel=fixed,upd=rule,phr=tpl"
    assert session.pending is None and not session.done

    text = session.next_question()
    assert text == schema.slot("climate").question
    assert session.pending == Question("climate")
    turn = session.observe("somewhere hot and tropical please")
    assert turn.index == 1 and turn.text == text and turn.metrics == {}
    assert understood(turn.parsed) and turn.belief is session.belief
    assert session.belief.map_value("climate") == "tropical"
    assert session.pending is None and session.turns == [turn]

    top = session.recommend(3)
    assert len(top) == 3
    assert all(rec.item.attrs["climate"] == "tropical" for rec in top)
    assert [rec.expected_utility for rec in top] == sorted(
        (rec.expected_utility for rec in top), reverse=True
    )


def test_a_pending_question_is_not_asked_again(schema):
    session = Session(schema, selector="random", seed=3)
    first = session.next_question()
    assert session.next_question() == first == session.next_question()
    session.observe("no idea")
    assert len(session.turns) == 1
    assert session.next_question() != first  # `random` never asks a slot twice


def test_an_answer_that_is_not_understood_leaves_the_belief(schema):
    session = Session(schema, selector="fixed")
    session.next_question()
    turn = session.observe("purple elephants")
    assert not understood(turn.parsed)
    assert session.belief.max_prob("climate") == pytest.approx(0.25)


def test_observe_needs_a_pending_question(schema):
    session = Session(schema)
    with pytest.raises(RuntimeError, match="next_question"):
        session.observe("tropical")
    session.next_question()
    session.observe("tropical")
    with pytest.raises(RuntimeError, match="next_question"):
        session.observe("tropical")


def test_stops_on_the_budget_and_goes_on_when_it_is_raised(schema):
    session = Session(schema, selector="fixed", budget=2, tau=1.1)
    for _ in range(2):
        assert session.next_question() is not None
        session.observe("no idea")
    assert session.next_question() is None
    assert session.done and session.stop_reason == BUDGET
    session.budget = 3
    assert session.next_question() == schema.slots[2].question and not session.done


def test_stops_when_confident_or_exhausted(schema):
    confident = Session(schema, tau=0.0)
    assert confident.next_question() is None and confident.stop_reason == CONFIDENT

    session = Session(schema, selector="fixed", budget=None, tau=1.1)
    while session.next_question() is not None:
        session.observe("no idea")
    assert session.stop_reason == EXHAUSTED and len(session.turns) == len(schema.slots)


def test_bad_arguments(schema):
    with pytest.raises(ValueError, match="budget"):
        Session(schema, budget=0)
    with pytest.raises(ValueError, match="unknown selector"):
        Session(schema, selector="nope")
    with pytest.raises(ValueError, match="LLM client"):
        Session(schema, updater="llm")
    with pytest.raises(ValueError, match="no catalog"):
        Session(schema).recommend()


def test_takes_component_objects(schema):
    class OnlyPace:
        name = "only_pace"

        def choose(self, belief, history, rng):
            return None if history else Question("pace")

    session = Session(schema, selector=OnlyPace(), updater=RuleUpdater(0.99), tau=1.1)
    assert session.arm == "sel=only_pace,upd=rule,phr=tpl"
    assert session.next_question() == schema.slot("pace").question
    session.observe("no idea")
    assert session.next_question() is None and session.stop_reason == EXHAUSTED


def test_from_config_defaults_to_the_packaged_demo():
    assert resources.demo_config() == DEMO
    assert resources.domain_dir("travel") == TRAVEL
    with pytest.raises(FileNotFoundError):
        resources.domain_dir("no_such_domain")

    config = load_config(DEMO)
    session = Session.from_config()
    assert session.arm == "sel=eig,upd=rule,phr=tpl"
    assert (session.budget, session.tau, session.seed) == (config.budgets[0], config.tau, 1)
    assert session.schema.domain == "travel" and len(session.recommend(5)) == 5

    other = Session.from_config(DEMO, selector="max_entropy", budget=8)
    assert other.arm == "sel=max_entropy,upd=rule,phr=tpl" and other.budget == 8
    assert Session.from_config(config).arm == session.arm


def test_from_config_with_llm_components_on_the_mock_backend(tmp_path):
    config = load_config(DEMO)
    client = build_client({"provider": "mock"}, config.prompts_dir, tmp_path / "cache.sqlite")
    try:
        session = Session.from_config(config, updater="llm", phraser="llm", client=client)
        assert session.arm == "sel=eig,upd=llm,phr=llm"
        assert "?" in session.next_question()
        turn = session.observe("I'd like the beach")
        assert "scores" in turn.parsed and understood(turn.parsed)
    finally:
        client.close()


@pytest.mark.parametrize("selector", ["fixed", "max_entropy", "eig"])
def test_a_session_fed_by_the_oracle_matches_the_episode(schema, selector):
    """The session is the episode loop: same questions, wording and beliefs, turn by turn."""
    persona = load_personas(TRAVEL / "personas.jsonl", split="eval")[0]
    simulator = OracleSimulator(schema, rho=0.8)
    seed, budget, tau = 7, 6, 0.9
    ep = run_episode(
        schema=schema,
        persona=persona,
        arm=f"sel={selector}",
        selector=make_selector(selector),
        updater=RuleUpdater(0.8),
        simulator=simulator,
        phraser=TemplatePhraser(schema),
        budget=budget,
        tau=tau,
        seed=seed,
    )

    session = Session(
        schema, selector=selector, updater=RuleUpdater(0.8), budget=budget, tau=tau, seed=seed
    )
    times_asked: dict[str, int] = {}
    while (text := session.next_question()) is not None:
        slot = session.pending.slot
        n = times_asked.get(slot, 0)
        times_asked[slot] = n + 1
        rng = derive_rng(seed, persona.id, "answer", slot, n)  # the episode's answer stream
        session.observe(simulator.respond(session.pending, text, persona, rng))

    assert len(ep.turns) > 1 and session.stop_reason == ep.stop_reason
    assert [(t.question, t.text, t.answer) for t in session.turns] == [
        (t.question, t.text, t.answer) for t in ep.turns
    ]
    assert session.belief.to_dict() == ep.turns[-1].belief.to_dict()


def test_pyproject_lists_every_package():
    """The package list is explicit (the bundled files are mapped in); it must not go stale."""
    with open(ROOT / "pyproject.toml", "rb") as f:
        setuptools = tomllib.load(f)["tool"]["setuptools"]
    src = ROOT / "src"
    found = {
        p.parent.relative_to(src).as_posix().replace("/", ".") for p in src.rglob("__init__.py")
    }
    bundled = {name for name in setuptools["packages"] if name.startswith("elicitor.bundled.")}
    assert set(setuptools["packages"]) - bundled == found
    assert {setuptools["package-dir"][name] for name in bundled} == {"data", "experiments"}
    # The package-data patterns pick up every file of the travel domain.
    patterns = setuptools["package-data"]["elicitor.bundled.data"]
    data = TRAVEL.parent
    matched = {p for pattern in patterns for p in data.glob(pattern) if p.is_file()}
    assert matched >= {p for p in TRAVEL.rglob("*") if p.is_file()}


def test_public_names_and_a_light_import():
    for name in elicitor.__all__:
        assert hasattr(elicitor, name)
    # The session path needs none of the analysis or LLM dependencies.
    code = (
        "import sys, elicitor; elicitor.Session.from_config().next_question(); "
        "print([m for m in ('numpy', 'scipy', 'matplotlib', 'pydantic', 'httpx') "
        "if m in sys.modules])"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
