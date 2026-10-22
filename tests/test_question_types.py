"""Choice and yes/no questions: candidates, templates, oracle, rule updater, mock (SPEC §6–§9)."""

import random

import pytest

from elicitor.answer_model import AnswerModel
from elicitor.belief import Belief
from elicitor.llm import build_client
from elicitor.llm.roles import build_roles
from elicitor.personas import Persona
from elicitor.phrasing import KIND_RULES, TemplatePhraser
from elicitor.questions import (
    CHOICE,
    KINDS,
    NO,
    OPEN,
    OTHER,
    YES,
    YESNO,
    Question,
    candidates,
    noise_free_outcome,
    outcomes,
)
from elicitor.simulator.oracle import OracleSimulator, render_outcome
from elicitor.updaters.llm import LLMUpdater
from elicitor.updaters.rule import RuleUpdater, parse_outcomes
from tests.conftest import TRAVEL

PROMPTS = TRAVEL / "prompts"


def all_candidates(schema):
    return [q for slot in schema.slots for q in candidates(slot)]


def test_candidate_set(schema):
    qs = all_candidates(schema)
    assert len(qs) == len(set(qs)) == 105
    for q in qs:
        slot = schema.slot(q.slot)
        if q.kind == CHOICE:
            assert len(q.values) in (2, 3) and set(q.values) <= set(slot.concrete_values)
        if q.kind == YESNO and slot.kind == "ordinal":
            assert q.values == slot.value_names[: len(q.values)]  # "below edge i"
    assert [q.kind for q in candidates(schema.slot("pace"), (YESNO, OPEN))][0] == OPEN


def test_question_validation():
    with pytest.raises(ValueError):
        Question("pace", "essay")
    with pytest.raises(ValueError):
        Question("pace", OPEN, ("relaxed",))
    with pytest.raises(ValueError):
        Question("pace", YESNO)
    assert Question("pace").to_dict() == {"slot": "pace", "kind": "open"}  # as in M1/M2


def test_templates(schema):
    phraser = TemplatePhraser(schema)
    for q in all_candidates(schema):
        text = phraser.render(q)
        assert text.endswith("?") and "{" not in text, text
        slot = schema.slot(q.slot)
        if q.kind == CHOICE:
            assert all(slot.value(v).label in text for v in q.values), text
    budget_under_200 = Question("budget", YESNO, ("low", "mid"))
    assert phraser.render(budget_under_200) == "Would you like to keep it under $200 a day?"
    weather = Question("climate", CHOICE, ("tropical", "cold"))
    assert phraser.render(weather) == (
        "Would you prefer hot, tropical weather; or cold, wintry weather?"
    )


def test_noise_free_oracle_answers_round_trip(schema):
    """The rule matcher reads exactly the outcome the oracle gave, for every question and value."""
    sim = OracleSimulator(schema, rho=1.0)
    for q in all_candidates(schema):
        slot = schema.slot(q.slot)
        for value in slot.value_names:
            persona = Persona(
                "t", "eval", "direct", {s.name: s.value_names[0] for s in schema.slots}
            )
            persona.truth[slot.name] = value
            answer = sim.respond(q, "", persona, random.Random(0))
            positive, negated, _ = parse_outcomes(slot, q, answer)
            assert (positive, negated) == ([noise_free_outcome(q, value)], []), (q, answer)


@pytest.mark.parametrize("kind", KINDS)
def test_rule_update_is_exact_bayes(schema, kind):
    """On oracle answers the rule updater's posterior is Bayes rule under the answer model."""
    rho = {"open": 0.6, "choice": 0.75, "yesno": 0.85}
    model = AnswerModel(rho)
    updater = RuleUpdater(rho)
    rng = random.Random(9)
    for q in [q for q in all_candidates(schema) if q.kind == kind]:
        slot = schema.slot(q.slot)
        prior = Belief(
            schema, {s.name: [rng.random() + 0.1 for _ in s.values] for s in schema.slots}
        )
        for outcome in outcomes(slot, q):
            answer = render_outcome(slot, q, outcome)
            posterior, record = updater.update(prior, q, "", answer)
            assert record["matched"]
            j = outcomes(slot, q).index(outcome)
            joint = [
                p * model.outcome_probs(slot, q, v)[j]
                for p, v in zip(prior.probs(slot.name), slot.value_names, strict=True)
            ]
            expected = [x / sum(joint) for x in joint]
            assert posterior.probs(slot.name) == pytest.approx(expected)


def test_yes_no_phrases(schema):
    slot = schema.slot("setting")
    beach = Question("setting", YESNO, ("beach",))
    cases = {
        "Yeah, sounds good to me.": [YES],
        "Hmm, hard to say. Maybe yes, that would suit me. I'm not sure yet.": [YES],
        "No, not really.": [NO],
        "I'd rather not, to be honest.": [NO],
        "The beach sounds lovely.": [YES],  # no yes/no word: read from the value named
        "I'd prefer the mountains.": [NO],
        "Honestly I don't mind.": [NO],  # no preference has the outcome "no"
    }
    for answer, expected in cases.items():
        assert parse_outcomes(slot, beach, answer)[0] == expected, answer
    # Ruling out the asked value counts against "yes"; ruling out another value says nothing.
    assert parse_outcomes(slot, beach, "Not the beach.")[:2] == ([], [YES])
    assert parse_outcomes(slot, beach, "Not the city.")[:2] == ([], [])
    under_200 = Question("budget", YESNO, ("low", "mid"))
    assert parse_outcomes(schema.slot("budget"), under_200, "About 150 a day")[0] == [YES]


def test_choice_answers(schema):
    slot = schema.slot("setting")
    q = Question("setting", CHOICE, ("beach", "mountains"))
    cases = {
        "The mountains, for sure.": ["mountains"],
        "Neither, honestly.": [OTHER],
        "None of those, really.": [OTHER],
        "Somewhere in the countryside.": [OTHER],
        "I don't mind.": [OTHER],
        "Hmm, hard to say.": [],
    }
    for answer, expected in cases.items():
        assert parse_outcomes(slot, q, answer)[0] == expected, answer
    assert parse_outcomes(slot, q, "Not the beach.")[:2] == ([], ["beach"])
    assert parse_outcomes(slot, q, "Not the city.")[:2] == ([], [])  # "other" has many values


def test_oracle_noise_per_question_type(schema):
    rho = {"open": 0.5, "choice": 0.5, "yesno": 0.9}
    sim = OracleSimulator(schema, rho=rho)
    persona = Persona("t", "eval", "direct", {s.name: s.value_names[0] for s in schema.slots})
    q = Question("setting", YESNO, ("beach",))
    rng = random.Random(1)
    n = 10000
    yes = sum(sim.respond(q, "", persona, rng).startswith("Yes") for _ in range(n)) / n
    assert yes == pytest.approx(0.9 + 0.1 / 2, abs=0.015)


# ------------------------------------------------------------------ LLM paths (mock backend)


def test_mock_llm_updater_reads_closed_answers(schema):
    updater = LLMUpdater(build_client({}, PROMPTS, None))
    prior = Belief.prior(schema)
    beach = Question("setting", YESNO, ("beach",))
    belief, record = updater.update(prior, beach, "Would the beach suit you?", "No, not really.")
    assert record["scores"]["beach"] == 2 and record["scores"]["city"] == 9
    assert belief.prob("setting", "beach") < prior.prob("setting", "beach")
    pick = Question("setting", CHOICE, ("beach", "city"))
    belief, _ = updater.update(prior, pick, "Beach or city?", "I'd pick a big city.")
    assert belief.map_value("setting") == "city"


def test_mock_simulator_answers_closed_questions(schema):
    from elicitor.simulator.llm import LLMSimulator, load_styles

    client = build_client({}, PROMPTS, None)
    sim = LLMSimulator(schema, client, load_styles(PROMPTS / "styles.toml"))
    q = Question("pace", YESNO, ("relaxed",))
    for style in ("direct", "vague", "rambling", "indifferent"):
        persona = Persona("t", "eval", style, {s.name: s.value_names[0] for s in schema.slots})
        answers = [sim.respond(q, "?", persona, random.Random(s)) for s in range(40)]
        read = [parse_outcomes(schema.slot("pace"), q, a)[0] for a in answers]
        assert all(r in ([YES], [NO]) for r in read), style
        assert read.count([YES]) > read.count([NO])  # the persona's pace is "relaxed"


def test_phrase_prompt_keeps_options_only_for_closed_questions(schema):
    role = build_roles(PROMPTS)["phrase"]
    slot = schema.slot("setting")
    for kind, rule in KIND_RULES.items():
        ctx = {"slot_description": slot.description, "template": "T?", "kind_rule": rule}
        user = role.prompt.render(ctx)[1]["content"]
        assert rule in user
        assert ('no "this or that" questions' in user) == (kind == OPEN)
