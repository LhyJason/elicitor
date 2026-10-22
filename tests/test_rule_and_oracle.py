import random
from collections import Counter

import pytest

from elicitor.belief import Belief
from elicitor.personas import Persona
from elicitor.questions import Question
from elicitor.simulator.oracle import OracleSimulator, render_answer
from elicitor.updaters.rule import RuleUpdater, find_mentions


def test_every_oracle_answer_is_understood(schema):
    """Round trip: the rule updater reads exactly the value the oracle named."""
    updater = RuleUpdater(rho=0.8)
    for slot in schema.slots:
        for value in slot.value_names:
            answer = render_answer(slot, value)
            assert find_mentions(slot, answer) == ([value], []), answer
            belief, record = updater.update(Belief.prior(schema), Question(slot.name), answer)
            assert record["matched"]
            assert belief.map_value(slot.name) == value


def test_rule_likelihood_matches_answer_model(schema):
    """With a uniform prior, the named value's posterior is P(true = a | answer names a)."""
    rho = 0.7
    slot = schema.slot("setting")
    n = len(slot.values)
    belief, _ = RuleUpdater(rho).update(Belief.prior(schema), Question("setting"), "The beach!")
    p_named_if_true = rho + (1 - rho) / n
    p_named_if_other = (1 - rho) / n
    expected = p_named_if_true / (p_named_if_true + (n - 1) * p_named_if_other)
    assert belief.prob("setting", "beach") == pytest.approx(expected)


def test_negation_counts_against(schema):
    slot = schema.slot("setting")
    assert find_mentions(slot, "Not the beach, I'd rather see a city.") == (["city"], ["beach"])
    assert find_mentions(slot, "No, mountains please") == (["mountains"], [])
    belief, _ = RuleUpdater().update(Belief.prior(schema), Question("setting"), "Avoid the coast.")
    assert belief.prob("setting", "beach") < min(
        belief.prob("setting", v) for v in slot.value_names if v != "beach"
    )


def test_no_preference_phrases(schema):
    slot = schema.slot("pace")
    assert find_mentions(slot, "Honestly I don't mind.") == (["no_preference"], [])
    assert find_mentions(slot, "Either way is fine") == (["no_preference"], [])


def test_numbers_fill_ordinal_bins(schema):
    updater = RuleUpdater()
    prior = Belief.prior(schema)
    cases = [
        ("budget", "Maybe around $150 per day?", "mid"),
        ("budget", "between 200 and 300", "high"),
        ("budget", "I can do 1,000 a day", "luxury"),
        ("trip_length", "5 days", "short"),
        ("trip_length", "12", "long"),
    ]
    for slot, answer, value in cases:
        belief, record = updater.update(prior, Question(slot), answer)
        assert belief.map_value(slot) == value, answer
        assert "number" in record


def test_unmatched_answer_leaves_belief_unchanged(schema):
    prior = Belief.prior(schema)
    belief, record = RuleUpdater().update(prior, Question("food"), "Hmm, hard to say.")
    assert belief is prior and not record["matched"]


def test_rule_updater_rejects_degenerate_rho():
    for rho in (0, 1, 1.5):
        with pytest.raises(ValueError):
            RuleUpdater(rho)


def persona(schema):
    return Persona("t1", "eval", "direct", {s.name: s.value_names[0] for s in schema.slots})


def test_noise_free_oracle_tells_the_truth(schema):
    sim = OracleSimulator(schema, rho=1.0)
    p = persona(schema)
    for slot in schema.slots:
        answer = sim.respond(Question(slot.name), "", p, random.Random(0))
        assert answer == render_answer(slot, p.truth[slot.name])


def test_oracle_noise_rate(schema):
    rho = 0.6
    sim = OracleSimulator(schema, rho=rho)
    slot = schema.slot("setting")
    p = persona(schema)
    rng = random.Random(42)
    n_draws = 20000
    counts = Counter(sim.respond(Question("setting"), "", p, rng) for _ in range(n_draws))
    truthful = counts[render_answer(slot, p.truth["setting"])] / n_draws
    assert truthful == pytest.approx(rho + (1 - rho) / len(slot.values), abs=0.015)
