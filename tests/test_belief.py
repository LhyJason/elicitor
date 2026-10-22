import math

import pytest

from elicitor.belief import Belief, entropy
from elicitor.metrics import belief_metrics


def test_entropy_known_values():
    assert entropy([1.0]) == 0
    assert entropy([1.0, 0.0, 0.0]) == 0
    assert entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert entropy([0.25] * 4) == pytest.approx(2.0)
    assert entropy([0.5, 0.25, 0.25]) == pytest.approx(1.5)


@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_uniform_entropy_is_log2_n(n):
    assert entropy([1 / n] * n) == pytest.approx(math.log2(n))


def test_entropy_is_maximal_for_uniform():
    n = 4
    for p in ([0.7, 0.1, 0.1, 0.1], [0.4, 0.4, 0.1, 0.1], [0.3, 0.3, 0.2, 0.2]):
        assert 0 < entropy(p) < math.log2(n)


def test_prior_is_normalized_and_uniform(schema):
    belief = Belief.prior(schema)
    for slot in schema.slots:
        probs = belief.probs(slot.name)
        assert sum(probs) == pytest.approx(1.0)
        assert belief.entropy(slot.name) == pytest.approx(math.log2(len(slot.values)))


def test_update_is_bayes_rule(schema):
    belief = Belief.prior(schema)
    slot = schema.slot("setting")  # 5 values incl. no_preference
    likelihood = [4.0, 1.0, 1.0, 1.0, 1.0]
    posterior = belief.update("setting", likelihood).probs("setting")
    assert posterior == pytest.approx([0.5, 0.125, 0.125, 0.125, 0.125])
    assert len(posterior) == len(slot.values)


def test_update_twice_multiplies_likelihoods(schema):
    belief = Belief.prior(schema)
    a, b = [2.0, 1.0, 1.0, 1.0], [3.0, 1.0, 2.0, 1.0]
    twice = belief.update("climate", a).update("climate", b)
    once = belief.update("climate", [x * y for x, y in zip(a, b, strict=True)])
    assert twice.probs("climate") == pytest.approx(once.probs("climate"))


def test_update_ignores_likelihood_scale(schema):
    belief = Belief.prior(schema)
    lik = [0.2, 0.1, 0.05, 0.05]
    scaled = [x * 37 for x in lik]
    assert belief.update("pace", lik).probs("pace") == pytest.approx(
        belief.update("pace", scaled).probs("pace")
    )


def test_update_leaves_other_slots_and_original_unchanged(schema):
    belief = Belief.prior(schema)
    new = belief.update("food", [1.0, 0.0, 0.0, 0.0])
    assert belief.probs("food") == schema.slot("food").prior
    for name in schema.slot_names:
        if name != "food":
            assert new.probs(name) == belief.probs(name)


def test_update_rejects_impossible_and_malformed_likelihoods(schema):
    belief = Belief.prior(schema).update("food", [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        belief.update("food", [0.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        belief.update("food", [1.0, 1.0])
    with pytest.raises(ValueError):
        belief.update("food", [1.0, -1.0, 1.0, 1.0])


def test_map_value_and_ties(schema):
    belief = Belief.prior(schema)
    assert belief.map_value("pace") == schema.slot("pace").value_names[0]
    belief = belief.update("pace", [1.0, 1.0, 3.0, 1.0])
    assert belief.map_value("pace") == "packed"
    assert belief.max_prob("pace") == pytest.approx(0.5)


def test_confidence_needs_every_slot(schema):
    belief = Belief.prior(schema)
    assert not belief.is_confident(0.9)
    for slot in schema.slots:
        assert not belief.is_confident(0.9)
        lik = [1.0] + [0.0] * (len(slot.values) - 1)
        belief = belief.update(slot.name, lik)
    assert belief.is_confident(0.9)


def test_belief_metrics_at_prior_and_at_truth(schema):
    truth = {s.name: s.value_names[1] for s in schema.slots}
    prior = Belief.prior(schema)
    m = belief_metrics(prior, truth)
    n_values = [len(s.values) for s in schema.slots]
    assert m["mass_on_truth"] == pytest.approx(sum(1 / n for n in n_values) / len(n_values))
    assert m["map_accuracy"] == 0  # ties go to the first value, truth is the second
    assert m["mean_entropy"] == pytest.approx(m["log_loss"])  # uniform: both are mean log2 n

    point = Belief(schema, {s.name: [float(v == truth[s.name]) for v in s.value_names]
                            for s in schema.slots})  # fmt: skip
    m = belief_metrics(point, truth)
    assert m == pytest.approx(
        {"mass_on_truth": 1, "map_accuracy": 1, "log_loss": 0, "mean_entropy": 0}
    )
