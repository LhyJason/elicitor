"""Recommender by expected utility and outcome metrics (SPEC §10, §12)."""

import random

import pytest

from elicitor.belief import Belief
from elicitor.catalog import Item, load_catalog
from elicitor.metrics import HIT_KS, OUTCOME_METRICS, outcome_metrics
from elicitor.personas import load_personas
from elicitor.recommender import Recommender, match, utility, violations
from tests.conftest import TRAVEL


@pytest.fixture(scope="module")
def recommender(schema):
    return Recommender(schema, load_catalog(TRAVEL / "catalog.jsonl"))


@pytest.fixture(scope="module")
def personas():
    return load_personas(TRAVEL / "personas.jsonl", split="eval")


def point_mass(schema, truth):
    return Belief(
        schema, {s.name: [float(v == truth[s.name]) for v in s.value_names] for s in schema.slots}
    )


def test_match_scores(schema):
    setting = schema.slot("setting")
    assert match(setting, "beach", "beach") == 1
    assert match(setting, "beach", "city") == 0
    assert match(setting, "beach", "no_preference") == 1
    budget = schema.slot("budget")
    assert [match(budget, "low", v) for v in budget.value_names] == pytest.approx(
        [1, 2 / 3, 1 / 3, 0]
    )


def test_violations(schema):
    truth = {s.name: s.value_names[0] for s in schema.slots} | {
        "budget": "mid",
        "trip_length": "week",
    }
    attrs = dict(truth)
    assert violations(schema, attrs, truth) == 0
    assert violations(schema, attrs | {"budget": "low"}, truth) == 0  # cheaper is fine (at_most)
    assert violations(schema, attrs | {"budget": "high"}, truth) == 1
    assert violations(schema, attrs | {"trip_length": "short"}, truth) == 1  # exact
    assert violations(schema, attrs | {"budget": "luxury", "trip_length": "long"}, truth) == 2


def test_expected_utility_of_a_point_mass_is_the_true_utility(schema, recommender, personas):
    for p in personas[:10]:
        eu = recommender.expected_utilities(point_mass(schema, p.truth))
        assert eu == pytest.approx(recommender.true_utilities(p.truth))
        assert eu == pytest.approx([utility(schema, i.attrs, p.truth) for i in recommender.catalog])


def test_knowing_the_truth_means_no_regret(schema, recommender, personas):
    for p in personas:
        ranking = recommender.rank(point_mass(schema, p.truth))
        m = outcome_metrics(recommender, ranking, p.truth)
        assert m["regret"] == pytest.approx(0, abs=1e-12)
        assert all(m[f"hit@{k}"] == 1 for k in HIT_KS)


def test_outcome_metrics_are_consistent(schema, recommender, personas):
    rng = random.Random(0)
    for p in personas:
        belief = Belief(schema, {s.name: [rng.random() for _ in s.values] for s in schema.slots})
        m = outcome_metrics(recommender, recommender.rank(belief), p.truth)
        assert set(m) == set(OUTCOME_METRICS)
        assert 0 <= m["regret"] <= 1
        hits = [m[f"hit@{k}"] for k in HIT_KS]
        assert hits == sorted(hits)  # a longer list can only hit more
        assert m["violations"] in (0, 1, 2)


def test_rank_ties_keep_catalog_order(schema):
    attrs = {s.name: s.concrete_values[0] for s in schema.slots}
    catalog = [Item("b", attrs), Item("a", attrs), Item("c", attrs)]
    rec = Recommender(schema, catalog)
    assert rec.rank(Belief.prior(schema)) == [0, 1, 2]
