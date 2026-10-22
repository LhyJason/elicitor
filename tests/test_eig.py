"""Answer model and analytic EIG (SPEC §7, Decision Q7 A), and the EIG selectors."""

import math
import random

import pytest

from elicitor.answer_model import AnswerModel, parse_rho
from elicitor.belief import Belief, entropy
from elicitor.episode import Turn
from elicitor.questions import OPEN, YESNO, Question, candidates, outcomes
from elicitor.selectors import make_selector
from elicitor.selectors.eig import EIGSelector


def random_belief(schema, rng, sparse=False):
    """Random full-support beliefs; with `sparse`, some values get (almost) no mass."""
    probs = {}
    for slot in schema.slots:
        weights = [rng.expovariate(1.0) for _ in slot.values]
        if sparse:
            weights = [w if rng.random() < 0.6 else 1e-9 for w in weights]
        probs[slot.name] = weights
    return Belief(schema, probs)


def all_candidates(schema):
    return [q for slot in schema.slots for q in candidates(slot)]


def brute_force_eig(model, belief, question):
    """Expected drop in the slot's entropy, by enumerating outcomes and applying Bayes rule."""
    slot = belief.schema.slot(question.slot)
    prior = belief.probs(slot.name)
    rows = [model.outcome_probs(slot, question, v) for v in slot.value_names]
    expected_posterior_entropy = 0.0
    for j in range(len(outcomes(slot, question))):
        joint = [p * row[j] for p, row in zip(prior, rows, strict=True)]
        p_outcome = sum(joint)
        if p_outcome > 0:
            expected_posterior_entropy += p_outcome * entropy([x / p_outcome for x in joint])
    return entropy(prior) - expected_posterior_entropy


# ------------------------------------------------------------------ answer model


def test_outcome_probabilities_sum_to_one(schema):
    model = AnswerModel({"open": 0.6, "choice": 0.75, "yesno": 0.9})
    for q in all_candidates(schema):
        slot = schema.slot(q.slot)
        for v in slot.value_names:
            probs = model.outcome_probs(slot, q, v)
            assert len(probs) == len(outcomes(slot, q))
            assert sum(probs) == pytest.approx(1.0)


def test_sampling_matches_outcome_probabilities(schema):
    model = AnswerModel({"open": 0.6, "choice": 0.75, "yesno": 0.9})
    slot = schema.slot("setting")
    q = Question("setting", "choice", ("beach", "city"))
    rng = random.Random(3)
    n = 20000
    draws = [model.sample(slot, q, "mountains", rng) for _ in range(n)]
    for outcome, p in zip(
        outcomes(slot, q), model.outcome_probs(slot, q, "mountains"), strict=True
    ):
        assert draws.count(outcome) / n == pytest.approx(p, abs=0.015)


def test_parse_rho():
    assert parse_rho(0.7) == {"open": 0.7, "choice": 0.7, "yesno": 0.7}
    assert parse_rho({"open": 0.5, "choice": 0.6, "yesno": 0.7})["yesno"] == 0.7
    for bad in (1.2, -0.1, {"open": 0.5}, {"open": 0.5, "choice": 0.6, "yesno": 0.7, "x": 1}):
        with pytest.raises(ValueError):
            parse_rho(bad)


# ------------------------------------------------------------------ EIG properties


def test_eig_equals_entropy_for_noise_free_open_questions(schema):
    """The M3 acceptance test: with rho = 1, asking slot s gains exactly H(b_s)."""
    model = AnswerModel(1.0)
    rng = random.Random(0)
    for trial in range(50):
        belief = random_belief(schema, rng, sparse=trial % 2 == 1)
        for slot in schema.slots:
            assert model.eig(belief, Question(slot.name)) == pytest.approx(
                belief.entropy(slot.name)
            )


def test_yes_no_eig_is_at_most_one_bit(schema):
    """The M3 acceptance test: a yes/no answer carries at most 1 bit, at any noise level."""
    rng = random.Random(1)
    yes_no = [q for q in all_candidates(schema) if q.kind == YESNO]
    for _ in range(50):
        belief = random_belief(schema, rng)
        model = AnswerModel(rng.random())
        for q in yes_no:
            assert 0 <= model.eig(belief, q) <= 1 + 1e-12


def test_noise_free_yes_no_eig_is_the_binary_entropy(schema):
    belief = random_belief(schema, random.Random(2))
    model = AnswerModel(1.0)
    for q in all_candidates(schema):
        if q.kind == YESNO:
            p_yes = sum(belief.prob(q.slot, v) for v in q.values)
            assert model.eig(belief, q) == pytest.approx(entropy([p_yes, 1 - p_yes]))


def test_eig_is_the_expected_entropy_drop(schema):
    """EIG = H(A) - H(A|V) equals E_A[H(V) - H(V | A)], computed by Bayes rule per outcome."""
    rng = random.Random(4)
    for _ in range(10):
        belief = random_belief(schema, rng)
        model = AnswerModel({"open": rng.random(), "choice": rng.random(), "yesno": rng.random()})
        for q in all_candidates(schema):
            assert model.eig(belief, q) == pytest.approx(
                brute_force_eig(model, belief, q), abs=1e-9
            )


def test_eig_bounds_and_noise(schema):
    rng = random.Random(5)
    belief = random_belief(schema, rng)
    for q in all_candidates(schema):
        slot = schema.slot(q.slot)
        bound = min(belief.entropy(q.slot), math.log2(len(outcomes(slot, q))))
        gains = [AnswerModel(rho).eig(belief, q) for rho in (0.0, 0.3, 0.6, 0.9, 1.0)]
        assert gains[0] == pytest.approx(0.0, abs=1e-12)  # pure noise tells nothing
        assert gains == sorted(gains)  # more reliable answers are worth more
        assert gains[-1] <= bound + 1e-12


# ------------------------------------------------------------------ selectors


def asked(belief, *questions):
    return [Turn(i, q, "", "", {}, belief, {}) for i, q in enumerate(questions, 1)]


def test_noise_free_eig_agrees_with_max_entropy(schema):
    """With rho = 1 no question beats the open question on the most uncertain slot."""
    eig = make_selector("eig", rho=1.0)
    max_entropy = make_selector("max_entropy")
    rng = random.Random(6)
    for _ in range(30):
        belief = random_belief(schema, rng)
        assert eig.choose(belief, [], rng) == max_entropy.choose(belief, [], rng)


def test_reliable_yes_no_beats_noisy_open_question(schema):
    """Two values left on one slot: a reliable yes/no settles it better than a noisy open one."""
    probs = {s.name: [1.0] + [1e-6] * (len(s.values) - 1) for s in schema.slots}
    probs["setting"] = [0.5, 0.5, 1e-6, 1e-6, 1e-6]
    belief = Belief(schema, probs)
    rng = random.Random(0)
    noisy_open = make_selector("eig", rho={"open": 0.5, "choice": 0.5, "yesno": 0.95})
    q = noisy_open.choose(belief, [], rng)
    assert q.slot == "setting" and q.kind == YESNO
    reliable_open = make_selector("eig", rho=0.95)
    assert reliable_open.choose(belief, [], rng) == Question("setting")


def test_eig_never_repeats_a_question_but_may_revisit_a_slot(schema):
    sel = make_selector("eig", rho=0.7)
    belief = Belief.prior(schema)
    rng = random.Random(0)
    history = []
    seen = set()
    while (q := sel.choose(belief, history, rng)) is not None:
        assert q not in seen
        seen.add(q)
        history += asked(belief, q)
    assert seen == set(all_candidates(schema))  # every question, then None


def test_eig_open_asks_open_questions_once_per_slot(schema):
    sel = make_selector("eig_open", rho=0.7)
    belief = Belief.prior(schema)
    rng = random.Random(0)
    history = []
    while (q := sel.choose(belief, history, rng)) is not None:
        assert q.kind == OPEN
        history += asked(belief, q)
    assert sorted(t.question.slot for t in history) == sorted(schema.slot_names)


def test_eig_ties_follow_candidate_order(schema):
    sel = EIGSelector(kinds=("open",), rho=1.0)
    # At the prior every 5-value slot ties; the first one in schema order wins.
    widest = max(len(s.values) for s in schema.slots)
    first = next(s.name for s in schema.slots if len(s.values) == widest)
    assert sel.choose(Belief.prior(schema), [], random.Random(0)) == Question(first)


def test_eig_selector_validates_kinds():
    with pytest.raises(ValueError):
        EIGSelector(kinds=())
    with pytest.raises(ValueError):
        EIGSelector(kinds=("open", "essay"))
