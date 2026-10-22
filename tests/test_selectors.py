import random

import pytest

from elicitor.belief import Belief
from elicitor.episode import Turn
from elicitor.questions import Question
from elicitor.selectors import make_selector


def asked(belief, *slots):
    return [Turn(i, Question(s), "", "", {}, belief, {}) for i, s in enumerate(slots, 1)]


def test_fixed_follows_schema_order(schema):
    belief = Belief.prior(schema)
    sel = make_selector("fixed")
    names = schema.slot_names
    assert sel.choose(belief, [], random.Random(0)) == Question(names[0])
    assert sel.choose(belief, asked(belief, names[0], names[2]), random.Random(0)) == Question(
        names[1]
    )


@pytest.mark.parametrize("name", ["fixed", "random", "max_entropy"])
def test_never_repeats_and_stops_when_exhausted(schema, name):
    belief = Belief.prior(schema)
    sel = make_selector(name)
    rng = random.Random(1)
    history = []
    for _ in schema.slots:
        q = sel.choose(belief, history, rng)
        assert q is not None and q.slot not in {t.question.slot for t in history}
        history += asked(belief, q.slot)
    assert sel.choose(belief, history, rng) is None


def test_random_is_seeded_and_covers_slots(schema):
    belief = Belief.prior(schema)
    sel = make_selector("random")
    first = [sel.choose(belief, [], random.Random(s)).slot for s in range(200)]
    again = [sel.choose(belief, [], random.Random(s)).slot for s in range(200)]
    assert first == again
    assert set(first) == set(schema.slot_names)


def test_max_entropy_picks_most_uncertain_slot(schema):
    belief = Belief.prior(schema)
    sel = make_selector("max_entropy")
    rng = random.Random(0)
    # At the prior, the slots with the most values have the largest entropy; ties keep
    # schema order.
    most = max(len(s.values) for s in schema.slots)
    widest = [s.name for s in schema.slots if len(s.values) == most]
    assert sel.choose(belief, [], rng).slot == widest[0]

    # Make every slot sharp except one 4-value slot: that one must be chosen.
    target = "pace"
    for slot in schema.slots:
        if slot.name != target:
            belief = belief.update(slot.name, [10.0] + [1.0] * (len(slot.values) - 1))
    assert sel.choose(belief, [], rng).slot == target
    assert sel.choose(belief, asked(belief, target), rng).slot != target


def test_unknown_selector():
    with pytest.raises(ValueError, match="unknown selector"):
        make_selector("eig_bayes")
    with pytest.raises(ValueError, match="needs an LLM client"):
        make_selector("eig_mc")
