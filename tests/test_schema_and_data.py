from collections import Counter

import pytest

from elicitor.catalog import load_catalog
from elicitor.datagen import check_data
from elicitor.personas import STYLES, load_personas
from elicitor.schema import CATEGORICAL, NO_PREFERENCE, ORDINAL, load_schema
from tests.conftest import TRAVEL


def test_travel_schema_shape(schema):
    kinds = Counter(s.kind for s in schema.slots)
    assert 6 <= kinds[CATEGORICAL] <= 8
    assert kinds[ORDINAL] == 2
    for slot in schema.slots:
        assert 3 <= len(slot.values) <= 5
        assert (NO_PREFERENCE in slot.value_names) == (slot.kind == CATEGORICAL)
        assert slot.question.endswith("?")
        if slot.kind == ORDINAL:
            assert len(slot.edges) == len(slot.values) - 1


def test_labels_are_keywords(schema):
    for slot in schema.slots:
        for value in slot.values:
            assert value.label.lower() in value.keywords


def test_schema_rejects_bad_edges(tmp_path):
    bad = tmp_path / "schema.toml"
    bad.write_text(
        'domain = "x"\n[slot.s]\nkind = "ordinal"\nquestion = "?"\nedges = [5, 1]\n'
        'values = [{name = "a", label = "a"}, {name = "b", label = "b"},'
        ' {name = "c", label = "c"}]\n'
    )
    with pytest.raises(ValueError, match="edges"):
        load_schema(bad)


def test_committed_data_matches_generator():
    assert check_data(TRAVEL) == []


def test_personas_are_stratified(schema):
    personas = load_personas(TRAVEL / "personas.jsonl")
    assert len({p.id for p in personas}) == len(personas) == 60
    splits = Counter(p.split for p in personas)
    assert splits == {"dev": 12, "eval": 48}
    for split in ("dev", "eval"):
        styles = Counter(p.style for p in personas if p.split == split)
        assert set(styles) == set(STYLES) and len(set(styles.values())) == 1
    for slot in schema.slots:
        counts = Counter(p.truth[slot.name] for p in personas)
        assert set(counts) == set(slot.value_names)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_catalog_covers_every_concrete_value(schema):
    items = load_catalog(TRAVEL / "catalog.jsonl")
    assert len(items) == 200
    for slot in schema.slots:
        values = Counter(item.attrs[slot.name] for item in items)
        assert set(values) == set(slot.concrete_values)
        assert NO_PREFERENCE not in values
