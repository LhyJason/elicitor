"""LLM updater, phraser and simulator, leakage, config and the mock suite end to end."""

import json
import random

import pytest

from elicitor.belief import Belief
from elicitor.experiments.config import Arm, load_config
from elicitor.experiments.runner import run_experiment
from elicitor.llm import build_client
from elicitor.llm.client import LLMClient
from elicitor.llm.providers import Completion
from elicitor.llm.roles import build_roles
from elicitor.metrics import leaked_slots
from elicitor.personas import load_personas
from elicitor.phrasing import LLMPhraser, make_phraser
from elicitor.questions import Question
from elicitor.simulator import make_simulator
from elicitor.updaters import make_updater
from elicitor.updaters.llm import LLMUpdater, likelihood
from tests.conftest import ROOT, TRAVEL

PROMPTS = TRAVEL / "prompts"
M2_MOCK = ROOT / "experiments" / "m2_mock.toml"
M2_SMOKE = ROOT / "experiments" / "m2_smoke_ollama.toml"


class AlwaysBad:
    name = "bad"

    def __init__(self):
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        return Completion(text="not json")


def mock_client():
    return build_client({}, PROMPTS, None)


def bad_client():
    return LLMClient(build_roles(PROMPTS), AlwaysBad(), {"agent": "m", "user": "m"})


# ------------------------------------------------------------------ LLM updater


def test_likelihood_mapping():
    lik = likelihood([0, 5, 10])
    assert lik[2] == 1.0
    assert lik[0] == pytest.approx(0.5 / 10.5)
    assert likelihood([3, 3, 3]) == [likelihood([3])[0]] * 3
    tempered = likelihood([0, 10], beta=2)
    assert tempered[0] == pytest.approx((0.5 / 10.5) ** 2)


def test_llm_updater_moves_mass_to_the_named_value(schema):
    updater = LLMUpdater(mock_client())
    prior = Belief.prior(schema)
    belief, record = updater.update(prior, Question("setting"), "Q?", "The mountains, for sure.")
    assert belief.map_value("setting") == "mountains"
    assert belief.prob("setting", "mountains") > prior.prob("setting", "mountains")
    assert record["scores"]["mountains"] == 9
    assert belief.probs("climate") == prior.probs("climate")


def test_llm_updater_without_information_changes_nothing(schema):
    updater = LLMUpdater(mock_client())
    prior = Belief.prior(schema)
    belief, record = updater.update(prior, Question("food"), "Q?", "Hmm, hard to say.")
    assert belief.probs("food") == pytest.approx(prior.probs("food"))
    assert set(record["scores"].values()) == {5}


def test_llm_updater_skips_empty_answers_and_survives_failures(schema):
    client = bad_client()
    updater = LLMUpdater(client)
    prior = Belief.prior(schema)
    belief, record = updater.update(prior, Question("pace"), "Q?", "   ")
    assert belief is prior and record["skipped"] == "empty answer"
    assert client.provider.calls == 0
    belief, record = updater.update(prior, Question("pace"), "Q?", "Slow, please.")
    assert belief is prior and "failed" in record
    assert client.stats["update"].failed_calls == 1


def test_make_updater_needs_a_client():
    with pytest.raises(ValueError, match="client"):
        make_updater("llm")
    with pytest.raises(ValueError, match="unknown updater"):
        make_updater("magic")


# ------------------------------------------------------------------ phraser and simulator


def test_llm_phraser_uses_the_reply_or_falls_back(schema):
    q = Question("pace")
    text = LLMPhraser(schema, mock_client()).render(q, random.Random(1))
    assert text.startswith("Quick question") and text.endswith("?")
    fallback = LLMPhraser(schema, bad_client()).render(q, random.Random(1))
    assert fallback == schema.slot("pace").question
    assert make_phraser("tpl", schema).render(q) == schema.slot("pace").question


def test_llm_simulator_answers_and_degrades_to_empty(schema):
    persona = load_personas(TRAVEL / "personas.jsonl", split="dev")[0]
    sim = make_simulator(schema, "llm", client=mock_client(), prompts_dir=PROMPTS)
    q = Question("setting")
    a = sim.respond(q, "Where to?", persona, random.Random(3))
    b = sim.respond(q, "Where to?", persona, random.Random(3))
    assert a == b and a
    broken = make_simulator(schema, "llm", client=bad_client(), prompts_dir=PROMPTS)
    assert broken.respond(q, "Where to?", persona, random.Random(3)) == ""
    with pytest.raises(ValueError, match="client"):
        make_simulator(schema, "llm")


def test_backstories_are_merged_from_their_own_file(tmp_path):
    src = (TRAVEL / "personas.jsonl").read_text(encoding="utf-8")
    (tmp_path / "personas.jsonl").write_text(src, encoding="utf-8")
    (tmp_path / "backstories.jsonl").write_text(
        json.dumps({"id": "p001", "backstory": "Loves trains."}) + "\n", encoding="utf-8"
    )
    personas = {p.id: p for p in load_personas(tmp_path / "personas.jsonl")}
    assert personas["p001"].backstory == "Loves trains."
    assert personas["p002"].backstory == ""


# ------------------------------------------------------------------ leakage


def test_leaked_slots(schema):
    assert leaked_slots(schema, "climate", "Somewhere hot, by the beach.") == ["setting"]
    assert leaked_slots(schema, "setting", "By the beach.") == []
    # "I don't mind" is a no_preference phrase in every slot; it is not a leak.
    assert leaked_slots(schema, "pace", "I don't mind, really.") == []


# ------------------------------------------------------------------ config


def test_arm_names_round_trip():
    arm = Arm("max_entropy", "llm", "llm")
    assert Arm.parse(arm.name) == arm
    for bad in ("sel=x,upd=y", "sel=x,upd=y,phr=z,extra=1", "nonsense"):
        with pytest.raises(ValueError):
            Arm.parse(bad)


def test_m2_configs_parse():
    mock = load_config(M2_MOCK)
    names = [a.name for a in mock.arms]
    assert len(names) == len(set(names)) == 3 * 2 + 1
    assert names[-1] == "sel=max_entropy,upd=llm,phr=llm"
    assert mock.needs_llm and mock.llm["provider"] == "mock"
    smoke = load_config(M2_SMOKE)
    assert smoke.split == "dev" and smoke.llm["provider"] == "ollama"
    assert load_config(ROOT / "experiments" / "m1_no_llm.toml").needs_llm is False


def test_extra_arm_already_in_grid_is_not_duplicated(tmp_path):
    text = M2_MOCK.read_text(encoding="utf-8").replace(
        'extra_arms = ["sel=max_entropy,upd=llm,phr=llm"]',
        'extra_arms = ["sel=fixed,upd=rule,phr=tpl"]',
    )
    path = tmp_path / "x.toml"
    data_dir = f'data_dir = "{TRAVEL.parent.as_posix()}"'
    path.write_text(text.replace('data_dir = "../data"', data_dir), encoding="utf-8")
    assert len(load_config(path).arms) == 6


# ------------------------------------------------------------------ mock suite


def test_mock_suite_runs_and_reruns_from_cache(tmp_path):
    config = load_config(M2_MOCK)
    rows = run_experiment(config, tmp_path, max_personas=4)
    assert len(rows) == len(config.arms) * len(config.budgets)
    stats = json.loads((tmp_path / "llm_stats.json").read_text(encoding="utf-8"))["roles"]
    assert set(stats) == {"update", "phrase", "simulate"}
    for role in stats.values():
        assert role["failed_calls"] == 0 and role["transport_errors"] == 0
    episodes_before = (tmp_path / "episodes.jsonl").read_bytes()

    run_experiment(config, tmp_path, max_personas=4)
    stats = json.loads((tmp_path / "llm_stats.json").read_text(encoding="utf-8"))["roles"]
    assert all(r["attempts"] == 0 and r["cache_hits"] == r["calls"] for r in stats.values())
    assert (tmp_path / "episodes.jsonl").read_bytes() == episodes_before


def test_mock_turns_record_text_answer_and_leaks(tmp_path):
    config = load_config(M2_MOCK)
    run_experiment(config, tmp_path, max_personas=2)
    rows = [
        json.loads(line)
        for line in (tmp_path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    llm_phrased = [r for r in rows if r["arm"].endswith("phr=llm")]
    turn = llm_phrased[0]["turns"][0]
    assert turn["text"].startswith("Quick question")
    assert turn["answer"] and "scores" in turn["parsed"] and "leaks" in turn


def test_mock_client_is_not_used_for_rule_only_runs(tmp_path):
    run_experiment(load_config(ROOT / "experiments" / "m1_no_llm.toml"), tmp_path, max_personas=1)
    assert not (tmp_path / "llm_stats.json").exists()
