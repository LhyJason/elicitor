"""M5: several servers, the parallel runner, per-arm budgets, eig_mc, calibration, backstories."""

import dataclasses
import json
import threading
import time

import pytest

from elicitor.backstories import flagged_slots, write_backstories
from elicitor.belief import Belief
from elicitor.calibration import calibrate, calibration_questions, rho_hat
from elicitor.cli import main
from elicitor.experiments.config import Arm, load_config
from elicitor.experiments.health import check_health
from elicitor.experiments.runner import run_experiment
from elicitor.llm import build_client
from elicitor.llm.cache import ReplyCache
from elicitor.llm.client import LLMClient
from elicitor.llm.providers import Completion, MockProvider, OllamaProvider, RoutedProvider
from elicitor.llm.roles import build_roles
from elicitor.personas import load_personas
from elicitor.questions import CHOICE, OPEN, YESNO
from elicitor.selectors import make_selector
from elicitor.simulator.llm import load_styles
from tests.conftest import ROOT, TRAVEL
from tests.test_llm_layer import GOOD_UPDATE, update_context

PROMPTS = TRAVEL / "prompts"
M5_MOCK = ROOT / "experiments" / "m5_mock.toml"
M3_EIG = ROOT / "experiments" / "m3_eig.toml"


@pytest.fixture(scope="module")
def roles():
    return build_roles(PROMPTS)


class Slow:
    """Provider that holds every request until released, counting what it was sent."""

    def __init__(self, name="ollama", text=GOOD_UPDATE):
        self.name = name
        self.text = text
        self.release = threading.Event()
        self.started = 0
        self.lock = threading.Lock()

    def complete(self, request):
        with self.lock:
            self.started += 1
        assert self.release.wait(5)
        return Completion(text=self.text)


# ---------------------------------------------------------------- servers


def test_routed_provider_picks_the_least_busy_server_of_the_group(roles, schema):
    agents, users = [Slow(), Slow()], [Slow()]
    routed = RoutedProvider({"agent": agents, "user": users})
    client = LLMClient(roles, routed, {"agent": "a", "user": "u"}, retry_wait_s=0.0)
    threads = [
        threading.Thread(target=client.call, args=("update", update_context(schema, f"x{i}")))
        for i in range(3)
    ]
    for t in threads:
        t.start()
    deadline = time.time() + 5
    while sum(s.started for s in agents) < 3 and time.time() < deadline:
        time.sleep(0.01)
    assert [s.started for s in agents] == [2, 1] and users[0].started == 0
    for s in agents:
        s.release.set()
    for t in threads:
        t.join()
    assert routed.sent == {"agent": [2, 1], "user": [0]}
    assert routed.in_flight == {"agent": [0, 0], "user": [0]}


def test_routed_provider_needs_one_provider_kind():
    with pytest.raises(ValueError, match="one provider kind"):
        RoutedProvider({"agent": [Slow("ollama")], "user": [Slow("mock")]})
    with pytest.raises(ValueError, match="at least one server"):
        RoutedProvider({"agent": [], "user": [Slow()]})


def test_server_lists_from_the_table_and_the_environment(monkeypatch):
    config = {"provider": "ollama", "models": {"agent": "a", "user": "u"}, "cache": False}
    table = {**config, "base_url": {"agent": ["http://h:1", "http://h:2"], "user": "http://h:3"}}
    provider = build_client(table, PROMPTS, None).provider
    assert isinstance(provider, RoutedProvider) and provider.name == "ollama"
    assert [p.base_url for p in provider.pools["agent"]] == ["http://h:1", "http://h:2"]
    monkeypatch.setenv("ELICITOR_SERVERS_AGENT", "http://h:4,http://h:5")
    monkeypatch.setenv("ELICITOR_SERVERS_USER", "http://h:6")
    provider = build_client(config, PROMPTS, None).provider
    assert [p.base_url for p in provider.pools["agent"]] == ["http://h:4", "http://h:5"]
    assert [p.base_url for p in provider.pools["user"]] == ["http://h:6"]
    monkeypatch.delenv("ELICITOR_SERVERS_USER")
    with pytest.raises(ValueError, match="exactly the groups"):
        build_client(config, PROMPTS, None)


def test_think_can_be_set_per_model_group(roles, schema):
    provider = OllamaProvider(think={"agent": False})
    client = LLMClient(roles, MockProvider(), {"agent": "a", "user": "u"})
    from elicitor.llm.providers import Request

    def payload(role):
        return provider.payload(
            Request(roles[role], {}, "m", [], {}, temperature=0, max_tokens=1, seed=0)
        )

    assert payload("update")["think"] is False
    assert "think" not in payload("simulate")
    client.close()


def test_identical_requests_in_flight_are_sent_once(tmp_path, roles, schema):
    slow = Slow()
    client = LLMClient(
        roles, slow, {"agent": "a", "user": "u"}, cache=ReplyCache(tmp_path / "c.sqlite")
    )
    replies = []
    tallies = []

    def call():
        with client.track() as tally:
            replies.append(client.call("update", update_context(schema)))
            tallies.append(tally["update"])

    threads = [threading.Thread(target=call) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    slow.release.set()
    for t in threads:
        t.join()
    assert slow.started == 1
    assert len({r.model_dump_json() for r in replies}) == 1
    assert sum(t.calls for t in tallies) == 4 and sum(t.cache_hits for t in tallies) == 3
    assert sum(t.attempts for t in tallies) == 1
    client.close()


# ---------------------------------------------------------------- runner


def test_parallel_run_writes_the_same_cells(tmp_path):
    config = dataclasses.replace(load_config(M5_MOCK), replicates=2)
    run_experiment(config, tmp_path / "one", max_personas=4)
    run_experiment(config, tmp_path / "many", max_personas=4, workers=6)
    one, many = tmp_path / "one", tmp_path / "many"
    names = sorted(p.name for p in (one / "cells").glob("*.jsonl"))
    assert names == sorted(p.name for p in (many / "cells").glob("*.jsonl"))
    for name in names:
        assert (one / "cells" / name).read_bytes() == (many / "cells" / name).read_bytes()
    for name in ("episodes.jsonl", "summary.csv"):
        assert (one / name).read_bytes() == (many / name).read_bytes()
    a = json.loads((one / "llm_stats.json").read_text(encoding="utf-8"))["roles"]
    b = json.loads((many / "llm_stats.json").read_text(encoding="utf-8"))["roles"]
    for role in a:
        assert (a[role]["calls"], a[role]["attempts"]) == (b[role]["calls"], b[role]["attempts"])


def test_arm_budgets_limit_an_arm(tmp_path):
    config = load_config(M5_MOCK)
    mc = Arm("eig_mc", "llm", "tpl")
    assert config.budgets_for(mc) == (5,) and config.budgets_for(Arm("eig", "llm")) == (3, 5)
    config = dataclasses.replace(
        config, selectors=("max_entropy",), updaters=("llm",), extra_arms=(mc,)
    )
    run_experiment(config, tmp_path, max_personas=2)
    episodes = [json.loads(line) for line in (tmp_path / "episodes.jsonl").open(encoding="utf-8")]
    assert {e["budget"] for e in episodes if e["arm"] == mc.name} == {5}
    assert len(episodes) == 2 * 2 + 2
    assert "episodes 6/6" in check_health(tmp_path).text()


def test_arm_budgets_are_checked(tmp_path):
    text = M5_MOCK.read_text(encoding="utf-8").replace(
        '"sel=eig_mc,upd=llm,phr=tpl" = [5]', '"sel=eig_mc,upd=llm,phr=tpl" = [4]'
    )
    bad = tmp_path / "bad.toml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="subset of budgets"):
        load_config(bad)


# ---------------------------------------------------------------- selectors


def test_entropy_choice_lists_the_options_of_the_most_uncertain_slot(schema):
    selector = make_selector("entropy_choice", rho={"open": 0.7, "choice": 0.8, "yesno": 0.9})
    belief = Belief.prior(schema)
    q = selector.choose(belief, [], None)
    assert q.kind == CHOICE
    slot = schema.slot(q.slot)
    assert q.slot == max(schema.slot_names, key=belief.entropy)
    if len(slot.concrete_values) == 3:
        assert set(q.values) == set(slot.concrete_values)


def test_eig_mc_rescores_the_shortlist_from_the_cache(tmp_path, roles, schema):
    client = LLMClient(
        roles, MockProvider(), {"agent": "m", "user": "m"}, cache=ReplyCache(tmp_path / "c")
    )
    rho = {"open": 0.7, "choice": 0.8, "yesno": 0.9}
    selector = make_selector("eig_mc", client=client, seed=1, rho=rho, shortlist=3, samples=2)
    belief = Belief.prior(schema)
    shortlist = [q for q, _ in sorted(selector.analytic.scores(belief, []), key=lambda x: -x[1])]
    with client.track() as first:
        q = selector.choose(belief, [], None)
    assert q in shortlist[:3]
    assert first["imagine"].attempts > 0
    with client.track() as second:
        assert selector.choose(belief, [], None) == q
    assert second["imagine"].attempts == 0 and second["update"].attempts == 0
    assert 0 < selector.estimate(belief, q) <= belief.entropy(q.slot) + 1e-9
    client.close()


# ---------------------------------------------------------------- calibration


def test_calibration_questions_cover_every_slot_and_type(schema):
    persona = load_personas(TRAVEL / "personas.jsonl", split="dev")[0]
    questions = calibration_questions(schema, persona, seed=1)
    assert len(questions) == 5 * len(schema.slots)
    kinds = [q.kind for q in questions]
    assert kinds.count(OPEN) == len(schema.slots)
    assert kinds.count(CHOICE) == kinds.count(YESNO) == 2 * len(schema.slots)
    assert questions == calibration_questions(schema, persona, seed=1)


def test_rho_hat_inverts_the_answer_model():
    import numpy as np

    rho, m = 0.6, np.array([2.0, 4.0, 5.0])
    correct = rho + (1 - rho) / m
    assert rho_hat(correct, 1 / m) == pytest.approx(rho)


def test_calibration_recovers_the_oracle_rho(tmp_path):
    truth = {"open": 0.7, "choice": 0.8, "yesno": 0.9}
    config = load_config(M3_EIG)
    result = calibrate(config, tmp_path, resamples=200)
    assert result["answers"] == 12 * 9 * 5 and set(result["rho"]) == {"rule"}
    for kind, rho in truth.items():
        r = result["rho"]["rule"][kind]
        assert abs(r["rho"] - rho) < 0.12, (kind, r)
        assert r["ci_low"] <= r["rho"] <= r["ci_high"]
    assert (tmp_path / "answers.jsonl").exists()


def test_calibration_runs_both_updaters_on_the_mock(tmp_path, capsys):
    assert main(["calibrate", str(M5_MOCK), "--out", str(tmp_path), "--workers", "4"]) == 0
    result = json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))
    assert set(result["rho"]) == {"rule", "llm"}
    assert result["llm"]["roles"]["simulate"]["calls"] == result["answers"]
    assert "rho" in capsys.readouterr().out


# ---------------------------------------------------------------- backstories


def test_backstories_avoid_the_slot_keywords(tmp_path, roles, schema):
    client = LLMClient(roles, MockProvider(), {"agent": "m", "user": "m"})
    personas = load_personas(TRAVEL / "personas.jsonl")
    path = tmp_path / "backstories.jsonl"
    stories = write_backstories(
        client, schema, personas, load_styles(PROMPTS / "styles.toml"), path, workers=4
    )
    assert [s.persona for s in stories] == [p.id for p in personas]
    assert not any(s.flagged for s in stories)
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(personas)
    assert flagged_slots(schema, "You love hiking at the beach.") == ["setting", "activity"]


def test_frozen_backstories_are_not_overwritten(capsys):
    if not (TRAVEL / "backstories.jsonl").exists():
        pytest.skip("the backstories are written on the cluster (SPEC §4)")
    assert main(["make-backstories", str(M5_MOCK)]) == 2
    assert "frozen" in capsys.readouterr().err
