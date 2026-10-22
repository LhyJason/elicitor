import json

import pytest

from elicitor.catalog import load_catalog
from elicitor.cli import main
from elicitor.episode import BUDGET, CONFIDENT, EXHAUSTED, run_episode
from elicitor.experiments.config import load_config
from elicitor.experiments.runner import run_experiment
from elicitor.metrics import OUTCOME_METRICS
from elicitor.personas import load_personas
from elicitor.phrasing import TemplatePhraser
from elicitor.recommender import Recommender
from elicitor.selectors import make_selector
from elicitor.selectors.eig import EIGSelector
from elicitor.simulator.oracle import OracleSimulator
from elicitor.updaters.rule import RuleUpdater
from tests.conftest import ROOT, TRAVEL

M1_CONFIG = ROOT / "experiments" / "m1_no_llm.toml"
M3_CONFIGS = [ROOT / "experiments" / "m3_eig.toml", ROOT / "experiments" / "m3_mock.toml"]


def episode(
    schema,
    selector="fixed",
    rho=1.0,
    budget=None,
    tau=1.1,
    seed=0,
    updater_rho=0.99,
    recommender=None,
    **selector_params,
):
    p = load_personas(TRAVEL / "personas.jsonl", split="eval")[0]
    return p, run_episode(
        schema=schema,
        persona=p,
        arm=f"sel={selector}",
        selector=make_selector(selector, **selector_params),
        updater=RuleUpdater(updater_rho),
        simulator=OracleSimulator(schema, rho=rho),
        phraser=TemplatePhraser(schema),
        budget=budget or len(schema.slots),
        tau=tau,
        seed=seed,
        recommender=recommender,
    )


@pytest.mark.parametrize("selector", ["fixed", "random", "max_entropy"])
def test_noise_free_episode_learns_everything(schema, selector):
    p, ep = episode(schema, selector)
    assert ep.stop_reason == BUDGET
    assert len(ep.turns) == len(schema.slots)
    assert ep.final_metrics["map_accuracy"] == 1
    assert ep.final_metrics["mass_on_truth"] > ep.initial_metrics["mass_on_truth"]
    for turn in ep.turns:
        assert turn.text == schema.slot(turn.question.slot).question


def test_noise_free_eig_episode_and_recommendation(schema):
    """With rho = 1 EIG asks open questions only; knowing everything means no regret."""
    recommender = Recommender(schema, load_catalog(TRAVEL / "catalog.jsonl"))
    p, ep = episode(schema, "eig", rho=1.0, recommender=recommender)
    assert [t.question.kind for t in ep.turns] == ["open"] * len(schema.slots)
    assert ep.final_metrics["map_accuracy"] == 1
    assert ep.final_metrics["regret"] == pytest.approx(0, abs=1e-12)
    assert ep.final_metrics["hit@1"] == 1
    assert ep.initial_metrics["regret"] > 0
    assert len(ep.to_dict()["recommended"]) == 10


def test_eig_asks_closed_questions_when_they_are_more_reliable(schema):
    rho = {"open": 0.7, "choice": 0.8, "yesno": 0.9}
    p = load_personas(TRAVEL / "personas.jsonl", split="eval")[0]
    ep = run_episode(
        schema=schema,
        persona=p,
        arm="sel=eig",
        selector=EIGSelector(rho=rho),
        updater=RuleUpdater(rho),
        simulator=OracleSimulator(schema, rho=rho),
        phraser=TemplatePhraser(schema),
        budget=5,
        tau=1.1,
        seed=0,
    )
    assert {t.question.kind for t in ep.turns} != {"open"}
    for t in ep.turns:
        assert t.text == TemplatePhraser(schema).render(t.question)


def test_budget_and_stop_reasons(schema):
    _, ep = episode(schema, budget=3)
    assert ep.stop_reason == BUDGET and len(ep.turns) == 3
    _, ep = episode(schema, budget=len(schema.slots) + 2)
    assert ep.stop_reason == EXHAUSTED and len(ep.turns) == len(schema.slots)
    # rho = 0.99 puts > 0.95 on each answered value, so tau = 0.95 stops once all are asked.
    _, ep = episode(schema, budget=len(schema.slots) + 2, tau=0.95)
    assert ep.stop_reason == CONFIDENT and len(ep.turns) == len(schema.slots)


def test_episode_is_deterministic(schema):
    _, a = episode(schema, "random", rho=0.5, seed=7)
    _, b = episode(schema, "random", rho=0.5, seed=7)
    _, c = episode(schema, "random", rho=0.5, seed=8)
    assert a.to_dict() == b.to_dict()
    assert a.to_dict() != c.to_dict()


def test_same_question_gets_same_answer_across_arms(schema):
    """Common random numbers (SPEC §11): answers depend on the slot, not on the arm."""
    _, fixed = episode(schema, "fixed", rho=0.3, seed=3)
    _, rand = episode(schema, "random", rho=0.3, seed=3)
    by_slot = {t.question.slot: t.answer for t in fixed.turns}
    for t in rand.turns:
        assert t.answer == by_slot[t.question.slot]


def test_full_m1_suite_runs(tmp_path):
    config = load_config(M1_CONFIG)
    rows = run_experiment(config, tmp_path)
    assert len(rows) == len(config.arms) * len(config.budgets)
    n_eval = len(load_personas(config.personas_path, split="eval"))
    assert all(r["episodes"] == config.replicates * n_eval for r in rows)
    # More questions -> more mass on the truth, for every arm.
    for arm in config.arms:
        mass = [r["mass_on_truth"] for r in rows if r["arm"] == arm.name]
        assert mass == sorted(mass)
    assert (tmp_path / "episodes.jsonl").exists() and (tmp_path / "summary.csv").exists()


@pytest.mark.parametrize("path", M3_CONFIGS, ids=lambda p: p.stem)
def test_m3_suites_run(tmp_path, path):
    config = load_config(path)
    rows = run_experiment(config, tmp_path, max_personas=3)
    assert len(rows) == len(config.arms) * len(config.budgets)
    assert any(arm.selector == "eig" for arm in config.arms)
    for row in rows:
        assert all(0 <= row[m] <= 2 for m in OUTCOME_METRICS)
    episodes = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert set(OUTCOME_METRICS) <= set(episodes[0]["final_metrics"])
    # The eig arms ask closed questions, so their code paths really run.
    kinds = {t["question"]["kind"] for e in episodes if "sel=eig," in e["arm"] for t in e["turns"]}
    assert kinds - {"open"}


def test_runner_output_is_reproducible(tmp_path):
    config = load_config(M1_CONFIG)
    run_experiment(config, tmp_path / "a", max_personas=3)
    run_experiment(config, tmp_path / "b", max_personas=3)
    for name in ("episodes.jsonl", "summary.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_cli_run_and_make_data_check(tmp_path, capsys):
    assert main(["run", str(M1_CONFIG), "--out", str(tmp_path), "--max-personas", "2"]) == 0
    assert "mass_on_truth" in capsys.readouterr().out
    assert main(["make-data", str(TRAVEL), "--check"]) == 0


def test_config_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('selecter = "typo"\n' + M1_CONFIG.read_text())
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(bad)
