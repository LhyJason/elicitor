import csv
import dataclasses
import json

import pytest

from elicitor.analysis.judge_report import judge_report, load_judge_run, pair_verdicts
from elicitor.analysis.report import RefusedToPublish
from elicitor.catalog import load_catalog
from elicitor.cli import main
from elicitor.experiments.config import Arm, load_config
from elicitor.experiments.runner import run_experiment
from elicitor.experiments.store import fingerprint
from elicitor.judge import (
    BEST,
    RANDOM,
    STATE,
    anchor_items,
    item_lines,
    load_judge_config,
    plan_fidelity,
    run_judge,
)
from elicitor.llm import LLMClient, build_client
from elicitor.llm.providers import Completion, MockProvider
from elicitor.llm.roles import ROLE_NAMES, JudgeReply, PairReply, build_roles
from elicitor.personas import load_personas
from elicitor.recommender import utility
from tests.conftest import ROOT, TRAVEL

M5_MOCK = ROOT / "experiments" / "m5_mock.toml"
PROMPTS = TRAVEL / "prompts"
EIG_RULE, EIG_LLM, EIG_MC = (
    "sel=eig,upd=rule,phr=tpl",
    "sel=eig,upd=llm,phr=tpl",
    "sel=eig_mc,upd=llm,phr=tpl",
)

JUDGE_FILE = """
name = "m7_test"
domain = "travel"
data_dir = "{data}"
seed = 9000

[judge]
source = "{source}"
replicate = 0
ordering = [5, 3]
constraints = [["{mc}", 5], ["{llm}", 5]]
pairs = [[["{llm}", 5], ["{llm}", 3]], [["{mc}", 5], ["{llm}", 5]]]

[judge.arms]
"{rule}" = [3, 5]
"{llm}" = [3, 5]
"{mc}" = [5]

[llm]
provider = "mock"
"""


def rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def source_run(tmp_path_factory):
    """A small mock run with the arms the judge study samples."""
    run = tmp_path_factory.mktemp("m7") / "source"
    config = dataclasses.replace(
        load_config(M5_MOCK),
        selectors=("eig",),
        extra_arms=(Arm.parse(EIG_MC),),
        arm_budgets={EIG_MC: (5,)},
    )
    run_experiment(config, run, max_personas=6)
    return run


def judge_file(tmp_path, source):
    path = tmp_path / "m7_test.toml"
    path.write_text(
        JUDGE_FILE.format(
            data=(ROOT / "data").as_posix(), source=source.as_posix(),
            rule=EIG_RULE, llm=EIG_LLM, mc=EIG_MC,
        ),
        encoding="utf-8",
    )  # fmt: skip
    return path


def mock_client(**kwargs):
    return build_client({"provider": "mock", **kwargs}, PROMPTS, None)


# ---------------------------------------------------------------- roles and client


def test_judge_roles_are_registered():
    roles = build_roles(PROMPTS)
    assert {"plan", "judge", "pair"} <= set(roles) == set(ROLE_NAMES)
    assert JudgeReply(reason="ok", score=10).score == 10
    with pytest.raises(ValueError):
        JudgeReply(reason="ok", score=11)
    with pytest.raises(ValueError):
        PairReply(reason="ok", choice="both")
    # The reason comes before the verdict in the schema sent to the server.
    assert list(JudgeReply.model_json_schema()["properties"]) == ["reason", "score"]


class Recorder(MockProvider):
    def __init__(self):
        self.seen = []

    def complete(self, request):
        self.seen.append((request.role.group, request.model))
        return super().complete(request)


def test_call_with_a_group_routes_and_counts_per_group():
    provider = Recorder()
    client = LLMClient(build_roles(PROMPTS), provider, {"agent": "qwen", "user": "llama"})
    context = {"profile": "- x", "item": "- y", "utility": 0.5}
    client.call("judge", context, seed=1)
    client.call("judge", context, seed=1, group="agent")
    assert provider.seen == [("user", "llama"), ("agent", "qwen")]
    stats = client.stats_dict()["roles"]
    assert stats["judge"]["calls"] == 1 and stats["judge@agent"]["calls"] == 1
    with pytest.raises(ValueError):
        client.call("judge", context, group="nobody")


def test_fingerprint_skips_only_the_judge_prompts(tmp_path):
    import shutil

    data = tmp_path / "data"
    shutil.copytree(TRAVEL, data / "travel")
    config = dataclasses.replace(load_config(M5_MOCK), data_dir=data)
    before = fingerprint(config, None)
    for name in ("plan.txt", "judge.txt", "pair.txt"):
        path = data / "travel" / "prompts" / name
        path.write_text(path.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8")
    assert fingerprint(config, None) == before
    path = data / "travel" / "prompts" / "simulate.txt"
    path.write_text(path.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8")
    assert fingerprint(config, None) != before


# ---------------------------------------------------------------- sample


def test_anchor_items(schema):
    catalog = load_catalog(TRAVEL / "catalog.jsonl")
    by_id = {item.id: item for item in catalog}
    for persona in load_personas(TRAVEL / "personas.jsonl")[:10]:
        anchors = anchor_items(schema, catalog, persona, seed=1)
        best = max(utility(schema, i.attrs, persona.truth) for i in catalog)
        u = {k: utility(schema, by_id[v].attrs, persona.truth) for k, v in anchors.items()}
        assert u[BEST] == pytest.approx(best) and u[RANDOM] < best
        assert anchors == anchor_items(schema, catalog, persona, seed=1)  # seeded


def test_item_lines_and_plan_fidelity(schema):
    item = load_catalog(TRAVEL / "catalog.jsonl")[0]
    lines = item_lines(schema, item.attrs).splitlines()
    assert len(lines) == len(schema.slots) and lines[-1].startswith("- trip length: ")
    plan = "Enjoy cold, snowy days in the countryside, and stay in a hostel."
    fidelity = plan_fidelity(schema, item.attrs, plan)
    assert {"climate", "setting", "lodging"} <= set(fidelity["named"])
    assert fidelity["other"] == []
    assert plan_fidelity(schema, item.attrs, "A hot beach.")["other"] == ["climate", "setting"]


def test_judge_file_checks(tmp_path, source_run):
    path = judge_file(tmp_path, source_run)
    config = load_judge_config(path)
    assert config.sources == [(EIG_RULE, 3), (EIG_RULE, 5), (EIG_LLM, 3), (EIG_LLM, 5),
                              (EIG_MC, 5)]  # fmt: skip
    assert load_judge_config(path, source="elsewhere").source.as_posix() == "elsewhere"
    bad = path.read_text(encoding="utf-8").replace("ordering = [5, 3]", "ordering = [8, 3]")
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="ordering"):
        load_judge_config(path)


# ---------------------------------------------------------------- end to end on the mock


@pytest.fixture(scope="module")
def judged(tmp_path_factory, source_run):
    tmp = tmp_path_factory.mktemp("m7_judge")
    config = load_judge_config(judge_file(tmp, source_run))
    client = mock_client()
    try:
        state = run_judge(config, client, tmp / "judge", workers=4)
    finally:
        client.close()
    return tmp / "judge", state


def test_judge_writes_every_planned_call(judged):
    out, state = judged
    assert state["problems"] == [] and state["judgments_failed"] == 0
    run = load_judge_run(out)
    assert len(run.personas) == 6
    assert state["scores"] == state["units"] * 6  # 2 judges x 3 presentations
    scores = [r for r in run.judgments if r["kind"] == "score"]
    assert len(scores) == state["scores"]
    utility_of = {(u["persona"], u["item"]): u["utility"] for u in run.units}
    assert all(r["score"] == round(10 * utility_of[(r["persona"], r["item"])]) for r in scores)
    # Every persona has both anchors, and a unit's sources say which arms chose it.
    for p in run.personas:
        sources = [s for u in run.units if u["persona"] == p for s in u["sources"]]
        assert sorted(sources) == sorted(
            [BEST, RANDOM, *(f"{a}@K{k}" for a, k in load_judge_config_text_sources(state))]
        )
    assert state["source"]["complete"] and state["source"]["provider"] == "mock"
    stats = json.loads((out / "llm_stats.json").read_text(encoding="utf-8"))["roles"]
    assert {"plan@agent", "plan@user", "judge@agent", "judge@user"} <= set(stats)


def load_judge_config_text_sources(state):
    from elicitor.judge import load_judge_config_text

    return load_judge_config_text(state["config"]).sources


def test_judge_rerun_is_identical(judged, source_run, tmp_path):
    out, _ = judged
    config = load_judge_config(judge_file(tmp_path, source_run))
    client = mock_client()
    try:
        run_judge(config, client, tmp_path / "again", workers=1)
    finally:
        client.close()
    for name in ("units.jsonl", "plans.jsonl", "judgments.jsonl"):
        assert (tmp_path / "again" / name).read_bytes() == (out / name).read_bytes()


def test_report_on_the_mock(judged, tmp_path):
    out, _ = judged
    written = judge_report(out, out=tmp_path / "report", resamples=200)
    names = {p.name for p in written}
    assert {"units.csv", "tests.csv", "weights.csv", "pairs.csv", "results.md",
            "score_vs_utility.png"} <= names  # fmt: skip
    tests = {(r["hypothesis"], r["a"]): r for r in rows(tmp_path / "report" / "tests.csv")}
    # The mock judge scores round(10 u): valid, and keeps the utility order at the ends.
    assert tests[("J1", "score")]["holds"] == "True"
    assert float(tests[("J2", "best item")]["diff"]) > 0
    # Plans and judges do not matter to the mock: no presentation effect, no self-preference.
    assert float(tests[("J4", "plans (mean of two)")]["diff"]) == 0
    assert tests[("J6", "first")]["holds"] == "False"
    md = (tmp_path / "report" / "results.md").read_text(encoding="utf-8")
    assert "## Frozen hypotheses" in md and "## Pairwise verdicts" in md
    # The mock may never feed committed tables.
    with pytest.raises(RefusedToPublish):
        judge_report(out, out=tmp_path / "results_summary" / "x", resamples=10)
    assert not (tmp_path / "results_summary").exists()


def test_incomplete_judge_folder_is_refused(judged, tmp_path):
    import shutil

    out, _ = judged
    copy = tmp_path / "copy"
    shutil.copytree(out, copy)
    (copy / STATE).unlink()
    with pytest.raises(ValueError, match="incomplete"):
        load_judge_run(copy)


class Faulty(MockProvider):
    """Plans by the agent model never validate; the pair judge always answers `first`."""

    def complete(self, request):
        if request.role.name == "plan" and request.role.group == "agent":
            return Completion(text="not json")
        if request.role.name == "pair":
            return Completion(text=PairReply(reason="x", choice="first").model_dump_json())
        return super().complete(request)


def test_failures_are_counted_and_position_bias_shows(source_run, tmp_path):
    config = load_judge_config(judge_file(tmp_path, source_run))
    client = LLMClient(build_roles(PROMPTS), Faulty(), {"agent": "a", "user": "u"})
    state = run_judge(config, client, tmp_path / "faulty")
    assert state["plans_failed"] == state["plans"] // 2
    run = load_judge_run(tmp_path / "faulty")
    plan_agent = [r for r in run.judgments if r.get("presentation") == "plan_agent"]
    # A judgement of a failed plan fails too, without a call and without a default score.
    assert plan_agent and all(r["failed"] and r["score"] is None for r in plan_agent)
    assert any("plan@agent: " in p and "failed calls" in p for p in state["problems"])
    verdicts = pair_verdicts(run)
    assert verdicts and all(v["winner_ab"] != v["winner_ba"] for v in verdicts)
    judge_report(tmp_path / "faulty", out=tmp_path / "report", resamples=50, make_figures=False)
    tests = {r["hypothesis"]: r for r in rows(tmp_path / "report" / "tests.csv")}
    assert float(tests["J6"]["mean_a"]) == 1.0 and float(tests["J6"]["mean_b"]) == 1.0
    assert tests["J5"]["n"] == "0"  # needs both plans


def test_cli_judge_and_report(source_run, tmp_path, capsys):
    path = judge_file(tmp_path, source_run)
    assert main(["judge", str(path), "--out", str(tmp_path / "j"), "--workers", "2"]) == 0
    assert main(["judge-report", str(tmp_path / "j"), "--no-figures"]) == 0
    assert (tmp_path / "j" / "report" / "tests.csv").exists()
    missing = ["--source", str(tmp_path / "none"), "--out", str(tmp_path / "k")]
    assert main(["judge", str(path), *missing]) == 2
    capsys.readouterr()
