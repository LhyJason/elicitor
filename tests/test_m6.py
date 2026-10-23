import csv
import dataclasses
import json

import numpy as np
import pytest

from elicitor.analysis.descriptive import persona_styles, spearman, updater_pairs
from elicitor.analysis.report import analyze, load_run
from elicitor.cli import main
from elicitor.demo import bar, belief_lines, demo_arm, run_demo
from elicitor.experiments.config import load_config
from elicitor.experiments.runner import run_experiment
from elicitor.llm import build_client
from elicitor.schema import load_schema
from tests.conftest import ROOT

M1 = ROOT / "experiments" / "m1_no_llm.toml"
M3_MOCK = ROOT / "experiments" / "m3_mock.toml"
DEMO = ROOT / "experiments" / "demo.toml"


def rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- descriptive tables


def test_updater_pairs_match_rule_and_llm_arms():
    arms = [
        "sel=fixed,upd=rule,phr=tpl",
        "sel=fixed,upd=llm,phr=tpl",
        "sel=eig,upd=llm,phr=llm",  # no rule arm with LLM wording
        "sel=eig,upd=rule,phr=tpl",
        "sel=eig,upd=llm,phr=tpl",
    ]
    assert updater_pairs(arms) == [
        ("sel=fixed,upd=rule,phr=tpl", "sel=fixed,upd=llm,phr=tpl"),
        ("sel=eig,upd=rule,phr=tpl", "sel=eig,upd=llm,phr=tpl"),
    ]


def test_spearman():
    x = np.array([0.1, 0.4, 0.2, 0.9])
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert spearman(x, x**2) == pytest.approx(1.0)
    assert np.isnan(spearman(x, np.zeros(4)))  # constant regret: nothing to rank
    assert np.isnan(spearman(x[:1], x[:1]))


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    run = tmp_path_factory.mktemp("m6") / "run"
    config = dataclasses.replace(load_config(M3_MOCK), budgets=(3,), replicates=2)
    run_experiment(config, run, max_personas=8)
    return run


def test_style_and_correlation_tables(mock_run, tmp_path):
    analyze(mock_run, out=tmp_path, make_figures=False, root=ROOT)
    styles = rows(tmp_path / "styles.csv")
    run = load_run(mock_run)
    personas = {
        json.loads(line)["id"]: json.loads(line)["style"]
        for line in (ROOT / "data" / "travel" / "personas.jsonl").read_text("utf-8").splitlines()
    }
    # The gap is LLM minus rule over the personas of one style, replicates averaged first.
    row = next(
        r for r in styles
        if r["metric"] == "mass_on_truth" and r["selector"] == "eig" and r["phraser"] == "tpl"
    )  # fmt: skip
    members = [p for p in run.state["personas"] if personas[p] == row["style"]]
    assert int(row["personas"]) == len(members)

    def mean(arm):
        values = [
            e["final_metrics"]["mass_on_truth"]
            for e in run.episodes
            if e["arm"] == arm and e["persona"] in members
        ]
        return sum(values) / len(values)

    gap = mean("sel=eig,upd=llm,phr=tpl") - mean("sel=eig,upd=rule,phr=tpl")
    assert float(row["gap"]) == pytest.approx(gap, abs=1e-5)
    assert float(row["ci_low"]) <= float(row["gap"]) <= float(row["ci_high"])
    assert {r["selector"] for r in styles} == {"max_entropy", "eig"}  # phr=llm has no rule pair

    correlations = rows(tmp_path / "correlations.csv")
    assert len(correlations) == len(run.arms)  # one budget
    assert all(int(r["episodes"]) == 16 for r in correlations)  # 8 personas x 2 replicates
    md = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "Updater gap by answer style (H3, descriptive)" in md
    assert "Final belief vs regret (H4, descriptive)" in md
    assert md.index("Paired comparisons") < md.index("H3, descriptive") < md.index("H4, descr")


def test_style_table_needs_the_persona_file(mock_run, tmp_path):
    assert persona_styles(load_run(mock_run), tmp_path) is None
    written = analyze(mock_run, out=tmp_path / "a", make_figures=False, root=tmp_path)
    assert "styles.csv" not in {p.name for p in written}
    assert "correlations.csv" in {p.name for p in written}


def test_rule_only_run_has_no_style_table(tmp_path):
    config = dataclasses.replace(load_config(M1), budgets=(3,), replicates=1)
    run_experiment(config, tmp_path / "run", max_personas=4)
    written = analyze(tmp_path / "run", make_figures=False, root=ROOT)
    assert "styles.csv" not in {p.name for p in written}


# ---------------------------------------------------------------- demo


def scripted(*answers):
    """A `read` function that returns the answers in order and records the prompts."""
    queue = list(answers)
    prompts = []

    def read(prompt):
        prompts.append(prompt)
        return queue.pop(0)

    return read, prompts


def test_bar_and_belief_lines(schema):
    assert bar(0.0) == "." * 20 and bar(1.0) == "#" * 20 and bar(0.5).count("#") == 10
    from elicitor.belief import Belief

    belief = Belief.prior(schema).update("climate", [8, 1, 1, 1])
    lines = belief_lines(schema, belief, asked="climate")
    assert lines[0].startswith("> climate") and "tropical" in lines[0]
    assert "(undecided)" in lines[1]  # setting is still at the uniform prior
    assert any(line.strip().startswith("cold") for line in lines)


def test_demo_updates_the_belief_and_recommends():
    config = load_config(DEMO)
    out = []
    read, prompts = scripted("somewhere hot and tropical please", "quit")
    belief = run_demo(
        config, demo_arm(config, selector="fixed"), budget=3, read=read, write=out.append
    )
    assert "[1/3] What kind of weather" in prompts[0] and len(prompts) == 2
    assert belief.map_value("climate") == "tropical" and belief.max_prob("climate") > 0.5
    text = "\n".join(out)
    assert "Stopped." in text and "Top recommendations" in text
    assert text.count("EU ") == 3


def test_demo_says_when_an_answer_is_not_understood():
    config = load_config(DEMO)
    out = []
    read, _ = scripted("purple elephants")
    belief = run_demo(
        config, demo_arm(config, selector="fixed"), budget=1, read=read, write=out.append
    )
    assert any("not understood" in line for line in out)
    assert belief.max_prob("climate") == pytest.approx(0.25)


def test_demo_default_arm_and_rejections():
    config = load_config(DEMO)
    assert demo_arm(config).name == "sel=eig,upd=rule,phr=tpl"
    with pytest.raises(ValueError, match="simulated user"):
        demo_arm(config, selector="eig_mc")
    with pytest.raises(ValueError, match="uses an LLM"):
        run_demo(config, demo_arm(config, updater="llm"), read=scripted()[0], write=print)


def test_demo_with_the_llm_updater_on_the_mock_backend(tmp_path):
    config = load_config(DEMO)
    client = build_client({"provider": "mock"}, config.prompts_dir, tmp_path / "cache.sqlite")
    try:
        read, prompts = scripted("I'd like the beach", "a big city", "")
        run_demo(
            config,
            demo_arm(config, updater="llm", phraser="llm"),
            budget=5,
            read=read,
            write=lambda s: None,
            client=client,
        )
    finally:
        client.close()
    assert len(prompts) == 3


def test_cli_demo(monkeypatch, capsys):
    answers = iter(["hot", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert main(["demo", str(DEMO), "--selector", "fixed", "--budget", "2"]) == 0
    out = capsys.readouterr().out
    assert "sel=fixed,upd=rule,phr=tpl" in out and "Top recommendations" in out

    def eof(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert main(["demo", str(DEMO)]) == 0  # Ctrl-D / closed stdin ends quietly
    assert main(["demo", str(DEMO), "--selector", "eig_mc"]) == 2
    assert main(["demo", str(ROOT / "missing.toml")]) == 2


def test_demo_schema_matches_the_domain():
    config = load_config(DEMO)
    assert load_schema(config.schema_path).domain == config.domain
