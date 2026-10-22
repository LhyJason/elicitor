import dataclasses
import json

import numpy as np
import pytest

import elicitor.experiments.runner as runner_module
from elicitor.analysis.report import (
    RefusedToPublish,
    analyze,
    curve,
    default_comparisons,
    episode_value,
    is_committed_path,
)
from elicitor.analysis.stats import bootstrap_mean_ci, holm, paired_test, wilcoxon
from elicitor.cli import main
from elicitor.experiments.config import load_config
from elicitor.experiments.health import check_health
from elicitor.experiments.runner import run_experiment
from elicitor.experiments.store import FingerprintMismatch, RunStore
from tests.conftest import ROOT

M1 = ROOT / "experiments" / "m1_no_llm.toml"
M3_MOCK = ROOT / "experiments" / "m3_mock.toml"


def small(path=M1, **changes):
    """A quick version of a suite: one budget, one replicate unless overridden."""
    config = load_config(path)
    return dataclasses.replace(config, **{"budgets": (3,), "replicates": 1, **changes})


# ---------------------------------------------------------------- resume


def test_second_run_skips_finished_cells(tmp_path):
    config = small()
    run_experiment(config, tmp_path, max_personas=2)
    messages = []
    run_experiment(config, tmp_path, max_personas=2, progress=messages.append)
    n = len(config.arms)
    assert messages == [f"resuming: {n} of {n} cells already finished"]


def test_interrupted_run_resumes_to_the_same_bytes(tmp_path, monkeypatch):
    config = small(M3_MOCK, replicates=2)
    run_experiment(config, tmp_path / "whole", max_personas=3)

    real = runner_module.run_episode
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 8:  # in the middle of the third cell (3 personas per cell)
            raise RuntimeError("node lost")
        return real(**kwargs)

    monkeypatch.setattr(runner_module, "run_episode", flaky)
    with pytest.raises(RuntimeError, match="node lost"):
        run_experiment(config, tmp_path / "split", max_personas=3)
    store = RunStore(tmp_path / "split")
    assert len(store.finished_cells()) == 2
    assert not (tmp_path / "split" / "episodes.jsonl").exists()
    live = store.read_live()
    # One task of the third cell had finished before the second one failed.
    assert "node lost" in live["error"]
    assert [entry["episodes"] for entry in live["cells"].values()] == [1]
    health = check_health(tmp_path / "split")
    assert not health.ok and "cells 2/" in health.text()

    monkeypatch.setattr(runner_module, "run_episode", real)
    run_experiment(config, tmp_path / "split", max_personas=3)
    for name in ("episodes.jsonl", "summary.csv"):
        assert (tmp_path / "split" / name).read_bytes() == (tmp_path / "whole" / name).read_bytes()
    assert store.read_live() is None
    # The LLM statistics add up over both jobs: every call is counted once.
    whole = json.loads((tmp_path / "whole" / "llm_stats.json").read_text(encoding="utf-8"))
    split = json.loads((tmp_path / "split" / "llm_stats.json").read_text(encoding="utf-8"))
    for role, counts in whole["roles"].items():
        assert split["roles"][role]["calls"] == counts["calls"]


def test_changed_settings_need_fresh(tmp_path):
    run_experiment(small(), tmp_path, max_personas=2)
    with pytest.raises(FingerprintMismatch):
        run_experiment(small(tau=0.8), tmp_path, max_personas=2)
    with pytest.raises(FingerprintMismatch):
        run_experiment(small(), tmp_path, max_personas=3)
    rows = run_experiment(small(tau=0.8), tmp_path, max_personas=2, fresh=True)
    assert len(rows) == len(small().arms)


def test_replicates_and_arms_can_be_added(tmp_path):
    run_experiment(small(selectors=("fixed",)), tmp_path, max_personas=2)
    messages = []
    bigger = small(selectors=("fixed", "random"), replicates=2)
    run_experiment(bigger, tmp_path, max_personas=2, progress=messages.append)
    assert messages[0] == "resuming: 1 of 4 cells already finished"
    assert len([m for m in messages if m.startswith("[")]) == 3


def test_cli_run_reports_a_mismatch(tmp_path, capsys):
    out = str(tmp_path)
    assert main(["run", str(M1), "--out", out, "--max-personas", "1"]) == 0
    assert main(["run", str(M1), "--out", out, "--max-personas", "2"]) == 2
    assert "--fresh" in capsys.readouterr().err
    assert main(["run", str(M1), "--out", out, "--max-personas", "2", "--fresh"]) == 0


# ---------------------------------------------------------------- health


def test_health_of_a_finished_mock_run(tmp_path, capsys):
    run_experiment(small(M3_MOCK), tmp_path, max_personas=2)
    health = check_health(tmp_path)
    assert health.ok
    text = health.text()
    assert "cells 5/5" in text and "simulate" in text and "health: OK" in text
    assert main(["health", str(tmp_path)]) == 0
    assert main(["health", str(tmp_path / "nothing")]) == 1


def test_health_flags_parse_failures(tmp_path):
    run_experiment(small(M3_MOCK), tmp_path, max_personas=2)
    store = RunStore(tmp_path)
    cell = store.finished_cells()[0]
    stats = store.read_cell_stats(cell)
    role = next(iter(stats["llm"]["roles"].values()))
    role["parse_failures"] = role["attempts"]  # 100 % of this cell's attempts
    store.stats_path(cell).write_text(json.dumps(stats), encoding="utf-8")
    health = check_health(tmp_path)
    assert not health.ok and "parse failure rate" in health.text()


# ---------------------------------------------------------------- statistics


def test_holm_matches_hand_computation():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == [1.0, 1.0]
    assert holm([]) == []


def test_float_noise_is_a_tie():
    a = np.array([0.1 + 0.2, 0.5, 0.25])
    b = np.array([0.3, 0.5, 0.25])
    r = paired_test(a, b)
    assert (r.wins, r.ties, r.losses) == (0, 3, 0) and r.p == 1.0
    assert wilcoxon(np.zeros(4)) == (0.0, 1.0)


def test_paired_test_direction_and_ci():
    rng = np.random.default_rng(1)
    b = rng.random(48)
    a = b + 0.1 + 0.01 * rng.standard_normal(48)
    up = paired_test(a, b)
    assert up.wins == 48 and up.p < 1e-6 and up.ci_low < up.diff < up.ci_high
    assert up.ci_low > 0
    down = paired_test(a, b, higher_is_better=False)
    assert down.losses == 48 and down.p == up.p
    assert bootstrap_mean_ci(a) == bootstrap_mean_ci(a)  # fixed seed


def test_default_comparisons_differ_in_one_component():
    arms = [
        "sel=fixed,upd=rule,phr=tpl",
        "sel=eig,upd=rule,phr=tpl",
        "sel=fixed,upd=llm,phr=tpl",
        "sel=eig,upd=llm,phr=tpl",
        "sel=eig,upd=llm,phr=llm",
    ]
    pairs = default_comparisons(arms)
    assert ("sel=eig,upd=rule,phr=tpl", "sel=fixed,upd=rule,phr=tpl") in pairs
    assert ("sel=eig,upd=llm,phr=llm", "sel=eig,upd=llm,phr=tpl") in pairs
    assert ("sel=eig,upd=llm,phr=tpl", "sel=fixed,upd=rule,phr=tpl") not in pairs
    assert len(pairs) == 5


def test_curve_carries_the_last_value_and_auc():
    episode = {
        "budget": 4,
        "n_turns": 2,
        "initial_metrics": {"map_accuracy": 0.0},
        "turns": [{"metrics": {"map_accuracy": 0.5}}, {"metrics": {"map_accuracy": 1.0}}],
        "final_metrics": {"map_accuracy": 1.0},
    }
    assert curve(episode, "map_accuracy") == [0.0, 0.5, 1.0, 1.0, 1.0]
    # Trapezoid over 0, .5, 1, 1, 1: (0 + 1) / 2 + .5 + 1 + 1 = 3; divided by K = 4.
    assert episode_value(episode, "map_auc") == pytest.approx(3 / 4)


# ---------------------------------------------------------------- analysis and publishing


def test_analysis_runs_on_mock_output_and_refuses_to_publish_it(tmp_path):
    run = tmp_path / "run"
    run_experiment(small(M3_MOCK, replicates=2, budgets=(3, 5)), run, max_personas=4)
    written = analyze(run)
    names = {p.name for p in written}
    assert {"arms.csv", "comparisons.csv", "curves.csv", "results.md"} <= names
    assert "curve_mass_on_truth_K5.png" in names
    md = (run / "analysis" / "results.md").read_text(encoding="utf-8")
    assert "LLM provider: mock" in md and "Paired comparisons" in md
    assert "![curve_regret_K3](curve_regret_K3.png)" in md

    with pytest.raises(RefusedToPublish) as refused:
        analyze(run, publish=True, root=tmp_path)
    assert any("mock" in r for r in refused.value.reasons)
    assert any("cut to 4 personas" in r for r in refused.value.reasons)
    with pytest.raises(RefusedToPublish):
        analyze(run, out=tmp_path / "results_summary" / "x", make_figures=False)
    assert not (tmp_path / "results_summary").exists()
    assert not (tmp_path / "docs").exists()


def test_complete_oracle_run_may_be_published(tmp_path):
    run = tmp_path / "run"
    run_experiment(small(), run)
    written = analyze(run, publish=True, root=tmp_path)
    assert tmp_path / "results_summary" / "m1_no_llm" / "results.md" in written
    assert (tmp_path / "docs" / "figures" / "m1_no_llm" / "curve_regret_K3.png").exists()
    md = (tmp_path / "results_summary" / "m1_no_llm" / "results.md").read_text(encoding="utf-8")
    assert "../../docs/figures/m1_no_llm/curve_regret_K3.png" in md
    # Same run, same tables.
    first = (tmp_path / "results_summary" / "m1_no_llm" / "comparisons.csv").read_bytes()
    analyze(run, publish=True, root=tmp_path)
    assert (tmp_path / "results_summary" / "m1_no_llm" / "comparisons.csv").read_bytes() == first


def test_incomplete_run_is_not_published(tmp_path, monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("stop")

    config = small()
    run_experiment(dataclasses.replace(config, selectors=("fixed",)), tmp_path)
    monkeypatch.setattr(runner_module, "run_episode", broken)
    with pytest.raises(RuntimeError):
        run_experiment(config, tmp_path)
    with pytest.raises(RefusedToPublish, match="incomplete"):
        analyze(tmp_path, publish=True, root=tmp_path)
    # Local analysis of the finished cells still works while a run is going.
    assert analyze(tmp_path, make_figures=False)


def test_committed_paths():
    assert is_committed_path("results_summary/m5")
    assert is_committed_path("docs/figures/m5/a.png")
    assert not is_committed_path("results/m5/analysis")
    assert not is_committed_path("docs/SPEC.md")


def test_cli_analyze_exit_codes(tmp_path, capsys):
    run = tmp_path / "run"
    run_experiment(small(M3_MOCK), run, max_personas=2)
    assert main(["analyze", str(run), "--no-figures"]) == 0
    capsys.readouterr()
    assert main(["analyze", str(run), "--out", str(tmp_path / "results_summary" / "x")]) == 2
    assert "mock" in capsys.readouterr().err
    assert main(["analyze", str(tmp_path / "missing")]) == 2


def test_analysis_table_in_the_experiment_file(tmp_path):
    text = M1.read_text(encoding="utf-8") + (
        "\n[analysis]\n"
        'comparisons = [["sel=random,upd=rule,phr=tpl", "sel=fixed,upd=rule,phr=tpl"]]\n'
        'metrics = ["regret"]\n'
        "bootstrap = 200\n"
    )
    data_dir = (ROOT / "data").as_posix()
    custom = tmp_path / "custom.toml"
    custom.write_text(
        text.replace('data_dir = "../data"', f'data_dir = "{data_dir}"'), encoding="utf-8"
    )
    config = dataclasses.replace(load_config(custom), budgets=(3,), replicates=1)
    run_experiment(config, tmp_path / "run", max_personas=3)
    analyze(tmp_path / "run", make_figures=False)
    rows = (
        (tmp_path / "run" / "analysis" / "comparisons.csv").read_text(encoding="utf-8").splitlines()
    )
    assert len(rows) == 2 and rows[1].startswith("regret,3,")

    bad = tmp_path / "bad.toml"
    bad.write_text(text.replace("sel=random", "sel=eig"), encoding="utf-8")
    with pytest.raises(ValueError, match="not in the grid"):
        load_config(bad)
