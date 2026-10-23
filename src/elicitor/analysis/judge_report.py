"""Report of a judge folder (SPEC Q10, M7): the frozen J1–J6 tests and the descriptive tables.

The unit of a test is the persona, as in §13: a persona's scores are averaged within each
condition, then compared with the paired Wilcoxon test (`analysis.stats.paired_test`). J2–J6
form one Holm family. The main judge is the user model with the `list` presentation.
Committed outputs (`results_summary/`, `docs/figures/`) are only written from a complete,
non-mock judge run on a source run that may itself be published.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sps

from elicitor.analysis.descriptive import spearman
from elicitor.analysis.report import RefusedToPublish, _csv, _p, _write, is_committed_path
from elicitor.analysis.stats import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, CI_LEVEL, holm, paired_test
from elicitor.jsonl import read_jsonl
from elicitor.judge import (
    BEST,
    LIST,
    PLAN_GROUPS,
    RANDOM,
    STATE,
    load_judge_config_text,
    source_label,
)
from elicitor.llm.roles import AGENT, USER

MAIN = (USER, LIST)  # (judge, presentation)
J1_BAR = 0.5
J6_BAR = 0.10
ALPHA = 0.05
UTILITY_BINS = (0.5, 0.6, 0.7, 0.8, 0.9)  # bin edges; the last bin [0.9, 1] includes 1
IMPLIED_WEIGHT_SCALE = 10  # score = 10 * u under the rubric, so each slot weighs 10 / |S|
# fmt: off
TEST_COLUMNS = ("hypothesis", "test", "a", "b", "n", "mean_a", "mean_b", "diff", "ci_low",
                "ci_high", "wins", "ties", "losses", "statistic", "p", "p_holm", "holds")
# fmt: on


@dataclass
class JudgeRun:
    root: Path
    state: dict[str, Any]
    units: list[dict[str, Any]]
    plans: list[dict[str, Any]]
    judgments: list[dict[str, Any]]

    @property
    def personas(self) -> list[str]:
        return list(dict.fromkeys(u["persona"] for u in self.units))

    @property
    def conditions(self) -> list[tuple[str, str]]:
        """(judge, presentation) pairs that were scored, main first."""
        seen = dict.fromkeys(
            (r["judge"], r["presentation"]) for r in self.judgments if r["kind"] == "score"
        )
        return sorted(seen, key=lambda c: (c != MAIN, c[0] != USER, c[1]))


def load_judge_run(judge_dir: str | Path) -> JudgeRun:
    root = Path(judge_dir)
    path = root / STATE
    if not path.exists():
        raise ValueError(f"{root} has no {STATE}: the judge run is incomplete; run it again")
    return JudgeRun(
        root=root,
        state=json.loads(path.read_text(encoding="utf-8")),
        units=read_jsonl(root / "units.jsonl"),
        plans=read_jsonl(root / "plans.jsonl"),
        judgments=read_jsonl(root / "judgments.jsonl"),
    )


def publish_problems(run: JudgeRun) -> list[str]:
    problems = []
    if run.state.get("provider") == "mock":
        problems.append("the judge run used the mock LLM provider")
    source = run.state["source"]
    if not source.get("complete"):
        problems.append(f"the source run {source['name']} was incomplete")
    if source.get("provider") == "mock":
        problems.append(f"the source run {source['name']} used the mock LLM provider")
    if source.get("split") != "eval":
        problems.append(f"the source run used the {source.get('split')!r} split, not 'eval'")
    if source.get("max_personas") is not None:
        problems.append(f"the source run was cut to {source['max_personas']} personas")
    return problems


# ---------------------------------------------------------------- scores


Scores = dict[tuple[str, str], dict[tuple[str, str], float]]  # (persona, item) -> cond -> score


def score_table(run: JudgeRun) -> Scores:
    table: Scores = defaultdict(dict)
    for r in run.judgments:
        if r["kind"] == "score" and not r["failed"]:
            table[(r["persona"], r["item"])][(r["judge"], r["presentation"])] = float(r["score"])
    return table


def spearman_ci(
    run: JudgeRun, scores: Scores, condition: tuple[str, str], resamples: int
) -> dict[str, Any]:
    """Pooled Spearman of score vs utility over units, with a bootstrap CI over personas."""
    by_persona: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for u in run.units:
        s = scores.get((u["persona"], u["item"]), {}).get(condition)
        if s is not None:
            by_persona[u["persona"]].append((u["utility"], s))
    personas = [p for p in run.personas if by_persona[p]]
    arrays = [np.array(by_persona[p], dtype=float) for p in personas]
    if not arrays:
        return {"n": 0, "estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    pooled = np.concatenate(arrays)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boots = []
    for _ in range(resamples):
        sample = np.concatenate([arrays[i] for i in rng.integers(0, len(arrays), len(arrays))])
        boots.append(spearman(sample[:, 0], sample[:, 1]))
    tail = (1 - CI_LEVEL) / 2 * 100
    low, high = np.nanpercentile(boots, [tail, 100 - tail])
    return {
        "n": int(pooled.shape[0]),
        "estimate": spearman(pooled[:, 0], pooled[:, 1]),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def _units_of(run: JudgeRun, persona: str, predicate) -> list[dict[str, Any]]:
    return [u for u in run.units if u["persona"] == persona and predicate(u["sources"])]


def _mean_score(scores: Scores, units: list[dict[str, Any]], condition) -> float | None:
    values = [scores.get((u["persona"], u["item"]), {}).get(condition) for u in units]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _paired_row(
    hypothesis: str, test: str, a_name: str, b_name: str, pairs: list[tuple[float, float]],
    resamples: int,
) -> dict[str, Any]:  # fmt: skip
    row: dict[str, Any] = {"hypothesis": hypothesis, "test": test, "a": a_name, "b": b_name}
    if not pairs:
        return {**row, "n": 0, "p": float("nan")}
    a, b = (np.array(v, dtype=float) for v in zip(*pairs, strict=True))
    r = paired_test(a, b, resamples=resamples)
    return {
        **row, "n": r.n, "mean_a": r.mean_a, "mean_b": r.mean_b, "diff": r.diff,
        "ci_low": r.ci_low, "ci_high": r.ci_high, "wins": r.wins, "ties": r.ties,
        "losses": r.losses, "statistic": r.statistic, "p": r.p,
    }  # fmt: skip


def pair_verdicts(run: JudgeRun) -> list[dict[str, Any]]:
    """One row per (persona, pair) asked in both orders, choices in item terms."""
    asked: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in run.judgments:
        if r["kind"] == "pair":
            asked[(r["persona"], r["pair"])][r["order"]] = r
    utility = {(u["persona"], u["item"]): u["utility"] for u in run.units}
    rows = []
    for (persona, index), orders in asked.items():
        ab, ba = orders.get("ab"), orders.get("ba")
        if ab is None or ba is None:
            continue
        a, b = ab["first"], ab["second"]

        def winner(r: dict[str, Any]) -> str | None:
            if r["failed"]:
                return None
            return {"first": r["first"], "second": r["second"]}.get(r["choice"], "equal")

        rows.append(
            {
                "persona": persona,
                "pair": index,
                "a": a,
                "b": b,
                "utility_a": utility[(persona, a)],
                "utility_b": utility[(persona, b)],
                "choice_ab": None if ab["failed"] else ab["choice"],
                "choice_ba": None if ba["failed"] else ba["choice"],
                "winner_ab": winner(ab),
                "winner_ba": winner(ba),
            }
        )
    return rows


def tests_table(run: JudgeRun, scores: Scores, resamples: int) -> list[dict[str, Any]]:
    config = load_judge_config_text(run.state["config"])
    rows: list[dict[str, Any]] = []

    # J1: validity of the main judge.
    j1 = spearman_ci(run, scores, MAIN, resamples)
    rows.append(
        {
            "hypothesis": "J1", "test": "Spearman(score, utility), user judge, list",
            "a": "score", "b": "utility", "n": j1["n"], "mean_a": j1["estimate"],
            "ci_low": j1["ci_low"], "ci_high": j1["ci_high"], "p": float("nan"),
            "holds": bool(j1["ci_low"] > J1_BAR),
        }
    )  # fmt: skip

    # J2: best > recommended at the high budget > at the low budget > random.
    high, low = config.ordering

    def at_budget(k: int):
        return lambda sources: any(s.endswith(f"@K{k}") for s in sources)

    levels = [
        ("best item", lambda s: BEST in s),
        (f"recommended at K = {high}", at_budget(high)),
        (f"recommended at K = {low}", at_budget(low)),
        ("random item", lambda s: RANDOM in s),
    ]
    for (name_a, pick_a), (name_b, pick_b) in zip(levels, levels[1:], strict=False):
        pairs = []
        for p in run.personas:
            a = _mean_score(scores, _units_of(run, p, pick_a), MAIN)
            b = _mean_score(scores, _units_of(run, p, pick_b), MAIN)
            if a is not None and b is not None:
                pairs.append((a, b))
        rows.append(_paired_row("J2", "ordering", name_a, name_b, pairs, resamples))

    # J3: source A above source B (equal regret, fewer constraint violations for A).
    label_a, label_b = (source_label(*s) for s in config.constraints)
    pairs = []
    for p in run.personas:
        a = _mean_score(scores, _units_of(run, p, lambda s: label_a in s), MAIN)
        b = _mean_score(scores, _units_of(run, p, lambda s: label_b in s), MAIN)
        if a is not None and b is not None:
            pairs.append((a, b))
    rows.append(_paired_row("J3", "constraints", label_a, label_b, pairs, resamples))

    # J4: plans above the list (user judge), on units with all three presentations.
    plan_conditions = [(USER, p) for p in PLAN_GROUPS]
    pairs = []
    for p in run.personas:
        plan_means, list_scores = [], []
        for u in _units_of(run, p, lambda s: True):
            got = scores.get((p, u["item"]), {})
            if MAIN in got and all(c in got for c in plan_conditions):
                plan_means.append(np.mean([got[c] for c in plan_conditions]))
                list_scores.append(got[MAIN])
        if plan_means:
            pairs.append((float(np.mean(plan_means)), float(np.mean(list_scores))))
    rows.append(_paired_row("J4", "presentation", "plans (mean of two)", "list", pairs, resamples))

    # J5: each judge favours its own model's plans (difference in differences).
    pairs = []
    for p in run.personas:
        agent_gaps, user_gaps = [], []
        for u in _units_of(run, p, lambda s: True):
            got = scores.get((p, u["item"]), {})
            needed = [(j, pr) for j in (AGENT, USER) for pr in PLAN_GROUPS]
            if all(c in got for c in needed):
                agent_gaps.append(got[(AGENT, "plan_agent")] - got[(AGENT, "plan_user")])
                user_gaps.append(got[(USER, "plan_agent")] - got[(USER, "plan_user")])
        if agent_gaps:
            pairs.append((float(np.mean(agent_gaps)), float(np.mean(user_gaps))))
    rows.append(
        _paired_row(
            "J5", "self-preference", "agent judge: own − other plan",
            "user judge: same plans", pairs, resamples,
        )
    )  # fmt: skip

    # J6: position bias in the pairwise test.
    verdicts = pair_verdicts(run)
    both = [v for v in verdicts if v["winner_ab"] is not None and v["winner_ba"] is not None]
    changed = sum(v["winner_ab"] != v["winner_ba"] for v in both)
    choices = [c for v in both for c in (v["choice_ab"], v["choice_ba"]) if c != "equal"]
    firsts = sum(c == "first" for c in choices)
    row: dict[str, Any] = {
        "hypothesis": "J6", "test": "position", "a": "first", "b": "second", "n": len(choices),
        "mean_a": firsts / len(choices) if choices else float("nan"),
        "mean_b": changed / len(both) if both else float("nan"),
        "wins": firsts, "losses": len(choices) - firsts,
        "p": float(sps.binomtest(firsts, len(choices), 0.5).pvalue) if choices else float("nan"),
    }  # fmt: skip
    rows.append(row)

    # Holm over J2–J6.
    family = [i for i, r in enumerate(rows) if r["hypothesis"] != "J1" and not np.isnan(r["p"])]
    for i, p_holm in zip(family, holm([rows[i]["p"] for i in family]), strict=True):
        rows[i]["p_holm"] = p_holm
    for r in rows:
        if r["hypothesis"] in ("J2", "J3", "J4", "J5"):
            r["holds"] = bool(r.get("diff", 0) > 0 and r.get("p_holm", 1.0) < ALPHA)
        elif r["hypothesis"] == "J6":
            r["holds"] = bool(r["n"] and r["mean_b"] >= J6_BAR and r.get("p_holm", 1.0) < ALPHA)
    return [{c: r.get(c, "") for c in TEST_COLUMNS} for r in rows]


# ---------------------------------------------------------------- descriptive tables


def spearman_table(run: JudgeRun, scores: Scores, resamples: int) -> list[dict[str, Any]]:
    return [
        {"judge": j, "presentation": p, **spearman_ci(run, scores, (j, p), resamples)}
        for j, p in run.conditions
    ]


def weights_table(run: JudgeRun, scores: Scores) -> list[dict[str, Any]]:
    """Least-squares fit of the score on the per-slot match scores, per condition, beside the
    weights the rubric implies (score = 10 u)."""
    slots = list(run.units[0]["matches"]) if run.units else []
    rows = [
        {"judge": "rubric", "presentation": "", "n": "", "term": "intercept", "weight": 0.0},
        *(
            {"judge": "rubric", "presentation": "", "n": "", "term": s,
             "weight": IMPLIED_WEIGHT_SCALE / len(slots)}
            for s in slots
        ),
    ]  # fmt: skip
    for condition in run.conditions:
        xs, ys = [], []
        for u in run.units:
            s = scores.get((u["persona"], u["item"]), {}).get(condition)
            if s is not None:
                xs.append([1.0, *(u["matches"][name] for name in slots)])
                ys.append(s)
        if len(ys) <= len(slots) + 1:
            continue
        coef, *_ = np.linalg.lstsq(np.array(xs), np.array(ys), rcond=None)
        for term, w in zip(["intercept", *slots], coef, strict=True):
            rows.append(
                {"judge": condition[0], "presentation": condition[1], "n": len(ys),
                 "term": term, "weight": float(w)}
            )  # fmt: skip
    return rows


def _bin(u: float) -> str:
    edges = UTILITY_BINS
    if u < edges[0]:
        return f"< {edges[0]:.1f}"
    for lo, hi in zip(edges, (*edges[1:], None), strict=True):
        if hi is None or u < hi:
            return f"{lo:.1f}–{hi:.1f}" if hi is not None else f"{lo:.1f}–1.0"
    raise AssertionError("unreachable")


BIN_ORDER = [_bin(0.0), *(_bin(e) for e in UTILITY_BINS)]


def leniency_table(run: JudgeRun, scores: Scores) -> list[dict[str, Any]]:
    """Mean score per utility bin and for each anchor, per condition."""
    rows = []
    for condition in run.conditions:
        groups: dict[str, list[float]] = defaultdict(list)
        for u in run.units:
            s = scores.get((u["persona"], u["item"]), {}).get(condition)
            if s is None:
                continue
            groups[_bin(u["utility"])].append(s)
            for anchor in (BEST, RANDOM):
                if anchor in u["sources"]:
                    groups[f"{anchor} item"].append(s)
        for group in [*BIN_ORDER, f"{BEST} item", f"{RANDOM} item"]:
            if groups.get(group):
                rows.append(
                    {"judge": condition[0], "presentation": condition[1], "group": group,
                     "n": len(groups[group]), "mean_score": float(np.mean(groups[group]))}
                )  # fmt: skip
    return rows


def pairs_table(run: JudgeRun) -> list[dict[str, Any]]:
    config = load_judge_config_text(run.state["config"])
    rows = []
    verdicts = pair_verdicts(run)
    for index, (a, b) in enumerate(config.pairs):
        mine = [v for v in verdicts if v["pair"] == index]
        both = [v for v in mine if v["winner_ab"] is not None and v["winner_ba"] is not None]
        consistent = [v for v in both if v["winner_ab"] == v["winner_ba"]]
        decided = [
            v for v in consistent
            if v["winner_ab"] != "equal" and abs(v["utility_a"] - v["utility_b"]) > 1e-9
        ]  # fmt: skip
        right = sum(
            (v["winner_ab"] == v["a"]) == (v["utility_a"] > v["utility_b"]) for v in decided
        )
        choices = [c for v in both for c in (v["choice_ab"], v["choice_ba"])]
        rows.append(
            {
                "pair": index,
                "a": source_label(*a),
                "b": source_label(*b),
                "personas": len(mine),
                "both_orders": len(both),
                "order_changed": len(both) - len(consistent),
                "order_changed_share": (len(both) - len(consistent)) / len(both) if both else "",
                "first": choices.count("first"),
                "second": choices.count("second"),
                "equal": choices.count("equal"),
                "a_won": sum(v["winner_ab"] == v["a"] for v in consistent),
                "b_won": sum(v["winner_ab"] == v["b"] for v in consistent),
                "decided_with_utility_gap": len(decided),
                "agrees_with_utility": right / len(decided) if decided else "",
            }
        )
    return rows


def plan_table(run: JudgeRun, slots: int) -> list[dict[str, Any]]:
    rows = []
    for group in dict.fromkeys(p["group"] for p in run.plans):
        mine = [p for p in run.plans if p["group"] == group]
        ok = [p for p in mine if not p["failed"]]
        rows.append(
            {
                "group": group,
                "plans": len(mine),
                "failed": len(mine) - len(ok),
                "named_share": float(np.mean([len(p["named"]) / slots for p in ok])) if ok else "",
                "other_share": float(np.mean([len(p["other"]) / slots for p in ok])) if ok else "",
            }
        )
    return rows


def units_table(run: JudgeRun, scores: Scores) -> list[dict[str, Any]]:
    rows = []
    for u in run.units:
        got = scores.get((u["persona"], u["item"]), {})
        rows.append(
            {
                "persona": u["persona"],
                "item": u["item"],
                "sources": ";".join(u["sources"]),
                "utility": u["utility"],
                "violations": u["violations"],
                **{f"m_{k}": v for k, v in u["matches"].items()},
                **{f"score_{j}_{p}": got.get((j, p), "") for j, p in run.conditions},
            }
        )
    return rows


# ---------------------------------------------------------------- output


def _num(v: Any, fmt: str = ".3f") -> str:
    if v == "" or v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return format(v, fmt)


def _markdown(run: JudgeRun, tables: dict[str, list[dict[str, Any]]], figure: Path | None,
              out: Path) -> str:  # fmt: skip
    s, src = run.state, run.state["source"]
    lines = [
        f"# Judge study: {s['name']}",
        "",
        f"- Source run: {src['name']} (replicate {src['replicate']}, {src['split']} split, "
        f"fingerprint `{src['fingerprint'][:16]}`), {s['personas']} personas, {s['units']} units",
        f"- Provider {s['provider']}, models {s['models']}; package version {s['version']}",
        f"- Calls planned: {s['plans']} plans, {s['scores']} scores, {s['pairs']} pairwise "
        f"({s['pairs_skipped']} pairs skipped: same item); failed: {s['plans_failed']} plans, "
        f"{s['judgments_failed']} judgements",
        "",
        "## Frozen hypotheses (SPEC Q13, M7)",
        "",
        "Persona means, paired Wilcoxon signed-rank (two-sided), Holm over J2–J6. J1 is a "
        f"bootstrap CI (bar {J1_BAR}); for J6, A is the share of `first` among non-`equal` "
        f"verdicts and B the share of pairs whose verdict changed with the order (bar {J6_BAR}).",
        "",
        "| | test | A | B | n | mean A | mean B | A − B [95 % CI] | p | p (Holm) | holds |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in tables["tests"]:
        diff = (
            f"{r['diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
            if r["diff"] != ""
            else f"[{_num(r['ci_low'])}, {_num(r['ci_high'])}]"
            if r["ci_low"] != ""
            else "–"
        )
        p = _p(r["p"]) if r["p"] != "" and not np.isnan(r["p"]) else "–"
        ph = _p(r["p_holm"]) if r["p_holm"] != "" else "–"
        lines.append(
            f"| {r['hypothesis']} | {r['test']} | {r['a']} | {r['b']} | {r['n']} | "
            f"{_num(r['mean_a'])} | {_num(r['mean_b'])} | {diff} | {p} | {ph} | "
            f"{'yes' if r['holds'] else 'no'} |"
        )
    lines += [
        "",
        "## Spearman with utility, every judge and presentation",
        "",
        "| judge | presentation | units | Spearman [95 % CI] |",
        "|---|---|---|---|",
    ]
    for r in tables["spearman"]:
        lines.append(
            f"| {r['judge']} | {r['presentation']} | {r['n']} | {_num(r['estimate'])} "
            f"[{_num(r['ci_low'])}, {_num(r['ci_high'])}] |"
        )
    lines += ["", "## Implied slot weights (least squares of score on m_s)", "",
              "The rubric implies score = 10 u, i.e. intercept 0 and the same weight on every "
              "slot.", ""]  # fmt: skip
    conditions = list(dict.fromkeys((r["judge"], r["presentation"]) for r in tables["weights"]))
    terms = list(dict.fromkeys(r["term"] for r in tables["weights"]))
    weight = {(r["judge"], r["presentation"], r["term"]): r["weight"] for r in tables["weights"]}
    lines.append("| term | " + " | ".join(f"{j} {p}".strip() for j, p in conditions) + " |")
    lines.append("|---|" + "---|" * len(conditions))
    for t in terms:
        cells = [_num(weight.get((j, p, t), ""), ".2f") for j, p in conditions]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += ["", "## Leniency: mean score by utility", ""]
    groups = list(dict.fromkeys(r["group"] for r in tables["leniency"]))
    mean = {(r["judge"], r["presentation"], r["group"]): r for r in tables["leniency"]}
    lines.append("| utility | " + " | ".join(f"{j} {p}" for j, p in run.conditions) + " |")
    lines.append("|---|" + "---|" * len(run.conditions))
    for g in groups:
        cells = []
        for j, p in run.conditions:
            r = mean.get((j, p, g))
            cells.append(f"{r['mean_score']:.2f} (n {r['n']})" if r else "–")
        lines.append(f"| {g} | " + " | ".join(cells) + " |")
    lines += ["", "## Pairwise verdicts", "",
              "| A | B | both orders | order changed | first / second / equal | A won / B won | "
              "agrees with utility |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in tables["pairs"]:
        lines.append(
            f"| `{r['a']}` | `{r['b']}` | {r['both_orders']} | {r['order_changed']} "
            f"({_num(r['order_changed_share'])}) | {r['first']} / {r['second']} / {r['equal']} | "
            f"{r['a_won']} / {r['b_won']} | {_num(r['agrees_with_utility'])} "
            f"(of {r['decided_with_utility_gap']}) |"
        )
    lines += ["", "## Plan fidelity (rule-matcher keywords, approximate)", "",
              "| writer | plans | failed | slots named | slots with another value |",
              "|---|---|---|---|---|"]  # fmt: skip
    for r in tables["plans"]:
        lines.append(
            f"| {r['group']} | {r['plans']} | {r['failed']} | {_num(r['named_share'])} | "
            f"{_num(r['other_share'])} |"
        )
    if figure is not None:
        import os

        rel = Path(os.path.relpath(figure, out)).as_posix()
        lines += ["", "## Score against utility", "", f"![{figure.stem}]({rel})"]
    return "\n".join(lines) + "\n"


def judge_report(
    judge_dir: str | Path,
    out: str | Path | None = None,
    figures_dir: str | Path | None = None,
    publish: bool = False,
    root: str | Path = ".",
    make_figures: bool = True,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[Path]:
    """Write the report of a judge folder; returns the files written."""
    run = load_judge_run(judge_dir)
    name = run.state["name"]
    if publish:
        out = Path(root) / "results_summary" / name
        figures_dir = Path(root) / "docs" / "figures" / name
    out = Path(out) if out is not None else run.root / "report"
    figures_dir = Path(figures_dir) if figures_dir is not None else out
    if is_committed_path(out) or (make_figures and is_committed_path(figures_dir)):
        problems = publish_problems(run)
        if problems:
            raise RefusedToPublish(problems)

    scores = score_table(run)
    slots = len(run.units[0]["matches"]) if run.units else 0
    tables = {
        "tests": tests_table(run, scores, resamples),
        "spearman": spearman_table(run, scores, resamples),
        "weights": weights_table(run, scores),
        "leniency": leniency_table(run, scores),
        "pairs": pairs_table(run),
        "plans": plan_table(run, slots),
    }
    written = [
        _write(out / "units.csv", _csv(units_table(run, scores))),
        _write(out / "tests.csv", _csv(tables["tests"])),
        _write(out / "weights.csv", _csv(tables["weights"])),
        _write(out / "pairs.csv", _csv(tables["pairs"])),
    ]
    figure = None
    if make_figures:
        figure = plot_score_vs_utility(
            run, tables["leniency"], figures_dir / "score_vs_utility.png"
        )
        written.append(figure)
    written.append(_write(out / "results.md", _markdown(run, tables, figure, out)))
    return written


def plot_score_vs_utility(run: JudgeRun, leniency: list[dict[str, Any]], path: Path) -> Path:
    from elicitor.analysis.figures import COLORS, GRID, INK, INK_MUTED, LINESTYLES, plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    judges = list(dict.fromkeys(j for j, _ in run.conditions))
    presentations = list(dict.fromkeys(p for _, p in run.conditions))
    x = list(range(len(BIN_ORDER)))
    for j, p in run.conditions:
        mean = {r["group"]: r["mean_score"] for r in leniency
                if r["judge"] == j and r["presentation"] == p}  # fmt: skip
        points = [(i, mean[g]) for i, g in enumerate(BIN_ORDER) if g in mean]
        if not points:
            continue
        ax.plot(
            [i for i, _ in points], [m for _, m in points],
            color=COLORS[judges.index(j)], linestyle=LINESTYLES[presentations.index(p)],
            linewidth=2, marker="o", markersize=4.5, markeredgecolor="white",
            markeredgewidth=1, label=f"{j} judge · {p}",
        )  # fmt: skip
    ax.set_xticks(x, BIN_ORDER)
    ax.set_xlabel("true utility of the item", color=INK_MUTED)
    ax.set_ylabel("mean judge score (0–10)", color=INK_MUTED)
    ax.set_ylim(0, 10.3)
    ax.set_title(f"{run.state['name']}: judge score against true utility", color=INK, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
    return path
