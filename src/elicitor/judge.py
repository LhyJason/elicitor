"""Judge study (SPEC Decision Q10 A, M7): LLM judges score the items a finished run recommended.

No dialogue is run. For each persona of the source run, the sample is the top item of a few
(arm, budget) sources plus two anchors (the best item and a random one). A *unit* is a distinct
(persona, item). Each unit is shown to each judge model in each presentation: the attribute
list, or a short plan that one of the two models wrote from the attributes alone. Pairs of
sources are also compared head to head, in both orders. The true utility of every unit is
known, which is what makes the judges checkable.

Outputs in the judge folder: `units.jsonl`, `plans.jsonl`, `judgments.jsonl`, `llm_stats.json`
and, written last, `judge_state.json`. A rerun is free through the reply cache.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from elicitor import __version__
from elicitor.catalog import Item, load_catalog
from elicitor.experiments.config import Arm
from elicitor.experiments.health import role_problems
from elicitor.experiments.store import RunStore, cell_id, sum_stats, write_text_atomic
from elicitor.jsonl import write_jsonl
from elicitor.llm import LLMClient, LLMParseError
from elicitor.llm.roles import AGENT, GROUPS, USER
from elicitor.personas import Persona, load_personas
from elicitor.recommender import match, utility, violations
from elicitor.rng import derive_rng, derive_seed
from elicitor.schema import NO_PREFERENCE, Schema, load_schema
from elicitor.simulator.llm import profile_lines
from elicitor.updaters.rule import find_mentions

LIST = "list"
PLAN_GROUPS = {"plan_agent": AGENT, "plan_user": USER}  # presentation -> group writing the plan
PRESENTATIONS = (LIST, *PLAN_GROUPS)
BEST, RANDOM = "best", "random"
ANCHORS = (BEST, RANDOM)
STATE = "judge_state.json"
_TIE = 1e-9  # utilities this close are equal (as for hit@k, SPEC §12)
_SEED_MOD = 2**31

_KEYS = {"name", "domain", "data_dir", "seed", "judge", "llm"}
_JUDGE_KEYS = {"source", "replicate", "arms", "anchors", "presentations", "judges", "pairs",
               "ordering", "constraints"}  # fmt: skip

Source = tuple[str, int]  # (arm name, budget)


def source_label(arm: str, budget: int) -> str:
    return f"{arm}@K{budget}"


# ---------------------------------------------------------------- judge files


@dataclass(frozen=True)
class JudgeConfig:
    name: str
    domain: str
    data_dir: Path
    seed: int
    source: Path  # run folder, relative to the working directory
    replicate: int
    arms: dict[str, tuple[int, ...]]  # arm name -> budgets whose top item is judged
    anchors: tuple[str, ...]
    presentations: tuple[str, ...]
    judges: tuple[str, ...]  # model groups
    pairs: tuple[tuple[Source, Source], ...]
    ordering: tuple[int, int]  # J2: recommended at the high budget, then the low one
    constraints: tuple[Source, Source]  # J3: source A, expected to score above source B
    llm: dict[str, Any]
    text: str  # the TOML text, for provenance

    @property
    def schema_path(self) -> Path:
        return self.data_dir / self.domain / "schema.toml"

    @property
    def personas_path(self) -> Path:
        return self.data_dir / self.domain / "personas.jsonl"

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / self.domain / "catalog.jsonl"

    @property
    def prompts_dir(self) -> Path:
        return self.data_dir / self.domain / "prompts"

    @property
    def sources(self) -> list[Source]:
        return [(arm, k) for arm, ks in self.arms.items() for k in ks]


def _source(raw: Any, path: Path) -> Source:
    if not (isinstance(raw, list) and len(raw) == 2):
        raise ValueError(f"{path}: a source is [arm name, budget], got {raw!r}")
    return Arm.parse(raw[0]).name, int(raw[1])


def load_judge_config(path: str | Path, source: str | Path | None = None) -> JudgeConfig:
    """Read a judge file; `source` overrides its run folder (`--source`)."""
    path = Path(path)
    return parse_judge_config(path.read_text(encoding="utf-8"), path, source)


def load_judge_config_text(text: str) -> JudgeConfig:
    """A judge file from its text, as kept in `judge_state.json` (the report needs no file)."""
    return parse_judge_config(text, Path(STATE))


def parse_judge_config(text: str, path: Path, source: str | Path | None = None) -> JudgeConfig:
    """`path` names the file in errors; data paths are relative to its folder."""
    raw = tomllib.loads(text)
    unknown = set(raw) - _KEYS
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    judge = dict(raw["judge"])
    unknown = set(judge) - _JUDGE_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown [judge] keys {sorted(unknown)}")
    config = JudgeConfig(
        name=raw["name"],
        domain=raw["domain"],
        data_dir=(path.parent / raw.get("data_dir", "../data")).resolve(),
        seed=int(raw["seed"]),
        source=Path(source if source is not None else judge["source"]),
        replicate=int(judge.get("replicate", 0)),
        arms={
            Arm.parse(name).name: tuple(sorted(int(k) for k in ks))
            for name, ks in judge["arms"].items()
        },
        anchors=tuple(judge.get("anchors", ANCHORS)),
        presentations=tuple(judge.get("presentations", PRESENTATIONS)),
        judges=tuple(judge.get("judges", (USER, AGENT))),
        pairs=tuple((_source(a, path), _source(b, path)) for a, b in judge.get("pairs", [])),
        ordering=tuple(int(k) for k in judge["ordering"]),  # type: ignore[arg-type]
        constraints=tuple(_source(s, path) for s in judge["constraints"]),  # type: ignore[arg-type]
        llm=dict(raw.get("llm", {})),
        text=text,
    )
    _check(config, path)
    return config


def _check(config: JudgeConfig, path: Path) -> None:
    def bad(what: str, values: tuple[str, ...], allowed: tuple[str, ...]) -> None:
        extra = set(values) - set(allowed)
        if extra:
            raise ValueError(f"{path}: unknown {what} {sorted(extra)}; choose from {allowed}")

    bad("anchors", config.anchors, ANCHORS)
    bad("presentations", config.presentations, PRESENTATIONS)
    bad("judges", config.judges, GROUPS)
    if USER not in config.judges or LIST not in config.presentations:
        raise ValueError(f"{path}: the main judge (user group, list presentation) is required")
    sources = set(config.sources)
    for source in [s for pair in config.pairs for s in pair] + list(config.constraints):
        if source not in sources:
            raise ValueError(f"{path}: {source_label(*source)} is not in [judge.arms]")
    if len(config.constraints) != 2:
        raise ValueError(f"{path}: constraints are two sources, A and B")
    budgets = {k for _, k in sources}
    if len(config.ordering) != 2 or not set(config.ordering) <= budgets:
        raise ValueError(f"{path}: ordering is [high budget, low budget] of the sample")


# ---------------------------------------------------------------- the sample


@dataclass
class Unit:
    persona: str
    item: str
    sources: list[str]  # source labels and anchors that chose this item
    utility: float
    violations: int
    matches: dict[str, float]  # per-slot match score m_s (SPEC §10)

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona": self.persona,
            "item": self.item,
            "sources": self.sources,
            "utility": round(self.utility, 6),
            "violations": self.violations,
            "matches": {k: round(v, 6) for k, v in self.matches.items()},
        }


def item_lines(schema: Schema, attrs: dict[str, str]) -> str:
    """The `list` presentation: one line per slot, its value's label."""
    return "\n".join(
        f"- {slot.name.replace('_', ' ')}: {slot.value(attrs[slot.name]).label}"
        for slot in schema.slots
    )


def read_picks(store: RunStore, config: JudgeConfig, personas: list[str]) -> dict:
    """{(persona, source label): top item id} from the source run's replicate cells."""
    picks: dict[tuple[str, str], str] = {}
    for arm, budgets in config.arms.items():
        cell = cell_id(arm, config.replicate)
        if not store.is_done(cell):
            raise ValueError(f"{store.root}: cell {cell} is not finished")
        for episode in store.read_cell(cell):
            if episode["budget"] in budgets:
                picks[(episode["persona"], source_label(arm, episode["budget"]))] = episode[
                    "recommended"
                ][0]
    missing = [
        (p, source_label(*s)) for p in personas for s in config.sources
        if (p, source_label(*s)) not in picks
    ]  # fmt: skip
    if missing:
        raise ValueError(f"{store.root}: no episode for {missing[0]} (and {len(missing) - 1} more)")
    return picks


def anchor_items(
    schema: Schema, catalog: list[Item], persona: Persona, seed: int
) -> dict[str, str]:
    """The best item (first of maximal utility, as hit@1 counts it) and a random item that is
    not of maximal utility, seeded by persona."""
    utilities = [utility(schema, item.attrs, persona.truth) for item in catalog]
    best = max(utilities)
    top = [i for i, u in enumerate(utilities) if u >= best - _TIE]
    rest = [i for i, u in enumerate(utilities) if u < best - _TIE]
    rng = derive_rng(seed, "random_item", persona.id)
    return {BEST: catalog[top[0]].id, RANDOM: catalog[rng.choice(rest)].id}


def build_units(
    config: JudgeConfig,
    schema: Schema,
    catalog: list[Item],
    personas: list[Persona],
    picks: dict[tuple[str, str], str],
) -> list[Unit]:
    """Units in persona order; within a persona, items in the order their first source is
    listed (sources as in [judge.arms], then the anchors)."""
    by_id = {item.id: item for item in catalog}
    units: list[Unit] = []
    for persona in personas:
        chosen = [(source_label(*s), picks[(persona.id, source_label(*s))]) for s in config.sources]
        anchors = anchor_items(schema, catalog, persona, config.seed)
        chosen += [(a, anchors[a]) for a in config.anchors]
        mine: dict[str, Unit] = {}
        for label, item_id in chosen:
            if item_id not in mine:
                attrs = by_id[item_id].attrs
                mine[item_id] = Unit(
                    persona=persona.id,
                    item=item_id,
                    sources=[],
                    utility=utility(schema, attrs, persona.truth),
                    violations=violations(schema, attrs, persona.truth),
                    matches={
                        s.name: match(s, attrs[s.name], persona.truth[s.name]) for s in schema.slots
                    },
                )
            mine[item_id].sources.append(label)
        units += mine.values()
    return units


def plan_fidelity(schema: Schema, attrs: dict[str, str], plan: str) -> dict[str, list[str]]:
    """Slots whose value the plan names, and slots where it names another value (rule matcher
    keywords, so approximate)."""
    named, other = [], []
    for slot in schema.slots:
        positive, _ = find_mentions(slot, plan)
        if attrs[slot.name] in positive:
            named.append(slot.name)
        if any(v not in (attrs[slot.name], NO_PREFERENCE) for v in positive):
            other.append(slot.name)
    return {"named": named, "other": other}


# ---------------------------------------------------------------- running


def _seed(*parts: object) -> int:
    return derive_seed(*parts) % _SEED_MOD


def run_judge(
    config: JudgeConfig,
    client: LLMClient,
    out_dir: str | Path,
    workers: int = 1,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Plan, judge and compare; writes the judge folder and returns its state (with the health
    problems of the LLM statistics under `problems`)."""
    say = progress or (lambda _msg: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / STATE).unlink(missing_ok=True)  # the folder is incomplete until the end

    store = RunStore(config.source)
    source_state = store.read_state()
    if source_state is None:
        raise ValueError(f"{config.source} is not a run folder (no run_state.json)")
    if source_state["replicates"] <= config.replicate:
        raise ValueError(f"{config.source} has no replicate {config.replicate}")
    schema = load_schema(config.schema_path)
    catalog = load_catalog(config.catalog_path)
    by_id = {item.id: item for item in catalog}
    all_personas = {p.id: p for p in load_personas(config.personas_path)}
    personas = [all_personas[pid] for pid in source_state["personas"]]
    picks = read_picks(store, config, [p.id for p in personas])
    units = build_units(config, schema, catalog, personas, picks)
    say(f"{len(units)} units from {len(personas)} personas of {source_state['name']}")

    stats: list[dict[str, Any] | None] = []

    def tracked(fn: Callable[[Any], dict[str, Any]]) -> Callable[[Any], dict[str, Any]]:
        def run(task: Any) -> dict[str, Any]:
            with client.track() as tally:
                row = fn(task)
            stats.append(client.stats_dict(tally))
            return row

        return run

    def pool_map(fn: Callable[[Any], dict[str, Any]], tasks: list[Any]) -> list[dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(tracked(fn), tasks))

    # Plans: once per item and writing model, reused across personas.
    plan_tasks = [
        (item_id, PLAN_GROUPS[p])
        for p in config.presentations if p in PLAN_GROUPS
        for item_id in dict.fromkeys(u.item for u in units)
    ]  # fmt: skip

    def write_plan(task: tuple[str, str]) -> dict[str, Any]:
        item_id, group = task
        attrs = by_id[item_id].attrs
        context = {
            "item": item_lines(schema, attrs),
            "labels": [s.value(attrs[s.name]).label for s in schema.slots],
        }
        row: dict[str, Any] = {"item": item_id, "group": group}
        try:
            reply = client.call(
                "plan", context, seed=_seed(config.seed, "plan", item_id), group=group
            )
        except LLMParseError:
            return {**row, "plan": None, "failed": True}
        plan = reply.plan.strip()  # type: ignore[attr-defined]
        return {**row, "plan": plan, "failed": False, **plan_fidelity(schema, attrs, plan)}

    plans = pool_map(write_plan, plan_tasks)
    plan_text = {(r["item"], r["group"]): r["plan"] for r in plans}
    say(f"{len(plans)} plans, {sum(r['failed'] for r in plans)} failed")

    profiles = {p.id: profile_lines(schema, p) for p in personas}

    def presented(item_id: str, presentation: str) -> str | None:
        if presentation == LIST:
            return item_lines(schema, by_id[item_id].attrs)
        return plan_text[(item_id, PLAN_GROUPS[presentation])]

    score_tasks = [
        (unit, presentation, group)
        for unit in units for presentation in config.presentations for group in config.judges
    ]  # fmt: skip

    def score(task: tuple[Unit, str, str]) -> dict[str, Any]:
        unit, presentation, group = task
        row: dict[str, Any] = {
            "kind": "score",
            "persona": unit.persona,
            "item": unit.item,
            "presentation": presentation,
            "judge": group,
        }
        text = presented(unit.item, presentation)
        if text is None:  # its plan failed: the judgement fails too, without a call
            return {**row, "score": None, "reason": "", "failed": True, "plan_failed": True}
        context = {"profile": profiles[unit.persona], "item": text, "utility": unit.utility}
        seed = _seed(config.seed, "judge", unit.persona, unit.item, presentation)
        try:
            reply = client.call("judge", context, seed=seed, group=group)
        except LLMParseError:
            return {**row, "score": None, "reason": "", "failed": True}
        return {**row, "score": reply.score, "reason": reply.reason, "failed": False}  # type: ignore[attr-defined]

    unit_by_key = {(u.persona, u.item): u for u in units}
    pair_tasks = []
    skipped_pairs = 0
    for persona in personas:
        for index, (a, b) in enumerate(config.pairs):
            item_a = picks[(persona.id, source_label(*a))]
            item_b = picks[(persona.id, source_label(*b))]
            if item_a == item_b:
                skipped_pairs += 1
                continue
            pair_tasks += [(persona.id, index, "ab", item_a, item_b),
                           (persona.id, index, "ba", item_b, item_a)]  # fmt: skip

    def compare(task: tuple[str, int, str, str, str]) -> dict[str, Any]:
        persona_id, index, order, first, second = task
        row: dict[str, Any] = {
            "kind": "pair",
            "persona": persona_id,
            "pair": index,
            "order": order,
            "first": first,
            "second": second,
        }
        context = {
            "profile": profiles[persona_id],
            "first": item_lines(schema, by_id[first].attrs),
            "second": item_lines(schema, by_id[second].attrs),
            "utility_first": unit_by_key[(persona_id, first)].utility,
            "utility_second": unit_by_key[(persona_id, second)].utility,
        }
        seed = _seed(config.seed, "pair", persona_id, first, second)
        try:
            reply = client.call("pair", context, seed=seed)
        except LLMParseError:
            return {**row, "choice": None, "reason": "", "failed": True}
        return {**row, "choice": reply.choice, "reason": reply.reason, "failed": False}  # type: ignore[attr-defined]

    def judge_task(task: tuple) -> dict[str, Any]:
        return score(task) if isinstance(task[0], Unit) else compare(task)

    judgments = pool_map(judge_task, [*score_tasks, *pair_tasks])
    failed = sum(r["failed"] for r in judgments)
    say(f"{len(score_tasks)} scores and {len(pair_tasks)} pairwise verdicts, {failed} failed")

    llm = sum_stats(stats)
    write_jsonl(out / "units.jsonl", (u.to_dict() for u in units))
    write_jsonl(out / "plans.jsonl", plans)
    write_jsonl(out / "judgments.jsonl", judgments)
    if llm is not None:
        write_text_atomic(out / "llm_stats.json", json.dumps(llm, indent=2) + "\n")

    cells = [cell_id(arm, r) for arm in source_state["arms"]
             for r in range(source_state["replicates"])]  # fmt: skip
    state = {
        "name": config.name,
        "version": __version__,
        "provider": config.llm.get("provider", "mock"),
        "models": (llm or {}).get("models"),
        "personas": len(personas),
        "units": len(units),
        "plans": len(plans),
        "plans_failed": sum(r["failed"] for r in plans),
        "scores": len(score_tasks),
        "pairs": len(pair_tasks),
        "pairs_skipped": skipped_pairs,
        "judgments_failed": failed,
        "config": config.text,
        "source": {
            "name": source_state["name"],
            "path": config.source.as_posix(),
            "fingerprint": source_state["fingerprint"],
            "split": source_state["split"],
            "max_personas": source_state.get("max_personas"),
            "provider": source_state.get("provider"),
            "complete": all(store.is_done(c) for c in cells),
            "replicate": config.replicate,
        },
    }
    write_text_atomic(out / STATE, json.dumps(state, indent=2) + "\n")
    return {**state, "llm": llm, "problems": role_problems(llm["roles"]) if llm else []}
