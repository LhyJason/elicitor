"""Measure the answer reliability rho per question type on the dev personas (SPEC §7, M5).

Every dev persona gets single questions from the prior belief, with template wording: per slot
the open question, a choice between two values and one between three, and two yes/no
questions. Each answer is read by every updater. A reading is correct when the true outcome has
the largest likelihood among the question's outcomes (ties shared). Under the answer model the
correct rate is a = rho + (1 - rho) / m, which gives

    rho_hat = (mean(a) - mean(1/m)) / (1 - mean(1/m)),

clipped to [0, 1], with a percentile bootstrap CI over personas.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from elicitor.belief import Belief
from elicitor.experiments.config import ExperimentConfig
from elicitor.jsonl import write_jsonl
from elicitor.metrics import leaked_slots
from elicitor.personas import Persona, load_personas
from elicitor.phrasing import template_question
from elicitor.questions import CHOICE, KINDS, YESNO, Question, candidates, noise_free_outcome
from elicitor.questions import outcomes as question_outcomes
from elicitor.rng import derive_rng
from elicitor.schema import Schema, load_schema
from elicitor.simulator import make_simulator
from elicitor.updaters import Updater, make_updater

UPDATERS = ("rule", "llm")
BOOTSTRAP = 2000
_TIE = 1e-9


def calibration_questions(schema: Schema, persona: Persona, seed: int) -> list[Question]:
    rng = derive_rng(seed, persona.id, "calibrate")
    out: list[Question] = []
    for slot in schema.slots:
        out.append(Question(slot.name))
        choices = candidates(slot, (CHOICE,))
        pairs = [q for q in choices if len(q.values) == 2]
        triples = [q for q in choices if len(q.values) == 3]
        out += [rng.choice(pairs), rng.choice(triples)] if triples else rng.sample(pairs, 2)
        out += rng.sample(candidates(slot, (YESNO,)), 2)
    return out


def reading(
    schema: Schema, updater: Updater, question: Question, text: str, answer: str, truth: str
) -> float:
    """1 when the true outcome alone has the largest likelihood, 1/j when j outcomes tie for it
    and the true one is among them, else 0."""
    prior = Belief.prior(schema)
    posterior, _ = updater.update(prior, question, text, answer)
    slot = schema.slot(question.slot)
    before, after = prior.probs(slot.name), posterior.probs(slot.name)
    ratio = {v: a / b for v, b, a in zip(slot.value_names, before, after, strict=True) if b > 0}
    by_outcome: dict[str, list[float]] = defaultdict(list)
    for value, r in ratio.items():
        by_outcome[noise_free_outcome(question, value)].append(r)
    scores = {w: sum(rs) / len(rs) for w, rs in by_outcome.items()}
    top = max(scores.values())
    best = [w for w, s in scores.items() if s >= top * (1 - _TIE)]
    truthful = noise_free_outcome(question, truth)
    return 1 / len(best) if truthful in best else 0.0


def rho_hat(correct: np.ndarray, chance: np.ndarray) -> float:
    c = chance.mean()
    return float(np.clip((correct.mean() - c) / (1 - c), 0.0, 1.0))


def estimate(rows: list[dict[str, Any]], updater: str, kind: str, resamples: int) -> dict:
    """rho_hat with a bootstrap CI over personas for one updater and question type."""
    rows = [r for r in rows if r["question"]["kind"] == kind]
    by_persona: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_persona[r["persona"]].append(r)
    personas = sorted(by_persona)

    def arrays(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        picked = [r for p in ids for r in by_persona[p]]
        correct = np.array([r["correct"][updater] for r in picked], dtype=float)
        chance = np.array([1 / r["outcomes"] for r in picked], dtype=float)
        return correct, chance

    correct, chance = arrays(personas)
    rng = np.random.default_rng(0)
    boots = [
        rho_hat(*arrays([personas[i] for i in rng.integers(0, len(personas), len(personas))]))
        for _ in range(resamples)
    ]
    low, high = np.percentile(boots, [2.5, 97.5])
    return {
        "answers": int(correct.size),
        "correct_rate": round(float(correct.mean()), 4),
        "chance": round(float(chance.mean()), 4),
        "rho": round(rho_hat(correct, chance), 4),
        "ci_low": round(float(low), 4),
        "ci_high": round(float(high), 4),
    }


def calibrate(
    config: ExperimentConfig,
    out_dir: str | Path,
    workers: int = 1,
    progress: Callable[[str], None] | None = None,
    resamples: int = BOOTSTRAP,
) -> dict[str, Any]:
    """Ask, read and estimate; writes `answers.jsonl` and `calibration.json` to `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema = load_schema(config.schema_path)
    personas = load_personas(config.personas_path, split="dev")
    kind = config.simulator.get("kind", "oracle")
    client = None
    # Without an [llm] table (an oracle file) only the rule updater reads the answers.
    if kind == "llm" or config.llm:
        from elicitor.llm import build_client

        client = build_client(config.llm, config.prompts_dir, out / "llm_cache.sqlite")
    sim_params = {k: v for k, v in config.simulator.items() if k != "kind"}
    simulator = make_simulator(
        schema, kind, client=client, prompts_dir=config.prompts_dir, **sim_params
    )
    updaters = {
        name: make_updater(name, client=client, **config.updater_params.get(name, {}))
        for name in UPDATERS
        if name == "rule" or client is not None
    }

    def ask(task: tuple[Persona, Question]) -> dict[str, Any]:
        persona, question = task
        slot = schema.slot(question.slot)
        text = template_question(slot, question)
        rng = derive_rng(config.seed, persona.id, "calibrate", json.dumps(question.to_dict()))
        answer = simulator.respond(question, text, persona, rng)
        truth = persona.truth[slot.name]
        return {
            "persona": persona.id,
            "style": persona.style,
            "question": question.to_dict(),
            "text": text,
            "answer": answer,
            "truth": truth,
            "outcomes": len(question_outcomes(slot, question)),
            "leaks": leaked_slots(schema, slot.name, answer),
            "correct": {
                name: reading(schema, u, question, text, answer, truth)
                for name, u in updaters.items()
            },
        }

    tasks = [(p, q) for p in personas for q in calibration_questions(schema, p, config.seed)]
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            rows = list(pool.map(ask, tasks))
        llm = client.stats_dict() if client is not None else None
    finally:
        if client is not None:
            client.close()

    write_jsonl(out / "answers.jsonl", rows)
    result: dict[str, Any] = {
        "name": config.name,
        "simulator": kind,
        "models": (llm or {}).get("models"),
        "personas": len(personas),
        "answers": len(rows),
        "leak_rate": round(sum(bool(r["leaks"]) for r in rows) / len(rows), 4),
        "rho": {name: {k: estimate(rows, name, k, resamples) for k in KINDS} for name in updaters},
        "llm": llm,
    }
    (out / "calibration.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    if progress:
        progress(format_calibration(result))
    return result


def format_calibration(result: dict[str, Any]) -> str:
    lines = [
        f"{result['answers']} answers from {result['personas']} dev personas, "
        f"leak rate {result['leak_rate']:.1%}",
        "updater  kind     answers  correct  chance   rho     95% CI",
    ]
    for name, kinds in result["rho"].items():
        for kind, r in kinds.items():
            lines.append(
                f"{name:<8} {kind:<8} {r['answers']:>7}  {r['correct_rate']:.3f}    "
                f"{r['chance']:.3f}    {r['rho']:.3f}   [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
            )
    return "\n".join(lines)
