# Elicitor — design spec (v1, 2026-10-22)

Design decisions are marked **Decision Qn** with the options considered and the choice; §0 indexes them.
The code is written from this file. A changed decision is recorded here before the code changes.

---

## 0. Decision index

| Q | Topic | Choice | Status |
|---|---|---|---|
| Q1 | Project name | Elicitor (package `elicitor`) | settled |
| Q2 | Domain | domain-agnostic core, travel as the first domain | settled |
| Q3 | Slot schema | 6–8 categorical + 2 binned numeric constraints, written fresh | settled |
| Q4 | Personas | generated from the schema with a seed; 12 dev + 48 eval | settled |
| Q5 | Belief representation | factorized categorical per slot | settled |
| Q6 | Belief updater | LLM likelihoods + Bayes rule; rule matcher as the no-LLM baseline | settled |
| Q7 | Question types and EIG | analytic EIG + one small Monte Carlo EIG arm; add choice questions | settled |
| Q8 | Stopping rule | budget K plus a confidence threshold τ | settled |
| Q9 | Question phrasing | templates by default, LLM phrasing as one ablation | settled |
| Q10 | Recommendation target | fixed synthetic catalog (objective regret); LLM judge optional | settled |
| Q11 | User simulator | LLM simulator for experiments, rule oracle for tests | settled |
| Q12 | Experiment grid | selector × updater factorial, 5 replicates, phrasing as a separate ablation | settled |
| Q13 | Hypotheses | frozen in this file before the full run | settled |
| Q14 | Models per role | agent qwen3:32b; simulator and judge llama3.3:70b (different family) | settled |
| Q15 | Structured outputs | pydantic models → JSON schema per role | settled |
| Q16 | Config and CLI | TOML experiment files + one `argparse` CLI | settled |
| Q17 | Packaging and tooling | `src/` layout, pyproject, ruff, pytest, CI on the mock backend | settled |
| Q18 | License | MIT | settled |
| Q19 | Results in the repo | summary tables + figures committed; raw runs as a release asset | settled |
| Q20 | Live demo | terminal chat where a human answers the questions | settled |

**Also fixed:** Python 3.12. No paid LLM APIs: every role runs on Ollama (local or on a Slurm GPU
node) or on a deterministic mock. English README with results. Sole author.

---

## 1. Goal and scope

**Pitch.** An LLM agent that learns what a user wants through a short dialogue. It keeps a
probability distribution (belief) over the user's preferences, picks each question by how much it
is expected to reduce uncertainty, updates the belief from the free-text answer, and then
recommends.

**Research question.** With a fixed budget of questions, how much do (a) principled question
selection and (b) LLM-based belief updates improve what the agent knows and what it recommends?

**Deliverables.** Installable Python package with a CLI; experiment suite that runs end to end on
the mock backend in under a minute and on a real model locally or on a cluster; README with results
and figures.

**Non-goals.** Paid APIs, fine-tuning, real human studies, a web front end (beyond Q20).

**Size limits.** The project stays small on purpose:
- Scope is M0–M6; M7 adds at most one extension.
- The full experiment finishes in one cluster job of at most 12 h wall time on at most 4 GPUs.
  Smoke runs stay under 1 h.
- If the measured rate (M5 smoke run) projects past 12 h, the grid is cut, in this order: fewer
  replicates (down to 3), drop K = 8, fewer Monte Carlo samples. The time limit is not extended.

## 2. Terms

slot · value · constraint · belief · question · answer · turn · episode · persona · catalog item ·
arm (one configuration of the agent) · replicate (one seed of an arm over all personas).

## 3. Domain and schema

The core knows nothing about travel; a domain is a schema file plus prompt templates.

> **Decision Q2 — domain.**
> A: travel planning only. B: domain-agnostic core, ship travel first. C: a different domain
> (restaurants, laptops, apartments).
> *Chosen B:* little extra work if the schema is data from the start, and a second domain later
> (§17, M7) shows the method generalizes.

> **Decision Q3 — slot schema.** How many slots, which values, whether every slot has an explicit
> "no preference" value, and how numeric constraints are handled.
> *Chosen:* 6–8 categorical preference slots with 3–5 values each, every one with a
> `no_preference` value; 2 numeric constraints (budget, trip length) binned into 4–5 ordinal bins
> so that entropy and EIG treat all slots the same way. Slot list written fresh for this project.

Schema file sketch (format only, slots TBD):

```toml
[slot.example_slot]
kind = "categorical"            # or "ordinal" for binned numeric constraints
values = ["a", "b", "c", "no_preference"]
prior = "uniform"               # or explicit weights
description = "what this slot means, used in prompts"
```

## 4. Personas

A persona = ground-truth value for every slot + an answer style + a short backstory.

> **Decision Q4 — persona generation and count.**
> *Chosen:* sample ground truth from the schema with a fixed seed (stratified so every value
> appears), cross with 3–4 answer styles (direct, vague, rambling, indifferent), have an LLM write a
> 2–3 sentence backstory once, then freeze everything to `data/<domain>/personas.jsonl`.
> Split 12 dev personas (prompt tuning only) and 48 eval personas (reported results only).

## 5. Belief model

For each slot $s$, a categorical distribution $b_s(v)$ over its values, starting from the schema
prior. Uncertainty of a slot: $H(b_s) = -\sum_v b_s(v)\log b_s(v)$.

> **Decision Q5 — representation.**
> A: factorized categorical per slot. B: joint distribution over slot pairs (captures correlations).
> C: Dirichlet pseudo-counts per slot.
> *Chosen A* for the main system. Note that with A and single-slot questions, EIG reduces almost
> to max-entropy (see §7); B or C are possible extensions.

## 6. Belief updaters

Interface: `update(belief, question, answer_text) -> belief` plus a log record of what was parsed.

> **Decision Q6 — updater family.**
> A: rule/keyword matcher (no LLM). B: LLM returns a likelihood score for each value of the asked
> slot(s); posterior $\propto$ prior $\times$ likelihood$^\beta$ (β tempers over-confidence).
> C: LLM writes the posterior directly.
> *Chosen B as the main updater and A as the no-LLM baseline;* C only if time allows. B keeps
> Bayes rule in code (testable, explainable) and the LLM only does language understanding.
> Answers may mention slots that were not asked; B should score those too (settled in M2 on dev personas).

## 7. Questions and selection policies

Question types: open question on one slot; choice between two or three values of one slot;
yes/no on a subset of values. (Multi-slot questions: not in v1.)

Selection policies: `fixed` (schema order), `random`, `max_entropy` (slot with the largest
$H(b_s)$), `eig` (question with the largest expected information gain).

**Why question types matter.** With only open single-slot questions and noise-free answers, the EIG
of asking slot $s$ is exactly $H(b_s)$, so `eig` and `max_entropy` pick the same question. They
differ only when answers are noisy with slot-dependent reliability, when question types carry
different amounts of information (a yes/no question gains at most 1 bit), or when slots are
correlated (Q5 B). The experiment must include at least one of these, or the EIG arm is a duplicate.

> **Decision Q7 — how EIG is computed.**
> A: analytic, under an answer model "the answer reveals the true value with probability ρ, else a
> random one". B: Monte Carlo, sampling simulated answers from an LLM and running the updater on
> each. *Chosen A as the main EIG selector* (cheap, exact, easy to explain), *plus B as one small
> extra arm* (`sel=eig_mc`): S = 3 sampled answers per candidate question, run only at K = 5 with
> the LLM updater. B costs about S × (number of candidate questions) × 2 LLM calls per turn, which
> is why it is kept to one setting (see §1, size limits).

> **Decision Q8 — when to stop asking.**
> *Chosen:* budget K (sweep K ∈ {3, 5, 8}, fewer than the number of slots so that ordering
> matters) plus early stop when every slot has $\max_v b_s(v) \ge \tau$. Report per-turn curves,
> not only the final turn.

## 8. Question phrasing

> **Decision Q9.** A: templates from the schema (deterministic). B: LLM rewrites the template into
> a natural question. *Chosen A by default, B as one ablation* to measure whether nicer phrasing
> changes how informative the simulated answers are.

## 9. User simulator

Answers a question in character, using only the persona; must not volunteer the whole profile.

> **Decision Q11.** A: LLM role-play prompted with ground truth + answer style. B: rule oracle that
> answers from ground truth with controlled noise ρ. *Chosen both:* A for experiments, B for unit
> tests and for a noise sweep where ρ is known exactly.
> Leakage check: count how often an answer mentions slots that were not asked.

## 10. Recommendation and evaluation target

> **Decision Q10.**
> A: LLM writes a free-text plan; an LLM judge scores it against the persona.
> B: a fixed synthetic catalog (e.g. 200 items with attribute values on every slot); the agent
> ranks items by expected utility under its belief; score = regret against the persona's best item,
> hit@k, constraint violations.
> *Chosen B as the primary metric* (objective, cheap, no judge bias) and A as a secondary,
> human-readable output with the judge run on a sample.
> Judge failures are counted as failures, never replaced by a default score.

## 11. Episode loop

```text
belief  = prior(schema)
history = []
for turn in 1..K:
    if confident(belief, τ): break
    q      = selector.choose(belief, history)
    text   = phraser.render(q)
    answer = simulator.respond(text, persona)
    belief = updater.update(belief, q, answer)
    history.append(turn record: q, text, answer, belief snapshot)
rec   = recommender.rank(belief, catalog)
score = evaluate(rec, persona, catalog)
```

Every component is chosen by config, and every random draw comes from one seeded generator per
episode, derived from (replicate seed, persona id, arm).

## 12. Metrics

- Belief, per turn: mean posterior mass on the true value; MAP accuracy over slots; log loss;
  mean entropy.
- Efficiency: turns until confident; area under the accuracy-vs-turn curve.
- Outcome: regret, hit@k, constraint violations (Q10).
- Operations, per role: calls, cache hits, parse failures, errors, latency, tokens.

## 13. Experiments

> **Decision Q12 — grid.** *Chosen:* selector {fixed, random, max_entropy, eig} × updater
> {rule, llm} = 8 arms, each × K values; phrasing (Q9) and simulator noise (Q11) as separate small
> ablations on the best arm. Arms are named by their components, e.g. `sel=eig,upd=llm,phr=tpl`.
> Plus the `sel=eig_mc` arm from Q7. 5 replicates (3 if the time limit in §1 requires it).
> Statistics: average replicates per persona, then paired Wilcoxon signed-rank over
> eval personas with bootstrap CIs and Holm correction across comparisons.

> **Decision Q13 — hypotheses** (draft; freeze before the full run):
> H1 `eig` > `fixed` and `random` on posterior mass on truth at budget K.
> H2 `eig` > `max_entropy` only in settings where they can differ (§7).
> H3 the LLM updater beats the rule updater, and the gap grows with vaguer answer styles.
> H4 better final belief → lower regret (episode-level correlation).

> **Decision Q14 — models per role.** *Chosen:* agent roles (updater, phrasing) on qwen3:32b;
> simulator and judge on llama3.3:70b, a different family, to limit self-preference. Both run on
> cluster GPUs (A40, 48 GB). Development uses the mock backend and a 4B-class model locally
> (6 GB VRAM). Thinking/reasoning output off for roles that must return JSON.

## 14. LLM layer

One client, providers `ollama` (native API), `openai_compatible` (any local server), `mock`
(deterministic, schema-valid replies for tests).

- Role registry: role name → prompt template, output schema, temperature, model.
- SQLite cache keyed by hash(provider, model, role, prompt, params, seed); a re-run is free.
- Structured output with retry on parse failure; every failure is logged per role.
- Call statistics written per run (see §12, operations).

> **Decision Q15.** *Chosen* pydantic v2 models for each role's output; the same model produces
> the JSON schema sent to the server and validates the reply.

## 15. Infrastructure

- Local: Ollama + small model; mock backend for CI and tests.
- Cluster: one Slurm job on up to 4 A40 GPUs, one Ollama server per GPU (two for the agent model,
  two for the simulator/judge model); the client spreads each role's calls over that role's
  servers. Submit script with a resume-by-default runner (finished (arm, replicate) cells are
  skipped), so a job that hits its time limit is resubmitted, not restarted.
- Run-health report: per-role error and parse rates, readable while a job runs.
- No account names, usernames or home paths in the repo; site settings live in an untracked
  `local.env`.

## 16. Repo layout and tooling

```text
src/<pkg>/
  schema.py  personas.py  belief.py  questions.py  catalog.py  episode.py  metrics.py
  updaters/   (rule.py, llm.py)
  selectors/  (fixed.py, random.py, entropy.py, eig.py)
  phrasing.py  simulator/ (oracle.py, llm.py)  recommender.py
  llm/        (client.py, providers.py, cache.py, roles.py, mock.py)
  experiments/ (config.py, runner.py)   analysis/ (stats.py, figures.py)
  cli.py
data/<domain>/   schema.toml  personas.jsonl  catalog.jsonl  prompts/
experiments/     *.toml
cluster/         slurm scripts, local.env.example
tests/           unit tests (belief math, EIG properties, updaters) + mock end-to-end
docs/            SPEC.md  figures/
results_summary/ committed tables
```

> **Decision Q16 — config and CLI.** *Chosen* TOML experiment files read into dataclasses, one
> `argparse` CLI with subcommands `run`, `analyze`, `health`, `demo`. No Hydra.

> **Decision Q17 — packaging and tooling.** *Chosen* pyproject + setuptools, plain venv/pip,
> ruff (lint + format), pytest, GitHub Actions running tests and the mock suite on every push.
> Few dependencies: numpy, scipy, pydantic, httpx, matplotlib, pandas.

> **Decision Q18 — license.** *Chosen* MIT.

## 17. Milestones

| M | Content | Done when |
|---|---|---|
| M0 | repo, pyproject, CI, empty package | CI green |
| M1 | schema, personas, belief, rule updater, fixed/random/entropy selectors, oracle simulator, catalog, episode loop | full suite runs with no LLM; belief and entropy unit tests pass |
| M2 | LLM layer (client, cache, mock, Ollama), LLM updater, phrasing, LLM simulator | mock suite in CI; local small-model smoke run with 0 errors |
| M3 | EIG selector + question types, recommender, metrics | EIG unit tests (EIG = H for noise-free open questions; EIG ≤ 1 bit for yes/no) |
| M4 | runner, run-health, analysis, figures | analysis runs on mock output and refuses to write committed tables from it |
| M5 | cluster jobs (≤ 4 A40), smoke run to measure the call rate, full run | all cells finished within the §1 time limit; error rate 0; parse failures < 1 % per role |
| M6 | README results, demo, release | fresh-clone test passes on a clean venv |
| M7 | optional, at most one: second domain, joint belief, or judge study | — |

## 18. Results and release

> **Decision Q19.** *Chosen* committing summary tables and figures; raw per-episode logs and the
> call cache as a GitHub release asset (or not at all).

> **Decision Q20 — demo.** *Chosen* a `demo` subcommand: terminal chat where a human answers,
> showing the belief bars after each turn. Useful in interviews.
