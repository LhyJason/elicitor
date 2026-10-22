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

*M3 additions.* An ordinal slot may set `threshold_question`, the template of its yes/no question
"below this bin edge?" with an `{edge}` placeholder (§8), and `constraint`, how a catalog item
can break it (§10): `"at_most"` (the item must not lie in a higher bin than the user's, e.g. the
budget) or `"exact"` (the item must lie in the user's bin, e.g. trip length; the default).
Only ordinal slots are constraints.

## 4. Personas

A persona = ground-truth value for every slot + an answer style + a short backstory.

> **Decision Q4 — persona generation and count.**
> *Chosen:* sample ground truth from the schema with a fixed seed (stratified so every value
> appears), cross with 3–4 answer styles (direct, vague, rambling, indifferent), have an LLM write a
> 2–3 sentence backstory once, then freeze everything to `data/<domain>/personas.jsonl`.
> Split 12 dev personas (prompt tuning only) and 48 eval personas (reported results only).
>
> *Backstories (M2):* kept in a separate `data/<domain>/backstories.jsonl` (persona id → text) and
> merged at load time, so `make-data --check` stays deterministic without an LLM. They are written
> once in M5 with the simulator model and then frozen; until then the simulator works without them.
>
> *Backstories in detail (M5):* `elicitor make-backstories <experiment file>` asks the `user`
> model group (role `backstory`, temperature 0.8) for every persona, dev and eval. The prompt gets
> only the answer style, never the ground truth, and asks for 2–3 sentences about who the person
> is (work, household, an everyday hobby) with nothing about travel, places, weather, food,
> lodging, money or time off. The backstory therefore cannot contradict the profile, and it adds
> no preference the simulator could leak. A backstory in which the leakage keywords (§9) find any
> slot value is drawn again with the next seed, up to 5 draws; the command reports any that stay
> flagged. The file is then committed and never regenerated: the command refuses to overwrite it
> without `--force`.
>
> *Written 10-22 (M5 smoke job, llama3.3):* 60 backstories, 61 calls, one redrawn, none still
> flagged; SHA-256 `c7ba7629…`. They are bland but on topic (jobs, household, a hobby; gardening
> and accounting recur). One defect is kept as frozen: p006's last sentence is cut off and ends
> in a stray non-ASCII character. p006 is a dev persona, so no reported number depends on it.

## 5. Belief model

For each slot $s$, a categorical distribution $b_s(v)$ over its values, starting from the schema
prior. Uncertainty of a slot: $H(b_s) = -\sum_v b_s(v)\log b_s(v)$.

> **Decision Q5 — representation.**
> A: factorized categorical per slot. B: joint distribution over slot pairs (captures correlations).
> C: Dirichlet pseudo-counts per slot.
> *Chosen A* for the main system. Note that with A and single-slot questions, EIG reduces almost
> to max-entropy (see §7); B or C are possible extensions.

## 6. Belief updaters

Interface: `update(belief, question, question_text, answer_text) -> belief` plus a log record of
what was parsed. The question text is passed because with LLM phrasing (§8) the wording varies.

> **Decision Q6 — updater family.**
> A: rule/keyword matcher (no LLM). B: LLM returns a likelihood score for each value of the asked
> slot(s); posterior $\propto$ prior $\times$ likelihood$^\beta$ (β tempers over-confidence).
> C: LLM writes the posterior directly.
> *Chosen B as the main updater and A as the no-LLM baseline;* C only if time allows. B keeps
> Bayes rule in code (testable, explainable) and the LLM only does language understanding.
> Answers may mention slots that were not asked; B should score those too (settled in M2 on dev personas).
>
> *B in detail (M2):* the LLM sees the slot description, its values with labels, the question as
> asked and the answer. It returns a short evidence quote and an integer score $s_v \in [0, 10]$
> per value: how well the answer fits a user whose true preference is $v$ (equal scores when the
> answer says nothing about the slot). Likelihood
> $\ell_v = \left(\frac{s_v + \delta}{10 + \delta}\right)^\beta$ with smoothing $\delta = 0.5$, so a
> score of 0 never rules a value out, and $\beta = 1$ by default. Temperature 0. An empty answer
> skips the call. *Unasked slots:* scored only if the leakage rate (§9) measured with the LLM
> simulator on the dev personas exceeds 10 % of answers; otherwise v1 scores the asked slot only.
> *Settled in M2 (dev smoke 3, §9): 3.9 % hand-checked, 8.9 % by keyword, so the asked slot only.*
>
> *Choice and yes/no questions (M3).* The rule updater first reads the answer as one or more
> outcomes of the question's answer model (§7): for a choice question, the offered values it
> names, else "other" when it names another value, a no-preference phrase or a "none of them"
> phrase; for a yes/no question, the first yes or no phrase, else "yes" when it names a value in
> the question's set and "no" when it names one outside it. The likelihood then has the same form
> as for open questions with the outcomes in place of the values: a value whose noise-free outcome
> was named keeps weight 1, every other value gets
> $\varepsilon = (1-\rho)/(m\rho + 1 - \rho)$ with $m$ the number of outcomes, and a negated
> outcome multiplies by $\varepsilon$. For open questions this is exactly the M1 updater. ρ can
> differ by question type. The LLM updater is unchanged: it already sees the question as asked,
> and its prompt for open questions stays byte-identical to M2.

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
>
> *A in detail (M3).* Each question $q$ on slot $s$ has a set of outcomes $\Omega_q$ and a
> noise-free answer $f_q(v) \in \Omega_q$ for every value $v$:
>
> | type | outcomes $\Omega_q$ | $f_q(v)$ |
> |---|---|---|
> | open | the slot's values | $v$ |
> | choice, offering a set $O$ of 2 or 3 concrete values | $O \cup \{\text{other}\}$ | $v$ if $v \in O$, else other |
> | yes/no on a set $S$ | {yes, no} | yes if $v \in S$, else no |
>
> Answer model: $P(\omega \mid v) = \rho_q [\omega = f_q(v)] + (1-\rho_q)/|\Omega_q|$, i.e. the
> noise-free outcome with probability $\rho_q$, else a uniformly random outcome. ρ is set per
> question type (`open`, `choice`, `yesno`). For open questions this is the M1 model. Then
> $\mathrm{EIG}(q) = H(A) - H(A \mid V)$ with $P(A = \omega) = \sum_v b_s(v) P(\omega \mid v)$,
> which gives EIG = $H(b_s)$ for a noise-free open question and EIG ≤ 1 bit for a yes/no one.
>
> - *Candidates per slot:* the open question; a choice between every pair and every triple of
>   concrete values (never `no_preference`); yes/no on each single concrete value of a categorical
>   slot, and on each "below edge $i$" threshold (the first $i$ bins) of an ordinal slot. 105
>   questions for the travel schema.
> - *Selector `eig`:* the candidate with the largest EIG over all slots; ties (within $10^{-12}$)
>   go to schema order, then open, choice, yes/no, then generation order. It never asks the same
>   question twice, but it may ask another question on a slot already asked (a yes/no check after
>   a noisy open answer), which only pays off because ρ < 1. `eig` takes `kinds` (default all
>   three) and `rho` from the experiment file's `[selector.eig]` table.
> - *Selector `eig_open`:* `eig` restricted to open questions, a diagnostic for H2: against
>   `max_entropy` it isolates what the noise model changes, against `eig` what the question types
>   add. Whether it enters the main grid is decided with the hypotheses (Q13).
> - *Why the arms differ.* With one ρ for all types, open questions nearly always win and `eig`
>   is close to `max_entropy`. Closed questions are easier to answer reliably, so the M3 suites use
>   $\rho$ = 0.7 / 0.8 / 0.9 for open / choice / yes/no, for both the oracle and the agent's model.
>   For the LLM simulator, ρ per type is estimated on the dev personas in M5 (how often one answer
>   alone puts the MAP on the true outcome) and frozen before the full run.
> - *`eig_mc` (M5):* scoring all 105 candidates would cost about 600 LLM calls per turn, so it
>   scores a shortlist of the top candidates by analytic EIG; the shortlist size is set in M5.
>
> *ρ per type in detail (M5).* `elicitor calibrate <experiment file>` asks the dev personas
> single questions from the prior belief with template wording: for every persona and slot the
> open question, one choice between two values and one between three (two pairs for a slot with
> only two concrete values), and two yes/no questions, drawn with a seeded generator (12 × 9 × 5
> = 540 answers). Each answer is read by both updaters. A reading is correct when the true
> outcome $f_q(t)$ has the largest likelihood among the outcomes (the mean likelihood of the
> values behind an outcome; a tie between $j$ outcomes counts $1/j$ when the true one is among
> them). Under the answer model the correct rate is $a = \rho + (1-\rho)/m$, so
> $\hat\rho = (\bar a - \overline{1/m}) / (1 - \overline{1/m})$ per type and updater, clipped to
> [0, 1], with a 95 % bootstrap CI over personas. Frozen use: the selectors (`eig`, `eig_mc` and
> the other EIG arms) take $\hat\rho$ measured with the LLM updater, the main updater; the rule
> updater takes its own $\hat\rho$ (its noise model is a statement about how well it reads).
> The calibration answers are written out, so leakage can be read on them too.
>
> *Measured (M5 smoke job, 10-22, llama3.3 simulator, 540 answers):*
>
> | type | ρ̂ LLM updater [95 % CI] | ρ̂ rule updater [95 % CI] |
> |---|---|---|
> | open | 0.988 [0.970, 1.000] | 0.792 [0.727, 0.859] |
> | choice | 0.974 [0.954, 0.990] | 0.827 [0.748, 0.899] |
> | yes/no | 0.847 [0.755, 0.935] | 0.643 [0.481, 0.778] |
>
> **Frozen** (rounded to two decimals): selectors 0.99 / 0.97 / 0.85, rule updater
> 0.79 / 0.83 / 0.64. The M3 assumption (closed questions more reliable) does not hold for this
> simulator: yes/no answers are the least reliable (a "would X suit you?" invites a hedged yes).
>
> *`eig_mc` in detail (M5).* Shortlist: the L = 5 candidates with the largest analytic EIG
> (frozen ρ). For each shortlisted question $q$ on slot $s$ and each value $v$ with $b_s(v) > 0$,
> the agent imagines S = 3 answers of a user whose preference on $s$ is $v$ (role `imagine`,
> agent model group, template wording, temperature 0.8) and reads each with the LLM updater,
> giving likelihoods $\ell_{v,j}$. Then
> $\widehat{\mathrm{EIG}}(q) = \sum_v b_s(v)\,\frac{1}{S}\sum_j
> \bigl[H(b_s) - H(b_s \cdot \ell_{v,j} / Z)\bigr]$, and the arm asks the shortlisted question
> with the largest estimate (ties to shortlist order). The imagined answers are seeded by
> (replicate seed, question, value, j) only, and neither prompt sees the belief, so both calls
> are cached across turns and personas: a replicate costs at most
> 105 × 5 × 3 × 2 ≈ 3,000 calls, in practice far fewer because the shortlists repeat. This is S
> answers per (question, value) instead of S per question, which removes the sampling noise of
> drawing $v$ and costs nothing extra after the cache. Run at K = 5 only (`arm_budgets`, §13).
> - *Selector `entropy_choice` (M5):* a heuristic baseline for H2. It takes the slot of
>   `max_entropy` and asks, of that slot's choice questions, the one with the largest analytic
>   EIG (for a slot with three concrete values, all three). If `eig` ≈ `entropy_choice`, the gain
>   of `eig` over `max_entropy` comes from listing the options, not from the EIG computation.

> **Decision Q8 — when to stop asking.**
> *Chosen:* budget K (sweep K ∈ {3, 5, 8}, fewer than the number of slots so that ordering
> matters) plus early stop when every slot has $\max_v b_s(v) \ge \tau$. Report per-turn curves,
> not only the final turn.

## 8. Question phrasing

> **Decision Q9.** A: templates from the schema (deterministic). B: LLM rewrites the template into
> a natural question. *Chosen A by default, B as one ablation* to measure whether nicer phrasing
> changes how informative the simulated answers are.
>
> *B in detail (M2):* the LLM gets the slot description and the template question (not the value
> labels, so it cannot list them), and returns one question that asks about that slot only and
> suggests no answer.
> Temperature 0.7, seeded per (replicate, persona, slot, ask count), so arms that ask the same
> question see the same wording. A reply that fails validation falls back to the template.
>
> *Templates for choice and yes/no (M3).* Choice: "Would you prefer A or B?" / "A, B, or C?",
> with semicolons between options when a label contains a comma. Yes/no on a value: "Would
> <label> suit you?". Yes/no on an ordinal threshold: the slot's `threshold_question` with the bin
> edge filled in (§3). With LLM phrasing, the prompt's rule "do not list or suggest possible
> answers" applies to open questions only; a choice question must keep exactly its options, a
> yes/no question must stay a yes/no question about its option. The open-question prompt stays
> byte-identical to M2.

## 9. User simulator

Answers a question in character, using only the persona; must not volunteer the whole profile.

> **Decision Q11.** A: LLM role-play prompted with ground truth + answer style. B: rule oracle that
> answers from ground truth with controlled noise ρ. *Chosen both:* A for experiments, B for unit
> tests and for a noise sweep where ρ is known exactly.
> Leakage check: count how often an answer mentions slots that were not asked.
>
> *A in detail (M2):* the prompt holds the full profile (every slot's true value as its label),
> the backstory if any, an answer-style guide and the question as asked; the instruction is to
> answer only that question, in character. The full profile is given on purpose, so leakage can
> happen and is measured rather than prevented. Temperature 0.8, seeded from the answer generator
> of §11 (common random numbers). A reply that fails validation becomes an empty answer, which the
> updaters treat as no information. Leakage is measured with the rule matcher's keywords on the
> unasked slots, so it is approximate (a generic word such as "walking" can be a false hit).
>
> *Dev smoke 1 (10-22, qwen3:32b as simulator, 12 dev personas, K = 5, 180 answers):* 62 answers
> (34 %) leaked, mostly `setting` (33) and `climate` (12). Reading them: most are real
> volunteering ("a place with hot, tropical weather for outdoor adventure" when asked about the
> setting); some are anecdotes in the rambling style that name a place without stating a
> preference ("a rooftop bar ... sunset over the mountains"). *Decision (user, 10-22):* tune the
> simulator prompt on the dev personas first (the prompt also names the topic asked, forbids
> other list items even in stories or side remarks, and the rambling style draws its stories from
> outside the trip preferences), rerun the smoke, and apply the 10 % rule of §6 to the new rate.
>
> *Dev smoke 2 (10-22, tuned prompt, same setup):* 0 errors again; keyword leakage 26 / 180
> answers (14.4 %). Reading the 19 distinct flagged answers: about 7 are real volunteering
> ("I'd like to camp somewhere by the beach" when asked about lodging), 4 answer the wrong topic,
> and about 6 are keyword false hits ("busy" in a remark about work; "rental car" matching the
> lodging keyword "rental"). The wrong-topic answers come from the schema text: `setting` was
> described as "the type of place the traveller wants to stay in" and `lodging` asked "where would
> you like to stay at night?", which overlap. *Decision (user, 10-22):* fix both texts, run smoke
> 3, and judge the 10 % rule on the hand-checked rate: flagged answers read and classified as real
> leaks (volunteering or wrong topic) or false hits, counted per answer like the keyword rate,
> which is reported beside it.
>
> *Dev smoke 3 (10-22, fixed slot texts, same setup):* 0 errors, 0 parse failures, 0 failed
> calls (360 calls, 625 s). Keyword leakage 16 / 180 answers (8.9 %). Of these, 7 are real leaks
> when borderline cases count as real ("as long as the weather is cold and it's winter" when asked
> about the setting) and 9 are false hits ("the hike I did last weekend", "walking her more");
> none answers the wrong topic. Hand-checked rate 7 / 180 = 3.9 %. **Settled:** below 10 %, so
> the v1 LLM updater scores the asked slot only (§6). Leakage is measured again in M5 with the
> real simulator model and reported with the results.
>
> *Choice and yes/no questions (M3).* The oracle draws an outcome from the answer model of §7
> with its own ρ per type and says it plainly ("I'd pick the mountains.", "None of those,
> really.", "Yes, that would suit me.", "No, not really."). The mock simulator does the same with
> ρ = 0.8 and its style wording. The LLM simulator's prompt is unchanged: the question text
> carries the type.
>
> *M5, llama3.3 as simulator (calibration answers, 12 dev personas, 540 answers):* keyword
> leakage 44 / 540 (8.2 %). Read by hand: 10 are real or borderline volunteering (camping
> lovers who add "being outdoors", a "relax and unwind" when asked about the setting), 1.9 %;
> the other 34 are keyword false hits, mostly "rental car" matching the lodging keyword "rental"
> and "busy" / "relax" in asides. Well under 10 %, so the M2 decision (score the asked slot only)
> stands. The smoke run's keyword leakage per arm is 0–14 % of answers (§17, M5).

## 10. Recommendation and evaluation target

> **Decision Q10.**
> A: LLM writes a free-text plan; an LLM judge scores it against the persona.
> B: a fixed synthetic catalog (e.g. 200 items with attribute values on every slot); the agent
> ranks items by expected utility under its belief; score = regret against the persona's best item,
> hit@k, constraint violations.
> *Chosen B as the primary metric* (objective, cheap, no judge bias) and A as a secondary,
> human-readable output with the judge run on a sample.
> Judge failures are counted as failures, never replaced by a default score.
>
> *B in detail (M3).* True utility of item $i$ for a user with true values $t$: the mean over
> slots of a match score, $u(i, t) = \frac{1}{|S|}\sum_s m_s(i_s, t_s) \in [0, 1]$. Categorical:
> $m_s = 1$ when $i_s = t_s$ or $t_s$ = `no_preference`, else 0. Ordinal with $n$ bins:
> $m_s = 1 - |i_s - t_s|/(n-1)$ (partial credit by bin distance). The recommender ranks the catalog
> by expected utility under the factorized belief,
> $\mathrm{EU}(i) = \frac{1}{|S|}\sum_s \sum_v b_s(v)\, m_s(i_s, v)$; ties keep catalog order.
> A belief that is a point mass on the truth therefore ranks a best item first. Hard constraints
> are not used for ranking; how often the top item breaks one is reported (§12).

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

Every component is chosen by config, and every random draw comes from a generator seeded by a hash
of its context. The selector's generator is derived from (replicate seed, persona id, arm). Each
simulated answer gets its own generator derived from (replicate seed, persona id, slot, how many
times that slot has been asked). Arms that ask the same question therefore get the same noisy
answer (common random numbers), which tightens the paired comparisons in §13.

## 12. Metrics

- Belief, per turn: mean posterior mass on the true value; MAP accuracy over slots; log loss;
  mean entropy.
- Efficiency: turns until confident; area under the accuracy-vs-turn curve. *M4:* the curve runs
  over turns 0 … K (turn 0 = the prior); an episode that stopped early keeps its last value up to
  K, since the belief no longer changes. The area is the trapezoid rule over the K + 1 points
  divided by K, so it lies in [0, 1] like MAP accuracy itself.
- Outcome: regret, hit@k, constraint violations (Q10). Computed after every turn, like the
  belief metrics. Regret $= \max_i u(i, t) - u(\text{top item}, t)$. hit@k (k = 1, 5, 10) = 1 when
  the top k items contain at least one item of maximal true utility. Violations = number of
  constraint slots (§3) the top item breaks: `at_most` when its bin is above the user's, `exact`
  when its bin differs.
- Operations, per role: calls, cache hits, parse failures, errors, latency, tokens.

## 13. Experiments

> **Decision Q12 — grid.** *Chosen:* selector {fixed, random, max_entropy, eig} × updater
> {rule, llm} = 8 arms, each × K values; phrasing (Q9) and simulator noise (Q11) as separate small
> ablations on the best arm. Arms are named by their components, e.g. `sel=eig,upd=llm,phr=tpl`.
> Plus the `sel=eig_mc` arm from Q7. 5 replicates (3 if the time limit in §1 requires it).
> In the experiment file the grid is `selectors × updaters × phrasers`, and single extra arms
> (the ablations) are listed by name in `extra_arms`. *M5:* `arm_budgets` maps an arm name to the
> budgets it runs at (default: all of `budgets`), so `eig_mc` runs at K = 5 only; it is part of
> the run fingerprint. Comparisons skip budgets an arm does not have.
> Statistics: average replicates per persona, then paired Wilcoxon signed-rank over
> eval personas with bootstrap CIs and Holm correction across comparisons.
>
> *Statistics in detail (M4).* The unit is the persona: each metric is first averaged over the
> replicates of a persona, so a test over 48 eval personas has n = 48 whatever the replicate
> count. For a comparison "arm A vs arm B" at budget K the per-persona differences A − B are
> rounded to 10 decimals (so float noise does not turn an exact tie into a tiny nonzero
> difference) and tested with the two-sided Wilcoxon signed-rank test (zero differences dropped,
> p = 1 when every difference is zero). The table reports the mean difference, a 95 % percentile
> bootstrap CI over personas (10 000 resamples, fixed seed), wins / ties / losses, p and the
> Holm-adjusted p. A Holm family is one metric: all comparisons and budgets of that metric in
> the run. Lower is better for log loss, entropy, regret and violations; the table says which.
> - *Comparisons.* An experiment file may list them in an `[analysis]` table
>   (`comparisons = [[A, B], …]`, plus `metrics`, `curve_metrics`, `bootstrap`). Without one, the
>   default is every pair of arms that differ in exactly one component (selector, updater or
>   phraser), with A the arm listed later. The frozen hypotheses (Q13) become the `[analysis]`
>   table of the full-run file.
> - *Outputs* of `elicitor analyze <run dir>`: `arms.csv` (per arm and budget: persona means
>   with bootstrap CIs, turns, confident share, leak rate, accuracy AUC), `comparisons.csv`,
>   `curves.csv` (per-turn means and standard errors over personas), `results.md` (the tables in
>   Markdown with the run's provenance) and per-turn curve figures (§16). By default they go to
>   `<run dir>/analysis/`.
> - *Committed outputs.* `--publish` writes to `results_summary/<name>/` and
>   `docs/figures/<name>/`. Any output path inside a `results_summary` or `docs/figures` folder
>   is only written from a run that is complete (every cell finished), used no mock provider,
>   covers the eval split and was not cut with `--max-personas`; otherwise analysis refuses and
>   writes nothing. The oracle suites qualify (their noise is known exactly, Q11); mock runs never
>   do.

> **Decision Q13 — hypotheses** (draft until M5; frozen below):
> H1 `eig` > `fixed` and `random` on posterior mass on truth at budget K.
> H2 `eig` > `max_entropy` only in settings where they can differ (§7).
> H3 the LLM updater beats the rule updater, and the gap grows with vaguer answer styles.
> H4 better final belief → lower regret (episode-level correlation).
>
> *Open before freezing (from M3):* under the assumed ρ 0.7 / 0.8 / 0.9 `eig` asks only choice
> questions, most listing every option of a slot, so its gain over `max_entropy` comes from the
> assumed reliability gap. Whether `eig_open` or an all-options-choice arm joins the main grid is
> decided here, with the ρ per type measured in M5, before the full run.
>
> **Frozen (M5, 10-22, before the full run).** The measured ρ (§7) reverses the M3 assumption:
> read by the LLM updater, open answers are the most reliable (0.99), choice 0.97, yes/no 0.85.
> With these values `eig` starts with open questions in `max_entropy`'s slot order. The main grid
> is `fixed`, `random`, `max_entropy`, `entropy_choice`, `eig` × {rule, llm} with template
> wording, plus `eig` + llm with LLM wording and `eig_mc` + llm at K = 5 (12 arms, 5 replicates,
> 48 eval personas). `eig_open` stays out: with one open ρ it asked exactly `max_entropy`'s
> questions in all M3 episodes. `entropy_choice` joins: `eig` no longer lists options, so it is
> the only arm that tests whether listing them helps. Primary metric: posterior mass on the true
> value; the others in the `[analysis]` table are secondary. The hypotheses, as directed claims:
> - H1: `eig` > `fixed` and > `random` on mass on truth at K = 5 and 8, with either updater.
> - H2: `eig` = `max_entropy` within noise (no Holm-significant difference on mass on truth),
>   because under the measured ρ they ask nearly the same questions; `entropy_choice` is not
>   better than `max_entropy`; `eig_mc` is not better than `eig` at K = 5.
> - H3: the LLM updater > the rule updater on mass on truth for every selector. Its second part
>   (the gap grows from direct to vague answer styles) is read descriptively from the episodes in
>   M6, by style, without a test.
> - H4: across episodes, higher final mass on truth goes with lower regret (Spearman correlation
>   per arm, descriptive, M6).
> - Phrasing (Q9): LLM wording changes mass on truth by less than 0.02 (no Holm-significant
>   difference).
> The 15 comparisons of H1–H3 and the phrasing ablation are the `[analysis]` table of
> `experiments/m5_full.toml`; Holm runs per metric over all of them and every budget.

> **Decision Q14 — models per role.** *Chosen:* agent roles (updater, phrasing) on qwen3:32b;
> simulator and judge on llama3.3:70b, a different family, to limit self-preference. Both run on
> cluster GPUs (A40, 48 GB). Development uses the mock backend and a 4B-class model locally
> (6 GB VRAM). Thinking/reasoning output off for roles that must return JSON.
> *M5:* the cluster's shared model folder holds llama3.3 as `llama3.3:latest` (the 70B model,
> Q4_K_M, 42.5 GB) and a Q8 tag too large for one A40, so the experiment files name `llama3.3:latest`.

## 14. LLM layer

One client, providers `ollama` (native API), `openai_compatible` (any local server), `mock`
(deterministic, schema-valid replies for tests).

- Role registry: role name → prompt template, output schema, temperature, model.
- SQLite cache keyed by hash(provider, model, role, prompt, params, seed); a re-run is free.
- Structured output with retry on parse failure; every failure is logged per role.
- Call statistics written per run (see §12, operations).

**M2 details.**
- Roles: `update` (§6 B), `phrase` (§8 B), `simulate` (§9 A). Each role belongs to a model group:
  `agent` (update, phrase) or `user` (simulate, later the judge); the experiment file maps groups
  to models in its `[llm]` table (Q14).
- Prompts are domain data: `data/<domain>/prompts/<role>.txt` with a system and a user part and
  `$name` placeholders, plus `styles.toml` with the answer-style guides.
- Cache key: SHA-256 of (provider, model, role, messages, JSON schema, temperature, max tokens,
  seed). Only replies that validated are stored.
- A call is tried up to 3 times; attempt $i$ uses seed + $i$, so a retry at temperature 0 is not a
  copy of the failed one. Counted per role: calls, cache hits, attempts, parse failures (attempts
  whose reply did not validate), failed calls (every attempt failed), transport errors, latency,
  prompt and completion tokens. Written to `llm_stats.json` next to the episodes.
- A failed call degrades as described in each role's section. A transport error (server down,
  HTTP error) after the retries stops the run, since no answer can be trusted after it.
- The mock provider calls a per-role Python function on the role's structured input (not on the
  prompt) and returns its output as JSON text, so the parse path is exercised: `update` scores
  values with the rule matcher's keywords, `simulate` answers like the oracle (ρ = 0.8) in a
  style-dependent wording, `phrase` returns the template with a short lead-in.
- Ollama's native API is called with `think: false` and the role's JSON schema as `format`;
  `num_ctx` is pinned because changing it reloads the model. HTTP clients ignore proxy settings
  from the environment (local servers only).

**M5 details.**
- New roles: `imagine` (agent group; §7 `eig_mc`: answer as a user whose preference on one slot
  is a given value) and `backstory` (user group; §4). `think` may be a table by model group
  (`think = { agent = false }`): a group not listed gets no `think` field, since llama3.3 has no
  thinking mode.
- *Several servers.* `base_url` may be a table of lists by group
  (`base_url = { agent = [...], user = [...] }`); the environment variables
  `ELICITOR_SERVERS_AGENT` and `ELICITOR_SERVERS_USER` (comma-separated) override it per group,
  as `ELICITOR_BASE_URL` overrides a single URL. Each request goes to the server of its role's
  group with the fewest requests in flight (ties to list order). The cache key does not contain
  the server, so a reply is the same whichever server produced it.
- *Concurrency.* The client is thread-safe. Requests with the same cache key are single-flight:
  while one is in flight, an identical one waits for its reply and counts as a cache hit. So
  when two arms ask the same question at the same time they still get the same answer (common
  random numbers, §11), as they would one after the other. Statistics are counted per task (see
  §15) and summed per cell.

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
- M2 adds the minimal pieces for one-GPU jobs: `cluster/submit.sh` (reads `local.env`, passes
  account and partition to `sbatch`), `cluster/serve_ollama.sh` (starts Ollama on a free port and
  waits until it answers) and `cluster/smoke.sbatch`. The environment variable
  `ELICITOR_BASE_URL`, when set, overrides the `base_url` of the experiment's `[llm]` table, so a
  job can point the client at the port it picked. M5 extends this to several servers.

**M4: resume and run health.**
- *Cells.* A cell is one (arm, replicate): every budget and every persona of that arm and
  replicate. `elicitor run` writes each finished cell to `cells/<arm>__r<replicate>.jsonl` (written
  to a temporary file and renamed, so a cell file is either complete or absent) together with
  `cells/<…>.stats.json` (its LLM call statistics and wall time). A run resumes by default:
  finished cells are skipped, and a cell that a killed job left half done is run again, cheaply,
  because its LLM replies are in the cache. `--fresh` discards the cells (not the cache).
- *Fingerprint.* `run_state.json` records a SHA-256 over everything that changes a cell's
  content: package version, domain, split, seed, budgets, τ, simulator, selector, updater and LLM
  settings (without `base_url`, `timeout` and `cache`, which do not change replies),
  `--max-personas` and the bytes of the domain's data and prompt files. A run into a folder with
  a different fingerprint stops with an error. The arm list and the replicate count are not
  part of it, so arms or replicates can be added to a finished run; cells of arms no longer in
  the file are kept but ignored.
- *Merged outputs.* When every cell is finished, `episodes.jsonl` (arm, replicate, budget,
  persona order), `summary.csv` and `llm_stats.json` (summed over cells, so a resumed run counts
  every job) are rebuilt from the cells. Episodes and summary are byte-identical to an
  uninterrupted run; the LLM statistics differ only in latency and in calls that a killed job
  already made (they show up as cache hits).
- *Live state.* After every episode the runner rewrites `live.json`: the cell in progress, its
  episode count and its LLM statistics so far (best effort; a failed write is skipped).
- *`elicitor health <run dir>`* reads only these files, so it works while a job runs: cells done
  / total, episodes, mean wall time per cell run in this folder and a naive ETA, the age of
  `live.json`, and per role calls, cache hits, parse-failure rate (per attempt), failed calls and
  transport errors. It exits 1 when a role has a failed call, a transport error, or a parse
  failure rate of 1 % or more (the M5 bar), else 0.

**M5: several GPUs.**
- *Parallel runner.* `elicitor run --workers N` runs tasks on N threads. A task is one
  (cell, persona): that persona's episodes at every budget of the arm, smallest budget first, so
  a longer episode finds the shorter one's turns in the cache. Tasks are queued cell by cell, in
  grid order; a cell is written as soon as its last task ends, with its episodes in the same
  order as before (budget, then persona). With the mock or the oracle the cells are byte-identical
  for any N. `--workers` is not part of the fingerprint. `live.json` lists every cell in
  progress.
- *Job scripts.* `cluster/serve_models.sh` starts one Ollama server per GPU (pinned with
  `CUDA_VISIBLE_DEVICES`, its own port, `OLLAMA_NUM_PARALLEL` per group, keep-alive forever,
  flash attention on, context length = the experiment's `num_ctx`), loads the group's model on
  each, logs `ollama ps` (the model must sit 100 % on the GPU) and exports the server lists.
  The models and `num_ctx` are read from the experiment file. `cluster/m5_smoke.sbatch` (4 GPUs,
  1 h): backstories, calibration, then the smoke run and `health`. `cluster/full.sbatch`
  (4 GPUs, 12 h): the full run, `health` and `analyze`. `cluster/smoke.sbatch` (one GPU, M2)
  uses the same helper with both groups on one server; the helper replaces `serve_ollama.sh`,
  and `cluster/job_setup.sh` holds the common setup (site settings, modules, venv).
- *CI* also runs the mock suite `m5_mock` (every arm of the full run, `--workers 4`) and
  `calibrate` on the oracle suite `m3_eig`, where the estimate must come out near the oracle's ρ.
- *Smoke run* (`experiments/m5_smoke.toml`): the full-run arms on the 12 dev personas, one
  replicate, with the M3 ρ table. Its wall time × 4 (personas) × replicates projects the full
  run; if the projection exceeds 11 h (1 h kept for loading and the tail), the grid is cut in the
  §1 order before submitting.

## 16. Repo layout and tooling

```text
src/<pkg>/
  schema.py  personas.py  belief.py  questions.py  answer_model.py  catalog.py  episode.py
  metrics.py
  updaters/   (rule.py, llm.py)
  selectors/  (fixed.py, random.py, entropy.py, eig.py)
  phrasing.py  simulator/ (oracle.py, llm.py)  recommender.py
  llm/        (client.py, providers.py, cache.py, roles.py, mock.py)
  experiments/ (config.py, runner.py, store.py, health.py)
  analysis/    (stats.py, report.py, figures.py)
  cli.py
data/<domain>/   schema.toml  personas.jsonl  catalog.jsonl  prompts/
experiments/     *.toml
cluster/         slurm scripts, local.env.example
tests/           unit tests (belief math, EIG properties, updaters) + mock end-to-end
docs/            SPEC.md  figures/
results_summary/ committed tables
```

> **Decision Q16 — config and CLI.** *Chosen* TOML experiment files read into dataclasses, one
> `argparse` CLI with subcommands `run`, `analyze`, `health`, `demo`, plus `make-data`, which
> regenerates the frozen persona and catalog files from the schema and a seed (`--check` only
> compares). No Hydra. M5 adds `make-backstories` (§4) and `calibrate` (§7).

> **Decision Q17 — packaging and tooling.** *Chosen* pyproject + setuptools, plain venv/pip,
> ruff (lint + format), pytest, GitHub Actions running tests and the mock suite on every push.
> Few dependencies: numpy, scipy, pydantic, httpx, matplotlib, pandas. Each is added in the
> milestone that first needs it (pydantic and httpx in M2; numpy, scipy and matplotlib in M4).
> pandas turned out unnecessary in M4 (the tables are small) and is not added.
> Figures: matplotlib with the non-interactive Agg backend, PNG at 150 dpi; one figure per
> curve metric and budget, one line per arm with a ±1.96 standard-error band over personas.

> **Decision Q18 — license.** *Chosen* MIT.

## 17. Milestones

| M | Content | Done when |
|---|---|---|
| M0 | repo, pyproject, CI, empty package | CI green |
| M1 | schema, personas, belief, rule updater, fixed/random/entropy selectors, oracle simulator, catalog, episode loop; also template phrasing, belief metrics and a minimal `run` (no resume) so the suite can run | full suite runs with no LLM; belief and entropy unit tests pass |
| M2 | LLM layer (client, cache, mock, Ollama), LLM updater, phrasing, LLM simulator | mock suite in CI; smoke run on a real model (dev personas) with 0 transport errors and 0 failed calls; parse failures and leakage reported. Changed 10-22: Ollama is not installed on the development laptop, so the M2 smoke runs as a one-GPU cluster job with qwen3:32b for both model groups (the agent model of Q14, already on the cluster; llama3.3:70b is first tested in M5). `experiments/m2_smoke_ollama.toml` (qwen3:4b) stays as the local option. **Done 10-22:** three smoke jobs, all with 0 errors, 0 parse failures and 0 failed calls; leakage settled after two prompt and text fixes (§9) |
| M3 | EIG selector + question types (templates, oracle, rule and LLM updaters, mock), recommender, outcome metrics | EIG unit tests (EIG = H for noise-free open questions; EIG ≤ 1 bit for yes/no); the m1 and m2 suites still run; an `eig` arm runs in CI (oracle suite `m3_eig` and mock suite `m3_mock`). **Done 10-22.** Oracle suite, ρ 0.7 / 0.8 / 0.9: `eig` raises mass on truth over `max_entropy` from 0.446 to 0.503 at K = 5 and from 0.570 to 0.644 at K = 8 (regret 0.182 → 0.161 and 0.126 → 0.098; at K = 3 regret is equal). It asks only choice questions: 69 % offer every concrete value of a 3-value slot, i.e. the open question with its options listed, and 31 % offer 3 of the 4 values of a 4-value slot. So in this suite its gain comes from the reliability gap assumed between open and closed questions, which makes the M5 estimate of ρ per type decisive for H2. `eig_open` asks exactly the questions of `max_entropy` in all 720 episodes (one ρ for every slot). The mock suite uses the same ρ table in the agent's model so that closed questions go through the LLM paths |
| M4 | runner (resume by default), run-health, analysis, figures; CLI `analyze` and `health` | analysis runs on mock output and refuses to write committed tables from it; an interrupted run resumed gives the same bytes as an uninterrupted one; the m1/m2/m3 suites still run in CI. **Done 10-22.** CI also resumes `m3_mock` (every cell skipped), runs `health` and `analyze` on it and checks that `--publish` refuses it. First paired reading of the oracle suite `m3_eig` (48 personas, 5 replicates, Holm over 30 tests per metric): `eig` − `max_entropy` mass on truth +0.031 / +0.057 / +0.073 at K = 3 / 5 / 8 (all p < 0.001 after Holm), log loss lower at K = 5 and 8, violations lower at K = 5 and 8; MAP accuracy and regret do not differ after Holm (regret −0.028 at K = 8, Holm p = 0.07) |
| M5 | cluster jobs (≤ 4 A40), first test of llama3.3:70b as simulator, frozen backstories, smoke run to measure the call rate and re-measure leakage, ρ per type measured on the dev personas and frozen, Q13 frozen, full run | all cells finished within the §1 time limit; error rate 0; parse failures < 1 % per role (`elicitor health` passes); `elicitor analyze --publish` accepts the run **Smoke job 10-22** (4 A40, `m5_smoke`): llama3.3:70b fits one A40 fully (42.6 GiB with 4 slots at 2048 context, loaded in 96 s); backstories 176 s, calibration 982 s, smoke run 1,067 s for 12 arms × 12 dev personas × 1 replicate (408 episodes); 0 parse failures, 0 failed calls, 0 transport errors in every role (simulate 325 requests, update 444, phrase 96, imagine 141; the rest cache hits), `health` OK. Projection for the full run: 20 × the smoke (4 × the personas, 5 replicates) ≈ 5.9 h plus loading, under the 11 h bar, so nothing is cut: 5 replicates, K = 3 / 5 / 8. |
| M6 | README results, demo, release | fresh-clone test passes on a clean venv |
| M7 | optional, at most one: second domain, joint belief, or judge study | — |

## 18. Results and release

> **Decision Q19.** *Chosen* committing summary tables and figures; raw per-episode logs and the
> call cache as a GitHub release asset (or not at all).

> **Decision Q20 — demo.** *Chosen* a `demo` subcommand: terminal chat where a human answers,
> showing the belief bars after each turn. Useful in interviews.
