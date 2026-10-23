# Elicitor

An LLM agent that learns what a user wants by asking a few well-chosen questions. It keeps a
probability distribution over the user's preferences, picks each question by expected information
gain, updates the distribution from the free-text answer, and then recommends. Everything runs on
open models served locally or on a GPU cluster; no paid API is needed.

Inspired by an EECS 545 team project.

## How it works

The domain is trip planning: 9 slots (climate, setting, pace, lodging, food, activity, transport,
plus budget and trip length as binned constraints), a 200-item synthetic catalog and 60 synthetic
users (12 for development, 48 for evaluation) with known ground truth and one of four answer
styles: direct, vague, rambling or indifferent.

One episode is a dialogue of at most K questions:

1. **Belief.** One categorical distribution per slot, starting from the prior.
2. **Question selection.** `fixed` (schema order), `random`, `max_entropy` (the slot with the
   most uncertainty), `entropy_choice` (the same slot, asked as a choice between its likeliest
   values), `eig` (the open, choice or yes/no question with the largest expected information gain
   under an answer-noise model, computed exactly) and `eig_mc` (expected gain estimated by
   sampling simulated answers from the LLM and running the updater on them).
3. **Simulated user.** llama3.3 70B role-plays the persona in its answer style and is told to
   answer only what was asked; how often it volunteers other slots is measured.
4. **Belief update.** Either a keyword matcher (`rule`) or the LLM (`llm`, qwen3 32B), which
   scores how well the answer fits each value; Bayes' rule is applied in code in both cases.
5. **Recommendation.** The catalog is ranked by expected utility under the belief and scored
   against the true preferences: regret, hit@k and constraint violations.

Question wording is a template by default; LLM rewording is one ablation. The full design and
every decision taken along the way are in [docs/SPEC.md](docs/SPEC.md).

## Results

Full run: 12 arms × 48 evaluation personas × 5 replicates at K = 3, 5 and 8 (8,160 episodes),
4.3 hours on 4 A40 GPUs, about 230,000 LLM calls (most served from the reply cache) with no failed
calls and no parse failures. Each metric is averaged over a persona's replicates first; arms are
compared with paired Wilcoxon tests over the 48 personas with Holm correction. The hypotheses
were frozen before the run (SPEC Q13). Complete tables:
[results_summary/m5_full/results.md](results_summary/m5_full/results.md).

At K = 5 (95 % bootstrap CIs in the full table):

| selector | updater | mass on truth | MAP accuracy | regret | violations |
|---|---|---|---|---|---|
| `fixed` | `llm` | 0.488 | 0.647 | 0.089 | 1.08 |
| `random` | `llm` | 0.511 | 0.654 | 0.101 | 0.73 |
| `max_entropy` | `llm` | 0.513 | 0.656 | 0.112 | 1.24 |
| `entropy_choice` | `llm` | 0.490 | 0.654 | 0.110 | 1.21 |
| `eig` | `llm` | 0.513 | 0.656 | 0.112 | 1.24 |
| `eig_mc` | `llm` | **0.541** | **0.668** | 0.110 | **0.34** |
| `eig` | `rule` | 0.507 | 0.569 | 0.119 | 1.20 |

Mass on truth is the mean probability the belief puts on the true value of each slot (the
primary metric); MAP accuracy is the share of slots whose most probable value is the true one;
violations count the constraint slots the top recommendation breaks (0 to 2).

![Probability on the true value per turn, K = 5](docs/figures/m5_full/curve_mass_on_truth_K5.png)

What held and what did not:

- **H1, principled selection beats fixed and random order: not supported.** `eig` beats `fixed`
  only at K = 5 with the LLM updater (+0.026 mass on truth, Holm p < 0.001). At K = 8 the two are
  identical, because both end up asking the same 8 of the 9 slots, and `random` is slightly ahead
  of `eig` with either updater (+0.010). With nine slots of 3 to 5 values each, question order
  has little room to matter.
- **H2, `eig` equals `max_entropy` under the measured answer noise: holds exactly.** The noise
  per question type was measured on the development personas before the run: the LLM updater
  reads the right value from 99 % of open answers, 97 % of choice answers and 85 % of yes/no
  answers. With open questions the most reliable, the question with the largest expected gain is
  always the open question on the most uncertain slot, so `eig` asked `max_entropy`'s questions in
  every episode and every metric is identical. (In an earlier run with a rule-based user and the
  opposite noise, 0.7 / 0.8 / 0.9, `eig` beat `max_entropy` by 0.057 at K = 5: its edge comes
  entirely from the noise model.) Listing the options (`entropy_choice`) is worse at K = 3
  (−0.03 with either updater) and at K = 5 with the LLM updater (−0.024).
- **`eig_mc` is the one selector that helps**, against the frozen prediction that it would not:
  +0.027 mass on truth over `eig` at K = 5 and 0.90 fewer constraint violations (both Holm
  p < 0.001). Regret and hit@1 do not change. It needs several LLM calls per candidate question,
  so it was run at K = 5 only; why its questions do better has not been analysed.
- **H3, the LLM updater beats the keyword updater: not supported on the primary metric.** Mass
  on truth does not differ for any selector. MAP accuracy does, for every selector: +0.04 to +0.08
  at K = 3, +0.06 to +0.10 at K = 5, +0.11 to +0.13 at K = 8 (all Holm p < 0.01). The LLM updater
  picks the right value more often without putting more probability on it. The breakdown by
  answer style (does the gap grow for vaguer answers?) is descriptive and not published yet.
- **H4, a better final belief goes with lower regret:** the per-arm Spearman correlations are
  not published yet.
- **Phrasing:** LLM rewording of the questions changes nothing measurable (largest difference
  −0.006 mass on truth, not significant).

Limits: the users are simulated by an LLM that knows its whole profile, there is one domain, the
belief treats slots as independent, and regret is measured on a synthetic catalog.

## Quick start

```bash
pip install -e .
elicitor demo                               # you answer the questions; no LLM needed
elicitor run experiments/m1_no_llm.toml     # no LLM needed; writes results/m1_no_llm/
elicitor run experiments/m2_mock.toml       # every LLM code path on the mock backend
elicitor run experiments/m3_eig.toml        # EIG vs the baselines, rule-based user, no LLM
```

Run these from the repo root. The demo asks up to five questions chosen by `eig`, shows the
belief after each answer and ends with the three best catalog items.
`elicitor demo --selector max_entropy --budget 8` changes the agent; with
[Ollama](https://ollama.com) running locally and `ollama pull qwen3:4b`,
`elicitor demo --updater llm` uses the LLM updater.

Then, for any run folder:

```bash
elicitor health results/m2_mock      # progress and per-role LLM failure rates (also mid-run)
elicitor analyze results/m3_eig      # tables, paired tests and curves in results/m3_eig/analysis/
```

A run into an existing folder resumes; `--fresh` starts over (the LLM reply cache is kept).
`analyze --publish` writes the committed tables in `results_summary/` and figures in
`docs/figures/`, and refuses runs that are incomplete, cut short or made with the mock backend.

## Running the full experiment

`experiments/m5_full.toml` is the run above. `cluster/` holds the Slurm scripts: copy
`cluster/local.env.example` to `cluster/local.env`, fill in your account and partition, then run
`cluster/submit.sh cluster/full.sbatch`. The job starts Ollama servers on the GPUs (qwen3:32b for
the agent, llama3.3 70B for the simulated user), runs the grid with resume, checks run health and
publishes the analysis. `elicitor calibrate` measures the answer noise per question type and
`elicitor make-backstories` writes the persona backstories; both results are frozen in the repo.

## Development

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest
```

CI runs the tests and the no-LLM and mock suites on Linux and Windows.

## License

MIT, see [LICENSE](LICENSE).
