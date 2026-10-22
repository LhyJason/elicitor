# Elicitor

An LLM agent that learns what a user wants by asking a few well-chosen questions. It keeps a
probability distribution over the user's preferences, picks each question by expected information
gain, updates the distribution from the free-text answer, and then recommends. Everything runs on
open models served locally or on a GPU cluster; no paid API is needed.

Inspired by an EECS 545 team project.

**Status:** early development. M1 built the no-LLM core (travel schema, personas, catalog,
factorized belief, keyword updater, fixed / random / max-entropy question selection, a rule-based
simulated user and the episode loop). M2 adds the LLM layer: an Ollama client with a SQLite reply
cache, structured outputs checked by pydantic, and a deterministic mock backend. On top of it sit
an LLM belief updater (the LLM scores the answer; Bayes' rule stays in code), LLM question
phrasing and an LLM simulated user. M3 adds choice and yes/no questions, an analytic
expected-information-gain selector over all question types, and a recommender that ranks a
200-item catalog by expected utility, scored by regret, hit@k and constraint violations. M4
makes runs resumable (finished arm × replicate cells are skipped, so a cluster job that runs out
of time is resubmitted rather than restarted), adds a run-health report that can be read while a
job runs, and an analysis step with paired statistics over personas (Wilcoxon signed-rank,
bootstrap CIs, Holm correction) and per-turn curves. The design is in
[docs/SPEC.md](docs/SPEC.md).

## Quick start

```bash
pip install -e .
elicitor run experiments/m1_no_llm.toml     # no LLM needed; writes results/m1_no_llm/
elicitor run experiments/m2_mock.toml       # every LLM code path on the mock backend
elicitor run experiments/m3_eig.toml        # EIG vs the baselines, rule-based user, no LLM
```

Then, for any run folder:

```bash
elicitor health results/m2_mock      # progress and per-role LLM failure rates (also mid-run)
elicitor analyze results/m3_eig      # tables, paired tests and curves in results/m3_eig/analysis/
```

A run into an existing folder resumes; `--fresh` starts over (the LLM reply cache is kept).
`analyze --publish` writes the committed tables in `results_summary/` and figures in
`docs/figures/`, and refuses runs that are incomplete, cut short or made with the mock backend.

The run prints mean metrics per arm and question budget: belief metrics (probability mass on the
true value, MAP accuracy, log loss and entropy in bits, averaged over slots) and outcome metrics
for the top recommendation (regret, hit@1/5/10, constraint violations), averaged over episodes. Runs that use
an LLM also print and save per-role call statistics (`llm_stats.json`).

With [Ollama](https://ollama.com) running locally, `ollama pull qwen3:4b` and then
`elicitor run experiments/m2_smoke_ollama.toml` runs a small smoke test on a real model.
`cluster/` has the same test as a one-GPU Slurm job.

## Development

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest
```

## License

MIT, see [LICENSE](LICENSE).
