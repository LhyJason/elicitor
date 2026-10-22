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
phrasing and an LLM simulated user. The design is in [docs/SPEC.md](docs/SPEC.md).

## Quick start

```bash
pip install -e .
elicitor run experiments/m1_no_llm.toml     # no LLM needed; writes results/m1_no_llm/
elicitor run experiments/m2_mock.toml       # every LLM code path on the mock backend
```

The run prints mean belief metrics per arm and question budget: probability mass on the true
value, MAP accuracy, log loss and entropy (bits), averaged over slots and episodes. Runs that use
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
