# Elicitor

An LLM agent that learns what a user wants by asking a few well-chosen questions. It keeps a
probability distribution over the user's preferences, picks each question by expected information
gain, updates the distribution from the free-text answer, and then recommends. Everything runs on
open models served locally or on a GPU cluster; no paid API is needed.

Inspired by an EECS 545 team project.

**Status:** early development. Milestone M1 is done: the no-LLM core (travel schema, personas,
catalog, factorized belief, keyword updater, fixed / random / max-entropy question selection, a
rule-based simulated user and the episode loop). The design is in [docs/SPEC.md](docs/SPEC.md).

## Quick start

```bash
pip install -e .
elicitor run experiments/m1_no_llm.toml     # no LLM needed; writes results/m1_no_llm/
```

The run prints mean belief metrics per arm and question budget: probability mass on the true
value, MAP accuracy, log loss and entropy (bits), averaged over slots and episodes.

## Development

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest
```

## License

MIT, see [LICENSE](LICENSE).
