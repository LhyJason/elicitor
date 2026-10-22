# Elicitor

An LLM agent that learns what a user wants by asking a few well-chosen questions. It keeps a
probability distribution over the user's preferences, picks each question by expected information
gain, updates the distribution from the free-text answer, and then recommends. Everything runs on
open models served locally or on a GPU cluster; no paid API is needed.

Inspired by an EECS 545 team project.

**Status:** early development (milestone M0: package skeleton and CI). The design is in
[docs/SPEC.md](docs/SPEC.md).

## Development

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest
```

## License

MIT, see [LICENSE](LICENSE).
