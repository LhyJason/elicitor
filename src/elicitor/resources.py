"""Files that ship with the package: the travel domain and the demo experiment file (SPEC Q21).

In a wheel they sit under `elicitor/bundled/`, laid out as in the repo (`data/<domain>/`,
`experiments/demo.toml`), so the demo file's relative `data_dir` resolves the same way. In a
checkout or an editable install there is no `bundled` folder and the repo root is used instead.
The files are not moved or copied in the repo: a run's fingerprint hashes `data/<domain>/`.
"""

from __future__ import annotations

from pathlib import Path


def root() -> Path:
    """The folder that holds `data/` and `experiments/`."""
    bundled = Path(__file__).resolve().parent / "bundled"
    return bundled if bundled.is_dir() else Path(__file__).resolve().parents[2]


def demo_config() -> Path:
    """The experiment file `elicitor demo` and `Session.from_config()` read by default."""
    path = root() / "experiments" / "demo.toml"
    if not path.is_file():
        raise FileNotFoundError(f"the demo experiment file is missing: {path}")
    return path


def domain_dir(domain: str = "travel") -> Path:
    """The folder of a domain that ships with the package (schema, catalog, prompts)."""
    path = root() / "data" / domain
    if not (path / "schema.toml").is_file():
        raise FileNotFoundError(f"no domain {domain!r} ships with the package: {path}")
    return path
