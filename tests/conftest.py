from pathlib import Path

import pytest

from elicitor.schema import Schema, load_schema

ROOT = Path(__file__).resolve().parents[1]
TRAVEL = ROOT / "data" / "travel"


@pytest.fixture(scope="session")
def schema() -> Schema:
    return load_schema(TRAVEL / "schema.toml")
