"""Generate a domain's frozen data files from its schema (`elicitor make-data`).

Counts and seeds live in the schema's `[data]` table, so the committed files can always be
regenerated and checked.
"""

from pathlib import Path

from elicitor.catalog import generate_catalog
from elicitor.jsonl import dumps_jsonl
from elicitor.personas import generate_personas
from elicitor.schema import load_schema


def expected_files(domain_dir: str | Path) -> dict[Path, str]:
    domain_dir = Path(domain_dir)
    schema = load_schema(domain_dir / "schema.toml")
    d = schema.data
    personas = generate_personas(schema, int(d["n_dev"]), int(d["n_eval"]), int(d["persona_seed"]))
    catalog = generate_catalog(schema, int(d["catalog_size"]), int(d["catalog_seed"]))
    return {
        domain_dir / "personas.jsonl": dumps_jsonl(p.to_dict() for p in personas),
        domain_dir / "catalog.jsonl": dumps_jsonl(i.to_dict() for i in catalog),
    }


def write_data(domain_dir: str | Path) -> list[Path]:
    files = expected_files(domain_dir)
    for path, text in files.items():
        path.write_text(text, encoding="utf-8", newline="\n")
    return list(files)


def check_data(domain_dir: str | Path) -> list[Path]:
    """Paths whose content differs from what the generator produces now."""
    stale = []
    for path, text in expected_files(domain_dir).items():
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            stale.append(path)
    return stale
