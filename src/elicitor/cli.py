"""Command-line entry point.

Subcommands (run, analyze, health, demo) are added milestone by milestone; see docs/SPEC.md §16.
M1 has `run` and `make-data`.
"""

import argparse
import sys
from pathlib import Path

from elicitor import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elicitor",
        description="Preference elicitation by expected information gain.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run an experiment file")
    run.add_argument("config", type=Path, help="experiment TOML file")
    run.add_argument("--out", type=Path, help="output directory (default: results/<name>)")
    run.add_argument("--max-personas", type=int, help="use only the first N personas (smoke runs)")

    data = sub.add_parser("make-data", help="regenerate frozen personas and catalog")
    data.add_argument("domain_dir", type=Path, help="domain folder, e.g. data/travel")
    data.add_argument(
        "--check", action="store_true", help="only check that the files match the generator"
    )
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    from elicitor.experiments.config import load_config
    from elicitor.experiments.runner import format_table, run_experiment

    config = load_config(args.config)
    out = args.out or Path("results") / config.name
    rows = run_experiment(config, out, max_personas=args.max_personas, progress=print)
    print(format_table(rows))
    print(f"wrote {out / 'episodes.jsonl'} and {out / 'summary.csv'}")
    return 0


def cmd_make_data(args: argparse.Namespace) -> int:
    from elicitor.datagen import check_data, write_data

    if args.check:
        stale = check_data(args.domain_dir)
        for path in stale:
            print(f"out of date: {path}", file=sys.stderr)
        return 1 if stale else 0
    for path in write_data(args.domain_dir):
        print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "make-data":
        return cmd_make_data(args)
    parser.print_help()
    return 0
