"""Command-line entry point.

Subcommands (run, analyze, health, demo) are added milestone by milestone; see docs/SPEC.md §16.
M1 added `run` and `make-data`, M4 `analyze` and `health`.
"""

import argparse
import json
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
    run.add_argument(
        "--fresh", action="store_true", help="discard finished cells instead of resuming"
    )

    analyze = sub.add_parser("analyze", help="paired statistics, tables and figures of a run")
    analyze.add_argument("run_dir", type=Path, help="run folder written by `elicitor run`")
    analyze.add_argument("--out", type=Path, help="table folder (default: <run_dir>/analysis)")
    analyze.add_argument("--figures", type=Path, help="figure folder (default: the table folder)")
    analyze.add_argument(
        "--publish",
        action="store_true",
        help="write to results_summary/<name>/ and docs/figures/<name>/ (complete real runs only)",
    )
    analyze.add_argument("--no-figures", action="store_true", help="tables only")

    health = sub.add_parser("health", help="progress and LLM failure rates of a run (also live)")
    health.add_argument("run_dir", type=Path, help="run folder written by `elicitor run`")

    data = sub.add_parser("make-data", help="regenerate frozen personas and catalog")
    data.add_argument("domain_dir", type=Path, help="domain folder, e.g. data/travel")
    data.add_argument(
        "--check", action="store_true", help="only check that the files match the generator"
    )
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    from elicitor.experiments.config import load_config
    from elicitor.experiments.health import format_roles
    from elicitor.experiments.runner import format_table, run_experiment
    from elicitor.experiments.store import FingerprintMismatch

    config = load_config(args.config)
    out = args.out or Path("results") / config.name
    try:
        rows = run_experiment(
            config, out, max_personas=args.max_personas, progress=print, fresh=args.fresh
        )
    except FingerprintMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_table(rows))
    llm_stats = out / "llm_stats.json"
    if llm_stats.exists():
        print(format_roles(json.loads(llm_stats.read_text(encoding="utf-8"))["roles"]))
    print(f"wrote {out / 'episodes.jsonl'} and {out / 'summary.csv'}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from elicitor.analysis.report import RefusedToPublish, analyze

    try:
        written = analyze(
            args.run_dir,
            out=args.out,
            figures_dir=args.figures,
            publish=args.publish,
            make_figures=not args.no_figures,
        )
    except RefusedToPublish as exc:
        print("refusing to write committed outputs:", file=sys.stderr)
        for reason in exc.reasons:
            print(f"  - {reason}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for path in written:
        print(f"wrote {path}")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    from elicitor.experiments.health import check_health

    health = check_health(args.run_dir)
    print(health.text())
    return 0 if health.ok else 1


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
    if args.command == "analyze":
        return cmd_analyze(args)
    if args.command == "health":
        return cmd_health(args)
    parser.print_help()
    return 0
