"""Command-line entry point.

Subcommands (run, analyze, health, demo) are added milestone by milestone; see docs/SPEC.md §16.
M1 added `run` and `make-data`, M4 `analyze` and `health`, M5 `calibrate` and `make-backstories`,
M6 `demo`, M7 `judge` and `judge-report`.
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
    run.add_argument(
        "--workers", type=int, default=1, help="threads running (cell, persona) tasks (default 1)"
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

    calib = sub.add_parser("calibrate", help="measure rho per question type on the dev personas")
    calib.add_argument("config", type=Path, help="experiment TOML file (simulator and [llm])")
    calib.add_argument(
        "--out", type=Path, help="output folder (default: results/<name>_calibration)"
    )
    calib.add_argument("--workers", type=int, default=1, help="threads asking questions")

    stories = sub.add_parser("make-backstories", help="write the frozen persona backstories")
    stories.add_argument("config", type=Path, help="experiment TOML file ([llm] user model)")
    stories.add_argument("--workers", type=int, default=1, help="threads writing backstories")
    stories.add_argument(
        "--force", action="store_true", help="overwrite an existing backstories file"
    )

    demo = sub.add_parser("demo", help="answer the agent's questions yourself (SPEC Q20)")
    demo.add_argument(
        "config", type=Path, nargs="?", help="experiment TOML file (default: experiments/demo.toml)"
    )
    demo.add_argument("--selector", help="question selector (default: the file's first)")
    demo.add_argument("--updater", help="belief updater, rule or llm (default: the file's first)")
    demo.add_argument("--phraser", help="question wording, tpl or llm (default: the file's first)")
    demo.add_argument("--budget", type=int, help="number of questions (default: the file's first)")

    judge = sub.add_parser("judge", help="LLM judges score a run's recommendations (SPEC Q10)")
    judge.add_argument("config", type=Path, help="judge TOML file")
    judge.add_argument("--out", type=Path, help="output directory (default: results/<name>)")
    judge.add_argument("--source", type=Path, help="source run folder (default: the file's)")
    judge.add_argument("--workers", type=int, default=1, help="threads making calls (default 1)")

    report = sub.add_parser("judge-report", help="tests and tables of a judge folder")
    report.add_argument("judge_dir", type=Path, help="folder written by `elicitor judge`")
    report.add_argument("--out", type=Path, help="table folder (default: <judge_dir>/report)")
    report.add_argument(
        "--publish",
        action="store_true",
        help="write to results_summary/<name>/ and docs/figures/<name>/ (complete real runs only)",
    )
    report.add_argument("--no-figures", action="store_true", help="tables only")

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
            config,
            out,
            max_personas=args.max_personas,
            progress=print,
            fresh=args.fresh,
            workers=args.workers,
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


def cmd_calibrate(args: argparse.Namespace) -> int:
    from elicitor.calibration import calibrate
    from elicitor.experiments.config import load_config

    config = load_config(args.config)
    out = args.out or Path("results") / f"{config.name}_calibration"
    calibrate(config, out, workers=args.workers, progress=print)
    print(f"wrote {out / 'calibration.json'} and {out / 'answers.jsonl'}")
    return 0


def cmd_make_backstories(args: argparse.Namespace) -> int:
    from elicitor.backstories import write_backstories
    from elicitor.experiments.config import load_config
    from elicitor.llm import build_client
    from elicitor.personas import BACKSTORIES, load_personas
    from elicitor.schema import load_schema
    from elicitor.simulator.llm import load_styles

    config = load_config(args.config)
    path = config.personas_path.with_name(BACKSTORIES)
    if path.exists() and not args.force:
        print(f"error: {path} exists and is frozen (SPEC §4); pass --force", file=sys.stderr)
        return 2
    cache = Path("results") / "backstories_cache.sqlite"
    client = build_client(config.llm, config.prompts_dir, cache)
    try:
        stories = write_backstories(
            client,
            load_schema(config.schema_path),
            load_personas(config.personas_path),
            load_styles(config.prompts_dir / "styles.toml"),
            path,
            workers=args.workers,
            progress=print,
        )
        print(client.stats_table())
    finally:
        client.close()
    return 1 if any(s.flagged for s in stories) else 0


def cmd_demo(args: argparse.Namespace) -> int:
    from elicitor.demo import CACHE, DEFAULT_CONFIG, demo_arm, run_demo
    from elicitor.experiments.config import load_config

    path = args.config or DEFAULT_CONFIG
    if not path.exists():
        print(f"error: {path} not found; run from the repo root or pass a file", file=sys.stderr)
        return 2
    config = load_config(path)
    try:
        arm = demo_arm(config, args.selector, args.updater, args.phraser)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    client = None
    if arm.uses_llm:
        from elicitor.llm import build_client

        client = build_client(config.llm, config.prompts_dir, CACHE)
    try:
        run_demo(config, arm, budget=args.budget, client=client)
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        if client is not None:
            client.close()
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    from elicitor.experiments.health import format_roles
    from elicitor.judge import load_judge_config, run_judge
    from elicitor.llm import build_client

    try:
        config = load_judge_config(args.config, source=args.source)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = args.out or Path("results") / config.name
    out.mkdir(parents=True, exist_ok=True)
    client = build_client(config.llm, config.prompts_dir, out / "llm_cache.sqlite")
    try:
        state = run_judge(config, client, out, workers=args.workers, progress=print)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        client.close()
    if state["llm"]:
        print(format_roles(state["llm"]["roles"]))
    print(f"wrote {out}")
    for problem in state["problems"]:
        print(f"  - {problem}", file=sys.stderr)
    return 1 if state["problems"] else 0


def cmd_judge_report(args: argparse.Namespace) -> int:
    from elicitor.analysis.judge_report import judge_report
    from elicitor.analysis.report import RefusedToPublish

    try:
        written = judge_report(
            args.judge_dir, out=args.out, publish=args.publish, make_figures=not args.no_figures
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
    if args.command == "calibrate":
        return cmd_calibrate(args)
    if args.command == "make-backstories":
        return cmd_make_backstories(args)
    if args.command == "demo":
        return cmd_demo(args)
    if args.command == "judge":
        return cmd_judge(args)
    if args.command == "judge-report":
        return cmd_judge_report(args)
    parser.print_help()
    return 0
