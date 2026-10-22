"""Command-line entry point.

Subcommands (run, analyze, health, demo) are added milestone by milestone; see docs/SPEC.md §16.
"""

import argparse

from elicitor import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="elicitor",
        description="Preference elicitation by expected information gain.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
