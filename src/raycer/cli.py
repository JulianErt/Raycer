"""Command line interface: ``raycer``."""

import argparse
from collections.abc import Sequence

import raycer


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(prog="raycer", description=__doc__)
    parser.add_argument("--version", action="version", version=f"raycer {raycer.__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("benchmark", help="measure steps per second (stub)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the exit code."""
    args = build_parser().parse_args(argv)
    if args.command is None:
        build_parser().print_help()
    return 0
