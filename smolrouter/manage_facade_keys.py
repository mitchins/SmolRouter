#!/usr/bin/env python3
"""Operator commands for managing router-owned facade keys."""

from __future__ import annotations

import argparse
from typing import Sequence

from . import facade_key_cli


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Operator commands for provisioning SmolRouter facade keys."
    )
    subparsers = parser.add_subparsers(dest="command")

    create_parser = subparsers.add_parser(
        "create",
        help="Generate and store a new facade key for an existing logical project id.",
        description=(
            "Generate and store a new SmolRouter facade key for an existing logical "
            "project id defined under routes.yaml facade_keys."
        ),
    )
    facade_key_cli.configure_create_parser(create_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "create":
        return facade_key_cli.run_create(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
