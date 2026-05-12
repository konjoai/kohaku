"""Kohaku command-line interface."""
from __future__ import annotations

import argparse
import sys


def _cmd_serve(args: argparse.Namespace) -> None:
    """Launch the FastAPI REST server."""
    try:
        from kohaku.server import serve
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    serve(host=args.host, port=args.port, capacity=args.capacity, dim=args.dim)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="kohaku",
        description="Kohaku HDC episodic memory — CLI",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    serve_p = sub.add_parser("serve", help="Start the REST API server")
    serve_p.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    serve_p.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    serve_p.add_argument("--capacity", type=int, default=1000, help="Memory capacity (default: 1000)")
    serve_p.add_argument("--dim", type=int, default=1024, help="HyperVector dimension (default: 1024)")
    serve_p.set_defaults(func=_cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the kohaku CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
