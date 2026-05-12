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


def _cmd_export(args: argparse.Namespace) -> None:
    """Export episodic memory from an .hkb file (or empty) as a graph."""
    from kohaku._pure import EpisodicMemory
    from kohaku.graph_export import GraphExportConfig, MemoryGraphExporter

    if args.from_file:
        from kohaku.persistence import load
        mem = load(args.from_file)
    else:
        mem = EpisodicMemory(capacity=1000)

    cfg = GraphExportConfig(similarity_threshold=args.threshold)
    exporter = MemoryGraphExporter(cfg)
    graph = exporter.export(mem)

    fmt = args.format.lower()
    out = args.out
    if out:
        exporter.save(graph, out)
        print(f"graph written to {out} ({graph.n_nodes} nodes, {graph.n_edges} edges)")
    elif fmt == "gexf":
        print(graph.to_gexf())
    else:
        print(graph.to_json())


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

    export_p = sub.add_parser("export", help="Export memory graph to JSON or GEXF")
    export_p.add_argument(
        "--format", choices=["json", "gexf"], default="json",
        help="Output format (default: json)",
    )
    export_p.add_argument(
        "--threshold", type=float, default=0.3,
        help="Cosine similarity threshold for edges (default: 0.3)",
    )
    export_p.add_argument(
        "--out", default=None,
        help="Output file path. Extension (.json/.gexf) selects format. "
             "Prints to stdout if omitted.",
    )
    export_p.add_argument(
        "--from", dest="from_file", default=None, metavar="FILE",
        help="Load EpisodicMemory from an .hkb or .json file before exporting.",
    )
    export_p.set_defaults(func=_cmd_export)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the kohaku CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
