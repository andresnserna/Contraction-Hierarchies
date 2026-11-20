"""Command-line entry point for point-to-point road-network routing."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from contraction_hierarchies import ContractionHierarchy
from graph_services import (
    Counters,
    compare_algorithms,
    find_path_ch,
    find_path_dijkstra,
    format_comparison,
    format_path_result,
    resolve_network,
)

DATASETS = ("tiny", "small", "medium")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find a shortest path with Dijkstra or Contraction Hierarchies, or "
            "compare both algorithms on the same pair of nodes. Exactly one of "
            "--find and --compare is required."
        ),
        epilog=(
            "Examples:\n"
            "  project_cli.py --find 12 31\n"
            "  project_cli.py --find rand --data small --algo dijkstra\n"
            "  project_cli.py --compare 12 31 --verbose\n"
            "  project_cli.py --compare rand --nodes custom-nodes.tsv --edges custom-edges.tsv\n\n"
            "Choose a dataset with --data tiny|small|medium (default: tiny), or "
            "provide both --nodes and --edges."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--find", nargs="+", metavar="NODE",
        help="find a path for U V, or use 'rand' to choose two distinct nodes",
    )
    modes.add_argument(
        "--compare", nargs="+", metavar="NODE",
        help="compare Dijkstra and CH for U V, or use 'rand' for two distinct nodes",
    )
    parser.add_argument("--nodes", type=Path, help="custom nodes TSV (requires --edges)")
    parser.add_argument("--edges", type=Path, help="custom edges TSV (requires --nodes)")
    parser.add_argument(
        "--data", type=str.lower, choices=DATASETS,
        help="bundled dataset: tiny, small, or medium (default: tiny)",
    )
    parser.add_argument(
        "--algo", choices=("dijkstra", "ch"),
        help="algorithm for --find: dijkstra or ch (default: ch); --compare always runs both",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print progress from the selected algorithm(s) and metrics to stderr",
    )
    parser.add_argument(
        "--counters", action="store_true",
        help="write operation, timing, and graph-size counters to counters.txt",
    )
    return parser


def _resolve_pair(spec: list[str] | None, parser: argparse.ArgumentParser) -> tuple[int, int] | None:
    if spec is None:
        parser.error(
            "you must specify a path to find with --find U V or --compare U V, "
            "or pass --find rand / --compare rand for a random pair"
        )
    if len(spec) == 1 and spec[0].lower() == "rand":
        return None
    if len(spec) != 2:
        parser.error("provide exactly two node IDs, or the single value 'rand'")
    try:
        return int(spec[0]), int(spec[1])
    except ValueError:
        parser.error("node IDs must be integers; use 'rand' for a random pair")


def _write_counters(counter: Counters) -> None:
    Path.cwd().joinpath("counters.txt").write_text(
        f"Relaxations\t{counter.relaxations}\n"
        f"Nodes_Settled\t{counter.nodes_settled}\n"
        f"PQ_Ops\t{counter.pq_ops}\n"
        f"Preprocessing_Time_Seconds\t{counter.preprocessing_time:.6f}\n"
        f"Query_Time_Seconds\t{counter.query_time:.6f}\n"
        f"Original_Directed_Edges\t{counter.original_edges}\n"
        f"CH_Shortcuts\t{counter.ch_shortcuts}\n"
        f"CH_Stored_Edges\t{counter.ch_edges}\n"
        f"CH_Query_Graph_Edges\t{counter.ch_query_edges}\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.find is None and args.compare is None:
        parser.error(
            "choose a path to find: use --find U V (or --find rand) or "
            "--compare U V (or --compare rand)"
        )
    if args.algo is not None and args.compare is not None:
        parser.error("--algo applies only to --find; --compare always runs both algorithms")
    if (args.nodes is None) != (args.edges is None):
        parser.error("provide both --nodes and --edges together")
    if args.data is not None and args.nodes is not None:
        parser.error("choose either --data or the --nodes/--edges pair")

    pair_spec = args.find if args.find is not None else args.compare
    pair = _resolve_pair(pair_spec, parser)
    try:
        network = resolve_network(args.data, args.nodes, args.edges)
        if pair is None:
            if len(network.nodes) < 2:
                parser.error("a random path requires a dataset with at least two nodes")
            pair = tuple(random.sample(sorted(network.nodes), 2))
        source, target = pair
        if source not in network.nodes:
            parser.error(f"source node {source} does not exist in this dataset")
        if target not in network.nodes:
            parser.error(f"target node {target} does not exist in this dataset")

        if args.verbose:
            print(f"Loaded {len(network.nodes)} nodes and {len(network.roads)} roads; route {source} -> {target}", file=sys.stderr)
        if args.find is not None:
            counter = Counters(verbose=args.verbose)
            result = (
                find_path_dijkstra(network, source, target, counter)
                if args.algo == "dijkstra"
                else find_path_ch(network, source, target, counter)
            )
            print(format_path_result(result))
            counters = [counter]
        else:
            comparison = compare_algorithms(
                network, source, target,
                ch=ContractionHierarchy(network),
                verbose=args.verbose,
            )
            print(format_comparison(comparison))
            counters = [comparison.dijkstra.counters, comparison.ch.counters]
        if args.counters:
            aggregate = Counters()
            for counter in counters:
                aggregate.add(counter)
            _write_counters(aggregate)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
