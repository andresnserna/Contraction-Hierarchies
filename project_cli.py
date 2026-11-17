"""Command-line entry point for COSC 4367 Project 2."""

from __future__ import annotations

import argparse
import sys
from math import inf
from pathlib import Path

from graph_services import Counters, all_pairs, critical_closure, evacuations
from road_network import RoadNetwork


DEFAULT_INSTANCE = Path(__file__).parent / "data" / "instances" / "tiny"
APSP_ALGORITHMS = {"dijkstra", "bellman-ford", "johnson", "floyd-warshall"}


def _format_distance(value: float) -> str:
    """TODO: add function doc here"""
    return "INF" if value == inf else str(int(value))


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser for legacy queries and planned path queries."""
    parser = argparse.ArgumentParser(
        description=(
            "Solve road-network queries. Use --query for evacuation, all-pairs, "
            "or critical-road analysis; use --find for a point-to-point path."
        ),
        epilog=(
            "Examples: --query evac; --find 12 31; --find rand --data TINY; "
            "--find 12 31 --algo dijkstra --verbose"
        ),
    )
    parser.add_argument("--nodes", type=Path, default=DEFAULT_INSTANCE / "nodes.tsv",
                        help="path to the input nodes TSV file (legacy queries default to the tiny instance)")
    parser.add_argument("--edges", type=Path, default=DEFAULT_INSTANCE / "edges.tsv",
                        help="path to the input edges TSV file (legacy queries default to the tiny instance)")
    parser.add_argument(
        "--data", choices=("TINY", "SMALL", "MEDIUM"), metavar="DATASET",
        help="select a bundled dataset (TINY, SMALL, or MEDIUM); planned to resolve its nodes.tsv and edges.tsv files",
    )
    parser.add_argument(
        "--query", choices=("evac", "apsp", "critical"),
        help="run a legacy query: evac assigns zones to shelters, apsp prints all-pairs distances, or critical finds the critical road",
    )
    parser.add_argument(
        "--find", nargs="+", metavar="U/V",
        help="find a shortest path between node IDs U and V, or pass the single value 'rand' to choose a random pair",
    )
    parser.add_argument("--close", nargs=2, type=int, metavar=("U", "V"),
                        help="show evacuation results with directed edge U->V closed")
    parser.add_argument(
        "--algo",
        choices=sorted(APSP_ALGORITHMS | {"ch"}),
        help="path algorithm for --find (ch or dijkstra; planned default: ch), or APSP algorithm for --query apsp (dijkstra, bellman-ford, johnson, or floyd-warshall)",
    )
    parser.add_argument("--counters", action="store_true", help="write operation counts to counters.txt")
    parser.add_argument(
        "--verbose", action="store_true",
        help="print the path search summary and metrics (relaxations, priority-queue operations, preprocessing time, and query time)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="run both Contraction Hierarchies and Dijkstra for --find, print both results and metrics, and report the winner",
    )
    return parser

def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.query is None and args.find is None:
        parser.error("specify either --query or --find")
    if args.query is not None and args.find is not None:
        parser.error("--query and --find cannot be used together")
    if args.compare and args.find is None:
        parser.error("--compare is only applicable to --find")
    if args.find is not None:
        # TODO: Validate the --find shape (two node IDs or the single token 'rand'),
        # choose/validate the dataset, and route this mode to the CH/Dijkstra flow.
        parser.error("--find execution is not implemented yet")
    if args.data is not None:
        # TODO: Resolve --data to its bundled nodes.tsv and edges.tsv paths.
        pass
    if args.algo is not None and args.query != "apsp":
        parser.error("--algo is only applicable to --query apsp")
    if args.close is not None and args.query != "evac":
        parser.error("--close is only applicable to --query evac")

    try:
        # TODO: Once --data and explicit --nodes/--edges are supported together,
        # validate their precedence and report missing or invalid input paths here.
        network = RoadNetwork.load(args.nodes, args.edges)
        counter = Counters()

        if args.query == "evac":
            excluded_arcs = frozenset()
            if args.close is not None:
                road = network.find_road(*args.close)
                excluded_arcs = frozenset({(road.id, *args.close)})
            for row in evacuations(network, excluded_arcs=excluded_arcs, counters=counter):
                shelter = "NONE" if row.shelter_id is None else network.nodes[row.shelter_id].name
                print(
                    f"{network.nodes[row.zone_id].name}\t{shelter}\t"
                    f"{_format_distance(row.minutes)}"
                )
        elif args.query == "apsp":
            distances = all_pairs(network, algorithm=args.algo or "dijkstra", counter=counter)
            for source in sorted(distances):
                for target in sorted(distances[source]):
                    if source != target:
                        print(
                            f"{source}\t{target}\t"
                            f"{_format_distance(distances[source][target])}"
                        )
        else:
            result = critical_closure(network, counter=counter)
            if result is None:
                print("NONE")
            else:
                print(f"{result.u}\t{result.v}\t{_format_distance(result.worst_minutes)}")

        if args.counters:
            counters_path = Path.cwd() / "counters.txt"
            counters_path.write_text(
                f"Relaxations\t{counter.relaxations}\nPQ_Ops\t{counter.pq_ops}\n",
                encoding="utf-8",
            )
        if args.verbose:
            print(
                f"V={len(network.nodes)} roads={len(network.roads)} "
                f"arcs={sum(map(len, network.adjacency().values()))} "
                f"relaxations={counter.relaxations} PQ_Ops={counter.pq_ops}",
                file=sys.stderr,
            )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
