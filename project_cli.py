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
    """TO-DO: add function doc here"""
    return "INF" if value == inf else str(int(value))


def _parser() -> argparse.ArgumentParser:
    """TO-DO: add function doc here"""
    parser = argparse.ArgumentParser(description="Solve Graphs as Networks queries.")
    parser.add_argument("--nodes", type=Path, default=DEFAULT_INSTANCE / "nodes.tsv",
                        help="path to nodes TSV file")
    parser.add_argument("--edges", type=Path, default=DEFAULT_INSTANCE / "edges.tsv",
                        help="path to edges TSV file")
    parser.add_argument("--query", required=True, choices=("evac", "apsp", "critical"),
                        help="query to run")
    parser.add_argument("--close", nargs=2, type=int, metavar=("U", "V"),
                        help="show evacuation results with directed edge U->V closed")
    parser.add_argument(
        "--algo",
        choices=sorted(APSP_ALGORITHMS),
        help="APSP algorithm (dijkstra, bellman-ford, johnson, or floyd-warshall)",
    )
    parser.add_argument("--counters", action="store_true", help="write operation counts to counters.txt")
    parser.add_argument("--verbose", action="store_true", help="print search summary to stderr")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.algo is not None and args.query != "apsp":
        parser.error("--algo is only applicable to --query apsp")
    if args.close is not None and args.query != "evac":
        parser.error("--close is only applicable to --query evac")

    try:
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
