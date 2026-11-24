"""Experiment runners for the graph query performance study."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Callable, TypeVar

from data import gen_graphs
from graph import Graph
from graph_services import ComparisonResult, Counters, compare_algorithms
from road_network import Node, Road, RoadNetwork

T = TypeVar("T")
REAL_LIFE_DIR = Path(__file__).resolve().parent / "data" / "real-life-examples"


def graph_to_road_network(graph: Graph) -> RoadNetwork:
    """Map graph labels to consecutive integer IDs for the routing services.

    Node.name retains the original label. Each adjacency entry becomes a
    one-way road, preserving asymmetric costs and bidirectional roads without
    duplicating arcs. Junction delays and capacities are zero. Road.minutes
    stores the original weight unchanged: distance inputs still measure
    distance, not time, and fractional weights are not rounded.
    """
    node_ids = {label: index for index, label in enumerate(graph.nodes)}
    nodes = {
        index: Node(index, str(label), "junction", 0)
        for label, index in node_ids.items()
    }
    roads = []
    for source, neighbors in graph.nodes.items():
        for target, weight in neighbors.items():
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Edge weights must be finite and nonnegative")
            roads.append(Road(len(roads), node_ids[source], node_ids[target],
                              weight, 0, True))
    return RoadNetwork(nodes, tuple(roads))


def compare_graph(graph: Graph, *, source=None, target=None) -> ComparisonResult:
    """Compare Dijkstra and CH for endpoints given as original graph labels.

    Defaults to the first and last nodes in insertion order. Result paths use
    consecutive integer IDs in that same order, as graph_to_road_network does.
    CH preprocessing and query metrics are supplied by compare_algorithms.
    """
    if not graph.nodes:
        raise ValueError("An experiment needs at least one node")
    node_ids = {label: index for index, label in enumerate(graph.nodes)}
    source = next(iter(graph.nodes)) if source is None else source
    target = next(reversed(graph.nodes)) if target is None else target
    if source not in node_ids or target not in node_ids:
        raise ValueError("Source and target must be existing graph node labels")
    network = graph_to_road_network(graph)
    return compare_algorithms(network, node_ids[source], node_ids[target])


def _generated_graph(generator, n, *, directed=False, seed=None, **parameters) -> Graph:
    """Build the project's Graph from a generator's node set and edge map."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    nodes, edges = generator(n, directed=directed, seed=seed, **parameters)
    graph = Graph()
    for node in sorted(nodes):
        graph.add_node(node)
    # Generators already include both arcs for undirected edges.
    for (source, target), attributes in edges.items():
        graph.add_edge(source, target, attributes["weight"])
    return graph


def experiment_erdos_renyi(n, probability, *, directed=False, seed=None,
                          source=None, target=None) -> ComparisonResult:
    """Compare routing on n Erdos-Renyi nodes with edge probability [0, 1]."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    graph = _generated_graph(gen_graphs.erdos_renyi, n, p=probability,
                             directed=directed, seed=seed)
    return compare_graph(graph, source=source, target=target)


def experiment_cycle(n, *, directed=False, seed=None, source=None, target=None) -> ComparisonResult:
    """Compare routing on a weighted cycle with n nodes."""
    graph = _generated_graph(gen_graphs.cycle, n, directed=directed, seed=seed)
    return compare_graph(graph, source=source, target=target)


def experiment_complete_graph(n, *, directed=False, seed=None, source=None, target=None) -> ComparisonResult:
    """Compare routing on a complete weighted graph with n nodes."""
    graph = _generated_graph(gen_graphs.complete_graph, n, directed=directed, seed=seed)
    return compare_graph(graph, source=source, target=target)


def experiment_path_graph(n, *, directed=False, seed=None, source=None, target=None) -> ComparisonResult:
    """Compare routing on a weighted path with n nodes."""
    graph = _generated_graph(gen_graphs.path_graph, n, directed=directed, seed=seed)
    return compare_graph(graph, source=source, target=target)


def experiment_random_trees(n, *, directed=False, seed=None, source=None, target=None) -> ComparisonResult:
    """Compare routing on a random weighted tree with n nodes."""
    graph = _generated_graph(gen_graphs.random_trees, n, directed=directed, seed=seed)
    return compare_graph(graph, source=source, target=target)


def experiment_balanced_trees(n, degree=3, *, directed=False, seed=None,
                             source=None, target=None) -> ComparisonResult:
    """Compare routing on a balanced n-node tree with degree children per node."""
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be a positive integer")
    graph = _generated_graph(gen_graphs.balanced_trees, n, degree=degree,
                             directed=directed, seed=seed)
    return compare_graph(graph, source=source, target=target)


def experiment_grid_graph(n, *, directed=False, seed=None, source=None, target=None) -> ComparisonResult:
    """Compare routing on an n-by-n grid containing n squared nodes."""
    graph = _generated_graph(gen_graphs.grid_graph, n, directed=directed, seed=seed)
    return compare_graph(graph, source=source, target=target)


def load_tsv_graph(directory) -> Graph:
    """Read node.tsv and edges.tsv into a distance-weighted Graph.

    Accept XML exports (id; u/v/distance or length) and generator exports
    (node_id; node_id_from/node_id_to/weight). Preserve node IDs as strings,
    isolated nodes, and the arc direction in each row. XML exports already
    contain both directions for undirected roads. Keep the minimum weight
    for parallel arcs, omit self-loops, and reject unknown endpoints or
    nonfinite/negative weights. Do not convert distance units.
    """
    directory = Path(directory)
    graph = Graph()
    with (directory / "node.tsv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        columns = set(reader.fieldnames or [])
        node_column = next((name for name in ("id", "node_id") if name in columns), None)
        if node_column is None:
            raise ValueError(f"{directory / 'node.tsv'}: expected id or node_id column")
        for line, row in enumerate(reader, start=2):
            node = row[node_column]
            if not node or graph.has_node(node):
                raise ValueError(f"{directory / 'node.tsv'}:{line}: empty or duplicate node ID")
            graph.add_node(node)

    with (directory / "edges.tsv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        columns = set(reader.fieldnames or [])
        schema = next((names for names in (
            ("u", "v", "distance"), ("u", "v", "length"),
            ("node_id_from", "node_id_to", "weight"),
        ) if set(names) <= columns), None)
        if schema is None:
            raise ValueError(f"{directory / 'edges.tsv'}: unsupported edge columns")
        source_column, target_column, weight_column = schema
        for line, row in enumerate(reader, start=2):
            source, target = row[source_column], row[target_column]
            if not graph.has_node(source) or not graph.has_node(target):
                raise ValueError(f"{directory / 'edges.tsv'}:{line}: unknown endpoint")
            try:
                weight = float(row[weight_column])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{directory / 'edges.tsv'}:{line}: invalid weight") from error
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{directory / 'edges.tsv'}:{line}: weight must be finite and nonnegative")
            if source == target:
                continue
            previous = graph.nodes[source].get(target)
            if previous is None:
                graph.add_edge(source, target, weight)
            elif weight < previous:
                graph.nodes[source][target] = weight
    return graph


def experiment_california(*, data_dir=None, source=None, target=None) -> ComparisonResult:
    """Load California TSVs and compare routing using original string endpoints."""
    graph = load_tsv_graph((REAL_LIFE_DIR if data_dir is None else Path(data_dir)) / "CA")
    return compare_graph(graph, source=source, target=target)


def experiment_new_york(*, data_dir=None, source=None, target=None) -> ComparisonResult:
    """Load New York TSVs and compare routing using original string endpoints."""
    graph = load_tsv_graph((REAL_LIFE_DIR if data_dir is None else Path(data_dir)) / "NY")
    return compare_graph(graph, source=source, target=target)


def experiment_texas(*, data_dir=None, source=None, target=None) -> ComparisonResult:
    """Load Texas TSVs and compare routing using original string endpoints."""
    graph = load_tsv_graph((REAL_LIFE_DIR if data_dir is None else Path(data_dir)) / "TX")
    return compare_graph(graph, source=source, target=target)


def experiment_merged(*, data_dir=None, source=None, target=None) -> ComparisonResult:
    """Load merged TSVs and compare routing using original string endpoints."""
    graph = load_tsv_graph((REAL_LIFE_DIR if data_dir is None else Path(data_dir)) / "merged")
    return compare_graph(graph, source=source, target=target)


@dataclass(frozen=True)
class TimedResult:
    """TODO: write function doc here"""

    value: object | None
    seconds: float
    timed_out: bool = False


class ThreadedRunner:
    """Run a callable in a worker while a separate thread measures its runtime."""

    @staticmethod
    def run(function: Callable[[Event], T], *, stop_after: float | None = 30.0) -> TimedResult:
        """TODO: write function doc here"""
        started = Event()
        finished = Event()
        stop_event = Event()
        measured: list[float] = []
        values: list[T] = []
        errors: list[BaseException] = []
        timed_out: list[bool] = []

        def clock() -> None:
            """TODO: write function doc here"""
            start_time = perf_counter()
            started.set()
            if stop_after is None:
                finished.wait()
                timed_out.append(False)
            elif finished.wait(stop_after):
                timed_out.append(False)
            else:
                stop_event.set()
                finished.wait()
                timed_out.append(True)
            measured.append(perf_counter() - start_time)

        def work() -> None:
            """TODO: write function doc here"""
            started.wait()
            try:
                values.append(function(stop_event))
            except TimeoutError:
                timed_out.append(True)
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        timer_thread = Thread(target=clock, name="experiment-clock")
        worker_thread = Thread(target=work, name="experiment-worker")
        timer_thread.start()
        worker_thread.start()
        try:
            worker_thread.join()
        except KeyboardInterrupt:
            stop_event.set()
            worker_thread.join()
            timer_thread.join()
            raise
        timer_thread.join()

        if errors:
            raise errors[0]
        return TimedResult(values[0] if values else None, measured[0], any(timed_out))


def with_stop(counters: Counters, stop_event: Event) -> Counters:
    """TODO: write function doc here"""
    counters.stop_event = stop_event
    return counters


def main() -> None:
    pass


if __name__ == "__main__":
    main()
