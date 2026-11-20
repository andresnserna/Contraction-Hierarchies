from __future__ import annotations
from dataclasses import dataclass
from math import inf
from threading import Event
from graph import Graph
from road_network import RoadNetwork
from time import perf_counter
from pathlib import Path
import sys
from contraction_hierarchies import ContractionHierarchy
from pathlib import Path

INSTANCES_DIR = Path(__file__).parent / "data" / "instances"


@dataclass
class Counters:
    """
    Counts edge relaxations and priority queue pushes/pops, passed into the selected query at execution
    """

    relaxations: int = 0
    nodes_settled: int = 0
    preprocessing_time: float = 0.0
    query_time: float = 0.0
    pq_ops: int = 0
    original_edges: int = 0
    ch_shortcuts: int = 0
    ch_edges: int = 0
    ch_query_edges: int = 0
    verbose: bool = False
    stop_event: Event | None = None 

    def check_timeout(self) -> None:
        """Raises TimeoutError if the stop event has been triggered."""
        if self.stop_event is not None and self.stop_event.is_set():
            raise TimeoutError("experiment run reached its stop limit")

    def progress(self, message: str) -> None:
        """Print algorithm progress when verbose mode is enabled."""
        if self.verbose:
            print(message, file=sys.stderr, flush=True)

    def add(self, other: Counters) -> None:
        """Combines metrics from another Counters instance into this one."""
        self.relaxations += other.relaxations
        self.nodes_settled += other.nodes_settled
        self.preprocessing_time += other.preprocessing_time
        self.query_time += other.query_time
        self.pq_ops += other.pq_ops
        self.original_edges = max(self.original_edges, other.original_edges)
        self.ch_shortcuts = max(self.ch_shortcuts, other.ch_shortcuts)
        self.ch_edges = max(self.ch_edges, other.ch_edges)
        self.ch_query_edges = max(self.ch_query_edges, other.ch_query_edges)


def _dijkstra_graph(network: RoadNetwork, *, reverse: bool = False, excluded_road_ids: frozenset[int] = frozenset(), excluded_arcs: frozenset[tuple[int, int, int]] = frozenset()) -> tuple[Graph, dict[tuple[int, int, int], int]]:
    """Build dr michaels adjacency-list graph and retain road identities."""
    graph = Graph()

    for node_id in sorted(network.nodes):
        graph.add_node(node_id)

    road_ids: dict[tuple[int, int, int], int] = {}
    
    for source, arcs in network.adjacency(
        reverse=reverse, excluded_road_ids=excluded_road_ids,
        excluded_arcs=excluded_arcs,
    ).items():
        for arc in arcs:
            graph.add_edge(source, arc.target, arc.weight)
            road_ids[(source, arc.target, arc.weight)] = arc.road_id

    return graph, road_ids


@dataclass(frozen=True)
class PathResult:
    """Shared result format returned by both Dijkstra and CH path searches."""

    algorithm: str
    source: int
    target: int
    cost: float
    path: list[int] | None
    counters: Counters


@dataclass(frozen=True)
class ComparisonResult:
    """Side-by-side comparison of Dijkstra and CH on the same query."""

    dijkstra: PathResult
    ch: PathResult
    distances_match: bool
    winner: str
    reason: str


def resolve_network(
    dataset: str | None = None,
    nodes_path: str | Path | None = None,
    edges_path: str | Path | None = None,
) -> RoadNetwork:
    """Load a RoadNetwork from either a named dataset or explicit TSV file paths."""
    if nodes_path is not None or edges_path is not None:
        if nodes_path is None or edges_path is None:
            raise ValueError("Both --nodes and --edges must be provided together.")
        return RoadNetwork.load(Path(nodes_path), Path(edges_path))

    name = dataset or "tiny"
    instance_dir = INSTANCES_DIR / name
    if not instance_dir.exists():
        raise ValueError(f"Unknown dataset '{name}'. Available: tiny, small, medium")

    return RoadNetwork.load(instance_dir / "nodes.tsv", instance_dir / "edges.tsv")


def _validate_nodes(network: RoadNetwork, u: int, v: int) -> None:
    """Raise a clear error if either endpoint is missing from the network."""
    if u not in network.nodes:
        raise ValueError(f"Source node {u} does not exist in this network.")
    if v not in network.nodes:
        raise ValueError(f"Target node {v} does not exist in this network.")


def find_path_dijkstra(
    network: RoadNetwork,
    u: int,
    v: int,
    counter: Counters | None = None,
) -> PathResult:
    """Run Dijkstra's algorithm from u to v and return a unified PathResult."""
    _validate_nodes(network, u, v)
    active_counter = counter if counter is not None else Counters()
    graph, _ = _dijkstra_graph(network)
    active_counter.original_edges = graph.get_num_edges()

    start = perf_counter()
    distances, parents = graph.dijkstra(u, counter=active_counter)
    active_counter.query_time += perf_counter() - start

    cost = distances[v]
    if cost == inf:
        return PathResult("dijkstra", u, v, inf, None, active_counter)

    path: list[int] = []
    current: int | None = v
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()

    return PathResult("dijkstra", u, v, cost, path, active_counter)


def find_path_ch(
    network: RoadNetwork,
    u: int,
    v: int,
    counter: Counters | None = None,
    ch: ContractionHierarchy | None = None,
) -> PathResult:
    """Run Contraction Hierarchies from u to v and return a unified PathResult."""
    _validate_nodes(network, u, v)
    active_counter = counter if counter is not None else Counters()

    hierarchy = ch if ch is not None else ContractionHierarchy(network)
    if not hierarchy.is_preprocessed:
        hierarchy.preprocess(counter=active_counter)

    cost, path = hierarchy.shortest_path(u, v, counter=active_counter)
    return PathResult("ch", u, v, cost, path, active_counter)


def compare_algorithms(
    network: RoadNetwork,
    u: int,
    v: int,
    ch: ContractionHierarchy | None = None,
    verbose: bool = False,
) -> ComparisonResult:
    """Run both Dijkstra and CH on (u, v), verify cost agreement, and pick a winner."""
    d_res = find_path_dijkstra(network, u, v, counter=Counters(verbose=verbose))
    ch_res = find_path_ch(network, u, v, counter=Counters(verbose=verbose), ch=ch)
    distances_match = d_res.cost == ch_res.cost

    # Select winner by query performance: fewer nodes settled, then fewer relaxations, then faster query time
    d_score = (d_res.counters.nodes_settled, d_res.counters.relaxations, d_res.counters.query_time)
    ch_score = (ch_res.counters.nodes_settled, ch_res.counters.relaxations, ch_res.counters.query_time)

    if not distances_match:
        winner = "none"
        reason = f"distance mismatch (Dijkstra={d_res.cost}, CH={ch_res.cost})"
    elif ch_score < d_score:
        winner = "ch"
        reason = (
            f"fewer query operations ({ch_res.counters.nodes_settled} vs "
            f"{d_res.counters.nodes_settled} nodes settled)"
        )
    elif d_score < ch_score:
        winner = "dijkstra"
        reason = (
            f"fewer query operations ({d_res.counters.nodes_settled} vs "
            f"{ch_res.counters.nodes_settled} nodes settled)"
        )
    else:
        winner = "tie"
        reason = "identical query operations and time"

    return ComparisonResult(
        dijkstra=d_res,
        ch=ch_res,
        distances_match=distances_match,
        winner=winner,
        reason=reason,
    )


def format_path_result(result: PathResult) -> str:
    """Format a single PathResult and its metrics for CLI output."""
    cost_str = "INF" if result.cost == inf else str(result.cost)
    path_str = "NONE (unreachable)" if result.path is None else " -> ".join(map(str, result.path))
    c = result.counters
    return (
        f"Algorithm:          {result.algorithm}\n"
        f"Route:              {result.source} -> {result.target}\n"
        f"Cost:               {cost_str}\n"
        f"Path:               {path_str}\n"
        f"Nodes Settled:      {c.nodes_settled}\n"
        f"Relaxations:        {c.relaxations}\n"
        f"PQ Ops:             {c.pq_ops}\n"
        f"Preprocessing Time: {c.preprocessing_time:.6f}s\n"
        f"Original Edges:     {c.original_edges}\n"
        f"CH Shortcuts:       {c.ch_shortcuts}\n"
        f"CH Stored Edges:    {c.ch_edges}\n"
        f"CH Query Edges:     {c.ch_query_edges}\n"
        f"Query Time:         {c.query_time:.6f}s"
    )


def format_comparison(comp: ComparisonResult) -> str:
    """Format a side-by-side comparison of Dijkstra and CH."""
    match_str = "MATCH" if comp.distances_match else "MISMATCH"
    return (
        f"=== Dijkstra ===\n{format_path_result(comp.dijkstra)}\n\n"
        f"=== Contraction Hierarchies ===\n{format_path_result(comp.ch)}\n\n"
        f"Cost Agreement: {match_str}\n"
        f"Winner:         {comp.winner} ({comp.reason})"
    )
