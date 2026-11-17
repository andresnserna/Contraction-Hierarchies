from __future__ import annotations
from dataclasses import dataclass
from math import inf
from threading import Event
from Floyd_Warshall import Weighted_Graph, floyd_warshall
from graph import Graph
from road_network import RoadNetwork
from time import perf_counter


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
    stop_event: Event | None = None 

    def check_timeout(self) -> None:
        """Raises TimeoutError if the stop event has been triggered."""
        if self.stop_event is not None and self.stop_event.is_set():
            raise TimeoutError("experiment run reached its stop limit")

    def add(self, other: Counters) -> None:
        """Combines metrics from another Counters instance into this one."""
        self.relaxations += other.relaxations
        self.nodes_settled += other.nodes_settled
        self.preprocessing_time += other.preprocessing_time
        self.query_time += other.query_time
        self.pq_ops += other.pq_ops


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

# TODO: Add find_path_ch(network, u, v, counter) and find_path_dijkstra(...)
# here; both should return a consistent path/cost/metrics result for formatting.
# TODO: Add compare_algorithms(network, u, v, ...) here to run both algorithms
# and select a winner using the cost and operation metrics described in the plan.
