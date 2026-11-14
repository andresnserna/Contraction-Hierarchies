"""Contraction Hierarchies preprocessing and point-to-point routing."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import inf
from typing import TYPE_CHECKING

from road_network import RoadNetwork

if TYPE_CHECKING:
    from graph_services import Counters


@dataclass(frozen=True)
class _CHEdge:
    """A directed CH edge; child edges retain the route represented by a shortcut."""

    source: int
    target: int
    weight: int
    middle_node: int | None = None
    left: _CHEdge | None = None
    right: _CHEdge | None = None


class ContractionHierarchy:
    """Preprocess a road network for repeated shortest-path queries."""

    def __init__(self, network: RoadNetwork) -> None:
        self.network = network
        self.contracted_graph: dict[int, dict[int, _CHEdge]] = {
            node_id: {} for node_id in network.nodes
        }
        self.rank: dict[int, int] = {}
        self.node_order: list[int] = []
        self.forward_upward: dict[int, dict[int, _CHEdge]] = {
            node_id: {} for node_id in network.nodes
        }
        # Edges are reversed here so a search from the target follows incoming
        # original-graph edges while still using ordinary outgoing adjacency.
        self.backward_upward: dict[int, dict[int, _CHEdge]] = {
            node_id: {} for node_id in network.nodes
        }
        self.shortcuts_added = 0
        self.is_preprocessed = False
        self._contracted: set[int] = set()

    def _build_contracted_graph(self) -> None:
        """Load original directed arcs and their effective travel-time weights."""
        self.contracted_graph = {node_id: {} for node_id in self.network.nodes}
        for source, arcs in self.network.adjacency().items():
            for arc in arcs:
                edge = _CHEdge(source, arc.target, arc.weight)
                old = self.contracted_graph[source].get(arc.target)
                if old is None or edge.weight < old.weight:
                    self.contracted_graph[source][arc.target] = edge

    def preprocess(self, counter: Counters | None = None) -> None:
        """Build the node hierarchy, shortcuts, and upward query graphs."""
        self._build_contracted_graph()
        self.rank.clear()
        self._contracted.clear()
        self.shortcuts_added = 0
        self.node_order = self.order_nodes()

        for rank, node_id in enumerate(self.node_order):
            self._contract_node(node_id, rank)

        self._build_upward_graphs()
        self.is_preprocessed = True

    def order_nodes(self, counter: Counters | None = None) -> list[int]:
        """Sort nodes by outgoing degree minus incoming degree in the road graph."""
        adjacency = self.network.adjacency()
        out_degree = {node: len(adjacency[node]) for node in self.network.nodes}
        in_degree = {node: 0 for node in self.network.nodes}
        for arcs in adjacency.values():
            for arc in arcs:
                in_degree[arc.target] += 1
        return sorted(
            self.network.nodes,
            key=lambda node: (out_degree[node] - in_degree[node], node),
        )

    def _edge_difference(self, node_id: int, counter: Counters | None = None) -> int:
        """Count required shortcuts minus incident active edges for a node."""
        incoming = [
            edge for source in self.contracted_graph
            if source not in self._contracted and source != node_id
            for target, edge in self.contracted_graph[source].items()
            if target == node_id
        ]
        outgoing = [
            edge for target, edge in self.contracted_graph[node_id].items()
            if target not in self._contracted and target != node_id
        ]
        shortcuts = 0
        for left in incoming:
            for right in outgoing:
                if left.source == right.target:
                    continue
                candidate = left.weight + right.weight
                if self._witness_search(
                    left.source, right.target, node_id, candidate
                ) > candidate:
                    shortcuts += 1
        return shortcuts - len(incoming) - len(outgoing)

    def _contract_node(
        self, node_id: int, rank: int, counter: Counters | None = None
    ) -> None:
        """Find required shortcuts through one node, then mark it contracted."""
        incoming = [
            edge for source in self.contracted_graph
            if source not in self._contracted and source != node_id
            for target, edge in self.contracted_graph[source].items()
            if target == node_id
        ]
        outgoing = [
            edge for target, edge in self.contracted_graph[node_id].items()
            if target not in self._contracted and target != node_id
        ]
        for left in incoming:
            for right in outgoing:
                if left.source == right.target:
                    continue
                candidate = left.weight + right.weight
                if self._witness_search(left.source, right.target, node_id, candidate) > candidate:
                    if self._add_shortcut(
                        left.source, right.target, candidate, node_id, left, right
                    ):
                        self.shortcuts_added += 1
        self.rank[node_id] = rank
        self._contracted.add(node_id)

    def _witness_search(
        self,
        source: int,
        target: int,
        avoided_node: int,
        max_distance: float,
        counter: Counters | None = None,
    ) -> float:
        """Find shortest uncontracted path avoiding a node, stopping above the candidate cost."""
        if source == avoided_node or target == avoided_node:
            return inf
        distances = {source: 0}
        heap = [(0, source)]
        settled: set[int] = set()
        while heap:
            distance, node = heapq.heappop(heap)
            if node in settled or distance != distances.get(node):
                continue
            # Since heap keys are ordered, no remaining route can be a witness
            # once its cheapest unsettled prefix already exceeds the candidate.
            if distance > max_distance:
                break
            if node == target:
                return distance
            settled.add(node)
            for neighbor, edge in self.contracted_graph[node].items():
                if neighbor == avoided_node or neighbor in self._contracted or neighbor in settled:
                    continue
                new_distance = distance + edge.weight
                if new_distance <= max_distance and new_distance < distances.get(neighbor, inf):
                    distances[neighbor] = new_distance
                    heapq.heappush(heap, (new_distance, neighbor))
        return inf

    def _add_shortcut(
        self, source: int, target: int, weight: int, middle_node: int,
        left: _CHEdge, right: _CHEdge,
    ) -> bool:
        """Store a cheaper shortcut and return whether the graph was updated."""
        old = self.contracted_graph[source].get(target)
        if old is not None and old.weight <= weight:
            return False
        self.contracted_graph[source][target] = _CHEdge(
            source, target, weight, middle_node, left, right
        )
        return True

    def _build_upward_graphs(self) -> None:
        """Split contracted graph into forward and reversed-backward upward edges."""
        self.forward_upward = {node: {} for node in self.network.nodes}
        self.backward_upward = {node: {} for node in self.network.nodes}
        for source, edges in self.contracted_graph.items():
            for target, edge in edges.items():
                if self.rank[source] < self.rank[target]:
                    self.forward_upward[source][target] = edge
                elif self.rank[source] > self.rank[target]:
                    self.backward_upward[target][source] = _CHEdge(
                        target, source, edge.weight, edge.middle_node, edge.left, edge.right
                    )

    def _upward_search(
        self,
        source: int,
        adjacency: dict[int, dict[int, _CHEdge]],
        counter: Counters | None = None,
    ) -> tuple[dict[int, float], dict[int, _CHEdge]]:
        """Run Dijkstra on an upward query graph and retain predecessor edges."""
        distances = {node: inf for node in adjacency}
        parent: dict[int, _CHEdge] = {}
        distances[source] = 0
        heap = [(0, source)]
        while heap:
            distance, node = heapq.heappop(heap)
            if distance != distances[node]:
                continue
            for target, edge in adjacency[node].items():
                candidate = distance + edge.weight
                if candidate < distances[target]:
                    distances[target] = candidate
                    parent[target] = edge
                    heapq.heappush(heap, (candidate, target))
        return distances, parent

    def shortest_path(
        self, source: int, target: int, counter: Counters | None = None
    ) -> tuple[float, list[int] | None]:
        """Return shortest distance and node path, or (inf, None) if unreachable."""
        if source not in self.network.nodes or target not in self.network.nodes:
            raise ValueError("Illegal node entered")
        if not self.is_preprocessed:
            self.preprocess()
        if source == target:
            return 0, [source]
        forward_dist, forward_parent = self._upward_search(
            source, self.forward_upward, counter
        )
        backward_dist, backward_parent = self._upward_search(
            target, self.backward_upward, counter
        )
        meeting = min(
            self.network.nodes,
            key=lambda node: (forward_dist[node] + backward_dist[node], node),
        )
        distance = forward_dist[meeting] + backward_dist[meeting]
        if distance == inf:
            return inf, None

        left_edges: list[_CHEdge] = []
        node = meeting
        while node != source:
            edge = forward_parent[node]
            left_edges.append(edge)
            node = edge.source
        left_edges.reverse()

        # backward_upward stores reversed traversal edges. Their original CH
        # edge runs from the current node toward the next node on the route.
        right_edges: list[_CHEdge] = []
        node = meeting
        while node != target:
            reverse_edge = backward_parent[node]
            original = self.contracted_graph[reverse_edge.target][reverse_edge.source]
            right_edges.append(original)
            node = reverse_edge.source

        path = [source]
        for edge in left_edges + right_edges:
            part = self._unpack_edge(edge)
            path.extend(part[1:])
        return distance, path

    def _unpack_edge(self, edge: _CHEdge) -> list[int]:
        """Expand a CH shortcut recursively into its original node sequence."""
        if edge.left is None or edge.right is None:
            return [edge.source, edge.target]
        left = self._unpack_edge(edge.left)
        right = self._unpack_edge(edge.right)
        return left + right[1:]
