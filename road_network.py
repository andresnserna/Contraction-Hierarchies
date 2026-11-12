"""Road-network data model and TSV loader for COSC 4367 Project 2."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Node:
    """TODO: add function doc here"""

    id: int
    name: str
    kind: str
    pass_time: int


@dataclass(frozen=True)
class Road:
    """TODO: add function doc here"""

    id: int
    u: int
    v: int
    minutes: int
    capacity: int
    oneway: bool


@dataclass(frozen=True)
class Arc:
    """TODO: add function doc here"""

    source: int
    target: int
    weight: int
    road_id: int


@dataclass
class RoadNetwork:
    """TODO: add function doc here"""

    nodes: dict[int, Node]
    roads: tuple[Road, ...]

    @classmethod
    def load(cls, nodes_file: str | Path, edges_file: str | Path) -> RoadNetwork:
        """Load the network from separate node and edge TSV files."""
        nodes_path = Path(nodes_file)
        roads_path = Path(edges_file)

        nodes: dict[int, Node] = {}
        with nodes_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter="\t")
            cls._require_columns(reader.fieldnames, {"id", "name", "kind", "pass_time"}, nodes_path)
            for line_number, row in enumerate(reader, start=2):
                try:
                    node_id = int(row["id"])
                    node = Node(
                        id=node_id,
                        name=row["name"],
                        kind=row["kind"],
                        pass_time=int(row["pass_time"]),
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{nodes_path}:{line_number}: invalid node row") from error
                if node_id in nodes:
                    raise ValueError(f"{nodes_path}:{line_number}: duplicate node id {node_id}")
                if not node.name or node.kind not in {"junction", "zone", "shelter"}:
                    raise ValueError(f"{nodes_path}:{line_number}: invalid name or node kind")
                if node.pass_time < 0:
                    raise ValueError(f"{nodes_path}:{line_number}: pass_time cannot be negative")
                if node.kind != "junction" and node.pass_time != 0:
                    raise ValueError(
                        f"{nodes_path}:{line_number}: only junctions may have pass_time"
                    )
                nodes[node_id] = node

        roads: list[Road] = []
        seen_roads: set[tuple[int, int, bool]] = set()
        with roads_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter="\t")
            cls._require_columns(
                reader.fieldnames, {"u", "v", "minutes", "capacity", "oneway"}, roads_path
            )
            for line_number, row in enumerate(reader, start=2):
                try:
                    u = int(row["u"])
                    v = int(row["v"])
                    minutes = int(row["minutes"])
                    capacity = int(row["capacity"])
                    oneway_value = int(row["oneway"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{roads_path}:{line_number}: invalid road row") from error
                if u not in nodes or v not in nodes:
                    raise ValueError(f"{roads_path}:{line_number}: road references an unknown node")
                if u == v or minutes <= 0 or capacity < 0 or oneway_value not in {0, 1}:
                    raise ValueError(f"{roads_path}:{line_number}: invalid road values")
                oneway = bool(oneway_value)
                key = (u, v, oneway) if oneway else (min(u, v), max(u, v), False)
                if key in seen_roads:
                    raise ValueError(f"{roads_path}:{line_number}: duplicate road {u}, {v}")
                seen_roads.add(key)
                roads.append(
                    Road(
                        id=len(roads),
                        u=u,
                        v=v,
                        minutes=minutes,
                        capacity=capacity,
                        oneway=oneway,
                    )
                )

        if not nodes:
            raise ValueError(f"{nodes_path}: no nodes found")
        return cls(nodes=nodes, roads=tuple(roads))

    @staticmethod
    def _require_columns(fieldnames: list[str] | None, required: set[str], path: Path) -> None:
        """TODO: add function doc here"""
        actual = set(fieldnames or ())
        missing = required - actual
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(sorted(missing))}")

    @property
    def zones(self) -> tuple[int, ...]:
        """TODO: add function doc here"""
        return tuple(sorted(node.id for node in self.nodes.values() if node.kind == "zone"))

    @property
    def shelters(self) -> tuple[int, ...]:
        """TODO: add function doc here"""
        return tuple(sorted(node.id for node in self.nodes.values() if node.kind == "shelter"))

    def adjacency(
        self,
        *,
        reverse: bool = False,
        excluded_road_ids: frozenset[int] = frozenset(),
        excluded_arcs: frozenset[tuple[int, int, int]] = frozenset(),
    ) -> dict[int, list[Arc]]:
        """Build adjacency lists, charging junction pass_time on departure."""

        result: dict[int, list[Arc]] = {node_id: [] for node_id in self.nodes}

        for road in self.roads:
            if road.id in excluded_road_ids:
                continue

            arcs = [(road.u, road.v)]

            if not road.oneway:
                arcs.append((road.v, road.u))

            for source, target in arcs:
                if (road.id, source, target) in excluded_arcs:
                    continue
                weight = road.minutes + self.nodes[source].pass_time
                if reverse:
                    source, target = target, source
                result[source].append(Arc(source, target, weight, road.id))

        for arcs in result.values():
            arcs.sort(key=lambda arc: (arc.target, arc.road_id))

        return result

    def find_road(self, u: int, v: int) -> Road:
        """Find a road that provides the directed arc u -> v."""
        for road in self.roads:
            if (road.u, road.v) == (u, v) or (
                not road.oneway and (road.u, road.v) == (v, u)
            ):
                return road
        raise ValueError(f"no road connects {u} to {v} in that direction")
