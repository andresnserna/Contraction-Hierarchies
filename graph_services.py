from __future__ import annotations
from dataclasses import dataclass
from math import inf
from threading import Event
from Floyd_Warshall import Weighted_Graph, floyd_warshall
from graph import Graph
from road_network import RoadNetwork


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


@dataclass(frozen=True)
class Evacuation:
    """
    using @dataclass so i don't need an _init heheheheheheheh and frozen so the record is immutible. 
    This is a class im treating as a record holder (so data?), called upon by the _multi_source_evacuation() func
    """
    zone_id: int
    shelter_id: int | None
    minutes: float
    path: tuple[int, ...] | None


@dataclass(frozen=True)
class CriticalClosure:
    """
    a class that holds the data for the --critical query
    """
    road_id: int
    u: int
    v: int
    worst_minutes: float
    worst_zone_id: int | None


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


def _multi_source_evacuation(network: RoadNetwork, *, excluded_road_ids: frozenset[int] = frozenset(), excluded_arcs: frozenset[tuple[int, int, int]] = frozenset(), counter: Counters | None = None) -> tuple[
    tuple[Evacuation, ...],
    dict[int, int | None],
    dict[int, int | None],
]:    
    """TO-DO: add function doc here"""

    # precondition: If there are no shelters, there is no destination to route anyone to.
    if not network.shelters:
        raise ValueError("the instance has no shelters")

    # Reverse every road so one search out from the shelters can answer the original zone -> shelter question. Keep the road IDs too, since we need
    # them later when we rebuild each zone's route.
    graph, road_ids = _dijkstra_graph(
        network, reverse=True, excluded_road_ids=excluded_road_ids,
        excluded_arcs=excluded_arcs,
    )

    # Add a virtual source that represents starting at all shelters at once.
    # The ID is one bigger than any real node so it won't collide with one.
    virtual_node = max(graph.nodes, default=-1) + 1
    graph.add_node(virtual_node)

    # adding the 0 cost edges to that vNode
    for shelter_id in network.shelters:
        graph.add_edge(virtual_node, shelter_id, 0)

    # Run Dijkstra once. In this reversed graph, the distance from a zone back to a shelter is the original zone-to-shelter cost.
    distances, _ = graph.dijkstra(virtual_node, counter=counter)

    # Dijkstra gives us the cheapest distance, but not which shelter owns each
    # node or the original route to it. These maps hold that info as we rebuild it.
    owner: dict[int, int | None] = {node_id: None for node_id in graph.nodes}
    parent: dict[int, int | None] = {node_id: None for node_id in graph.nodes}
    parent_road: dict[int, int | None] = {node_id: None for node_id in graph.nodes}

    # A shelter owns itself, which gives the route-labeling step a starting point.
    for shelter_id in network.shelters:
        owner[shelter_id] = shelter_id

    # Save the incoming roads for each node. Since the graph is reversed, these are the next steps back toward a shelter in the original direction.
    incoming: dict[int, list[tuple[int, int, int]]] = {
        node_id: [] for node_id in graph.nodes
    }

    for source, neighbors in graph.nodes.items():
        if source == virtual_node:
            continue
        for target, weight in neighbors.items():
            road_id = road_ids[(source, target, weight)]
            incoming[target].append((source, weight, road_id))

    # Process closest nodes first. With positive road times, a node's best predecessor is always closer, so its shelter label is ready by this point.
    for node_id in sorted(
        (node for node in graph.nodes if node != virtual_node),
        key=lambda node: (distances[node], node),
    ):
        # Shelters already own themselves; unreachable nodes have no route to label.
        if owner[node_id] is not None or distances[node_id] == inf:
            continue
        options = []

        # Keep only roads that preserve the shortest distance. If there is a tie, min() below picks the lowest shelter ID, then a stable predecessor/road.
        for predecessor, weight, road_id in incoming[node_id]:
            predecessor_owner = owner[predecessor]
            if (
                predecessor_owner is not None
                and distances[predecessor] + weight == distances[node_id]
            ):
                options.append((predecessor_owner, predecessor, road_id))
        
        if options:
            owner[node_id], parent[node_id], parent_road[node_id] = min(options) #<<< when multiple shortest routes are possible, min(options) picks the option with the lowest shelter ID first

    # Build one result row per zone, including an explicit NONE result when no shelter can be reached.
    rows = []

    for zone_id in network.zones:
        shelter_id = owner[zone_id]
        distance = distances[zone_id]

        if shelter_id is None or distance == inf:
            rows.append(Evacuation(zone_id, None, inf, None))
            continue

        # Follow the saved original-direction parents from the zone to its shelter.
        path = [zone_id]
        current = zone_id

        while current != shelter_id:
            next_node = parent[current]
            if next_node is None:
                raise RuntimeError(f"broken reverse-search parent chain for zone {zone_id}")
            path.append(next_node)
            current = next_node

        rows.append(Evacuation(zone_id, shelter_id, distance, tuple(path)))

    return tuple(rows), parent, parent_road


def evacuations(network: RoadNetwork, *, excluded_road_ids: frozenset[int] = frozenset(), excluded_arcs: frozenset[tuple[int, int, int]] = frozenset(), counters: Counters | None = None) -> tuple[Evacuation, ...]:
    """
    Find each zone's cheapest route to a shelter with one reverse Dijkstra. this is a convenience wrapper for the query
    """

    result = _multi_source_evacuation(network, excluded_road_ids=excluded_road_ids, excluded_arcs=excluded_arcs, counter=counters)
    return result[0]


def naive_per_zone(network: RoadNetwork, *, excluded_road_ids: frozenset[int] = frozenset(), counters: Counters | None = None) -> tuple[Evacuation, ...]:
    """Find each zone's closest shelter by running Dijkstra once per zone."""
    if not network.shelters:
        raise ValueError("the instance has no shelters")

    # This is the simple version: start at each zone separately and see how far that one Dijkstra run can get to every shelter.
    graph, _ = _dijkstra_graph(network, excluded_road_ids=excluded_road_ids)
    rows = []
    for zone_id in network.zones:
        distances, parent = graph.dijkstra(zone_id, counter=counters)

        # Compare distance first, then shelter ID so ties go to the lower ID.
        shelter_id = min(network.shelters,
            key=lambda candidate: (distances[candidate], candidate),
        )
        distance = distances[shelter_id]

        if distance == inf:
            rows.append(Evacuation(zone_id, None, inf, None))
            continue

        # Parents point back toward the zone, so walk from shelter to zone and flip the list at the end to return the route in zone -> shelter order.
        path = []
        current: int | None = shelter_id

        while current is not None:
            path.append(current)

            if current == zone_id:
                break

            current = parent[current]

        if not path or path[-1] != zone_id:
            raise RuntimeError(f"broken Dijkstra parent chain for zone {zone_id}")
        
        path.reverse()
        rows.append(Evacuation(zone_id, shelter_id, distance, tuple(path)))

    return tuple(rows)


def naive_per_shelter(network: RoadNetwork, *, excluded_road_ids: frozenset[int] = frozenset(), counters: Counters | None = None) -> tuple[Evacuation, ...]:
    """Find each zone's closest shelter by running reverse Dijkstra per shelter."""
    if not network.shelters:
        raise ValueError("the instance has no shelters")

    # Flip the roads around, then each shelter can search outward to find which zones can reach it in the original graph.
    graph, _ = _dijkstra_graph(network, reverse=True, excluded_road_ids=excluded_road_ids)

    # Save each shelter's distances and parents so we can compare shelters zone
    # by zone afterward. The same counters object collects all of these runs.
    by_shelter = {shelter_id: graph.dijkstra(shelter_id, counter=counters) for shelter_id in network.shelters}

    rows = []
    for zone_id in network.zones:
        # Pick the closest shelter; the shelter ID breaks any distance tie.
        shelter_id = min(network.shelters,
            key=lambda candidate: (by_shelter[candidate][0][zone_id], candidate),
        )

        distances, parent = by_shelter[shelter_id]
        distance = distances[zone_id]

        if distance == inf:
            rows.append(Evacuation(zone_id, None, inf, None))
            continue

        # On this reversed graph, following parents from the zone goes back to the shelter, which is already the route direction we need to return.
        path = [zone_id]
        current = zone_id
        while current != shelter_id:
            current = parent[current]
            if current is None:
                raise RuntimeError(f"broken reverse-search parent chain for zone {zone_id}")
            path.append(current)
        rows.append(Evacuation(zone_id, shelter_id, distance, tuple(path)))

    return tuple(rows)


def all_pairs(network: RoadNetwork, *, algorithm: str = "dijkstra", counter: Counters | None = None) -> dict[int, dict[int, float]]:
    """
    Compute all-pairs travel times; endpoint junction costs are not charged.
    """

    if algorithm == "dijkstra":
        graph, _ = _dijkstra_graph(network)
        distances = {
            source: graph.dijkstra(source, counter=counter)[0]
            for source in sorted(network.nodes)
        }

    elif algorithm == "bellman-ford":
        graph, _ = _dijkstra_graph(network)
        distances = {}
        for source in sorted(network.nodes):
            result = graph.bellman_ford(source, counter=counter)
            if result is None:
                raise ValueError("Bellman-Ford found a negative cycle")
            distances[source] = result[0]

    elif algorithm == "johnson":
        graph = Graph()
        for node_id in sorted(network.nodes):
            graph.add_node(node_id)
        for source, arcs in network.adjacency().items():
            for arc in arcs:
                graph.add_edge(source, arc.target, arc.weight)
        result = graph.johnsons_apsp(counter=counter)
        if result is None:
            raise ValueError("Johnson's algorithm found a negative cycle")
        distances = result[0]

    elif algorithm == "floyd-warshall":
        node_ids = sorted(network.nodes)
        indexes = {node_id: index for index, node_id in enumerate(node_ids)}
        graph = Weighted_Graph(len(node_ids))
        for source, arcs in network.adjacency().items():
            for arc in arcs:
                graph.add_directed_edge(indexes[source], indexes[arc.target], arc.weight)
        matrix, _ = floyd_warshall(graph, verbose=False, counter=counter)
        distances = {
            source: {
                target: matrix[indexes[source]][indexes[target]]
                for target in node_ids
            }
            for source in node_ids
        }
        
    else:
        raise ValueError(f"unsupported APSP algorithm: {algorithm}")

    for source, row in distances.items():
        source_cost = network.nodes[source].pass_time
        for target, distance in row.items():
            if source != target and distance != inf:
                row[target] = distance - source_cost
    return distances


def benchmark_all_pairs(network: RoadNetwork, *, algorithm: str, counter: Counters) -> None:
    """Run an APSP algorithm while discarding results to bound benchmark memory."""
    if algorithm == "dijkstra":
        graph, _ = _dijkstra_graph(network)
        for source in sorted(network.nodes):
            counter.check_timeout()
            graph.dijkstra(source, counter=counter)
        return

    if algorithm == "johnson":
        graph = Graph()
        for node_id in sorted(network.nodes):
            graph.add_node(node_id)
        for source, arcs in network.adjacency().items():
            counter.check_timeout()
            for arc in arcs:
                graph.add_edge(source, arc.target, arc.weight)
        if graph.johnsons_apsp(counter=counter, store_results=False) is None:
            raise ValueError("Johnson's algorithm found a negative cycle")
        return

    if algorithm == "floyd-warshall":
        node_ids = sorted(network.nodes)
        indexes = {node_id: index for index, node_id in enumerate(node_ids)}
        graph = Weighted_Graph(len(node_ids), counter=counter)
        for source, arcs in network.adjacency().items():
            counter.check_timeout()
            for arc in arcs:
                graph.add_directed_edge(indexes[source], indexes[arc.target], arc.weight)
        floyd_warshall(graph, verbose=False, counter=counter)
        return

    raise ValueError(f"unsupported APSP algorithm: {algorithm}")


def critical_closure(network: RoadNetwork, *, counter: Counters | None = None) -> CriticalClosure | None:
    """
    Find the closure producing the largest post-closure time for any zone.
    """
    result = _multi_source_evacuation(network, counter=counter)
    baseline, parent, parent_road = result
    reachable = {row.zone_id for row in baseline if row.minutes != inf}

    if not reachable:
        return None

    candidates: set[tuple[int, int, int]] = set()

    for row in baseline:
        if row.zone_id not in reachable:
            continue

        current = row.zone_id

        while current != row.shelter_id:
            road_id = parent_road[current]
            next_node = parent[current]

            if road_id is None or next_node is None:
                raise RuntimeError(f"broken shortest-path forest for zone {row.zone_id}")
            
            candidates.add((road_id, current, next_node))
            current = next_node

    worst = None
    
    for road_id, u, v in candidates:
        changed = evacuations(
            network,
            excluded_arcs=frozenset({(road_id, u, v)}),
            counters=counter,
        )
        worst_zone = max(
            (row for row in changed if row.zone_id in reachable),
            key=lambda row: (row.minutes, -row.zone_id),
        )
        candidate = CriticalClosure(
            road_id=road_id,
            u=u,
            v=v,
            worst_minutes=worst_zone.minutes,
            worst_zone_id=worst_zone.zone_id,
        )
        if worst is None or (candidate.worst_minutes, -candidate.u, -candidate.v) > (
            worst.worst_minutes,
            -worst.u,
            -worst.v,
        ):
            worst = candidate
    return worst
