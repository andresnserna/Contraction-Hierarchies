"""Tests for graph generators, shortest paths, and shipped road instances."""

from __future__ import annotations

import csv
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from math import inf
from pathlib import Path

from data import gen_graphs
from graph import Graph
from project_cli import main as cli_main
from graph_services import all_pairs, critical_closure, evacuations
from road_network import Node, Road, RoadNetwork

from contraction_hierarchies import ContractionHierarchy
from graph_services import Counters, _dijkstra_graph


INSTANCE = Path(__file__).parent / "data" / "instances" / "tiny"
NODES = INSTANCE / "nodes.tsv"
EDGES = INSTANCE / "edges.tsv"


class GraphGeneratorTests(unittest.TestCase):
    """Check each generated topology in directed and undirected form."""

    def check_graph(self, generator, n, directed, **kwargs):
        """Validate weights, reproducibility, and all-pairs shortest paths."""
        nodes, edges = generator(n, directed=directed, seed=7, **kwargs)
        self.assertEqual((nodes, edges), generator(n, directed=directed, seed=7, **kwargs))
        count = n * n if generator is gen_graphs.grid_graph else n
        self.assertEqual(nodes, set(range(count)))

        graph = Graph()
        matrix = Weighted_Graph(count)
        for node in sorted(nodes):
            graph.add_node(node)
        for (u, v), attributes in edges.items():
            self.assertIn(u, nodes)
            self.assertIn(v, nodes)
            self.assertNotEqual(u, v)
            weight = attributes["weight"]
            self.assertIsInstance(weight, int)
            self.assertGreaterEqual(weight, 1)
            self.assertLessEqual(weight, 20)
            if not directed:
                self.assertEqual(edges[(v, u)]["weight"], weight)
            graph.add_edge(u, v, weight)
            matrix.add_directed_edge(u, v, weight)

        expected, _ = floyd_warshall(matrix, verbose=False)
        distances = {}
        for source in sorted(nodes):
            distances[source], _ = graph.dijkstra(source)
            self.assertEqual(distances[source], dict(enumerate(expected[source])))
        return nodes, edges, distances

    def test_erdos_renyi(self):
        """Probability extremes give empty/full graphs; intermediate p is sparse."""
        n = 12
        for directed in (True, False):
            for p in (0, 0.3, 1):
                with self.subTest(directed=directed, p=p):
                    _, edges, distances = self.check_graph(
                        gen_graphs.erdos_renyi, n, directed, p=p,
                    )
                    maximum = n * (n - 1) // (2 if directed else 1)
                    if p == 0:
                        self.assertEqual(edges, {})
                        self.assertEqual(distances[0][n - 1], inf)
                    elif p == 1:
                        expected = {(u, v) for u in range(n) for v in range(n)
                                    if u < v or (not directed and u > v)}
                        self.assertEqual(set(edges), expected)
                    else:
                        self.assertGreater(len(edges), 0)
                        self.assertLess(len(edges), maximum)
                    if directed:
                        self.assertTrue(all(u < v for u, v in edges))

    def test_cycle(self):
        """A cycle wraps to node zero and reaches every node in either mode."""
        n = 8
        for directed in (True, False):
            with self.subTest(directed=directed):
                _, edges, distances = self.check_graph(gen_graphs.cycle, n, directed)
                expected = {(u, (u + 1) % n) for u in range(n)}
                if not directed:
                    expected |= {(v, u) for u, v in expected}
                self.assertEqual(set(edges), expected)
                self.assertTrue(all(d < inf for row in distances.values() for d in row.values()))

    def test_complete_graph(self):
        """Every distinct ordered pair has an edge, including asymmetric costs."""
        n = 8
        for directed in (True, False):
            with self.subTest(directed=directed):
                _, edges, _ = self.check_graph(gen_graphs.complete_graph, n, directed)
                self.assertEqual(set(edges), {(u, v) for u in range(n)
                                             for v in range(n) if u != v})
                if directed:
                    self.assertTrue(any(edges[(u, v)] != edges[(v, u)] for u, v in edges))

    def test_path_graph(self):
        """Only consecutive nodes connect; endpoint distance is their weight sum."""
        n = 8
        for directed in (True, False):
            with self.subTest(directed=directed):
                _, edges, distances = self.check_graph(gen_graphs.path_graph, n, directed)
                expected = {(u, u + 1) for u in range(n - 1)}
                if not directed:
                    expected |= {(v, u) for u, v in expected}
                self.assertEqual(set(edges), expected)
                cost = sum(edges[(u, u + 1)]["weight"] for u in range(n - 1))
                self.assertEqual(distances[0][n - 1], cost)
                self.assertEqual(distances[n - 1][0], inf if directed else cost)

    def test_random_trees(self):
        """Each new node has one earlier parent, so all nodes can reach the root."""
        n = 12
        for directed in (True, False):
            with self.subTest(directed=directed):
                _, edges, distances = self.check_graph(gen_graphs.random_trees, n, directed)
                self.assertEqual(len(edges), (n - 1) * (1 if directed else 2))
                for u in range(1, n):
                    self.assertEqual(len([v for a, v in edges if a == u and v < u]), 1)
                    self.assertLess(distances[u][0], inf)
                    self.assertEqual(distances[0][u] == inf, directed)

    def test_balanced_trees(self):
        """Children have the expected parent for binary and ternary trees."""
        n = 13
        for directed in (True, False):
            for degree in (2, 3):
                with self.subTest(directed=directed, degree=degree):
                    _, edges, distances = self.check_graph(
                        gen_graphs.balanced_trees, n, directed, degree=degree,
                    )
                    expected = {((v - 1) // degree, v) for v in range(1, n)}
                    if not directed:
                        expected |= {(v, u) for u, v in expected}
                    self.assertEqual(set(edges), expected)
                    self.assertTrue(all(d < inf for d in distances[0].values()))
                    self.assertEqual(distances[n - 1][0] == inf, directed)

    def test_grid_graph(self):
        """An n-by-n grid connects adjacent cells without wrapping rows."""
        n = 3
        for directed in (True, False):
            with self.subTest(directed=directed):
                _, edges, distances = self.check_graph(gen_graphs.grid_graph, n, directed)
                expected = {(r * n + c, r * n + c + 1)
                            for r in range(n) for c in range(n - 1)}
                expected |= {(r * n + c, (r + 1) * n + c)
                             for r in range(n - 1) for c in range(n)}
                if not directed:
                    expected |= {(v, u) for u, v in expected}
                self.assertEqual(set(edges), expected)
                self.assertTrue(all(d < inf for d in distances[0].values()))
                self.assertEqual(distances[n * n - 1][0] == inf, directed)


class ProjectQueryTests(unittest.TestCase):
    """TODO: add function doc here"""

    @classmethod
    def setUpClass(cls) -> None:
        """TODO: add function doc here"""
        cls.network = RoadNetwork.load(NODES, EDGES)

    def test_load(self) -> None:
        """TODO: add function doc here"""
        self.assertEqual(len(self.network.nodes), 20)
        self.assertEqual(len(self.network.roads), 26)
        self.assertEqual(sum(not road.oneway for road in self.network.roads), 21)
        self.assertEqual(sum(road.oneway for road in self.network.roads), 5)
        self.assertEqual(len(self.network.zones), 5)
        self.assertEqual(len(self.network.shelters), 3)
        self.assertEqual(sum(node.pass_time > 0 for node in self.network.nodes.values()), 2)

    def test_ch_distances_match_dijkstra_for_all_pairs(self) -> None:
        """Compare CH and Dijkstra distances for every node pair on test graphs."""
        # (Due at checkpoint)
        dijkstra_apsp = all_pairs(self.network, algorithm="dijkstra")
        ch = ContractionHierarchy(self.network)
        ch.preprocess()
        
        for source in self.network.nodes:
            # Grab the pass_time of the starting node
            source_cost = self.network.nodes[source].pass_time
            
            for target in self.network.nodes:
                if source == target:
                    continue
                    
                ch_distance, _ = ch.shortest_path(source, target)
                
                # APSP strips the source node's pass_time, so we must strip it from ch_distance too
                adjusted_ch = ch_distance - source_cost if ch_distance != inf else inf
                
                self.assertEqual(
                    adjusted_ch, 
                    dijkstra_apsp[source].get(target, inf),
                    f"Mismatch {source}->{target}: CH={adjusted_ch}, APSP={dijkstra_apsp[source].get(target, inf)}"
                )

    def test_ch_paths_are_valid_and_have_dijkstra_distance(self) -> None:
        """Verify CH paths are valid routes with the same shortest distance as Dijkstra."""
        # (Due at checkpoint)
        ch = ContractionHierarchy(self.network)
        ch.preprocess()
        adjacency = self.network.adjacency()
        
        for source in self.network.nodes:
            for target in self.network.nodes:
                ch_distance, ch_path = ch.shortest_path(source, target)
                
                if ch_distance == inf:
                    self.assertIsNone(ch_path)
                    continue
                
                self.assertIsNotNone(ch_path)
                self.assertEqual(ch_path[0], source)
                self.assertEqual(ch_path[-1], target)
                
                # Sum the physical edge weights to prove the path matches the distance
                path_weight = 0
                for i in range(len(ch_path) - 1):
                    u, v = ch_path[i], ch_path[i + 1]
                    arcs = adjacency.get(u, [])
                    path_weight += min((arc.weight for arc in arcs if arc.target == v), default=inf)
                
                self.assertEqual(path_weight, ch_distance)

    def test_ch_shortcuts_unpack_to_original_paths(self) -> None:
        """Verify shortcut-based CH routes unpack into valid original-road paths."""
        # (Due at checkpoint)
        ch = ContractionHierarchy(self.network)
        ch.preprocess()
        adjacency = self.network.adjacency()
        
        for source in self.network.nodes:
            for target in self.network.nodes:
                _, ch_path = ch.shortest_path(source, target)
                if ch_path is None:
                    continue
                    
                # Verify every step in the path exists directly in the original network
                for i in range(len(ch_path) - 1):
                    u, v = ch_path[i], ch_path[i + 1]
                    is_neighbor = any(arc.target == v for arc in adjacency.get(u, []))
                    self.assertTrue(is_neighbor, f"Invalid step {u} -> {v} not in original network.")

    def test_dijkstra_and_ch_counters_match_known_graph(self) -> None:
        """Check Dijkstra and CH operation counters against hand-computed totals."""
        # (Due at checkpoint)
        # Using a custom 3-node line graph so we can hand-compute the exact expected counter totals
        from graph_services import _dijkstra_graph, Counters
        
        network = RoadNetwork(
            nodes={
                0: Node(0, "A", "junction", 0),
                1: Node(1, "B", "junction", 0),
                2: Node(2, "C", "junction", 0),
            },
            roads=(
                Road(0, 0, 1, 5, 10, True),
                Road(1, 1, 2, 5, 10, True),
            ),
        )
        
        # Test Dijkstra
        graph, _ = _dijkstra_graph(network)
        d_counter = Counters()
        graph.dijkstra(0, counter=d_counter)
        
        self.assertEqual(d_counter.nodes_settled, 3) # 0, 1, and 2 are settled
        self.assertEqual(d_counter.relaxations, 2)   # 0->1 and 1->2 are relaxed
        
        # Test CH
        ch_counter = Counters()
        ch = ContractionHierarchy(network)
        ch.preprocess(counter=ch_counter)
        
        self.assertGreater(ch_counter.preprocessing_time, 0.0)
        
        query_counter = Counters()
        ch.shortest_path(0, 2, counter=query_counter)
        self.assertGreater(query_counter.nodes_settled, 0)
        self.assertGreater(query_counter.query_time, 0.0)

    def test_verbose_summary_goes_to_stderr(self) -> None:
        """TODO: redo this test impl"""
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = cli_main([
                # "--nodes", str(NODES), "--edges", str(EDGES),
                # "--query", "evac", "--verbose",
            ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(stdout.getvalue().splitlines()), len(self.network.zones))
        self.assertIn("V=20 roads=26 arcs=47", stderr.getvalue())
        self.assertRegex(stderr.getvalue(), r"relaxations=\d+ PQ_Ops=\d+")
        self.assertNotIn("V=20", stdout.getvalue())

    def test_cli_rejects_incompatible_flags(self) -> None:
        """TODO: redo this test impl"""
        invalid_args = (
            # ["--nodes", str(NODES), "--edges", str(EDGES), "--query", "evac", "--algo", "dijkstra"],
            # ["--nodes", str(NODES), "--edges", str(EDGES), "--query", "critical", "--close", "6", "7"],
            # ["--nodes", str(NODES), "--edges", str(EDGES), "--query", "evac", "--close", "6", "999"],
        )
        for argv in invalid_args:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli_main(argv)
                self.assertEqual(error.exception.code, 2)

        with tempfile.TemporaryDirectory() as directory:
            nodes_path = Path(directory) / "nodes.tsv"
            edges_path = Path(directory) / "edges.tsv"
            nodes_path.write_text("id\tname\tkind\n0\tZone\tzone\n", encoding="utf-8")
            edges_path.write_text("u\tv\tminutes\tcapacity\toneway\n", encoding="utf-8")
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli_main([
                        "--nodes", str(nodes_path), "--edges", str(edges_path),
                        "--query", "evac",
                    ])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
