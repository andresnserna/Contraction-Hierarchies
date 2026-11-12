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
from Floyd_Warshall import Weighted_Graph, floyd_warshall
from graph import Graph
from project_cli import main as cli_main
from graph_services import all_pairs, critical_closure, evacuations
from road_network import Node, Road, RoadNetwork


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
    """TO-DO: add function doc here"""

    @classmethod
    def setUpClass(cls) -> None:
        """TO-DO: add function doc here"""
        cls.network = RoadNetwork.load(NODES, EDGES)

    def test_load(self) -> None:
        """TO-DO: add function doc here"""
        self.assertEqual(len(self.network.nodes), 20)
        self.assertEqual(len(self.network.roads), 26)
        self.assertEqual(sum(not road.oneway for road in self.network.roads), 21)
        self.assertEqual(sum(road.oneway for road in self.network.roads), 5)
        self.assertEqual(len(self.network.zones), 5)
        self.assertEqual(len(self.network.shelters), 3)
        self.assertEqual(sum(node.pass_time > 0 for node in self.network.nodes.values()), 2)

    def test_evac(self) -> None:
        """TO-DO: add function doc here"""
        expected = {}
        with (INSTANCE / "expected.evac.tsv").open(encoding="utf-8") as file:
            for zone, shelter, minutes in csv.reader(file, delimiter="\t"):
                expected[zone] = (shelter, int(minutes))

        actual = {
            self.network.nodes[row.zone_id].name: (
                self.network.nodes[row.shelter_id].name if row.shelter_id is not None else "NONE",
                row.minutes,
            )
            for row in evacuations(self.network)
        }
        self.assertEqual(actual, expected)

    def test_apsp_algorithms(self) -> None:
        """TO-DO: add function doc here"""
        expected = {}

        with (INSTANCE / "expected.apsp.tsv").open(encoding="utf-8") as file:
            for source, target, minutes in csv.reader(file, delimiter="\t"):
                expected[(int(source), int(target))] = int(minutes)
                
        for algorithm in ("dijkstra", "bellman-ford", "johnson"):
            result = all_pairs(self.network, algorithm=algorithm)
            actual = {
                (source, target): result[source][target]
                for source in sorted(result)
                for target in sorted(result[source])
                if source != target
            }
            self.assertEqual(actual, expected)

    def test_floyd_warshall(self) -> None:
        """TO-DO: add function doc here"""
        dijkstra_result = all_pairs(self.network, algorithm="dijkstra")
        floyd_result = all_pairs(self.network, algorithm="floyd-warshall")
        self.assertEqual(floyd_result, dijkstra_result)

    def test_cli_outputs(self) -> None:
        """Run the actual CLI on both shipped TSV datasets and compare its rows with the expected shelter assignments and all-pairs travel times."""
        for instance_name in ("tiny", "small"):
            with self.subTest(instance=instance_name):
                instance = Path(__file__).parent / "data" / "instances" / instance_name
                nodes_path = instance / "nodes.tsv"
                edges_path = instance / "edges.tsv"

                for query, expected_name in (
                    ("evac", "expected.evac.tsv"),
                    ("apsp", "expected.apsp.tsv"),
                ):
                    with self.subTest(query=query):
                        expected = (instance / expected_name).read_text(
                            encoding="utf-8"
                        ).splitlines()
                        output = io.StringIO()
                        with redirect_stdout(output):
                            exit_code = cli_main([
                                "--nodes", str(nodes_path),
                                "--edges", str(edges_path),
                                "--query", query,
                            ])

                        self.assertEqual(exit_code, 0)
                        self.assertEqual(output.getvalue().splitlines(), expected)

    def test_critical(self) -> None:
        """TO-DO: add function doc here"""
        with (INSTANCE / "expected.critical.tsv").open(encoding="utf-8") as file:
            expected = next(csv.reader(file, delimiter="\t"))
        result = critical_closure(self.network)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual([result.u, result.v, result.worst_minutes], [int(x) for x in expected])

    def test_unreachable(self) -> None:
        """TO-DO: add function doc here"""
        network = RoadNetwork(nodes=self.network.nodes, roads=())
        result = evacuations(network)
        self.assertTrue(any(row.minutes == inf for row in result))

    def test_route_tie(self) -> None:
        """TO-DO: add function doc here"""
        network = RoadNetwork(
            nodes={
                0: Node(0, "Zone", "zone", 0),
                1: Node(1, "First", "shelter", 0),
                2: Node(2, "Second", "shelter", 0),
            },
            roads=(
                Road(0, 0, 1, 4, 10, True),
                Road(1, 0, 2, 4, 10, True),
            ),
        )
        result = evacuations(network)
        self.assertEqual((result[0].shelter_id, result[0].minutes), (1, 4))

    def test_close(self) -> None:
        """
        Verify --close removes only the requested arc of a two-way road.
        """
        network = RoadNetwork.load(NODES, EDGES)
        road = network.find_road(6, 7)
        self.assertFalse(road.oneway)

        baseline = network.adjacency()
        self.assertIn(7, {arc.target for arc in baseline[6]})
        self.assertIn(6, {arc.target for arc in baseline[7]})

        closed = network.adjacency(excluded_arcs=frozenset({(road.id, 6, 7)}))
        self.assertNotIn(7, {arc.target for arc in closed[6]})
        self.assertIn(6, {arc.target for arc in closed[7]})

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = cli_main([
                "--nodes", str(NODES), "--edges", str(EDGES),
                "--query", "evac", "--close", "6", "7",
            ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "Riverside\tJefferson_Elem\t53",
                "Millbrook\tJefferson_Elem\t41",
                "Eastgate\tRiverside_HS\t19",
                "Harbor_Flats\tJefferson_Elem\t28",
                "Old_Town\tCivic_Center\t7",
            ],
        )

    def test_critical_tie_with_inf_prefers_lower_edge_ids(self) -> None:
        """If multiple shortest-path edges produce INF, choose the lower (u, v)."""
        network = RoadNetwork(
            nodes={
                0: Node(0, "Zone", "zone", 0),
                1: Node(1, "Junction", "junction", 0),
                3: Node(3, "Shelter", "shelter", 0),
            },
            roads=(
                Road(0, 0, 1, 2, 10, True),
                Road(1, 1, 3, 3, 10, True),
            ),
        )

        result = critical_closure(network)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual((result.u, result.v), (0, 1))
        self.assertEqual(result.worst_minutes, inf)

        with tempfile.TemporaryDirectory() as directory:
            nodes_path = Path(directory) / "nodes.tsv"
            edges_path = Path(directory) / "edges.tsv"
            nodes_path.write_text(
                "id\tname\tkind\tpass_time\n"
                "0\tZone\tzone\t0\n"
                "1\tJunction\tjunction\t0\n"
                "3\tShelter\tshelter\t0\n",
                encoding="utf-8",
            )
            edges_path.write_text(
                "u\tv\tminutes\tcapacity\toneway\n"
                "0\t1\t2\t10\t1\n"
                "1\t3\t3\t10\t1\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = cli_main([
                    "--nodes", str(nodes_path), "--edges", str(edges_path),
                    "--query", "critical",
                ])
            self.assertEqual(exit_code, 0)
            self.assertEqual(output.getvalue().strip(), "0\t1\tINF")

    def test_counters_file_and_non_pq_algorithms_report_zero(self) -> None:
        """Bellman-Ford and Floyd-Warshall write zero priority-queue operations."""
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                for algorithm in ("bellman-ford", "floyd-warshall"):
                    with self.subTest(algorithm=algorithm):
                        with redirect_stdout(io.StringIO()):
                            exit_code = cli_main([
                                "--nodes", str(NODES), "--edges", str(EDGES),
                                "--query", "apsp", "--algo", algorithm, "--counters",
                            ])
                        self.assertEqual(exit_code, 0)
                        counters = Path("counters.txt").read_text(encoding="utf-8")
                        self.assertRegex(counters, r"(?m)^Relaxations\t\d+$")
                        self.assertRegex(counters, r"(?m)^PQ_Ops\t0$")
            finally:
                os.chdir(original_cwd)

    def test_verbose_summary_goes_to_stderr(self) -> None:
        """TO-DO: add function doc here"""
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = cli_main([
                "--nodes", str(NODES), "--edges", str(EDGES),
                "--query", "evac", "--verbose",
            ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(stdout.getvalue().splitlines()), len(self.network.zones))
        self.assertIn("V=20 roads=26 arcs=47", stderr.getvalue())
        self.assertRegex(stderr.getvalue(), r"relaxations=\d+ PQ_Ops=\d+")
        self.assertNotIn("V=20", stdout.getvalue())

    def test_cli_rejects_incompatible_flags_invalid_ids_and_bad_input(self) -> None:
        """TO-DO: add function doc here"""
        invalid_args = (
            ["--nodes", str(NODES), "--edges", str(EDGES), "--query", "evac", "--algo", "dijkstra"],
            ["--nodes", str(NODES), "--edges", str(EDGES), "--query", "critical", "--close", "6", "7"],
            ["--nodes", str(NODES), "--edges", str(EDGES), "--query", "evac", "--close", "6", "999"],
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
