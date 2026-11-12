"""Focused tests against the shipped tiny instance and expected outputs."""

from __future__ import annotations

import csv
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from math import inf
from pathlib import Path

from project_cli import main as cli_main
from graph_services import all_pairs, critical_closure, evacuations
from road_network import Node, Road, RoadNetwork


INSTANCE = Path(__file__).parent / "data" / "instances" / "tiny"
NODES = INSTANCE / "nodes.tsv"
EDGES = INSTANCE / "edges.tsv"


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

    def test_evac(self) -> None:
        """TODO: add function doc here"""
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
        """TODO: add function doc here"""
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
        """TODO: add function doc here"""
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
        """TODO: add function doc here"""
        with (INSTANCE / "expected.critical.tsv").open(encoding="utf-8") as file:
            expected = next(csv.reader(file, delimiter="\t"))
        result = critical_closure(self.network)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual([result.u, result.v, result.worst_minutes], [int(x) for x in expected])

    def test_unreachable(self) -> None:
        """TODO: add function doc here"""
        network = RoadNetwork(nodes=self.network.nodes, roads=())
        result = evacuations(network)
        self.assertTrue(any(row.minutes == inf for row in result))

    def test_route_tie(self) -> None:
        """TODO: add function doc here"""
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
        """TODO: add function doc here"""
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
        """TODO: add function doc here"""
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
