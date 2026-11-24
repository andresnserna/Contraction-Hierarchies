"""Check experiment graph construction and real-life TSV loading."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import experiments as exp
from graph import Graph
from graph_services import ComparisonResult, compare_algorithms


class GeneratedExperimentTests(unittest.TestCase):
    def setUp(self):
        # Inspect graph setup independently from the algorithm comparisons below.
        patcher = patch.object(exp, "compare_graph", side_effect=lambda graph, **kwargs: graph)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_all_topologies_and_reproducible_weights(self):
        cases = (
            (exp.experiment_erdos_renyi, {"probability": 1}, 4, 12),
            (exp.experiment_cycle, {}, 4, 8),
            (exp.experiment_complete_graph, {}, 4, 12),
            (exp.experiment_path_graph, {}, 4, 6),
            (exp.experiment_random_trees, {}, 4, 6),
            (exp.experiment_balanced_trees, {"degree": 2}, 4, 6),
            (exp.experiment_grid_graph, {}, 16, 48),
        )
        for function, kwargs, nodes, arcs in cases:
            with self.subTest(function=function.__name__):
                graph = function(4, seed=7, **kwargs)
                self.assertIsInstance(graph, Graph)
                self.assertEqual(graph.get_num_nodes(), nodes)
                self.assertEqual(graph.get_num_edges(), arcs)
                self.assertEqual(graph.nodes, function(4, seed=7, **kwargs).nodes)
                for u, neighbors in graph.nodes.items():
                    for v, weight in neighbors.items():
                        self.assertEqual(graph.nodes[v][u], weight)
                        self.assertTrue(1 <= weight <= 20)

    def test_probability_zero_keeps_isolated_nodes(self):
        graph = exp.experiment_erdos_renyi(5, 0, seed=7)
        self.assertEqual(graph.get_num_nodes(), 5)
        self.assertEqual(graph.get_num_edges(), 0)

    def test_directed_path(self):
        graph = exp.experiment_path_graph(4, directed=True, seed=7)
        self.assertEqual(graph.get_num_edges(), 3)
        self.assertEqual(graph.dijkstra(3)[0][0], float("inf"))
        self.assertLess(graph.dijkstra(0)[0][3], float("inf"))

    def test_bad_parameters(self):
        for n in (-1, 2.5, True):
            with self.assertRaises(ValueError):
                exp.experiment_cycle(n)
        for probability in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                exp.experiment_erdos_renyi(3, probability)
        with self.assertRaises(ValueError):
            exp.experiment_balanced_trees(4, degree=0)


class RealLifeExperimentTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(exp, "compare_graph", side_effect=lambda graph, **kwargs: graph)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_dataset(self, root, name, node_header="id", edge_header="u\tv\tdistance"):
        folder = root / name
        folder.mkdir()
        (folder / "node.tsv").write_text(f"{node_header}\n{name}_0\n{name}_1\nisolated\n")
        (folder / "edges.tsv").write_text(
            f"{edge_header}\n{name}_0\t{name}_1\t9\n"
            f"{name}_0\t{name}_1\t2.5\n{name}_1\t{name}_0\t4\n"
        )
        return folder

    def test_each_dataset_function_uses_its_own_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (("CA", exp.experiment_california), ("NY", exp.experiment_new_york),
                     ("TX", exp.experiment_texas), ("merged", exp.experiment_merged))
            for name, function in cases:
                self.write_dataset(root, name)
                graph = function(data_dir=root)
                self.assertEqual(graph.get_num_nodes(), 3)
                self.assertEqual(graph.get_num_edges(), 2)
                self.assertEqual(graph.nodes[f"{name}_0"][f"{name}_1"], 2.5)
                self.assertEqual(graph.nodes[f"{name}_1"][f"{name}_0"], 4)
                self.assertEqual(graph.nodes["isolated"], {})
            with patch.object(exp, "REAL_LIFE_DIR", root):
                self.assertIn("NY_0", exp.experiment_new_york().nodes)

    def test_generator_and_length_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, node_header, edge_header in (
                ("generated", "node_id", "node_id_from\tnode_id_to\tweight"),
                ("tiger", "id", "u\tv\tlength"),
            ):
                folder = self.write_dataset(root, name, node_header, edge_header)
                graph = exp.load_tsv_graph(folder)
                self.assertEqual(graph.nodes[f"{name}_0"][f"{name}_1"], 2.5)

    def test_missing_files_and_invalid_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                exp.experiment_california(data_dir=root)
            folder = self.write_dataset(root, "NY")
            for row in ("NY_0\tunknown\t1", "NY_0\tNY_1\t-1",
                        "NY_0\tNY_1\tnan", "NY_0\tNY_1\tinvalid"):
                (folder / "edges.tsv").write_text(f"u\tv\tdistance\n{row}\n")
                with self.subTest(row=row), self.assertRaises(ValueError):
                    exp.load_tsv_graph(folder)


class AlgorithmComparisonTests(unittest.TestCase):
    def test_all_generated_experiments_run_both_algorithms(self):
        cases = (
            (exp.experiment_erdos_renyi, {"probability": 0.4}),
            (exp.experiment_cycle, {}), (exp.experiment_complete_graph, {}),
            (exp.experiment_path_graph, {}), (exp.experiment_random_trees, {}),
            (exp.experiment_balanced_trees, {"degree": 2}),
            (exp.experiment_grid_graph, {}),
        )
        for directed in (False, True):
            for function, kwargs in cases:
                with self.subTest(function=function.__name__, directed=directed):
                    result = function(4, directed=directed, seed=7, **kwargs)
                    self.assertIsInstance(result, ComparisonResult)
                    self.assertTrue(result.distances_match)
                    self.assertEqual(result.dijkstra.algorithm, "dijkstra")
                    self.assertEqual(result.ch.algorithm, "ch")
                    self.assertGreaterEqual(result.ch.counters.preprocessing_time, 0)
                    self.assertGreaterEqual(result.dijkstra.counters.query_time, 0)

    def test_conversion_preserves_fractional_asymmetric_and_zero_costs(self):
        graph = Graph()
        for label in ("n10", "n2", "n99", "isolated"):
            graph.add_node(label)
        graph.add_edge("n10", "n2", 0.125)
        graph.add_edge("n2", "n10", 7.5)
        graph.add_edge("n2", "n99", 0)
        graph.add_edge("n10", "n99", 10)
        network = exp.graph_to_road_network(graph)
        self.assertEqual([node.name for node in network.nodes.values()], list(graph.nodes))
        self.assertEqual(len(network.nodes), 4)
        self.assertEqual(len(network.roads), 4)
        for label, index in zip(graph.nodes, network.nodes):
            expected, _ = graph.dijkstra(label)
            for other, target in zip(graph.nodes, network.nodes):
                result = compare_algorithms(network, index, target)
                self.assertEqual(result.dijkstra.cost, expected[other])
                self.assertEqual(result.ch.cost, expected[other])
        result = exp.compare_graph(graph, source="n10", target="n99")
        self.assertEqual(result.ch.cost, 0.125)
        self.assertEqual(result.ch.path, [0, 1, 2])

    def test_all_real_life_functions_use_comparison_service(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, function in (
                ("CA", exp.experiment_california), ("NY", exp.experiment_new_york),
                ("TX", exp.experiment_texas), ("merged", exp.experiment_merged),
            ):
                folder = root / name
                folder.mkdir()
                (folder / "node.tsv").write_text("id\nn0\nn1\nisolated\n")
                (folder / "edges.tsv").write_text("u\tv\tdistance\nn0\tn1\t2.5\n")
                with patch.object(exp, "compare_algorithms", wraps=compare_algorithms) as service:
                    result = function(data_dir=root, source="n0", target="n1")
                    service.assert_called_once()
                self.assertTrue(result.distances_match)
                self.assertEqual(result.ch.cost, 2.5)
                self.assertEqual(result.dijkstra.path, [0, 1])

    def test_unreachable_singleton_empty_and_invalid_endpoints(self):
        result = exp.experiment_path_graph(4, directed=True, source=3, target=0)
        self.assertTrue(result.distances_match)
        self.assertEqual(result.ch.cost, float("inf"))
        self.assertIsNone(result.ch.path)
        result = exp.experiment_path_graph(1)
        self.assertEqual(result.ch.cost, 0)
        self.assertEqual(result.ch.path, [0])
        with self.assertRaises(ValueError):
            exp.experiment_path_graph(0)
        with self.assertRaises(ValueError):
            exp.experiment_path_graph(3, source=99)


if __name__ == "__main__":
    unittest.main()
