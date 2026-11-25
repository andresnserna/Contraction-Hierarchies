import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from read_data import decompress_graph, decompress_graphs, xml_to_tsv


class XMLExportTests(unittest.TestCase):
    def test_distance_export_preserves_nodes_and_shortest_arcs(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "roads.xml"
            source.write_text('''<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
              <key id="w" for="edge" attr.name="distance" attr.type="float"/>
              <graph edgedefault="undirected">
                <node id="n0"/><node id="n1"/><node id="isolated"/>
                <edge source="n0" target="n1"><data key="w">8</data></edge>
                <edge source="n1" target="n0"><data key="w">2.5</data></edge>
                <edge source="n0" target="n0"><data key="w">1</data></edge>
              </graph></graphml>''')
            nodes, edges = xml_to_tsv(source)
            self.assertEqual(nodes.name, "node.tsv")
            self.assertEqual(edges.name, "edges.tsv")
            with nodes.open(newline="") as file:
                self.assertEqual(list(csv.DictReader(file, delimiter="\t")),
                                 [{"id": "n0"}, {"id": "n1"}, {"id": "isolated"}])
            with edges.open(newline="") as file:
                self.assertEqual(list(csv.DictReader(file, delimiter="\t")), [
                    {"u": "n0", "v": "n1", "distance": "2.5"},
                    {"u": "n1", "v": "n0", "distance": "2.5"},
                ])
            before = edges.read_bytes()
            with patch("read_data.load_graph_xml") as loader:
                self.assertEqual(xml_to_tsv(source), (nodes, edges))
                loader.assert_not_called()
            self.assertEqual(edges.read_bytes(), before)

    def test_directed_length_and_custom_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "roads.xml"
            source.write_text('''<graphml>
              <key id="w" for="edge" attr.name="length" attr.type="double"/>
              <graph edgedefault="directed">
                <edge source="a" target="b"><data key="w">1.25</data></edge>
              </graph></graphml>''')
            destination = Path(directory) / "export"
            _, edges = xml_to_tsv(source, destination, "length")
            self.assertEqual(edges.parent, destination)
            self.assertEqual(edges.read_text(), "u\tv\tlength\na\tb\t1.25\n")

    def test_partial_export_creates_only_missing_file(self):
        for existing_name in ("node.tsv", "edges.tsv"):
            with self.subTest(existing=existing_name), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "roads.xml"
                source.write_text('''<graphml><graph edgedefault="directed">
                  <node id="isolated"/>
                </graph></graphml>''')
                destination = Path(directory) / "output"
                destination.mkdir()
                existing = destination / existing_name
                existing.write_text("previous output\n")
                nodes, edges = xml_to_tsv(source, destination)
                self.assertEqual(existing.read_text(), "previous output\n")
                if existing_name == "node.tsv":
                    self.assertEqual(edges.read_text(), "u\tv\tdistance\n")
                else:
                    self.assertEqual(nodes.read_text(), "id\nisolated\n")

    def test_decompression_skips_existing_xml_in_custom_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compressed_dir = root / "compressed"
            compressed_dir.mkdir()
            source = compressed_dir / "roads.xml.zst"
            # Invalid compressed bytes prove that existing output bypasses decoding.
            source.write_bytes(b"not a zstd file")
            destination = root / "output"
            destination.mkdir()
            xml = destination / "roads.xml"
            xml.write_text("existing XML\n")
            self.assertEqual(decompress_graph(source, destination), xml)
            self.assertEqual(decompress_graphs(compressed_dir, destination), [xml])
            self.assertEqual(xml.read_text(), "existing XML\n")
            self.assertEqual(source.read_bytes(), b"not a zstd file")


if __name__ == "__main__":
    unittest.main()
