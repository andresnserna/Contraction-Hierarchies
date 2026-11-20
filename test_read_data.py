import csv
from pathlib import Path
import tempfile
import unittest

from read_data import xml_to_tsv


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
            with self.assertRaises(FileExistsError):
                xml_to_tsv(source)
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


if __name__ == "__main__":
    unittest.main()
