from pathlib import Path
import csv
import math
import shutil
import xml.etree.ElementTree as ET

from graph import Graph


def load_graph_xml(path, weight_attribute="distance"):
    """Read an uncompressed GraphML XML file into the project's Graph.

    Preserve string node IDs (including isolated nodes) and edge direction.
    Use ``length`` for the newer TIGER files, whose lengths are in kilometers.
    Parallel edges are reduced to their minimum weight; self-loops are skipped
    because Graph does not support them. Other attributes are not loaded.
    XML elements are discarded as they are processed to limit memory use.
    """
    result = Graph()
    weight_keys = {}
    graph_element = None
    directed_default = False

    def add_arc(source, target, weight):
        existing = result.nodes[source].get(target)
        if existing is None:
            result.add_edge(source, target, weight)
        elif weight < existing:
            result.nodes[source][target] = weight

    with open(path, "rb") as xml:
        for event, element in ET.iterparse(xml, events=("start", "end")):
            tag = element.tag.rsplit("}", 1)[-1]
            if event == "start":
                if tag == "graph":
                    if graph_element is not None:
                        raise ValueError("Nested GraphML graphs are not supported")
                    graph_element = element
                    directed_default = element.get("edgedefault") == "directed"
                continue

            if tag == "key":
                if (element.get("attr.name") == weight_attribute
                        and element.get("for", "all") in ("edge", "all")):
                    default = next(
                        (child.text for child in element
                         if child.tag.rsplit("}", 1)[-1] == "default"),
                        None,
                    )
                    weight_keys[element.attrib["id"]] = default
                element.clear()
            elif tag == "node":
                result.add_node(element.attrib["id"])
                graph_element.remove(element)
            elif tag == "edge":
                source = element.attrib["source"]
                target = element.attrib["target"]
                value = next((v for v in weight_keys.values() if v is not None), None)
                for child in element:
                    if (child.tag.rsplit("}", 1)[-1] == "data"
                            and child.get("key") in weight_keys):
                        value = child.text
                try:
                    weight = float(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Edge {source!r} -> {target!r} has missing or invalid "
                        f"{weight_attribute!r} weight"
                    ) from error
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError(f"Edge {source!r} -> {target!r} needs a finite, nonnegative weight")
                result.add_node(source)
                result.add_node(target)
                if source != target:
                    add_arc(source, target, weight)
                    directed = element.get("directed")
                    is_directed = directed_default if directed is None else directed in ("true", "1")
                    if not is_directed:
                        add_arc(target, source, weight)
                graph_element.remove(element)

    if graph_element is None:
        raise ValueError("No GraphML graph found")
    return result


def xml_to_tsv(path, output_dir=None, weight_attribute="distance"):
    """Save GraphML as node.tsv (id) and edges.tsv (u, v, weight).

    The weight column uses weight_attribute as its header, with no unit
    conversion. Node IDs remain strings. Undirected roads produce two rows,
    one per direction, matching Graph's adjacency lists. As in load_graph_xml,
    parallel edges retain the minimum weight and self-loops are omitted.
    By default, output goes in a folder named after the XML file.
    These are distance graph tables, not the evacuation CLI's minutes/capacity
    input format. Return the two output Paths.
    """
    source = Path(path)
    graph = load_graph_xml(source, weight_attribute)
    destination = source.with_suffix("") if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    nodes_path = destination / "node.tsv"
    edges_path = destination / "edges.tsv"

    # Exclusive creation protects previous exports from being overwritten.
    if nodes_path.exists() or edges_path.exists():
        raise FileExistsError(f"TSV output already exists in {destination}")
    with nodes_path.open("x", newline="", encoding="utf-8") as nodes_file:
        writer = csv.writer(nodes_file, delimiter="\t")
        writer.writerow(["id"])
        for node in graph.nodes:
            writer.writerow([node])
    with edges_path.open("x", newline="", encoding="utf-8") as edges_file:
        writer = csv.writer(edges_file, delimiter="\t")
        writer.writerow(["u", "v", weight_attribute])
        for source_node, neighbors in graph.nodes.items():
            for target_node, weight in neighbors.items():
                writer.writerow([source_node, target_node, weight])
    return nodes_path, edges_path


def decompress_graph(path, output_dir=None):
    """Save a .xml.zst file as .xml, keeping the compressed original.

    Return the saved Path. Existing output files are never overwritten.
    """
    import zstandard as zstd

    source = Path(path)
    if not source.name.endswith(".xml.zst"):
        raise ValueError("Expected a .xml.zst file")

    destination_dir = source.parent if output_dir is None else Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.with_suffix("").name

    with source.open("rb") as compressed:
        with destination.open("xb") as output:
            try:
                with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
                    shutil.copyfileobj(reader, output)
            except BaseException:
                destination.unlink()
                raise
    return destination


def decompress_graphs(directory, output_dir=None):
    """Decompress all .xml.zst files directly inside a directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    return [
        decompress_graph(path, output_dir)
        for path in sorted(directory.glob("*.xml.zst"))
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert road GraphML XML files to TSV.")
    parser.add_argument(
        "path", nargs="?", type=Path,
        default=Path(__file__).parent / "data" / "real-life-examples",
        help="An XML file or directory of XML files",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--weight", default="distance", help="Edge attribute to export")
    parser.add_argument("--decompress", action="store_true", help="Decompress .xml.zst files instead")
    args = parser.parse_args()
    if args.decompress:
        paths = (decompress_graphs(args.path, args.output_dir) if args.path.is_dir()
                 else [decompress_graph(args.path, args.output_dir)])
        for saved_path in paths:
            print(saved_path)
    else:
        if not args.path.exists():
            parser.error(f"Path does not exist: {args.path}")
        sources = sorted(args.path.glob("*.xml")) if args.path.is_dir() else [args.path]
        if not sources:
            parser.error(f"No XML files found in {args.path}")
        for source in sources:
            destination = args.output_dir
            if destination is not None and args.path.is_dir():
                destination = destination / source.stem
            print(f"Converting {source}...", flush=True)
            for saved_path in xml_to_tsv(source, destination, args.weight):
                print(saved_path)
