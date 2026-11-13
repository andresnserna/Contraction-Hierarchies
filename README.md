# Contraction Hierarchies — CLI Guide

This project models a road network as a directed, weighted graph and answers evacuation, all-pairs travel-time, and critical-road-closure queries. The command-line program is `Graphs as Networks/project_cli.py`.

## Run the program

From the repository root, run a query with Python:

```sh
python3 "Graphs as Networks/project_cli.py" --query evac
```

If no data files are specified, the program loads the shipped `tiny` instance. To select another instance, pass its node and edge TSV files together:

```sh
python3 "Graphs as Networks/project_cli.py" \
  --nodes "Graphs as Networks/data/instances/small/nodes.tsv" \
  --edges "Graphs as Networks/data/instances/small/edges.tsv" \
  --query evac
```

Use `--help` to see the available flags and choices.

## Input and graph model

The CLI reads two tab-separated files:

- `nodes.tsv` columns: `id`, `name`, `kind`, `pass_time`. `kind` is `zone`, `shelter`, or `junction`.
- `edges.tsv` columns: `u`, `v`, `minutes`, `capacity`, `oneway`. The current travel-time queries use `minutes`; `capacity` is loaded but not used in these queries. `oneway` is `1` for a one-way road and `0` for a two-way road.

Each node becomes a graph vertex. A one-way road contributes the arc `u → v`; a two-way road contributes both `u → v` and `v → u`. The cost of leaving a junction includes that junction's `pass_time`. Zone and shelter pass times are zero. The loader checks required columns, IDs, node kinds, endpoint references, and road values, and reports invalid input as a CLI error.

## Queries and options

### Evacuation assignments: `--query evac`

Prints one row per zone: the zone name, its nearest reachable shelter name (or `NONE`), and travel time in minutes (or `INF` if no shelter is reachable). Rows are tab-separated, with no header.

The implementation reverses the road arcs, connects a virtual source to every shelter with zero-cost arcs, and runs one Dijkstra search. This computes the nearest shelter for every zone in one pass. If shelters tie in travel time, the lower shelter ID wins.

```sh
python3 "Graphs as Networks/project_cli.py" --query evac
```

### Close one directed road arc: `--close U V`

Use with `--query evac` to recompute assignments with the directed arc `U → V` removed:

```sh
python3 "Graphs as Networks/project_cli.py" --query evac --close 6 7
```

On a two-way road, this closes only the requested direction; the reverse arc remains available. On a one-way road, only its existing direction can be closed. The node IDs must be connected by a road in the requested direction.

### All-pairs travel times: `--query apsp`

Prints a row for every ordered pair of distinct nodes: source ID, destination ID, and shortest travel time (or `INF` if unreachable). Rows are tab-separated, with no header. The default algorithm is Dijkstra from every node.

Choose an algorithm with `--algo`:

| Value | Algorithm |
| --- | --- |
| `dijkstra` | Dijkstra from each node; default |
| `bellman-ford` | Bellman-Ford from each node |
| `johnson` | Johnson's all-pairs algorithm |
| `floyd-warshall` | Floyd-Warshall |

Example:

```sh
python3 "Graphs as Networks/project_cli.py" --query apsp --algo floyd-warshall
```

`--algo` applies only to `apsp`. All input road costs and junction delays are nonnegative, but the alternative algorithms are available for comparison and the empirical study.

### Critical closure: `--query critical`

Prints the directed edge `u`, `v` whose closure produces the largest post-closure travel time for any zone, followed by that worst time. If the network has no reachable zones to evaluate, it prints `NONE`. A disconnected zone after a tested closure contributes `INF`, which ranks above finite times. Equal worst times are resolved by the lower `(u, v)` ID pair.

The search first finds the shortest-path forest for the baseline evacuation assignments, then tests only directed arcs in that forest. For each candidate arc, it closes that direction and reruns the evacuation query. For a two-way road, the reverse direction remains open. Zones that were already unreachable in the baseline are excluded from the comparison; if a previously reachable zone becomes unreachable after a closure, its `INF` time is considered worst. The reported value is the worst resulting travel time, not the increase from baseline.

```sh
python3 "Graphs as Networks/project_cli.py" --query critical
```

### Operation counts and search summary

Add `--counters` to write `counters.txt` in the current working directory. It contains tab-separated `Relaxations` and `PQ_Ops` counts. A relaxation is an edge considered for a distance improvement. `PQ_Ops` counts priority-queue pushes and pops; algorithms that do not use a priority queue report zero for this metric.

```sh
python3 "Graphs as Networks/project_cli.py" --query apsp --algo floyd-warshall --counters
```

Add `--verbose` to print the vertex count, road-row count, directed-arc count, relaxations, and priority-queue operations to standard error. Query rows remain on standard output, so they can still be redirected or piped separately.

```sh
python3 "Graphs as Networks/project_cli.py" --query evac --verbose
```

`--counters` and `--verbose` may be used together. `--close` is valid only with `evac`; `--algo` is valid only with `apsp`. Invalid arguments or invalid input are reported as command-line errors.

## Test coverage map

The test suite is `Graphs as Networks/test_project.py`. This table maps the implemented CLI and graph requirements to the existing tests. “Partial” means a nearby behavior is tested, but the listed CLI behavior or edge case does not have a direct assertion.

| Ability | Existing coverage | Coverage status |
| --- | --- | --- |
| Load node and road TSV data; recognize zones, shelters, one-way and two-way roads, and junction delays | `test_load` | Covered for the shipped tiny fixture; malformed-file validation is not covered |
| Compute nearest-shelter assignments and travel times | `test_evac`, `test_cli_outputs` | Covered at function level on tiny and CLI level on tiny and small |
| Compute all-pairs distances with Dijkstra, Bellman-Ford, Johnson, and Floyd-Warshall | `test_apsp_algorithms`, `test_floyd_warshall`, `test_cli_outputs` | Algorithms are covered against expected values or Dijkstra; CLI output uses the default algorithm only |
| Find the critical closure and worst post-closure time | `test_critical` | Function result covered on tiny; the `--query critical` CLI output is not directly tested |
| Represent unreachable evacuation zones as `NONE` / `INF` | `test_unreachable` | `INF` is tested at function level; CLI formatting for unreachable zones is not directly tested |
| Break equal nearest-shelter routes by lower shelter ID | `test_route_tie` | Covered at function level |
| Close only `U → V` on a two-way road and keep `V → U` | `test_close` | Arc behavior and CLI evacuation output are covered |
| Critical-closure ties, including `INF`, use the lowest `(u, v)` | `test_critical_tie_with_inf_prefers_lower_edge_ids` | Covered at function and CLI levels |
| `--counters` output and zero `PQ_Ops` for non-priority-queue algorithms | `test_counters_file_and_non_pq_algorithms_report_zero` | Covered for Bellman-Ford and Floyd-Warshall |
| `--verbose` summary and stderr separation | `test_verbose_summary_goes_to_stderr` | Covered |
| CLI rejects incompatible flags, invalid IDs, or malformed input | `test_cli_rejects_incompatible_flags_invalid_ids_and_bad_input` | Covered for representative cases |

The tests use the standard library `unittest` framework. From the project directory, run them with:

```sh
cd "Graphs as Networks"
python3 -m unittest test_project.py
```

