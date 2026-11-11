"""Matplotlib visualizations for the graph algorithm experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


APPROACHES = ("perzone", "pershelter", "vnode")


def draw_experiment_3_1(data_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Draw and save grouped runtime and operation-count bar charts."""
    values: dict[tuple[str, str], dict[str, float]] = {}
    datasets: list[str] = []
    with data_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            if row["dataset"] not in datasets:
                datasets.append(row["dataset"])
            values[(row["dataset"], row["approach"])] = {
                "runtime": float(row["runtime_seconds"]),
                "operations": float(row["op_count"]),
            }

    x = np.arange(len(datasets))
    width = 0.24
    colors = ("#3568a8", "#e18b32", "#4b9b67")

    def display_name(dataset: str) -> str:
        """TO-DO: add function doc here"""
        if dataset.startswith("generated_z"):
            return dataset.removeprefix("generated_").replace("z", "Z").replace("_s", "/S")
        return dataset

    def draw(metric: str, title: str, ylabel: str, filename: str) -> Path:
        """TO-DO: add function doc here"""
        figure, axis = plt.subplots(
            figsize=(max(10, len(datasets) * 1.5), 5.5), layout="constrained"
        )
        for index, approach in enumerate(APPROACHES):
            heights = [values[(dataset, approach)][metric] for dataset in datasets]
            bars = axis.bar(
                x + (index - 1) * width,
                heights,
                width,
                label=approach,
                color=colors[index],
            )
            axis.bar_label(bars, fmt="%.3g", padding=3, fontsize=7, rotation=90)
        axis.set_title(title)
        axis.set_xlabel("Dataset (generated labels show zones / shelters)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [display_name(dataset) for dataset in datasets])
        axis.legend(title="Approach")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        path = output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

    runtime_path = draw(
        "runtime",
        "Experiment 3.1: Evacuation Runtime",
        "Runtime (seconds)",
        "experiment3.1_runtime.png",
    )
    operations_path = draw(
        "operations",
        "Experiment 3.1: Operation Count",
        "Operations (relaxations + priority queue operations)",
        "experiment3.1_operations.png",
    )
    return runtime_path, operations_path


def draw_experiment_3_2(data_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Draw APSP runtime and operation-count comparisons against actual V."""
    rows: list[dict[str, str]] = []
    with data_path.open(newline="", encoding="utf-8") as file:
        rows.extend(csv.DictReader(file, delimiter="\t"))

    vertices = sorted({int(row["V"]) for row in rows})
    algorithms = ("dijkstra", "floyd-warshall", "johnson")
    colors = {"dijkstra": "#3568a8", "floyd-warshall": "#e18b32", "johnson": "#4b9b67"}
    x = np.asarray(vertices, dtype=float)
    gaps = np.diff(x)
    default_gap = gaps[-1] if len(gaps) else 1.0
    widths = np.asarray([
        (min(gaps[index - 1], gaps[index]) if index > 0 and index < len(gaps)
         else gaps[0] if index == 0 and len(gaps)
         else gaps[-1] if len(gaps) else default_gap) * 0.68
        for index in range(len(x))
    ])

    def draw(metric: str, title: str, ylabel: str, filename: str) -> Path:
        """TO-DO: add function doc here"""
        figure, axis = plt.subplots(figsize=(12, 6.5), layout="constrained")
        missing: dict[int, list[str]] = {}
        for algorithm_index, algorithm in enumerate(algorithms):
            for index, vertex_count in enumerate(vertices):
                record = next(
                    (row for row in rows
                     if int(row["V"]) == vertex_count
                     and row["algorithm"] == algorithm),
                    None,
                )
                if record is None or record["status"] == "memory_limit" or not record[metric]:
                    if record and record["status"] == "memory_limit":
                        missing.setdefault(vertex_count, []).append("FW memory limit")
                    continue
                value = float(record[metric])
                bar_width = widths[index] / 3.4
                center = x[index] + (algorithm_index - 1) * bar_width
                bars = axis.bar(
                    center, value, bar_width * 0.92,
                    color=colors[algorithm],
                    label=algorithm if index == 0 else None,
                )
                if record["status"] == "timeout":
                    axis.annotate(
                        "timeout", (center, value), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, rotation=90,
                    )
                elif len(vertices) <= 5:
                    axis.bar_label(bars, fmt="%.3g", padding=2, fontsize=7)

        axis.set_title(title)
        axis.set_xlabel("Vertices (V)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [f"{vertex:,}" for vertex in vertices], rotation=25, ha="right")
        axis.legend(title="Algorithm")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        if missing:
            labels = [f"V={vertex:,}: {', '.join(statuses)}" for vertex, statuses in missing.items()]
            axis.text(
                0.01, 0.99, "Floyd–Warshall omitted above the safe dense-matrix limit\n" + "\n".join(labels),
                transform=axis.transAxes, ha="left", va="top", fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
        path = output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

    runtime_path = draw(
        "runtime_seconds", "Experiment 3.2: All-Pairs Runtime",
        "Runtime (seconds)", "experiment3.2_runtime.png",
    )
    operations_path = draw(
        "op_count", "Experiment 3.2: All-Pairs Operation Count",
        "Operations (relaxations + priority queue operations)",
        "experiment3.2_operations.png",
    )
    return runtime_path, operations_path


def draw_experiment_3_3(data_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Draw APSP runtime and operation counts against directed edge count E."""
    rows: list[dict[str, str]] = []
    with data_path.open(newline="", encoding="utf-8") as file:
        rows.extend(csv.DictReader(file, delimiter="\t"))

    edge_counts = sorted({int(row["E"]) for row in rows})
    algorithms = ("dijkstra", "floyd-warshall", "johnson")
    colors = {"dijkstra": "#3568a8", "floyd-warshall": "#e18b32", "johnson": "#4b9b67"}
    x = np.asarray(edge_counts, dtype=float)
    gaps = np.diff(x)
    widths = np.asarray([
        (min(gaps[i - 1], gaps[i]) if i > 0 and i < len(gaps)
         else gaps[0] if i == 0 and len(gaps)
         else gaps[-1] if len(gaps) else 1.0) * 0.68
        for i in range(len(x))
    ])

    def draw(metric: str, title: str, ylabel: str, filename: str) -> Path:
        """TO-DO: add function doc here"""
        figure, axis = plt.subplots(figsize=(11, 6), layout="constrained")
        for algorithm_index, algorithm in enumerate(algorithms):
            for index, edge_count in enumerate(edge_counts):
                record = next(
                    (row for row in rows
                     if int(row["E"]) == edge_count
                     and row["algorithm"] == algorithm),
                    None,
                )
                if record is None or record["status"] == "memory_limit" or not record[metric]:
                    continue
                value = float(record[metric])
                bar_width = widths[index] / 3.4
                center = x[index] + (algorithm_index - 1) * bar_width
                bars = axis.bar(
                    center, value, bar_width * 0.92,
                    color=colors[algorithm],
                    label=algorithm if index == 0 else None,
                )
                if record["status"] == "timeout":
                    axis.annotate(
                        "timeout", (center, value), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, rotation=90,
                    )
                elif len(edge_counts) <= 5:
                    axis.bar_label(bars, fmt="%.3g", padding=2, fontsize=7)

        axis.set_title(title)
        axis.set_xlabel("Directed edges (E)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [f"{edge_count:,}" for edge_count in edge_counts], rotation=25, ha="right")
        axis.legend(title="Algorithm")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        path = output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return path

    runtime_path = draw(
        "runtime_seconds", "Experiment 3.3: Runtime vs. Edge Density",
        "Runtime (seconds)", "experiment3.3_runtime.png",
    )
    operations_path = draw(
        "op_count", "Experiment 3.3: Operation Count vs. Edge Density",
        "Operations (relaxations + priority queue operations)",
        "experiment3.3_operations.png",
    )
    return runtime_path, operations_path
