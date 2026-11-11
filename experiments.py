"""Experiment runners for the graph query performance study."""

from __future__ import annotations

import csv
import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Callable, TypeVar

from graph_services import Counters, benchmark_all_pairs, evacuations, naive_per_shelter, naive_per_zone
from road_network import RoadNetwork
from visuals import draw_experiment_3_1

T = TypeVar("T")


@dataclass(frozen=True)
class TimedResult:
    value: object | None
    seconds: float
    timed_out: bool = False


class ThreadedRunner:
    """Run a callable in a worker while a separate thread measures its runtime."""

    @staticmethod
    def run(function: Callable[[Event], T], *, stop_after: float | None = 30.0) -> TimedResult:
        """TO-DO: add function doc here"""
        started = Event()
        finished = Event()
        stop_event = Event()
        measured: list[float] = []
        values: list[T] = []
        errors: list[BaseException] = []
        timed_out: list[bool] = []

        def clock() -> None:
            """TO-DO: add function doc here"""
            start_time = perf_counter()
            started.set()
            if stop_after is None:
                finished.wait()
                timed_out.append(False)
            elif finished.wait(stop_after):
                timed_out.append(False)
            else:
                stop_event.set()
                finished.wait()
                timed_out.append(True)
            measured.append(perf_counter() - start_time)

        def work() -> None:
            """TO-DO: add function doc here"""
            started.wait()
            try:
                values.append(function(stop_event))
            except TimeoutError:
                timed_out.append(True)
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        timer_thread = Thread(target=clock, name="experiment-clock")
        worker_thread = Thread(target=work, name="experiment-worker")
        timer_thread.start()
        worker_thread.start()
        try:
            worker_thread.join()
        except KeyboardInterrupt:
            stop_event.set()
            worker_thread.join()
            timer_thread.join()
            raise
        timer_thread.join()

        if errors:
            raise errors[0]
        return TimedResult(values[0] if values else None, measured[0], any(timed_out))


def with_stop(counters: Counters, stop_event: Event) -> Counters:
    """TO-DO: add function doc here"""
    counters.stop_event = stop_event
    return counters


def main() -> None:
    """TO-DO: add function doc here"""
    pass


if __name__ == "__main__":
    main()
