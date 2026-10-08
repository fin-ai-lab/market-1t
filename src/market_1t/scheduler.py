"""Memory-budgeted process pool used by the conversion and snapshot stages.

Tasks run in spawned worker processes, largest first; a task only starts
while the estimated peak memory of all running tasks stays within a budget.
A worker that dies (e.g. killed by the OOM killer) costs only its own tasks,
which are retried once on their own.
"""

from __future__ import annotations

import bisect
import logging
import multiprocessing
import os
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from typing import Any

GiB = 1024**3


def available_memory_bytes() -> int:
    """``MemAvailable`` from /proc/meminfo (falls back to total RAM)."""
    try:
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:  # macOS and other systems without /proc
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return 8 * GiB


def reset_peak_rss() -> None:
    """Reset this process' peak RSS (VmHWM) so it can be measured per task."""
    try:
        with open("/proc/self/clear_refs", "w") as clear_refs:
            clear_refs.write("5")
    except OSError:
        pass


def peak_rss_bytes() -> int:
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def thread_environment(threads: int) -> dict[str, str]:
    """Environment that caps the native thread pools of a worker process.

    Spawned workers inherit these before Polars/NumPy initialise, which avoids
    the thread explosion ("can't start new thread") seen with many workers.
    """
    value = str(max(1, threads))
    return {
        "POLARS_MAX_THREADS": value,
        "OMP_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "NUMEXPR_MAX_THREADS": value,
    }


@dataclass
class Task:
    """A unit of work; ``memory`` is its estimated peak memory in bytes."""

    key: Any
    memory: int
    payload: Any
    attempts: int = field(default=0)


class WorkerDied(RuntimeError):
    """A worker process exited abruptly (for example, killed by the OOM killer)."""


# Abort when worker pools keep breaking without any task completing in between
# (e.g. every worker crashes on start-up); tasks are then reported as failed.
MAX_FRUITLESS_REBUILDS = 3


def run_tasks(
    tasks: Sequence[Task],
    func: Callable[[Any], Any],
    *,
    max_workers: int,
    memory_budget: int,
    initializer: Callable | None = None,
    initargs: tuple = (),
    env: dict[str, str] | None = None,
    logger: logging.Logger | None = None,
) -> Iterator[tuple[Task, Any, BaseException | None]]:
    """Run ``func(task.payload)`` in spawned worker processes.

    Tasks start largest first and a task only starts while the estimated
    memory of all running tasks stays within ``memory_budget`` (one task always
    runs, however large).  Results are yielded as ``(task, result, error)`` in
    completion order; every task is yielded exactly once.  If a worker dies
    abruptly, the pool is rebuilt and each affected task is retried once on its
    own; tasks lost twice are reported as ``WorkerDied``.
    """
    if env:
        os.environ.update(env)
    pending = sorted(tasks, key=lambda t: t.memory)
    estimates = [t.memory for t in pending]

    def requeue(task: Task) -> None:
        position = bisect.bisect_right(estimates, task.memory)
        estimates.insert(position, task.memory)
        pending.insert(position, task)

    if max_workers <= 1:
        if initializer is not None:
            initializer(*initargs)
        while pending:
            task = pending.pop()
            estimates.pop()
            try:
                result = func(task.payload)
            except Exception as error:  # noqa: BLE001 - reported to caller
                yield task, None, error
            else:
                yield task, result, None
        return

    context = multiprocessing.get_context("spawn")

    def new_executor():
        return ProcessPoolExecutor(max_workers, mp_context=context, initializer=initializer, initargs=initargs)

    executor = new_executor()
    running: dict[Any, Task] = {}
    reserved = 0
    fruitless_rebuilds = 0  # pool rebuilds since a task last completed
    try:
        while pending or running:
            broken = False
            while pending and len(running) < max_workers:
                if not running:
                    index = len(pending) - 1
                else:
                    index = bisect.bisect_right(estimates, memory_budget - reserved) - 1
                    if index < 0:
                        break
                task = pending.pop(index)
                estimates.pop(index)
                try:
                    future = executor.submit(func, task.payload)
                except BrokenProcessPool:
                    # A worker died since the last wait(): the task never
                    # started, so it goes back to the queue for the new pool.
                    requeue(task)
                    broken = True
                    break
                task.attempts += 1
                running[future] = task
                reserved += task.memory
            if not broken:
                done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for future in done:
                    task = running.pop(future)
                    reserved -= task.memory
                    try:
                        result = future.result()
                    except BrokenProcessPool:
                        broken = True
                        running[future] = task  # handled below with the rest
                        reserved += task.memory
                        continue
                    except Exception as error:  # noqa: BLE001 - reported to caller
                        fruitless_rebuilds = 0
                        yield task, None, error
                        continue
                    fruitless_rebuilds = 0
                    yield task, result, None
            if broken:
                # Every in-flight task is lost when a worker dies.  Rebuild the
                # pool and retry each lost task once, alone.
                lost = list(running.values())
                running.clear()
                reserved = 0
                executor.shutdown(wait=False, cancel_futures=True)
                fruitless_rebuilds += 1
                if fruitless_rebuilds > MAX_FRUITLESS_REBUILDS:
                    error = WorkerDied(
                        f"worker processes died {fruitless_rebuilds} times in a row without completing a task"
                    )
                    if logger:
                        logger.error("%s; giving up on %d remaining tasks", error, len(lost) + len(pending))
                    remaining = lost + pending[::-1]
                    pending.clear()
                    estimates.clear()
                    for task in remaining:
                        yield task, None, error
                    return
                executor = new_executor()
                for task in lost:
                    if task.attempts >= 2:
                        yield task, None, WorkerDied("worker process died twice while running this task")
                    else:
                        if logger:
                            logger.warning("Worker died; retrying %s alone", task.key)
                        task.memory = max(task.memory, memory_budget)
                        requeue(task)
    except BaseException:
        # Stop promptly on errors/interrupts: do not wait for running tasks.
        processes = getattr(executor, "_processes", None) or {}
        for process in list(processes.values()):
            process.terminate()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)


def default_max_workers() -> int:
    return max(1, min(32, (os.cpu_count() or 2) // 2))


def default_memory_budget() -> int:
    return int(0.5 * available_memory_bytes())


def human_bytes(n: float) -> str:
    return f"{n / GiB:.1f} GiB"
