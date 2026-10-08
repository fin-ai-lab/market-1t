"""The conversion and snapshot stages: find the work, run it on worker processes, report.

Both stages are resumable: existing outputs are skipped and every output is
written atomically, so an interrupted run can simply be restarted.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from . import convert, snapshot
from . import scheduler as sched
from .conditions import TradeRules
from .layout import QUOTES, TRADES, Layout, freq_label

logger = logging.getLogger(__name__)

# Peak memory of one conversion with the default settings is ~3.3 GiB (a
# 10 GB quotes file); budget a little more per concurrent file.
CONVERSION_TASK_BYTES = int(2.5 * sched.GiB)

# Estimated peak memory of one snapshot task: worker baseline plus a multiple
# of the compressed Parquet input (measured peak is ~15x on the largest inputs).
SNAPSHOT_TASK_BASE_BYTES = int(0.6 * sched.GiB)
SNAPSHOT_TASK_BYTES_PER_INPUT_BYTE = 20


def _run(tasks, func, *, max_workers, memory_budget, initializer, initargs, threads, log_file, on_result):
    """Run tasks with a progress bar; ``on_result(task, result, error)`` returns the result record."""
    results = []
    with open(log_file, "a") if log_file else nullcontext() as log:
        with logging_redirect_tqdm(), tqdm(total=len(tasks), miniters=1) as progress:
            for task, result, error in sched.run_tasks(
                tasks,
                func,
                max_workers=max_workers,
                memory_budget=memory_budget,
                initializer=initializer,
                initargs=initargs,
                env=sched.thread_environment(threads),
                logger=logger,
            ):
                record = on_result(task, result, error)
                results.append(record)
                if log:
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                progress.update(1)
    return results


# ---------------------------------------------------------------------------
# Conversion: flat files -> ticker-partitioned Parquet
# ---------------------------------------------------------------------------

_CONVERT_OPTIONS: dict = {}


def _init_conversion_worker(options: dict) -> None:
    _CONVERT_OPTIONS.update(options)


def _convert_task(payload: dict) -> dict:
    result = dict(payload)
    sched.reset_peak_rss()
    started = time.time()
    try:
        stats = convert.convert_to(
            payload["fn_in"],
            payload["fn_out"],
            payload["feed"],
            _CONVERT_OPTIONS["n_partitions"],
            chunk_bytes=_CONVERT_OPTIONS["chunk_mb"] * convert.MiB,
            parse_threads=_CONVERT_OPTIONS["parse_threads"],
            write_threads=_CONVERT_OPTIONS["write_threads"],
            buffer_bytes=_CONVERT_OPTIONS["buffer_mb"] * convert.MiB,
        )
        result["rows"] = stats.rows
        result["decompressor"] = stats.decompressor
    except Exception as error:  # noqa: BLE001 - reported per file
        result["error"] = str(type(error))
        result["error_msg"] = str(error)
    result["seconds"] = round(time.time() - started, 3)
    result["peak_rss_bytes"] = sched.peak_rss_bytes()
    return result


def discover_conversion_tasks(layout: Layout, feeds: Sequence[str], dates: Sequence[str], buffer_mb: int):
    """Flat files of the requested days whose Parquet output does not exist yet."""
    wanted = set(dates)
    tasks, n_available = [], {}
    for feed in feeds:
        if feed not in convert.COLS_BY_FEED:
            raise ValueError(f"Feed `{feed}` is not supported. Must be one of: {', '.join(convert.COLS_BY_FEED)}")
        pattern = os.path.join(str(layout.flatfiles), layout.product, feed, "*", "*", "*.csv.gz")
        files = sorted(fn for fn in glob.glob(pattern) if os.path.basename(fn).split(".")[0] in wanted)
        n_available[feed] = len(files)
        for fn in files:
            date = os.path.basename(fn).split(".")[0]
            fn_out = str(layout.parquet_day(feed, date))
            if os.path.exists(fn_out):
                continue
            size = os.stat(fn).st_size
            payload = {"feed": feed, "date": date, "fn_in": fn, "fn_out": fn_out, "size_in": size}
            # Largest files first: the scheduler starts tasks in decreasing
            # ``memory`` order, so break ties by input size.
            memory = CONVERSION_TASK_BYTES + buffer_mb * convert.MiB + size // 1024
            tasks.append(sched.Task(key=(feed, date), memory=memory, payload=payload))
    return tasks, n_available


def run_conversion(
    layout: Layout,
    dates: Sequence[str],
    feeds: Sequence[str] = (QUOTES, TRADES),
    *,
    n_partitions: int = 100,
    max_workers: int = 4,
    memory_budget: int | None = None,
    parse_threads: int = 8,
    write_threads: int = 6,
    chunk_mb: int = 64,
    buffer_mb: int = 1024,
    log_file: str | None = None,
) -> int:
    """Convert the flat files of ``dates`` to partitioned Parquet; returns an exit status."""
    for name, value in (
        ("max_workers", max_workers),
        ("parse_threads", parse_threads),
        ("write_threads", write_threads),
        ("chunk_mb", chunk_mb),
        ("buffer_mb", buffer_mb),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    convert.check_partition_hash()
    if convert.DECOMPRESSOR != "isal":
        logger.warning("The isal package is not installed; gzip decompression will be ~6x slower")
    memory_budget = memory_budget or sched.default_memory_budget()

    started = time.time()
    tasks, n_available = discover_conversion_tasks(layout, feeds, dates, buffer_mb)
    for feed in feeds:
        todo = sum(1 for t in tasks if t.payload["feed"] == feed)
        if todo == 0:
            logger.info("%s: all %d flat files of the requested days are converted", feed, n_available[feed])
        else:
            logger.info("%s: %d of %d flat files to convert", feed, todo, n_available[feed])
    if not tasks:
        return 0
    logger.info(
        "Converting %d files using up to %d concurrent files within %s (%d parse + %d write threads each)",
        len(tasks),
        max_workers,
        sched.human_bytes(memory_budget),
        parse_threads,
        write_threads,
    )

    def on_result(task, result, error):
        if error is not None:
            result = {**task.payload, "error": str(type(error)), "error_msg": str(error)}
        if "error" in result:
            logger.error("Failed to convert %s: %s", result["fn_in"], result["error_msg"])
        else:
            logger.info(
                "Converted %s (%.2f GB, %s rows) in %.0fs",
                result["fn_in"],
                result["size_in"] / 1e9,
                f"{result['rows']:,}",
                result["seconds"],
            )
        return result

    options = {
        "n_partitions": n_partitions,
        "chunk_mb": chunk_mb,
        "parse_threads": parse_threads,
        "write_threads": write_threads,
        "buffer_mb": buffer_mb,
    }
    results = _run(
        tasks,
        _convert_task,
        max_workers=max_workers,
        memory_budget=memory_budget,
        initializer=_init_conversion_worker,
        initargs=(options,),
        threads=1,
        log_file=log_file,
        on_result=on_result,
    )
    failures = [r for r in results if "error" in r]
    logger.info("Converted %d/%d files in %.0fs", len(results) - len(failures), len(results), time.time() - started)
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Snapshots: partitioned Parquet -> regularly sampled snapshots
# ---------------------------------------------------------------------------

_RULES: TradeRules | None = None
_INIT_ERROR: BaseException | None = None


def _init_snapshot_worker(conditions_path: str | None, threads: int) -> None:
    # Failures are reported by each task instead of killing the worker, which
    # the scheduler would otherwise treat as a crash.
    global _RULES, _INIT_ERROR
    try:
        import pyarrow as pa

        pa.set_cpu_count(threads)
        pa.set_io_thread_count(max(1, threads))
        try:
            # Hand freed Arrow memory back to the OS right away so that the
            # scheduler's per-task estimates are not inflated by idle workers.
            pa.jemalloc_set_decay_ms(0)
        except NotImplementedError:
            pass
        _RULES = TradeRules.load(conditions_path)
    except Exception as error:  # noqa: BLE001 - surfaced by _snapshot_task
        _INIT_ERROR = error


def _snapshot_task(payload: dict) -> dict:
    result = {k: payload[k] for k in ("date", "partition", "freqs", "input_bytes")}
    sched.reset_peak_rss()
    started = time.time()
    try:
        if _INIT_ERROR is not None:
            raise RuntimeError(f"worker initialisation failed: {_INIT_ERROR!r}")
        if not os.path.isdir(payload["trades_path"]):
            raise FileNotFoundError(
                f"No trades for {payload['date']} partition {payload['partition']}: {payload['trades_path']}"
            )
        layout: Layout = payload["layout"]
        inputs = snapshot.read_inputs(payload["quotes_path"], payload["trades_path"])
        for freq, table in snapshot.iter_snapshots(*inputs, payload["freqs"], _RULES):
            inputs = None  # prepared inputs are kept by the generator; free the raw tables
            fn_out = str(layout.snapshot_file(freq, payload["date"], payload["partition"]))
            for stale in glob.glob(f"{fn_out}.tmp-*"):
                os.remove(stale)
            snapshot.write_table(table, fn_out)
            result.setdefault("rows", {})[freq_label(freq)] = table.num_rows
            del table
        result["processed"] = True
    except Exception as error:  # noqa: BLE001 - reported per partition
        result["processed"] = False
        result["error"] = {"type": str(type(error)), "msg": str(error)}
    result["seconds"] = round(time.time() - started, 3)
    result["peak_rss_bytes"] = sched.peak_rss_bytes()
    return result


def _dir_bytes(path: str) -> int:
    try:
        return sum(entry.stat().st_size for entry in os.scandir(path) if entry.is_file())
    except FileNotFoundError:
        return 0


def _existing_outputs(layout: Layout, freq: float, date: str) -> set:
    """Partition names of a day that already have a snapshot for ``freq``."""
    try:
        entries = [
            entry for entry in os.scandir(layout.snapshot_day(freq, date)) if entry.name.startswith("partition=")
        ]
    except FileNotFoundError:
        return set()
    return {entry.name for entry in entries if os.path.exists(os.path.join(entry.path, "0.parquet"))}


def discover_snapshot_tasks(
    layout: Layout, dates: Sequence[str], freqs: Sequence[float], n_partitions: int, overwrite: bool = False
) -> tuple[list[sched.Task], dict[str, list[int]]]:
    """Tasks for every (date, partition) missing at least one frequency."""

    def scan(date):
        quotes_day = str(layout.parquet_day(QUOTES, date))
        trades_day = str(layout.parquet_day(TRADES, date))
        try:
            available = {entry.name for entry in os.scandir(quotes_day) if entry.is_dir()}
        except FileNotFoundError:
            available = set()
        done = {freq: set() if overwrite else _existing_outputs(layout, freq, date) for freq in freqs}
        tasks, missing = [], []
        for partition in range(n_partitions):
            name = f"partition={partition}"
            if name not in available:
                missing.append(partition)
                continue
            needed = [freq for freq in freqs if name not in done[freq]]
            if not needed:
                continue
            quotes_path = os.path.join(quotes_day, name)
            trades_path = os.path.join(trades_day, name)
            input_bytes = _dir_bytes(quotes_path) + _dir_bytes(trades_path)
            payload = {
                "date": date,
                "partition": partition,
                "freqs": needed,
                "quotes_path": quotes_path,
                "trades_path": trades_path,
                "layout": layout,
                "input_bytes": input_bytes,
            }
            memory = SNAPSHOT_TASK_BASE_BYTES + SNAPSHOT_TASK_BYTES_PER_INPUT_BYTE * input_bytes
            tasks.append(sched.Task(key=(date, partition), memory=memory, payload=payload))
        return date, tasks, missing

    all_tasks, missing_by_date = [], {}
    with ThreadPoolExecutor(32) as pool:
        for date, tasks, missing in pool.map(scan, dates):
            all_tasks.extend(tasks)
            if missing:
                missing_by_date[date] = missing
    return all_tasks, missing_by_date


def parse_freqs(value: str) -> list[float]:
    """Comma-separated positive frequencies in Hz, deduplicated."""
    freqs: list[float] = []
    for item in value.split(","):
        if not item.strip():
            continue
        freq = float(item)
        if not freq > 0:
            raise ValueError(f"Frequencies must be positive, got {item!r}")
        if freq not in freqs:
            freqs.append(freq)
    if not freqs:
        raise ValueError("At least one frequency is required")
    labels = [freq_label(f) for f in freqs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Frequencies map to duplicate output directories: {labels}")
    return freqs


def run_snapshots(
    layout: Layout,
    dates: Sequence[str],
    freqs: Sequence[float],
    *,
    n_partitions: int = 100,
    max_workers: int | None = None,
    memory_budget: int | None = None,
    threads_per_worker: int = 1,
    conditions_path: str | None = None,
    overwrite: bool = False,
    log_file: str | None = None,
) -> int:
    """Build the snapshots of ``freqs`` for ``dates``; returns an exit status."""
    max_workers = max_workers or sched.default_max_workers()
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if threads_per_worker < 1:
        raise ValueError("threads_per_worker must be at least 1")
    TradeRules.load(conditions_path)  # fail fast on a bad rules file
    memory_budget = memory_budget or sched.default_memory_budget()

    started = time.time()
    tasks, missing_by_date = discover_snapshot_tasks(layout, dates, freqs, n_partitions, overwrite)
    incomplete = sorted(d for d, parts in missing_by_date.items() if len(parts) < n_partitions)
    absent = sorted(d for d, parts in missing_by_date.items() if len(parts) == n_partitions)
    if absent:
        shown = ", ".join(absent[:10]) + ("..." if len(absent) > 10 else "")
        logger.warning("%d dates have no quote data: %s", len(absent), shown)
    if incomplete:
        shown = ", ".join(incomplete[:10]) + ("..." if len(incomplete) > 10 else "")
        logger.warning("%d dates are missing data for at least one partition: %s", len(incomplete), shown)
    labels = ", ".join(freq_label(f) for f in freqs)
    if not tasks:
        logger.info("All snapshots (%s) exist for the %d requested dates", labels, len(dates))
        return 0
    logger.info(
        "Creating %s snapshots for %d partitions of %d dates using up to %d workers within %s",
        labels,
        len(tasks),
        len({t.key[0] for t in tasks}),
        max_workers,
        sched.human_bytes(memory_budget),
    )

    def on_result(task, result, error):
        if error is not None:
            result = {**{k: task.payload[k] for k in ("date", "partition", "freqs")}, "processed": False}
            result["error"] = {"type": str(type(error)), "msg": str(error)}
        if "error" in result:
            logger.error(
                "Failed to process %s partition=%s: %s",
                result["date"],
                result["partition"],
                json.dumps(result["error"]),
            )
        return result

    results = _run(
        tasks,
        _snapshot_task,
        max_workers=max_workers,
        memory_budget=memory_budget,
        initializer=_init_snapshot_worker,
        initargs=(conditions_path, threads_per_worker),
        threads=threads_per_worker,
        log_file=log_file,
        on_result=on_result,
    )
    all_errors = [r for r in results if "error" in r]
    missing_input = sorted({r["date"] for r in all_errors if "FileNotFound" in r["error"]["type"]})
    other = [r for r in all_errors if "FileNotFound" not in r["error"]["type"]]
    if other:
        logger.error("%d partitions were not processed successfully", len(other))
    if missing_input:
        logger.warning(
            "%d dates lack trade data for some partitions: %s", len(missing_input), ", ".join(missing_input[:10])
        )
    peak = max((r.get("peak_rss_bytes", 0) for r in results), default=0)
    logger.info(
        "Processed %d/%d partitions in %.0fs (largest worker peak RSS %s)",
        len(results) - len(all_errors),
        len(results),
        time.time() - started,
        sched.human_bytes(peak),
    )
    return 1 if all_errors else 0
