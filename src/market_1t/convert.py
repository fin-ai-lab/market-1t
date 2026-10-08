"""Bounded-memory conversion of Massive (formerly Polygon.io) flat files to partitioned Parquet.

A gzip CSV flat file flows through three threads:

1. decompression (ISA-L via the ``isal`` package, zlib otherwise) into a small
   ring of reusable buffers cut at line boundaries,
2. parsing of each buffer with Arrow's multi-threaded CSV reader and hashing of
   its tickers to partitions,
3. routing of the rows to one buffer per partition; full buffers are written
   as Parquet row groups by a pool of writer threads.

Rows keep their file order within each partition and memory is bounded by the
chunk size, the queue lengths and the partition buffer budget, independently
of the file size (~2-3 GiB for a 10 GB quotes file).

Output::

    {output}/{date}.parquet/partition={p}/00000000.parquet

with the feed's columns in schema order plus ``partition`` (uint64), where
``partition = polars.col("ticker").hash(42) % n_partitions``.

Missing values follow the Polars CSV reader: only empty unquoted fields are
null (Arrow's default null tokens would turn the real ticker ``NA``, or
strings such as ``NULL``, into nulls).  Blank lines are skipped.  The output
matches the original conversion (``tests/reference/legacy_conversion.py``).
"""

from __future__ import annotations

import os
import queue
import shutil
import threading
import zlib
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

try:  # ISA-L inflate is ~6x faster than zlib.
    from isal import isal_zlib as _inflate_lib

    DECOMPRESSOR = "isal"
except ImportError:  # pragma: no cover - depends on the environment
    _inflate_lib = zlib
    DECOMPRESSOR = "zlib"

# Feed schemas in CSV column order and the columns kept in the Parquet files.
SCHEMAS_BY_FEED = {
    "quotes_v1": {
        "ticker": pa.large_string(),
        "ask_exchange": pa.int64(),
        "ask_price": pa.float64(),
        "ask_size": pa.int64(),
        "bid_exchange": pa.int64(),
        "bid_price": pa.float64(),
        "bid_size": pa.int64(),
        "conditions": pa.large_string(),
        "indicators": pa.large_string(),
        "participant_timestamp": pa.int64(),
        "sequence_number": pa.int64(),
        "sip_timestamp": pa.int64(),
        "tape": pa.int64(),
        "trf_timestamp": pa.int64(),
    },
    "trades_v1": {
        "ticker": pa.large_string(),
        "conditions": pa.large_string(),
        "correction": pa.int64(),
        "exchange": pa.int64(),
        "id": pa.int64(),
        "participant_timestamp": pa.int64(),
        "price": pa.float64(),
        "sequence_number": pa.int64(),
        "sip_timestamp": pa.int64(),
        # SIP flat files can contain fractional-share trades (for example
        # size=0.140000), so this must not be parsed as an integer.
        "size": pa.float64(),
        "tape": pa.int64(),
        "trf_id": pa.int64(),
        "trf_timestamp": pa.int64(),
    },
}

COLS_BY_FEED = {
    "quotes_v1": [
        "ticker",
        "sip_timestamp",
        "sequence_number",
        "ask_exchange",
        "ask_price",
        "ask_size",
        "bid_exchange",
        "bid_price",
        "bid_size",
        "conditions",
        "indicators",
    ],
    "trades_v1": ["ticker", "sip_timestamp", "sequence_number", "exchange", "price", "size", "conditions"],
}

PARTITION_SEED = 42
MiB = 1024**2

# Parquet encoding.  Delta encoding suits the near-monotonic timestamp and
# sequence columns; the rest are dictionary encoded.  With zstd level 3 the
# files are ~10% smaller than Polars' defaults and ~30% smaller than Arrow's
# defaults, and decode just as fast.
DELTA_COLUMNS = ("sip_timestamp", "sequence_number")
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 3


def output_columns(feed: str) -> list[str]:
    """Kept columns in schema (not projection) order, as Polars writes them."""
    keep = set(COLS_BY_FEED[feed])
    return [column for column in SCHEMAS_BY_FEED[feed] if column in keep]


def output_schema(feed: str) -> pa.Schema:
    types = SCHEMAS_BY_FEED[feed]
    return pa.schema([pa.field(c, types[c]) for c in output_columns(feed)] + [pa.field("partition", pa.uint64())])


def ticker_partitions(tickers: Sequence[str | None], n_partitions: int) -> np.ndarray:
    """Partition of each ticker under the reference partitioning.

    Polars documents ``hash`` as stable only within one Polars version; see
    ``check_partition_hash``.
    """
    series = pl.Series("ticker", list(tickers), dtype=pl.String)
    return (series.hash(PARTITION_SEED) % n_partitions).to_numpy().astype(np.int64)


# Partitions of a few tickers under the reference partitioning (n_partitions=100),
# as found in Parquet data converted in 2008, 2024 and 2026.  A null ticker
# lands in 42.
_KNOWN_PARTITIONS = {
    "A": 42,
    "AAPL": 52,
    "BRK.A": 68,
    "GME": 98,
    "MSFT": 69,
    "NA": 96,
    "QQQ": 46,
    "SPY": 77,
    "TSLA": 86,
    "ZVZZT": 39,
    None: 42,
}


def check_partition_hash() -> None:
    """Fail fast if the installed Polars hashes tickers differently."""
    tickers = list(_KNOWN_PARTITIONS)
    got = dict(zip(tickers, ticker_partitions(tickers, 100).tolist(), strict=True))
    if got != _KNOWN_PARTITIONS:
        raise RuntimeError(
            f"polars {pl.__version__} assigns tickers to different partitions than the reference partitioning "
            f"(expected {_KNOWN_PARTITIONS}, got {got}); Polars changed its hash in 1.26, use the versions pinned in "
            "uv.lock (`uv sync`)."
        )


# ---------------------------------------------------------------------------
# Decompression into newline-aligned chunks.
# ---------------------------------------------------------------------------


def iter_gzip_blocks(path: str, read_bytes: int = MiB // 2, max_output: int = 4 * MiB) -> Iterator[bytes]:
    """Decompressed blocks of a (possibly multi-member) gzip file.

    Blocks of a few MiB are markedly faster than large ones: large output
    buffers are page-faulted in afresh for every call.
    """
    with open(path, "rb", buffering=0) as raw:
        decompressor = _inflate_lib.decompressobj(31)
        started = False  # has the current member received any input?
        pending = b""
        while True:
            if not pending:
                pending = raw.read(read_bytes)
                if not pending:
                    break
            started = True
            block = decompressor.decompress(pending, max_output)
            pending = decompressor.unconsumed_tail
            if block:
                yield block
            if decompressor.eof:
                # Concatenated gzip members: continue with the next one.  The
                # rest of the input is ``unused_data`` (with an output limit it
                # is also echoed in ``unconsumed_tail``).  Like the gzip module,
                # tolerate NUL padding after a member.
                pending = decompressor.unused_data.lstrip(b"\x00")
                decompressor = _inflate_lib.decompressobj(31)
                started = False
                while not pending:
                    more = raw.read(read_bytes)
                    if not more:
                        break
                    pending = more.lstrip(b"\x00")
        if started:
            tail = decompressor.flush()
            if tail:
                yield tail
            if not decompressor.eof:
                raise EOFError(f"Compressed file ended before the end-of-stream marker was reached: {path}")


class Stopped(Exception):
    """Raised inside a pipeline stage when another stage has failed."""


class BufferRing:
    """A fixed set of reusable byte buffers (recycling avoids page faults)."""

    def __init__(self, count: int, capacity: int, stop: threading.Event | None = None):
        self.capacity = capacity
        self.stop = stop or threading.Event()
        self._free: queue.Queue[np.ndarray] = queue.Queue()
        for _ in range(count):
            self._free.put(np.empty(capacity, dtype=np.uint8))

    def acquire(self) -> np.ndarray:
        while True:
            try:
                return self._free.get(timeout=0.2)
            except queue.Empty:
                if self.stop.is_set():
                    raise Stopped() from None

    def release(self, buffer: np.ndarray) -> None:
        self._free.put(buffer)


def iter_line_chunks(blocks: Iterator[bytes], chunk_bytes: int, ring: BufferRing) -> Iterator[tuple[np.ndarray, int]]:
    """Copy blocks into ring buffers; yield ``(buffer, length)`` cut after a newline.

    Chunks hold at least ``chunk_bytes`` (except the last) and end with a
    newline unless the input does not.  The consumer releases each buffer to
    the ring once it no longer needs its contents.
    """
    buffer = ring.acquire()
    filled = 0
    last_newline = -1
    try:
        for block in blocks:
            size = len(block)
            if filled + size > ring.capacity:
                raise ValueError(f"CSV line longer than {ring.capacity - chunk_bytes} bytes")
            buffer[filled : filled + size] = np.frombuffer(block, dtype=np.uint8)
            position = block.rfind(b"\n")
            if position >= 0:
                last_newline = filled + position
            filled += size
            if filled >= chunk_bytes and last_newline >= 0:
                cut = last_newline + 1
                following = ring.acquire()
                rest = filled - cut
                following[:rest] = buffer[cut:filled]
                chunk = buffer
                buffer, filled, last_newline = following, rest, -1
                yield chunk, cut
    except BaseException:
        ring.release(buffer)
        raise
    if filled:
        yield buffer, filled
    else:
        ring.release(buffer)


# ---------------------------------------------------------------------------
# Partitioned writing.
# ---------------------------------------------------------------------------


def _copy_table(table: pa.Table) -> pa.Table:
    """Compact deep copy, so a small slice no longer pins its parent chunk."""
    return pa.Table.from_arrays([pa.concat_arrays(column.chunks) for column in table.columns], schema=table.schema)


@dataclass
class _PartitionBuffer:
    tables: list[pa.Table] = field(default_factory=list)
    rows: int = 0
    bytes: int = 0


class PartitionedWriter:
    """Buffers rows per partition and writes row groups in background threads.

    Writes of one partition always run on the same thread, so each file is
    appended to in order.  ``buffer_bytes`` bounds the rows held in memory
    (buffered plus queued for writing).
    """

    def __init__(
        self,
        root: str,
        schema: pa.Schema,
        n_partitions: int,
        *,
        row_group_rows: int = 1024 * 1024,
        buffer_bytes: int = 1024 * MiB,
        n_threads: int = 6,
    ):
        self.root = root
        self.schema = schema
        self.n_partitions = n_partitions
        self.row_group_rows = row_group_rows
        self.buffer_bytes = buffer_bytes
        delta = [c for c in DELTA_COLUMNS if c in schema.names]
        self.writer_options = dict(
            compression=COMPRESSION,
            compression_level=COMPRESSION_LEVEL,
            use_dictionary=[c for c in schema.names if c not in delta],
            column_encoding={c: "DELTA_BINARY_PACKED" for c in delta},
            write_statistics=True,
        )
        self.buffers = [_PartitionBuffer() for _ in range(n_partitions)]
        self.buffered = 0
        self.writers: dict[int, pq.ParquetWriter] = {}
        self.rows_written = np.zeros(n_partitions, dtype=np.int64)
        self.executors = [ThreadPoolExecutor(1) for _ in range(max(1, n_threads))]
        self.inflight: deque = deque()  # (future, nbytes)
        self.inflight_bytes = 0

    def add(self, table: pa.Table, partitions: np.ndarray) -> None:
        """Append ``table`` rows; ``partitions`` holds each row's partition."""
        n = table.num_rows
        if n == 0:
            return
        breaks = np.flatnonzero(partitions[1:] != partitions[:-1]) + 1
        if len(breaks) > 256:
            # Not sorted by ticker: group rows by partition, keeping order.
            order = np.argsort(partitions, kind="stable")
            table = table.take(pa.array(order))
            partitions = partitions[order]
            breaks = np.flatnonzero(partitions[1:] != partitions[:-1]) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [n]))
        for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
            piece = table.slice(start, end - start)
            if 2 * (end - start) < n:
                # Copy small pieces so they do not pin the whole parsed chunk.
                piece = _copy_table(piece)
            self._append(int(partitions[start]), piece)
        self._enforce_budget()

    def _append(self, partition: int, piece: pa.Table) -> None:
        buf = self.buffers[partition]
        buf.tables.append(piece)
        buf.rows += piece.num_rows
        nbytes = piece.nbytes
        buf.bytes += nbytes
        self.buffered += nbytes
        if buf.rows >= self.row_group_rows:
            self._flush(partition)

    def _enforce_budget(self) -> None:
        while self.buffered + self.inflight_bytes > self.buffer_bytes:
            if self.inflight and (self.inflight_bytes > self.buffer_bytes // 2 or self.buffered == 0):
                self._wait_oldest()
                continue
            largest = max(range(self.n_partitions), key=lambda p: self.buffers[p].bytes)
            if self.buffers[largest].bytes == 0:
                break
            self._flush(largest)

    def _wait_oldest(self) -> None:
        future, nbytes = self.inflight.popleft()
        future.result()
        self.inflight_bytes -= nbytes

    def _flush(self, partition: int) -> None:
        buf = self.buffers[partition]
        if buf.rows == 0:
            return
        table = pa.concat_tables(buf.tables) if len(buf.tables) > 1 else buf.tables[0]
        self.buffered -= buf.bytes
        self.buffers[partition] = _PartitionBuffer()
        executor = self.executors[partition % len(self.executors)]
        self.inflight.append((executor.submit(self._write, partition, table), buf.bytes))
        self.inflight_bytes += buf.bytes
        self.rows_written[partition] += table.num_rows

    def _write(self, partition: int, table: pa.Table) -> None:
        writer = self.writers.get(partition)
        if writer is None:
            directory = os.path.join(self.root, f"partition={partition}")
            os.makedirs(directory, exist_ok=False)
            writer = pq.ParquetWriter(os.path.join(directory, "00000000.parquet"), self.schema, **self.writer_options)
            self.writers[partition] = writer
        column = pa.array(np.full(table.num_rows, partition, dtype=np.uint64))
        writer.write_table(pa.Table.from_arrays(table.columns + [column], schema=self.schema))

    def close(self) -> np.ndarray:
        """Flush everything, close the files and return rows per partition."""
        try:
            for partition in range(self.n_partitions):
                self._flush(partition)
            while self.inflight:
                self._wait_oldest()
        finally:
            for executor in self.executors:
                executor.shutdown(wait=True)
            for writer in self.writers.values():
                writer.close()
        return self.rows_written

    def abort(self) -> None:
        for executor in self.executors:
            executor.shutdown(wait=True, cancel_futures=True)
        for writer in self.writers.values():
            try:
                writer.close()
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass


# ---------------------------------------------------------------------------
# File conversion.
# ---------------------------------------------------------------------------


@dataclass
class ConversionStats:
    rows: int
    rows_per_partition: np.ndarray
    decompressor: str


def _chunks_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    return np.concatenate([chunk.to_numpy(zero_copy_only=False) for chunk in column.chunks])


def row_partitions(tickers: pa.ChunkedArray, cache: dict[str | None, int], n_partitions: int) -> np.ndarray:
    """Partition of every row, hashing each distinct ticker only once."""
    distinct = pc.unique(tickers)
    values = distinct.to_pylist()  # includes None if there are null tickers
    new = [v for v in values if v not in cache]
    if new:
        cache.update(zip(new, ticker_partitions(new, n_partitions).tolist(), strict=True))
    lookup = np.array([cache[v] for v in values], dtype=np.int64)
    index = pc.index_in(tickers, value_set=distinct, skip_nulls=False)
    return lookup[_chunks_to_numpy(index)]


def _put(q: queue.Queue, item, stop: threading.Event) -> None:
    while True:
        try:
            q.put(item, timeout=0.2)
            return
        except queue.Full:
            if stop.is_set():
                raise Stopped() from None


def _get(q: queue.Queue, stop: threading.Event):
    while True:
        try:
            return q.get(timeout=0.2)
        except queue.Empty:
            if stop.is_set():
                raise Stopped() from None


class _Stage(threading.Thread):
    """Pipeline thread; a failure is recorded and stops the whole pipeline."""

    def __init__(self, target, stop: threading.Event):
        super().__init__(daemon=True)
        # Note: threading.Thread defines _stop() and _target; avoid those names.
        self.stage_fn, self.stop_event = target, stop
        self.error: BaseException | None = None

    def run(self):
        try:
            self.stage_fn()
        except Stopped:
            pass
        except BaseException as error:  # noqa: BLE001 - re-raised by the main thread
            self.error = error
            self.stop_event.set()


def convert_file(
    path: str,
    output_dir: str,
    feed: str,
    n_partitions: int,
    *,
    chunk_bytes: int = 64 * MiB,
    parse_threads: int = 8,
    write_threads: int = 6,
    buffer_bytes: int = 1024 * MiB,
    row_group_rows: int = 1024 * 1024,
    queue_chunks: int = 2,
) -> ConversionStats:
    """Convert one gzip CSV flat file into ``output_dir/partition=<p>/00000000.parquet``.

    ``output_dir`` must not exist.  Raises if the file is malformed or if any
    partition ends up without rows (a day of the SIP feeds fills all of them).
    """
    columns = output_columns(feed)
    types = SCHEMAS_BY_FEED[feed]
    convert_options = pacsv.ConvertOptions(
        column_types={c: types[c] for c in columns},
        include_columns=columns,
        null_values=[""],
        strings_can_be_null=True,
        quoted_strings_can_be_null=False,
    )
    pa.set_cpu_count(max(1, parse_threads))

    stop = threading.Event()
    ring = BufferRing(queue_chunks + 3, chunk_bytes + 16 * MiB, stop)
    raw_chunks: queue.Queue = queue.Queue(maxsize=queue_chunks)
    parsed: queue.Queue = queue.Queue(maxsize=queue_chunks)

    def decompress():
        for item in iter_line_chunks(iter_gzip_blocks(path), chunk_bytes, ring):
            _put(raw_chunks, item, stop)
        _put(raw_chunks, None, stop)

    def parse():
        cache: dict[str | None, int] = {}
        read_options = None
        while True:
            item = _get(raw_chunks, stop)
            if item is None:
                _put(parsed, None, stop)
                return
            buffer, length = item
            start = 0
            if read_options is None:
                head = buffer[: min(length, 1 << 16)].tobytes()
                end = head.find(b"\n")
                if end < 0:
                    raise ValueError(f"CSV header line not found in {path}")
                header = head[:end].decode("utf-8").rstrip("\r").split(",")
                missing = [c for c in columns if c not in header]
                if missing:
                    raise ValueError(f"{path} lacks columns {missing}")
                read_options = pacsv.ReadOptions(use_threads=True, block_size=4 * MiB, column_names=header)
                start = end + 1
            data = pa.py_buffer(buffer).slice(start, length - start)
            try:
                table = pacsv.read_csv(
                    pa.BufferReader(data), read_options=read_options, convert_options=convert_options
                )
            finally:
                del data
                ring.release(buffer)  # parsed tables do not reference the input
            if table.num_rows:
                _put(parsed, (table, row_partitions(table.column("ticker"), cache, n_partitions)), stop)

    os.makedirs(output_dir, exist_ok=False)
    writer = PartitionedWriter(
        output_dir,
        output_schema(feed),
        n_partitions,
        row_group_rows=row_group_rows,
        buffer_bytes=buffer_bytes,
        n_threads=write_threads,
    )
    stages = [_Stage(decompress, stop), _Stage(parse, stop)]

    def join_stages():
        for stage in stages:
            if stage.ident is not None:  # started
                stage.join()

    total_rows = 0
    try:
        for stage in stages:
            stage.start()
        while True:
            item = _get(parsed, stop)
            if item is None:
                break
            table, partitions = item
            writer.add(table, partitions)
            total_rows += table.num_rows
        rows_per_partition = writer.close()
    except BaseException as error:
        stop.set()
        writer.abort()
        join_stages()
        failures = [stage.error for stage in stages if stage.error is not None]
        if failures:
            raise failures[0] from (None if isinstance(error, Stopped) else error)
        raise
    finally:
        stop.set()
        join_stages()
    if total_rows == 0:
        raise RuntimeError(f"Conversion produced no rows for {path}")
    missing = int(np.sum(rows_per_partition == 0))
    if missing:
        raise RuntimeError(f"Expected {n_partitions} partition directories/files; {missing} partitions have no rows")
    return ConversionStats(rows=total_rows, rows_per_partition=rows_per_partition, decompressor=DECOMPRESSOR)


def convert_to(path: str, fn_out: str, feed: str, n_partitions: int, **options) -> ConversionStats:
    """Convert atomically: build ``fn_out`` in a temporary directory, then rename."""
    parent = os.path.dirname(fn_out)
    os.makedirs(parent, exist_ok=True)
    base = os.path.basename(fn_out)
    for stale in os.listdir(parent):
        if stale.startswith(f"{base}.tmp-"):
            shutil.rmtree(os.path.join(parent, stale), ignore_errors=True)
    temp_out = f"{fn_out}.tmp-{os.getpid()}"
    try:
        stats = convert_file(path, temp_out, feed, n_partitions, **options)
        os.rename(temp_out, fn_out)
        return stats
    finally:
        if os.path.exists(temp_out):
            shutil.rmtree(temp_out, ignore_errors=True)
