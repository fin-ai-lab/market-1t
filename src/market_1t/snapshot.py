"""Vectorised snapshot engine for Massive (formerly Polygon.io) quotes and trades.

For one (date, partition) of the ticker-partitioned Parquet data, builds the
snapshots of any number of sampling frequencies from a single read: per
ticker and interval, the last NBBO quote plus OHLCV and VWAP aggregates of the
eligible trades, forward filled per ticker.

The output reproduces, bit for bit, the original pandas implementation
(``tests/reference/legacy_snapshot.py``), including its numerical artefacts,
which are part of the published data and are reproduced deliberately:

* Interval assignment divides the int64 SIP timestamp by the interval in
  float64 (``np.ceil(ts / interval_ns)``), so timestamps within ~128ns of a
  boundary can land in the neighbouring interval.
* The quote/trade outer merge upcasts ``sip_timestamp_quote`` and
  ``sip_timestamp_trade`` to float64 whenever either side is missing for some
  interval of the partition, rounding every value of that column to float64
  precision (a multiple of 256ns for current timestamps).  ``n`` becomes
  float64 under the same condition.
* Interval sums use pandas' Kahan-compensated group sum; a naive sum differs in
  the last bit for many multi-trade intervals.
* The mean of (open, high, low, close) used for ``vwap`` is evaluated as
  ``(((o + h) + l) + c) / count`` with missing values treated as zero.
* ``first``/``last``/``max``/``min``/``sum`` skip missing values, NaN produced
  by arithmetic is treated as missing (and forward filled) and missing floats
  are written as Parquet nulls.
* Quotes and trades whose ticker is null are ignored (pandas drops null group
  keys), but a null-ticker trade can still be the first eligible trade that
  fixes the session date used for the Form T correction.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pytz import timezone

from .conditions import HAS_FORM_T, HIGH_LOW, OPEN_CLOSE, VOLUME, TradeRules

EST = timezone("US/Eastern")

QUOTE_COLUMNS = ["ticker", "sip_timestamp", "sequence_number", "ask_price", "ask_size", "bid_price", "bid_size"]
TRADE_COLUMNS = ["ticker", "sip_timestamp", "sequence_number", "conditions", "price", "size"]

OUTPUT_COLUMNS = [
    "ticker",
    "ts_interval",
    "sip_timestamp_quote",
    "ask_price",
    "ask_size",
    "bid_price",
    "bid_size",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "n",
    "sip_timestamp_trade",
    "vwap_all",
    "vwap",
    "vwap_ol",
]
QUOTE_VALUE_COLUMNS = ["ask_price", "ask_size", "bid_price", "bid_size"]
TRADE_PRICE_COLUMNS = ["open", "high", "low", "close"]
VWAP_COLUMNS = ["vwap_all", "vwap", "vwap_ol"]
_INT_OUTPUT_COLUMNS = {"ts_interval", "sip_timestamp_quote", "sip_timestamp_trade"}

_INT_PATTERN = r"^-?[0-9]+$"
_FLOAT_PATTERN = r"^-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)$"


def interval_ns_for(freq_hz: float) -> int:
    """Interval length exactly as computed by the legacy code."""
    return int(1e9 / freq_hz)


def legacy_market_hours(first_timestamp: int) -> tuple[int, int]:
    """Regular session bounds used for the Form T volume correction.

    The legacy code derives the session date from the UTC calendar date of the
    first eligible trade of the partition (in file order) and adds 9.5 absolute
    hours to local midnight.  Both are kept for fidelity.
    """
    date_str = pd.to_datetime(first_timestamp).strftime("%Y-%m-%d")
    ts_open = int(
        (pd.to_datetime(date_str).tz_localize(EST) + pd.Timedelta(9.5, "h")).tz_convert("UTC").timestamp() * 1e9
    )
    ts_close = ts_open + int((16 - 9.5) * 60 * 60 * 1e9)
    return ts_open, ts_close


# ---------------------------------------------------------------------------
# Input.  The partitioned Parquet written over the years is not uniformly
# typed (some days store every quote column as strings, older trade files store
# ``size`` as int64), so every column is coerced with the semantics of the
# legacy ``astype(int)`` / ``astype(float)`` calls.
# ---------------------------------------------------------------------------


def read_partition(path: str, columns: Sequence[str], dictionary_columns: Sequence[str] = ()) -> pa.Table:
    """Read one ``partition=<p>`` directory like ``pandas.read_parquet(path)``.

    ``dictionary_columns`` are decoded to dictionary arrays, which carries the
    same values while avoiding per-row string materialisation.
    """
    files = [e.name for e in os.scandir(path) if not e.name.startswith((".", "_"))]
    if len(files) == 1 and files[0].endswith(".parquet") and os.path.isfile(os.path.join(path, files[0])):
        source = os.path.join(path, files[0])
        return pq.ParquetFile(source, read_dictionary=list(dictionary_columns)).read(columns=list(columns))
    # Any other layout: defer to the dataset discovery used by pandas.
    return pq.read_table(path, columns=list(columns), read_dictionary=list(dictionary_columns))


def _chunks_to_numpy(column: pa.ChunkedArray, dtype) -> np.ndarray:
    if column.num_chunks == 1:
        return column.chunk(0).to_numpy(zero_copy_only=False).astype(dtype, copy=False)
    if column.num_chunks == 0:
        return np.zeros(0, dtype=dtype)
    return np.concatenate([chunk.to_numpy(zero_copy_only=False).astype(dtype, copy=False) for chunk in column.chunks])


def _is_string_type(arrow_type: pa.DataType) -> bool:
    return pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type)


def as_int64(column: pa.ChunkedArray, name: str) -> np.ndarray:
    """Equivalent of ``Series.astype(int)`` for the timestamp/sequence columns."""
    arrow_type = column.type
    if column.null_count:
        raise ValueError(f"Cannot convert non-finite values (NA or inf) to integer in column '{name}'")
    if pa.types.is_integer(arrow_type):
        return _chunks_to_numpy(column, np.int64)
    if pa.types.is_floating(arrow_type):
        values = _chunks_to_numpy(column, np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Cannot convert non-finite values (NA or inf) to integer in column '{name}'")
        return values.astype(np.int64)
    if _is_string_type(arrow_type):
        series = pl.from_arrow(column)
        if series.str.contains(_INT_PATTERN).all():
            return series.cast(pl.Int64, strict=True).to_numpy()
        # Unusual spellings (whitespace, '+', underscores): defer to Python.
        return np.array([int(value) for value in series.to_list()], dtype=np.int64)
    raise TypeError(f"Unsupported type {arrow_type} for integer column '{name}'")


def as_float64(column: pa.ChunkedArray, name: str) -> np.ndarray:
    """Equivalent of ``Series.astype(float)``: NaN marks missing values."""
    arrow_type = column.type
    if pa.types.is_floating(arrow_type) or pa.types.is_integer(arrow_type):
        # Nulls become NaN, exactly like pandas' conversion of Arrow data.
        return _chunks_to_numpy(column, np.float64)
    if pa.types.is_null(arrow_type):
        return np.full(len(column), np.nan)
    if _is_string_type(arrow_type):
        series = pl.from_arrow(column)
        if series.drop_nulls().str.contains(_FLOAT_PATTERN).all():
            # Plain decimals: correctly rounded parsing is identical to float().
            return series.cast(pl.Float64, strict=True).fill_null(np.nan).to_numpy()
        return np.array([np.nan if v is None else float(v) for v in series.to_list()], dtype=np.float64)
    raise TypeError(f"Unsupported type {arrow_type} for float column '{name}'")


def _dictionary_chunks(column: pa.ChunkedArray, name: str) -> list[pa.DictionaryArray]:
    """Dictionary-encoded chunks of a string column."""
    if pa.types.is_dictionary(column.type):
        chunks = column.chunks
    elif _is_string_type(column.type) or pa.types.is_null(column.type):
        chunks = [chunk.dictionary_encode() for chunk in column.chunks]
    else:
        raise TypeError(f"Unsupported type {column.type} for string column '{name}'")
    for chunk in chunks:
        if not (_is_string_type(chunk.dictionary.type) or pa.types.is_null(chunk.dictionary.type)):
            raise TypeError(f"Unsupported dictionary type {chunk.dictionary.type} for column '{name}'")
    return chunks


def dictionary_values(column: pa.ChunkedArray, name: str) -> set:
    values = set()
    for chunk in _dictionary_chunks(column, name):
        values.update(v for v in chunk.dictionary.to_pylist() if v is not None)
    return values


def map_dictionary(column: pa.ChunkedArray, name: str, mapping, null_value: int) -> np.ndarray:
    """Map every row of a string column through ``mapping(value) -> int``."""
    out = []
    for chunk in _dictionary_chunks(column, name):
        lookup = np.array(
            [mapping(v) if v is not None else null_value for v in chunk.dictionary.to_pylist()] + [null_value]
        )
        indices = chunk.indices.to_numpy(zero_copy_only=False)
        if chunk.indices.null_count:
            indices = np.where(chunk.indices.is_valid().to_numpy(zero_copy_only=False), indices, len(lookup) - 1)
        out.append(lookup[indices.astype(np.int64, copy=False)])
    if not out:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(out) if len(out) > 1 else out[0]


# ---------------------------------------------------------------------------
# NumPy helpers on data sorted by group.
# ---------------------------------------------------------------------------


def _run_starts(*keys: np.ndarray) -> np.ndarray:
    """Start offsets of runs of identical key tuples (data sorted by keys)."""
    n = len(keys[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    brk = np.empty(n, dtype=bool)
    brk[0] = True
    np.not_equal(keys[0][1:], keys[0][:-1], out=brk[1:])
    for key in keys[1:]:
        brk[1:] |= key[1:] != key[:-1]
    return np.flatnonzero(brk)


def _group_first_valid(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    n = len(values)
    idx = np.where(np.isnan(values), n, np.arange(n))
    first = np.minimum.reduceat(idx, starts)
    ok = first < n
    return np.where(ok, values[np.where(ok, first, 0)], np.nan)


def _group_last_valid(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    idx = np.where(np.isnan(values), -1, np.arange(len(values)))
    last = np.maximum.reduceat(idx, starts)
    ok = last >= 0
    return np.where(ok, values[np.where(ok, last, 0)], np.nan)


def _group_codes(starts: np.ndarray, n: int) -> np.ndarray:
    return np.repeat(np.arange(len(starts)), np.diff(np.append(starts, n)))


def _group_extreme(values: np.ndarray, starts: np.ndarray, maximum: bool) -> np.ndarray:
    """NaN-skipping group max/min with pandas' first-wins tie handling."""
    if np.any((values == 0) & np.signbit(values)):
        # -0.0 == 0.0 but the two differ in the output; defer to pandas, whose
        # strict comparison keeps the first of equal values.
        grouped = pd.Series(values).groupby(_group_codes(starts, len(values)), sort=False)
        return (grouped.max() if maximum else grouped.min()).to_numpy()
    return (np.fmax if maximum else np.fmin).reduceat(values, starts)


def _group_kahan_sum(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """NaN-skipping group sums identical to pandas' Kahan-compensated group_sum.

    ``values`` has shape (rows, columns) and is sorted by group.  With at most
    two values the compensation term never feeds back, so the result is
    ``(0.0 + v1) + v2``; larger groups go through pandas' own kernel.
    """
    n_rows = values.shape[0]
    sizes = np.diff(np.append(starts, n_rows))
    first = values[starts]
    sums = np.where(np.isnan(first), 0.0, 0.0 + first)
    pair = sizes >= 2
    if pair.any():
        second = values[starts[pair] + 1]
        sums[pair] = np.where(np.isnan(second), sums[pair], sums[pair] + second)
    big = sizes >= 3
    if big.any():
        rows = np.flatnonzero(np.repeat(big, sizes))
        codes = np.repeat(np.arange(np.count_nonzero(big)), sizes[big])
        sums[big] = pd.DataFrame(values[rows]).groupby(codes, sort=False).sum().to_numpy()
    return sums


def _forward_fill(values: np.ndarray, valid: np.ndarray, run_start: np.ndarray, fill) -> tuple[np.ndarray, np.ndarray]:
    """Forward fill within runs; positions with no earlier valid value get ``fill``.

    Returns the filled values and their validity.
    """
    if valid.all():
        return values, valid
    src = np.maximum.accumulate(np.where(valid, np.arange(len(values)), -1))
    ok = src >= run_start
    return np.where(ok, values[np.where(ok, src, 0)], fill), ok


# ---------------------------------------------------------------------------
# Snapshot construction.
# ---------------------------------------------------------------------------


@dataclass
class PreparedQuotes:
    tid: np.ndarray  # ticker ids (sort like the ticker strings)
    ts: np.ndarray  # int64 SIP timestamps
    values: dict[str, np.ndarray]  # float64, NaN marks missing
    has_missing: bool


@dataclass
class PreparedTrades:
    tid: np.ndarray
    ts: np.ndarray
    prices: dict[str, np.ndarray]  # open/high/low/close, NaN when ineligible
    n: np.ndarray  # int64
    sums: np.ndarray  # (rows, 6): volume, _dol_vol_all, _vol, _dol_vol, _vol_ol, _dol_vol_ol
    market_hours: tuple[int, int]


def _is_lexsorted(tid: np.ndarray, ts: np.ndarray, seq: np.ndarray, block: int = 1 << 20) -> bool:
    """Whether rows are ordered by (tid, ts, seq); scans in cache-sized blocks."""
    for start in range(0, len(tid) - 1, block):
        stop = min(start + block, len(tid) - 1)
        prev, nxt = slice(start, stop), slice(start + 1, stop + 1)
        tid_step = tid[nxt] - tid[prev]
        ts_step = ts[nxt] - ts[prev]
        if np.any(tid_step < 0) or np.any((tid_step == 0) & (ts_step < 0)):
            return False
        ties = np.flatnonzero((tid_step == 0) & (ts_step == 0)) + start
        if ties.size and np.any(seq[ties + 1] < seq[ties]):
            return False
    return True


def _sort_order(tid: np.ndarray, ts: np.ndarray, seq: np.ndarray) -> np.ndarray | None:
    """Stable (ticker, sip_timestamp, sequence_number) order; None if already sorted."""
    if _is_lexsorted(tid, ts, seq):
        return None
    return np.lexsort((seq, ts, tid))


def prepare_quotes(table: pa.Table, ticker_ids) -> PreparedQuotes:
    ts = as_int64(table.column("sip_timestamp"), "sip_timestamp")
    seq = as_int64(table.column("sequence_number"), "sequence_number")
    values = {}
    for col in QUOTE_VALUE_COLUMNS:
        values[col] = as_float64(table.column(col), col)
    # NBBO sizes are reported in round lots; convert to shares.
    values["ask_size"] = values["ask_size"] * 100
    values["bid_size"] = values["bid_size"] * 100
    tid = map_dictionary(table.column("ticker"), "ticker", ticker_ids.__getitem__, -1)

    # pandas drops rows whose group key (ticker) is null.
    keep = tid >= 0
    if not keep.all():
        tid, ts, seq = tid[keep], ts[keep], seq[keep]
        values = {k: v[keep] for k, v in values.items()}
    order = _sort_order(tid, ts, seq)
    if order is not None:
        tid, ts = tid[order], ts[order]
        values = {k: v[order] for k, v in values.items()}
    has_missing = any(np.isnan(v).any() for v in values.values())
    return PreparedQuotes(tid=tid, ts=ts, values=values, has_missing=has_missing)


def prepare_trades(table: pa.Table, ticker_ids, rules: TradeRules) -> PreparedTrades:
    ts = as_int64(table.column("sip_timestamp"), "sip_timestamp")
    seq = as_int64(table.column("sequence_number"), "sequence_number")
    price = as_float64(table.column("price"), "price")
    size = as_float64(table.column("size"), "size")
    # Condition rules are evaluated once per distinct conditions string.
    bits = map_dictionary(table.column("conditions"), "conditions", rules.bits, rules.bits(None))

    with np.errstate(invalid="ignore"):
        size_ge1 = size >= 1  # NaN compares False, as in pandas
    updates_high_low = ((bits & HIGH_LOW) != 0) & size_ge1
    updates_open_close = ((bits & OPEN_CLOSE) != 0) & size_ge1
    updates_volume = (bits & VOLUME) != 0
    keep = np.flatnonzero(updates_high_low | updates_open_close | updates_volume)
    if keep.size == 0:
        # The legacy code fails with ``.iloc[0]`` on the empty selection.
        raise IndexError("single positional indexer is out-of-bounds")
    ts, seq, price, size, bits = ts[keep], seq[keep], price[keep], size[keep], bits[keep]
    updates_high_low, updates_open_close, updates_volume = (
        updates_high_low[keep],
        updates_open_close[keep],
        updates_volume[keep],
    )

    # Form T trades only count towards volume outside regular hours.  The
    # session is derived from the first eligible trade in file order.
    ts_open, ts_close = legacy_market_hours(int(ts[0]))
    updates_volume &= ~((ts >= ts_open) & (ts <= ts_close) & ((bits & HAS_FORM_T) != 0))

    with np.errstate(invalid="ignore", over="ignore"):
        open_ = np.where(updates_open_close, price, np.nan)
        high = np.where(updates_high_low, price, np.nan)
        prices = {"open": open_, "high": high, "low": high, "close": open_}
        volume = size * updates_volume.astype(np.float64)
        # Row mean of [open, high, low, close] as pandas computes it: missing
        # values are zero-filled and the four columns are summed left to right.
        valid_count = np.zeros(len(price))
        total = np.zeros(len(price))
        for col in TRADE_PRICE_COLUMNS:
            valid = ~np.isnan(prices[col])
            valid_count += valid
            total = total + np.where(valid, prices[col], 0.0)
        with np.errstate(divide="ignore"):
            mean_ohlc = total / valid_count
        mean_ohlc[valid_count == 0] = np.nan
        vol_nan = np.where(volume == 0, np.nan, volume)
        odd_lot = (volume < 100).astype(np.float64)
        sums = np.column_stack(
            [
                volume,  # volume and _vol_all
                price * volume,  # _dol_vol_all
                (~np.isnan(mean_ohlc)).astype(np.float64) * vol_nan,  # _vol
                mean_ohlc * vol_nan,  # _dol_vol
                odd_lot * vol_nan,  # _vol_ol
                (price * odd_lot) * vol_nan,  # _dol_vol_ol
            ]
        )
    n = (~np.isnan(volume)).astype(np.int64)

    tid = map_dictionary(table.column("ticker"), "ticker", ticker_ids.__getitem__, -1)[keep]
    valid_ticker = tid >= 0
    if not valid_ticker.all():
        tid, ts, seq, n, sums = (
            tid[valid_ticker],
            ts[valid_ticker],
            seq[valid_ticker],
            n[valid_ticker],
            sums[valid_ticker],
        )
        prices = {k: v[valid_ticker] for k, v in prices.items()}
    order = _sort_order(tid, ts, seq)
    if order is not None:
        tid, ts, n, sums = tid[order], ts[order], n[order], sums[order]
        prices = {k: v[order] for k, v in prices.items()}
    return PreparedTrades(tid=tid, ts=ts, prices=prices, n=n, sums=sums, market_hours=(ts_open, ts_close))


def _interval(ts: np.ndarray, interval_ns: int) -> np.ndarray:
    # Float64 division as in the legacy ``np.ceil(ts / interval_ns)``.
    x = ts / interval_ns
    np.ceil(x, out=x)
    out = x.astype(np.int64)
    out *= interval_ns
    return out


@dataclass
class Aggregates:
    tid: np.ndarray
    ts_interval: np.ndarray
    columns: dict[str, np.ndarray]


def aggregate_quotes(quotes: PreparedQuotes, interval_ns: int) -> Aggregates:
    """Last quote (per column, skipping missing values) of each ticker interval."""
    if len(quotes.ts) == 0:
        empty_float = np.zeros(0, dtype=np.float64)
        columns = {"sip_timestamp_quote": np.zeros(0, dtype=np.int64)}
        columns.update({col: empty_float for col in quotes.values})
        return Aggregates(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), columns)
    ts_interval = _interval(quotes.ts, interval_ns)
    starts = _run_starts(quotes.tid, ts_interval)
    last = np.append(starts[1:], len(ts_interval)) - 1
    columns = {"sip_timestamp_quote": quotes.ts[last]}
    for col, values in quotes.values.items():
        columns[col] = _group_last_valid(values, starts) if quotes.has_missing else values[last]
    return Aggregates(quotes.tid[starts], ts_interval[starts], columns)


def aggregate_trades(trades: PreparedTrades, interval_ns: int) -> Aggregates:
    """Per-interval OHLCV/VWAP aggregates of the eligible trades."""
    ts_interval = _interval(trades.ts, interval_ns)
    starts = _run_starts(trades.tid, ts_interval)
    if len(starts) == 0:
        # The legacy per-ticker loop concatenates an empty list.
        raise ValueError("No objects to concatenate")
    last = np.append(starts[1:], len(ts_interval)) - 1
    volume, dol_vol_all, vol, dol_vol, vol_ol, dol_vol_ol = _group_kahan_sum(trades.sums, starts).T
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap_all = dol_vol_all / volume
        vwap = dol_vol / vol
        vwap_ol = dol_vol_ol / vol_ol
    columns = {
        "open": _group_first_valid(trades.prices["open"], starts),
        "high": _group_extreme(trades.prices["high"], starts, maximum=True),
        "low": _group_extreme(trades.prices["low"], starts, maximum=False),
        "close": _group_last_valid(trades.prices["close"], starts),
        "volume": volume,
        "n": np.add.reduceat(trades.n, starts),
        "sip_timestamp_trade": trades.ts[last],
        "vwap_all": vwap_all,
        "vwap": vwap,
        "vwap_ol": vwap_ol,
    }
    return Aggregates(trades.tid[starts], ts_interval[starts], columns)


def _union_positions(q: Aggregates, t: Aggregates) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sorted union of (ticker, interval) keys and each side's output rows."""
    tid = np.concatenate([q.tid, t.tid])
    ts_interval = np.concatenate([q.ts_interval, t.ts_interval])
    order = np.lexsort((ts_interval, tid))
    tid, ts_interval = tid[order], ts_interval[order]
    new_key = np.empty(len(order), dtype=bool)
    if len(order):
        new_key[0] = True
        new_key[1:] = (tid[1:] != tid[:-1]) | (ts_interval[1:] != ts_interval[:-1])
    row = np.cumsum(new_key) - 1
    out_rows = np.empty(len(order), dtype=np.int64)
    out_rows[order] = row
    keep = np.flatnonzero(new_key)
    return tid[keep], ts_interval[keep], out_rows[: len(q.tid)], out_rows[len(q.tid) :]


def merge_snapshot(q: Aggregates, t: Aggregates, tickers: Sequence[str]) -> pa.Table:
    """Outer merge quotes and trades per interval, forward fill per ticker."""
    tid, ts_interval, q_rows, t_rows = _union_positions(q, t)
    n_rows = len(tid)
    if n_rows == 0:
        raise ValueError("No objects to concatenate")
    has_quote = np.zeros(n_rows, dtype=bool)
    has_quote[q_rows] = True
    has_trade = np.zeros(n_rows, dtype=bool)
    has_trade[t_rows] = True
    starts = _run_starts(tid)
    run_start = np.repeat(starts, np.diff(np.append(starts, n_rows)))

    def scatter(values, rows, fill):
        out = np.full(n_rows, fill, dtype=values.dtype)
        out[rows] = values
        return out

    # pandas upcasts int64 columns with missing rows to float64 in the merge,
    # which rounds every timestamp in the column (and makes ``n`` float).
    ts_quote = scatter(q.columns["sip_timestamp_quote"], q_rows, 0)
    if not has_quote.all():
        ts_quote = ts_quote.astype(np.float64).astype(np.int64)
    ts_trade = scatter(t.columns["sip_timestamp_trade"], t_rows, 0)
    if not has_trade.all():
        ts_trade = ts_trade.astype(np.float64).astype(np.int64)
    ts_quote, valid_quote = _forward_fill(ts_quote, has_quote, run_start, 0)
    ts_trade, valid_trade = _forward_fill(ts_trade, has_trade, run_start, 0)

    # Data start after the first valid quote and trade of each ticker.
    keep = np.flatnonzero(valid_quote & valid_trade)
    out = {
        "ticker": pa.array(tickers, pa.string()).take(pa.array(tid[keep]))
        if len(tickers)
        else pa.array([], pa.string()),
        "ts_interval": ts_interval[keep],
        "sip_timestamp_quote": ts_quote[keep],
        "sip_timestamp_trade": ts_trade[keep],
    }
    for col in QUOTE_VALUE_COLUMNS:
        values = scatter(q.columns[col], q_rows, np.nan)
        out[col] = _forward_fill(values, ~np.isnan(values), run_start, np.nan)[0][keep]
    for col in TRADE_PRICE_COLUMNS + VWAP_COLUMNS:
        values = scatter(t.columns[col], t_rows, np.nan)
        out[col] = _forward_fill(values, ~np.isnan(values), run_start, np.nan)[0][keep]
    out["volume"] = scatter(t.columns["volume"], t_rows, 0.0)[keep]
    n = scatter(t.columns["n"], t_rows, 0)[keep]
    out["n"] = n.astype(np.float64) if not has_trade.all() else n
    return build_table(out)


def iter_snapshots(
    quotes: pa.Table,
    trades: pa.Table,
    freqs_hz: Iterable[float],
    rules: TradeRules,
) -> Iterator[tuple[float, pa.Table]]:
    """Build the snapshot of one (date, partition) for every frequency.

    ``quotes``/``trades`` hold ``QUOTE_COLUMNS``/``TRADE_COLUMNS`` with any of
    the column types found in the archive.  Yields ``(freq_hz, table)`` with
    Arrow tables whose schema and pandas metadata match the legacy output; the
    inputs are prepared once and shared by all frequencies.
    """
    tickers = sorted(
        dictionary_values(quotes.column("ticker"), "ticker") | dictionary_values(trades.column("ticker"), "ticker")
    )
    ticker_ids = {ticker: i for i, ticker in enumerate(tickers)}
    prepared_quotes = prepare_quotes(quotes, ticker_ids)
    prepared_trades = prepare_trades(trades, ticker_ids, rules)
    del quotes, trades
    for freq in freqs_hz:
        interval_ns = interval_ns_for(freq)
        yield (
            freq,
            merge_snapshot(
                aggregate_quotes(prepared_quotes, interval_ns),
                aggregate_trades(prepared_trades, interval_ns),
                tickers,
            ),
        )


def create_snapshots(
    quotes: pa.Table, trades: pa.Table, freqs_hz: Iterable[float], rules: TradeRules
) -> dict[float, pa.Table]:
    return dict(iter_snapshots(quotes, trades, freqs_hz, rules))


def read_inputs(quotes_path: str, trades_path: str) -> tuple[pa.Table, pa.Table]:
    """Read the quote and trade columns of one (date, partition)."""
    quotes = read_partition(quotes_path, QUOTE_COLUMNS, dictionary_columns=["ticker"])
    trades = read_partition(trades_path, TRADE_COLUMNS, dictionary_columns=["ticker", "conditions"])
    return quotes, trades


def snapshot_partition(
    quotes_path: str, trades_path: str, freqs_hz: Iterable[float], rules: TradeRules
) -> dict[float, pa.Table]:
    """Read one (date, partition) and build its snapshots."""
    return create_snapshots(*read_inputs(quotes_path, trades_path), freqs_hz, rules)


# ---------------------------------------------------------------------------
# Output.
# ---------------------------------------------------------------------------

_METADATA_CACHE: dict[bool, dict[bytes, bytes]] = {}


def _pandas_metadata(n_is_float: bool) -> dict[bytes, bytes]:
    """Schema metadata that ``DataFrame.to_parquet`` attaches to the legacy files."""
    if n_is_float not in _METADATA_CACHE:
        sample = pd.DataFrame(
            {
                col: np.array(["x"], dtype=object)
                if col == "ticker"
                else np.zeros(
                    1, dtype=np.int64 if col in _INT_OUTPUT_COLUMNS or (col == "n" and not n_is_float) else np.float64
                )
                for col in OUTPUT_COLUMNS
            }
        )
        _METADATA_CACHE[n_is_float] = pa.Schema.from_pandas(sample, preserve_index=False).metadata
    return _METADATA_CACHE[n_is_float]


def build_table(columns: dict[str, object]) -> pa.Table:
    """Arrow table equivalent to ``pa.Table.from_pandas`` on the legacy frame."""
    n_is_float = columns["n"].dtype == np.float64
    if len(columns["ts_interval"]) == 0:
        # pandas infers a null-typed ticker column for an empty frame.
        frame = pd.DataFrame(
            {col: np.asarray(columns[col]) if col != "ticker" else np.array([], dtype=object) for col in OUTPUT_COLUMNS}
        )
        return pa.Table.from_pandas(frame, preserve_index=False)
    arrays = []
    fields = []
    for col in OUTPUT_COLUMNS:
        values = columns[col]
        if col == "ticker":
            arrays.append(values)
            fields.append(pa.field(col, pa.string()))
        elif values.dtype == np.int64:
            arrays.append(pa.array(values, pa.int64()))
            fields.append(pa.field(col, pa.int64()))
        else:
            # NaN is written as null, as pandas does.
            arrays.append(pa.array(values, pa.float64(), from_pandas=True))
            fields.append(pa.field(col, pa.float64()))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=_pandas_metadata(n_is_float)))


def write_table(table: pa.Table, fn_out: str) -> None:
    """Atomically write a snapshot with the legacy writer settings."""
    os.makedirs(os.path.dirname(fn_out), exist_ok=True)
    temp_out = f"{fn_out}.tmp-{os.getpid()}"
    try:
        # DataFrame.to_parquet(index=False) defaults: snappy, dictionary on.
        pq.write_table(table, temp_out, compression="snappy")
        os.replace(temp_out, fn_out)
    finally:
        if os.path.exists(temp_out):
            os.remove(temp_out)
