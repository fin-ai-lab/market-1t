"""Synthetic Polygon quotes/trades partitions that exercise the edge cases of
the snapshot logic, written in the archive's Parquet layout."""

import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

NS = 1_000_000_000

# Condition fields seen in the flat files plus deliberately awkward ones:
# None (regular sale), Form T (12), Stock Option (35), Trade Thru Exempt (41),
# codes outside every rule set (e.g. 7 affects volume only, 16 nothing).
CONDITIONS = [None, None, None, "37", "12,37", "12", "14,41", "35", "2,12", "7", "16", "4,12,37", "10", "53", "1", "5"]

TICKER_POOL = ["A", "AA", "AAPL", "AB.C", "BRK.A", "BRK.B", "MSFT", "NA", "SPY", "Z", "ZVZZT", "ZZZ"]


def session_bounds(date: str):
    """UTC nanoseconds of 04:00 and 20:00 US/Eastern on ``date``."""
    local = pd.Timestamp(date).tz_localize("US/Eastern")
    return int((local + pd.Timedelta(hours=4)).value), int((local + pd.Timedelta(hours=20)).value)


def _near_boundaries(rng, start, end, n):
    """Timestamps within a few hundred ns of whole seconds (float rounding edge)."""
    base = rng.integers(start // NS + 1, end // NS, n) * NS
    return base + rng.choice([-257, -256, -255, -129, -128, -127, -1, 0, 1, 127, 128, 129, 255, 256, 257], n)


def make_partition(
    date: str,
    seed: int,
    n_tickers: int = 6,
    quotes_per_ticker: int = 400,
    trades_per_ticker: int = 250,
    null_ticker_rows: bool = False,
    missing_values: bool = False,
    unsorted: bool = False,
    late_first_trade: bool = False,
    signed_zero: bool = False,
):
    """Return (quotes, trades) DataFrames in flat-file order."""
    rng = np.random.default_rng(seed)
    start, end = session_bounds(date)
    tickers = sorted(rng.choice(TICKER_POOL, size=min(n_tickers, len(TICKER_POOL)), replace=False).tolist())
    quotes: list[pd.DataFrame] = []
    trades: list[pd.DataFrame] = []
    seq = 0
    for k, ticker in enumerate(tickers):
        base = float(rng.choice([0.5, 3.21, 17.0, 101.3, 4250.0]))
        # Quotes: bursts inside a few seconds plus spread-out updates.
        nq = int(rng.integers(quotes_per_ticker // 2, quotes_per_ticker + 1))
        ts = np.concatenate(
            [
                rng.integers(start, end, nq // 2),
                _near_boundaries(rng, start, end, nq // 4),
                rng.integers(start + 5 * 3600 * NS, start + 5 * 3600 * NS + 20 * NS, nq - nq // 2 - nq // 4),
            ]
        )
        ts.sort()
        q = pd.DataFrame(
            {
                "ticker": ticker,
                "ask_exchange": rng.integers(0, 20, nq),
                "ask_price": np.round(base * (1 + rng.normal(0, 0.01, nq)), 4),
                "ask_size": rng.integers(0, 60, nq),
                "bid_exchange": rng.integers(0, 20, nq),
                "bid_price": np.round(base * (1 - np.abs(rng.normal(0, 0.01, nq))), 4),
                "bid_size": rng.integers(0, 60, nq),
                "conditions": "1,81",
                "indicators": None,
                "sequence_number": np.arange(seq, seq + nq),
                "sip_timestamp": ts,
            }
        )
        seq += nq
        # Duplicate timestamps within a ticker.
        dup = rng.choice(nq - 1, size=max(1, nq // 20), replace=False)
        q.loc[dup + 1, "sip_timestamp"] = q.loc[dup, "sip_timestamp"].to_numpy()
        if missing_values:
            for col in ["ask_price", "bid_size", "bid_price"]:
                q[col] = q[col].astype("float64")
                q.loc[rng.random(nq) < 0.05, col] = np.nan
        quotes.append(q)

        nt = int(rng.integers(trades_per_ticker // 2, trades_per_ticker + 1))
        tts = np.concatenate(
            [
                rng.integers(start, end, nt // 2),
                _near_boundaries(rng, start, end, nt // 4),
                # A busy stretch (many trades per interval, Kahan sums matter).
                rng.integers(start + 6 * 3600 * NS, start + 6 * 3600 * NS + 3 * NS, nt - nt // 2 - nt // 4),
            ]
        )
        tts.sort()
        size = rng.choice([1.0, 100.0, 37.0, 0.14, 0.074081, 250.0, 5000.0, 0.0, 99.0, 1.5], nt)
        price = np.round(base * (1 + rng.normal(0, 0.01, nt)), 6)
        t = pd.DataFrame(
            {
                "ticker": ticker,
                "conditions": rng.choice(np.array(CONDITIONS, dtype=object), nt),
                "exchange": rng.integers(1, 20, nt),
                "price": price,
                "sequence_number": np.arange(seq, seq + nt),
                "sip_timestamp": tts,
                "size": size,
            }
        )
        seq += nt
        # Equal timestamps with sequence numbers out of file order.
        dup = rng.choice(nt - 2, size=max(1, nt // 15), replace=False)
        t.loc[dup + 1, "sip_timestamp"] = t.loc[dup, "sip_timestamp"].to_numpy()
        swap = t.loc[dup, "sequence_number"].to_numpy().copy()
        t.loc[dup, "sequence_number"] = t.loc[dup + 1, "sequence_number"].to_numpy()
        t.loc[dup + 1, "sequence_number"] = swap
        if missing_values:
            t.loc[rng.random(nt) < 0.03, "price"] = np.nan
            t.loc[rng.random(nt) < 0.03, "size"] = np.nan
        if signed_zero and k == 0:
            # Price-eligible trades (regular sale, round lot) in one interval.
            rows = t.index[:6]
            t.loc[rows, "conditions"] = None
            t.loc[rows, "size"] = 100.0
            t.loc[rows, "sip_timestamp"] = t.loc[rows[0], "sip_timestamp"]
            t.loc[rows, "price"] = [0.0, -0.0, 0.0, np.inf, 1.0, -np.inf]
        if late_first_trade and k == 0:
            # The first eligible trade in file order falls after 00:00 UTC of
            # the next day (possible after 19:00 EST), which moves the session
            # used for the Form T correction.
            t["sip_timestamp"] = end - int(0.5 * 3600 * NS) + np.sort(rng.integers(0, 1800 * NS, nt))
            t["conditions"] = rng.choice(np.array([None, "12", "12,37"], dtype=object), nt)
        trades.append(t)

    quotes_df = pd.concat(quotes, ignore_index=True)
    trades_df = pd.concat(trades, ignore_index=True)
    if null_ticker_rows:
        for df in (quotes_df, trades_df):
            idx = rng.choice(len(df), size=max(1, len(df) // 50), replace=False)
            df.loc[idx, "ticker"] = None
    if unsorted:
        quotes_df = quotes_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        trades_df = trades_df.sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    return quotes_df, trades_df


QUOTE_TYPES = {
    "ticker": pa.large_string(),
    "ask_exchange": pa.int64(),
    "ask_price": pa.float64(),
    "ask_size": pa.int64(),
    "bid_exchange": pa.int64(),
    "bid_price": pa.float64(),
    "bid_size": pa.int64(),
    "conditions": pa.large_string(),
    "indicators": pa.large_string(),
    "sequence_number": pa.int64(),
    "sip_timestamp": pa.int64(),
}
TRADE_TYPES = {
    "ticker": pa.large_string(),
    "conditions": pa.large_string(),
    "exchange": pa.int64(),
    "price": pa.float64(),
    "sequence_number": pa.int64(),
    "sip_timestamp": pa.int64(),
    "size": pa.float64(),
}


NUMERIC_QUOTE_COLUMNS = (
    "ask_exchange",
    "ask_price",
    "ask_size",
    "bid_exchange",
    "bid_price",
    "bid_size",
    "sequence_number",
    "sip_timestamp",
)


def _as_text(value) -> str | None:
    """Number as the archive's string-typed days store it ("0", "75.82", ...)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


def _table(
    df: pd.DataFrame, types: dict[str, pa.DataType], overrides: dict[str, pa.DataType] | None = None
) -> pa.Table:
    types = dict(types, **(overrides or {}))
    arrays = []
    for name, typ in types.items():
        values = df[name]
        if pa.types.is_large_string(typ) and name in NUMERIC_QUOTE_COLUMNS:
            arrays.append(pa.array([_as_text(v) for v in values.tolist()], typ))
        elif pa.types.is_integer(typ) and values.dtype.kind == "f":
            arrays.append(pa.array(values.to_numpy(), pa.float64(), from_pandas=True).cast(typ))
        elif values.dtype == object:
            arrays.append(pa.array(values.tolist(), typ))
        else:
            arrays.append(pa.array(values.to_numpy(), typ))
    return pa.Table.from_arrays(arrays, names=list(types))


def write_partition(
    root: str, date: str, partition: int, quotes: pd.DataFrame, trades: pd.DataFrame, schema: str = "modern"
):
    """Write the partition like the archive; ``schema`` picks an archived variant."""
    q_over, t_over = None, None
    if schema == "strings":
        q_over = {c: pa.large_string() for c in QUOTE_TYPES if c not in ("ticker", "conditions", "indicators")}
    elif schema == "int_size":
        t_over = {"size": pa.int64()}
        trades = trades.assign(size=np.floor(trades["size"]))
    yr, mth, _ = date.split("-")
    paths = {}
    for feed, df, types, over in (
        ("quotes_v1", quotes, QUOTE_TYPES, q_over),
        ("trades_v1", trades, TRADE_TYPES, t_over),
    ):
        directory = os.path.join(root, "us_stocks_sip", feed, yr, mth, f"{date}.parquet", f"partition={partition}")
        os.makedirs(directory, exist_ok=True)
        table = _table(df, types, over)
        table = table.append_column("partition", pa.array(np.full(table.num_rows, partition, dtype=np.uint64)))
        pq.write_table(table, os.path.join(directory, "00000000.parquet"), row_group_size=max(1, table.num_rows // 3))
        paths[feed] = directory
    return paths["quotes_v1"], paths["trades_v1"]
