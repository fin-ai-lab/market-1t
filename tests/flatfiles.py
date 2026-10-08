"""Synthetic Massive flat files (gzip CSV) for the conversion and CLI tests."""

import gzip
import os

import numpy as np
import pyarrow.parquet as pq

TRADES_HEADER = (
    "ticker,conditions,correction,exchange,id,participant_timestamp,price,sequence_number,sip_timestamp,size,tape,"
    "trf_id,trf_timestamp"
)
QUOTES_HEADER = (
    "ticker,ask_exchange,ask_price,ask_size,bid_exchange,bid_price,bid_size,conditions,indicators,"
    "participant_timestamp,sequence_number,sip_timestamp,tape,trf_timestamp"
)
# Includes real tickers that look like null tokens (NA, NULL, N/A, nan).
TICKERS = ["A", "AAPL", "BRK.A", "NA", "NULL", "N/A", "nan", "SPY", "ZZZ"] + [f"T{i:03d}" for i in range(150)]
# Condition fields, including a quoted empty string (never seen in real files,
# where the original snapshot code cannot parse it).
CONDITIONS = ['"12,37"', "", '""', "37", '"14,41"', "1", '"4,12,37"']
REAL_CONDITIONS = [c for c in CONDITIONS if c != '""']


def trades_csv(
    rng, n, sort=True, start=1_700_000_000_000_000_000, span=86_400_000_000_000, tickers=TICKERS, conditions=CONDITIONS
):
    tickers = rng.choice(tickers, n)
    ts = rng.integers(start, start + span, n)
    if sort:
        order = np.lexsort((ts, tickers))
        tickers, ts = tickers[order], ts[order]
    sizes = ["1.000000", "100.000000", "0.140000", "37.000000", "0.074081"]
    rows = [TRADES_HEADER]
    for i in range(n):
        rows.append(
            f"{tickers[i]},{rng.choice(conditions)},0,{rng.integers(1, 20)},{i},{ts[i] - 100},"
            f"{rng.uniform(1, 500):.6f},{i},{ts[i]},{rng.choice(sizes)},1,0,0"
        )
    return "\n".join(rows) + "\n"


def quotes_csv(rng, n, start=1_700_000_000_000_000_000, span=86_400_000_000_000, tickers=TICKERS):
    tickers = rng.choice(tickers, n)
    ts = rng.integers(start, start + span, n)
    order = np.lexsort((ts, tickers))
    tickers, ts = tickers[order], ts[order]
    indicators = ["", "604", '"1,2"']
    rows = [QUOTES_HEADER]
    for i in range(n):
        indicator = rng.choice(indicators)
        rows.append(
            f"{tickers[i]},{rng.integers(0, 20)},{rng.uniform(1, 500):.4f},{rng.integers(0, 50)},{rng.integers(0, 20)},"
            f'{rng.uniform(1, 500):.4f},{rng.integers(0, 50)},"1,81",{indicator},{ts[i] - 5},{i},{ts[i]},1,0'
        )
    return "\n".join(rows) + "\n"


def write_gzip(path, text, members=1, trailing_padding=b""):
    """Write ``text`` as a (possibly multi-member) gzip file."""
    data = text.encode()
    step = len(data) // members + 1
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as out:
        for k in range(members):
            out.write(gzip.compress(data[k * step : (k + 1) * step]))
        out.write(trailing_padding)


def read_partitions(root):
    """``{"partition=<p>": table}`` of a converted day (one file per partition)."""
    tables = {}
    for name in sorted(os.listdir(root)):
        files = sorted(os.listdir(os.path.join(root, name)))
        assert len(files) == 1, (root, name, files)
        tables[name] = pq.ParquetFile(os.path.join(root, name, files[0])).read()
    return tables
