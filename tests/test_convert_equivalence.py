"""The streaming converter must reproduce the Polars-based conversion of the
original code (tests/reference/legacy_conversion.py)."""

import gzip
import os
import zlib

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
from flatfiles import TRADES_HEADER, quotes_csv, read_partitions, trades_csv, write_gzip

from market_1t import convert

N_PARTITIONS = 8
# Arrow's default null tokens, which the original streaming path applied to strings.
ARROW_NULL_TOKENS = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
]


@pytest.fixture
def converters(tmp_path, legacy_conversion):
    def legacy(fn_in, feed, streaming):
        fn_out = tmp_path / f"legacy_{'stream' if streaming else 'eager'}_{os.path.basename(fn_in)}.parquet"
        result = legacy_conversion.process_file(
            {
                "feed": feed,
                "fn_in": str(fn_in),
                "fn_out": str(fn_out),
                "size_in": os.path.getsize(fn_in),
                "n_partitions": N_PARTITIONS,
                "streaming_threshold_bytes": 0 if streaming else 10**15,
                "streaming_block_bytes": 1 << 16,
            }
        )
        assert "error" not in result, result
        return read_partitions(fn_out)

    def new(fn_in, feed, **options):
        fn_out = tmp_path / f"new_{os.path.basename(fn_in)}.parquet"
        convert.convert_to(str(fn_in), str(fn_out), feed, N_PARTITIONS, **options)
        return read_partitions(fn_out)

    return legacy, new


@pytest.fixture
def matches_reference(tmp_path, converters):
    legacy, new = converters

    def run(text, feed, members=1, padding=b""):
        fn = tmp_path / f"{feed}_{len(os.listdir(tmp_path))}.csv.gz"
        write_gzip(str(fn), text, members, padding)
        ours = new(fn, feed, chunk_bytes=1 << 15, buffer_bytes=1 << 20, row_group_rows=2000)
        eager = legacy(fn, feed, streaming=False)
        assert sorted(ours) == sorted(eager)
        for name in ours:
            assert ours[name].schema.names == eager[name].schema.names
            assert [f.type for f in ours[name].schema] == [f.type for f in eager[name].schema]
            assert ours[name].equals(eager[name]), f"{feed} {name} differs from the Polars conversion"
        return fn, ours

    return run


def test_trades(matches_reference):
    matches_reference(trades_csv(np.random.default_rng(1), 20_000), "trades_v1")


def test_quotes(matches_reference):
    matches_reference(quotes_csv(np.random.default_rng(2), 20_000), "quotes_v1")


def test_unsorted_input(matches_reference):
    matches_reference(trades_csv(np.random.default_rng(3), 20_000, sort=False), "trades_v1")


def test_multi_member_gzip(matches_reference):
    matches_reference(trades_csv(np.random.default_rng(4), 20_000), "trades_v1", members=3)


def test_nul_padding_after_last_member(tmp_path, matches_reference, converters):
    # Tolerated like the gzip module (the Polars reader cannot read such files,
    # so compare with the unpadded file instead).
    _, new = converters
    text = trades_csv(np.random.default_rng(4), 5_000)
    _, reference = matches_reference(text, "trades_v1", members=2)
    fn = tmp_path / "padded.csv.gz"
    write_gzip(str(fn), text, members=2, trailing_padding=b"\0" * 64)
    padded = new(fn, "trades_v1", chunk_bytes=1 << 15)
    assert sorted(padded) == sorted(reference)
    assert all(padded[name].equals(reference[name]) for name in padded)


def test_no_trailing_newline(matches_reference):
    matches_reference(trades_csv(np.random.default_rng(5), 5_000).rstrip("\n"), "trades_v1")


def test_zlib_fallback(monkeypatch, matches_reference):
    monkeypatch.setattr(convert, "_inflate_lib", zlib)
    matches_reference(trades_csv(np.random.default_rng(6), 5_000), "trades_v1")


def test_streaming_reference_differs_only_by_null_tokens(matches_reference, converters):
    legacy, _ = converters
    fn, ours = matches_reference(trades_csv(np.random.default_rng(7), 20_000), "trades_v1")
    streaming = legacy(fn, "trades_v1", streaming=True)
    token_rows = pl.concat([pl.from_arrow(t) for t in ours.values()]).filter(pl.col("ticker").is_in(ARROW_NULL_TOKENS))
    assert token_rows.height > 0
    for name in ours:
        mine = pl.from_arrow(ours[name]).filter(~pl.col("ticker").is_in(ARROW_NULL_TOKENS))
        mine = mine.with_columns(
            pl.when(pl.col("conditions").is_in(ARROW_NULL_TOKENS))
            .then(None)
            .otherwise(pl.col("conditions"))
            .alias("conditions")
        )
        theirs = pl.from_arrow(streaming[name]).filter(pl.col("ticker").is_not_null())
        assert mine.equals(theirs), name


def test_blank_lines_are_skipped_like_the_streaming_reference(tmp_path, converters):
    # Blank lines carry no data.  The original streaming path skipped them; its
    # Polars path turned each into an all-null row.
    legacy, new = converters
    text = trades_csv(np.random.default_rng(9), 5_000).replace(',"",', ",,")
    for token in ("NA", "NULL", "N/A", "nan"):
        text = text.replace(f"\n{token},", "\nAAPL,")
    lines = text.split("\n")
    fn = tmp_path / "blank.csv.gz"
    write_gzip(str(fn), "\n".join(lines[:1000] + [""] + lines[1000:]) + "\n\n")
    ours = new(fn, "trades_v1", chunk_bytes=1 << 15)
    streaming = legacy(fn, "trades_v1", streaming=True)
    assert sorted(ours) == sorted(streaming)
    assert all(ours[name].equals(streaming[name]) for name in ours)


def test_partition_hash_matches_reference():
    convert.check_partition_hash()


def test_partition_hash_guard_rejects_a_different_hash(monkeypatch):
    # What a Polars release with another string hash (1.26+) looks like.
    monkeypatch.setattr(convert, "ticker_partitions", lambda tickers, n: np.arange(len(tickers)))
    with pytest.raises(RuntimeError, match="uv.lock"):
        convert.check_partition_hash()


# -- decompression and chunking ------------------------------------------------


@pytest.mark.parametrize("inflate", ["isal", "zlib"])
def test_gzip_chunking(tmp_path, monkeypatch, inflate):
    if inflate == "zlib":
        monkeypatch.setattr(convert, "_inflate_lib", zlib)
    rng = np.random.default_rng(0)
    lines = [f"T{rng.integers(0, 9999):04d},{i},{rng.uniform():.6f},1,0,0" for i in range(50_000)]
    base = ("h1,h2,h3,h4,h5,h6\n" + "\n".join(lines)).encode()
    fn = str(tmp_path / "chunks.gz")
    for trailing_newline in (True, False):
        data = base + (b"\n" if trailing_newline else b"")
        for members in (1, 3):
            step = len(data) // members + 1
            with open(fn, "wb") as out:
                for k in range(members):
                    out.write(gzip.compress(data[k * step : (k + 1) * step]))
                out.write(b"\0" * 16)
            for read_bytes, max_output, chunk_bytes in (
                (1 << 10, 1 << 12, 1 << 10),
                (1 << 19, 1 << 22, 1 << 16),
                (1 << 19, 1 << 22, 1 << 30),
            ):
                ring = convert.BufferRing(3, chunk_bytes + (16 << 20))
                chunks = []
                for buffer, length in convert.iter_line_chunks(
                    convert.iter_gzip_blocks(fn, read_bytes, max_output), chunk_bytes, ring
                ):
                    chunks.append(buffer[:length].tobytes())
                    ring.release(buffer)
                assert b"".join(chunks) == data
                assert all(chunk.endswith(b"\n") for chunk in chunks[:-1])
                assert chunks[-1].endswith(b"\n") == trailing_newline


# -- failures ------------------------------------------------------------------

GOOD = trades_csv(np.random.default_rng(8), 5_000)


@pytest.mark.parametrize(
    "blob, error",
    [
        (gzip.compress(GOOD.encode())[:-500], EOFError),
        (gzip.compress((GOOD + "X,1,2\n").encode()), pa.ArrowInvalid),
        (gzip.compress(GOOD.replace(",size,", ",sz,", 1).encode()), ValueError),
        (gzip.compress((GOOD + "AAPL,,0,1,1,1,abc,1,2,1.0,1,0,0\n").encode()), pa.ArrowInvalid),
        (gzip.compress((TRADES_HEADER + "\nAAPL,,0,1,1,1,1.0,1,2,1.0,1,0,0\n").encode()), RuntimeError),
    ],
    ids=["truncated_gzip", "malformed_row", "missing_column", "invalid_number", "missing_partitions"],
)
def test_bad_input_fails_cleanly(tmp_path, blob, error):
    fn = tmp_path / "in.csv.gz"
    fn.write_bytes(blob)
    fn_out = tmp_path / "out.parquet"
    with pytest.raises(error):
        convert.convert_to(str(fn), str(fn_out), "trades_v1", N_PARTITIONS, chunk_bytes=1 << 14)
    assert list(tmp_path.glob("out.parquet*")) == [], "partial output left behind"
