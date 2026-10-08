"""The snapshot engine must write byte-identical files to the original pandas
implementation (tests/reference/legacy_snapshot.py)."""

import filecmp

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import synthetic

from market_1t import snapshot
from market_1t.layout import freq_label

FREQS = [1.0, 0.125, 1000.0, 3.0]


@pytest.fixture
def check(tmp_path, legacy_snapshot, legacy_conditions, rules):
    """Write a partition, build it with both implementations and compare."""

    def run(quotes, trades, date="2021-01-27", partition=7, schema="modern", freqs=FREQS):
        quotes_path, trades_path = synthetic.write_partition(str(tmp_path), date, partition, quotes, trades, schema)
        try:
            new, new_error = snapshot.snapshot_partition(quotes_path, trades_path, freqs, rules), None
        except Exception as error:  # noqa: BLE001
            new, new_error = None, error
        for freq in freqs:
            try:
                reference, _ = legacy_snapshot.create_snapshot(
                    date, partition, freq_hz=freq, data_dir=str(tmp_path), conditions_fn=legacy_conditions
                )
            except Exception as error:  # noqa: BLE001
                assert new_error is not None, f"reference failed ({error!r}) but market_1t did not"
                assert type(new_error) is type(error), f"{new_error!r} vs {error!r}"
                continue
            assert new_error is None, f"market_1t failed ({new_error!r}) but the reference did not"
            ref_fn, new_fn = tmp_path / f"ref_{freq}.parquet", tmp_path / f"new_{freq}.parquet"
            reference.to_parquet(ref_fn, index=False)
            snapshot.write_table(new[freq], str(new_fn))
            if not filecmp.cmp(ref_fn, new_fn, shallow=False):
                ref_table, new_table = pq.ParquetFile(ref_fn).read(), pq.ParquetFile(new_fn).read()
                pytest.fail(
                    f"{freq}Hz output differs: "
                    f"schema equal={ref_table.schema.equals(new_table.schema, check_metadata=True)} "
                    f"rows {ref_table.num_rows} vs {new_table.num_rows}; content equal={ref_table.equals(new_table)}"
                )
        return new_error

    return run


# -- generated partitions ----------------------------------------------------

GENERATED = {
    "basic": ("2021-01-27", {}, "modern"),
    "summer_session": ("2020-07-17", {}, "modern"),
    "half_day": ("2019-11-29", {}, "modern"),
    "missing_values": ("2021-01-27", {"missing_values": True}, "modern"),
    "null_tickers": ("2021-01-27", {"null_ticker_rows": True}, "modern"),
    "unsorted_file_order": ("2021-01-27", {"unsorted": True}, "modern"),
    "first_eligible_trade_after_utc_midnight": ("2021-01-27", {"late_first_trade": True}, "modern"),
    "signed_zero_and_infinite_prices": ("2021-01-27", {"signed_zero": True}, "modern"),
    "string_typed_quotes": ("2016-07-18", {"missing_values": True}, "strings"),
    "integer_trade_sizes": ("2012-08-01", {}, "int_size"),
}


@pytest.mark.parametrize("name", list(GENERATED))
def test_generated(check, name):
    date, options, schema = GENERATED[name]
    seed = list(GENERATED).index(name) + 1
    check(*synthetic.make_partition(date, seed=seed, **options), date=date, schema=schema)


def test_randomised(check):
    rng = np.random.default_rng(2024)
    for seed in range(12):
        options = {
            name: bool(rng.random() < 0.3)
            for name in ("missing_values", "null_ticker_rows", "unsorted", "late_first_trade")
        }
        date = str(rng.choice(["2008-10-10", "2016-03-14", "2021-01-27", "2026-07-01"]))
        quotes, trades = synthetic.make_partition(date, seed=100 + seed, n_tickers=int(rng.integers(1, 8)), **options)
        schema = str(rng.choice(["modern", "strings", "int_size"]))
        check(quotes, trades, date=date, partition=int(rng.integers(0, 100)), schema=schema, freqs=[1.0, 0.25])


# -- hand-built corner cases -------------------------------------------------


def frames(quote_ts, trade_ts, conditions=None, sizes=None, tickers=("AAPL",)):
    quote_rows, trade_rows = [], []
    for ticker in tickers:
        for i, ts in enumerate(quote_ts):
            quote_rows.append(
                dict(
                    ticker=ticker,
                    ask_exchange=1,
                    ask_price=10.0 + i,
                    ask_size=3,
                    bid_exchange=2,
                    bid_price=9.0 + i,
                    bid_size=4,
                    conditions="1",
                    indicators=None,
                    sequence_number=i,
                    sip_timestamp=ts,
                )
            )
        for i, ts in enumerate(trade_ts):
            trade_rows.append(
                dict(
                    ticker=ticker,
                    conditions=(conditions or [None] * len(trade_ts))[i],
                    exchange=4,
                    price=9.5 + 0.1 * i,
                    sequence_number=1000 + i,
                    sip_timestamp=ts,
                    size=(sizes or [100.0] * len(trade_ts))[i],
                )
            )
    return pd.DataFrame(quote_rows), pd.DataFrame(trade_rows)


START = synthetic.session_bounds("2021-01-27")[0]


def test_every_interval_has_quotes_and_trades(check):
    # No missing side after the merge: timestamps are not rounded and n stays int64.
    ts = [START + k * synthetic.NS + 123 for k in range(20)]
    check(*frames(ts, ts), freqs=[1.0])


def test_trades_only_in_quote_intervals(check):
    quote_ts = [START + k * synthetic.NS + 777 for k in range(30)]
    check(*frames(quote_ts, quote_ts[::3]), freqs=[1.0, 0.125])


def test_form_t_during_and_outside_regular_hours(check):
    ts = [int(START + 3600 * synthetic.NS * h) for h in (1, 2, 5.5, 6, 11.5, 12.5, 15.9)]
    check(*frames(ts, ts, conditions=["12", "12,37", "12", "12,37", "12", None, "12"]), freqs=[1.0])


def test_session_boundaries_inclusive(check):
    ts_open, ts_close = snapshot.legacy_market_hours(START)
    ts = [ts_open - 1, ts_open, ts_open + 1, ts_close - 1, ts_close, ts_close + 1]
    check(*frames(ts, ts, conditions=["12"] * 6), freqs=[1.0, 1000.0])


def test_fractional_and_zero_sizes(check):
    ts = [START + 5 * synthetic.NS + k for k in range(8)]
    check(*frames(ts, ts, sizes=[0.14, 0.0, 1.0, 0.5, 99.0, 100.0, 0.074081, 250.0]), freqs=[1.0, 0.125])


def test_no_eligible_trades_fails_like_reference(check):
    ts = [START + k * synthetic.NS for k in range(5)]
    assert isinstance(check(*frames(ts, ts, conditions=["16"] * 5), freqs=[1.0]), IndexError)


def test_only_null_ticker_trades_fails_like_reference(check):
    ts = [START + k * synthetic.NS for k in range(5)]
    quotes, trades = frames(ts, ts)
    trades["ticker"] = None
    assert isinstance(check(quotes, trades, freqs=[1.0]), ValueError)


def test_no_ticker_with_both_quotes_and_trades(check):
    # Every row is dropped: the output is an empty file whose ticker column is
    # null-typed, as pandas infers it.
    ts = [START + k * synthetic.NS for k in range(10)]
    quotes, trades = frames(ts, ts, tickers=("AAPL", "MSFT"))
    check(quotes[quotes.ticker == "AAPL"], trades[trades.ticker == "MSFT"], freqs=[1.0])


def test_all_quote_tickers_null(check):
    ts = [START + k * synthetic.NS for k in range(10)]
    quotes, trades = frames(ts, ts)
    quotes["ticker"] = None
    check(quotes, trades, freqs=[1.0])


def test_quotes_without_trades_for_some_tickers(check):
    ts = [START + k * synthetic.NS for k in range(10)]
    quotes, trades = frames(ts, ts, tickers=("AAPL", "MSFT", "ZZZ"))
    check(quotes[quotes.ticker != "ZZZ"], trades[trades.ticker != "MSFT"], freqs=[1.0])


# -- helpers -------------------------------------------------------------------


def test_string_numbers_parse_like_pandas():
    floats = [
        "0",
        "75.82",
        "1e-05",
        " 12.5",
        "+3",
        "1_000.5",
        "nan",
        "inf",
        "-0.0",
        ".5",
        "5.",
        None,
        "123456789.123456789",
    ]
    got = snapshot.as_float64(pa.chunked_array([pa.array(floats, pa.large_string())]), "x")
    expected = pd.Series(floats, dtype=object).astype(float).to_numpy()
    assert np.array_equal(np.isnan(got), np.isnan(expected))
    assert np.array_equal(got.view(np.int64)[~np.isnan(got)], expected.view(np.int64)[~np.isnan(expected)])
    ints = ["1785740693712443934", " 12", "+7", "007", "1_000", "-5"]
    got = snapshot.as_int64(pa.chunked_array([pa.array(ints, pa.large_string())]), "x")
    assert np.array_equal(got, pd.Series(ints, dtype=object).astype(int).to_numpy())


def test_missing_integers_fail_like_pandas():
    with pytest.raises(ValueError):
        snapshot.as_int64(pa.chunked_array([pa.array([1, None], pa.int64())]), "sip_timestamp")


def test_group_reductions_match_pandas_bitwise():
    def same(a, b):
        na, nb = np.isnan(a), np.isnan(b)
        return np.array_equal(na, nb) and np.array_equal(a[~na].view(np.int64), b[~nb].view(np.int64))

    rng = np.random.default_rng(7)
    for _ in range(300):
        sizes = rng.integers(1, 8, size=rng.integers(1, 300))
        starts = np.r_[0, np.cumsum(sizes)[:-1]]
        n = int(sizes.sum())
        values = rng.choice([np.nan, -0.0, 0.0, np.inf, -np.inf, 1e16, 1.0, -1.0, 0.1, 3.3, 1e-300, 2.5e8], size=(n, 6))
        values = np.where(rng.random((n, 6)) < 0.5, rng.normal(0, 1e6, (n, 6)), values)
        codes = np.repeat(np.arange(len(starts)), sizes)
        with np.errstate(invalid="ignore"):
            assert same(
                pd.DataFrame(values).groupby(codes, sort=False).sum().to_numpy(),
                snapshot._group_kahan_sum(values, starts),
            )
        x = values[:, 0]
        grouped = pd.Series(x).groupby(codes, sort=False)
        assert same(grouped.max().to_numpy(), snapshot._group_extreme(x, starts, True))
        assert same(grouped.min().to_numpy(), snapshot._group_extreme(x, starts, False))
        assert same(grouped.first().to_numpy(), snapshot._group_first_valid(x, starts))
        assert same(grouped.last().to_numpy(), snapshot._group_last_valid(x, starts))


def test_sortedness_check_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n = int(rng.integers(0, 40))
        tid = np.sort(rng.integers(0, 4, n)) if rng.random() < 0.7 else rng.integers(0, 4, n)
        ts, seq = rng.integers(0, 5, n), rng.integers(0, 5, n)
        if rng.random() < 0.5:
            order = np.lexsort((seq, ts, tid))
            tid, ts, seq = tid[order], ts[order], seq[order]
        keys = list(zip(tid.tolist(), ts.tolist(), seq.tolist(), strict=True))
        expected = all(keys[i] <= keys[i + 1] for i in range(len(keys) - 1))
        for block in (1, 3, 1 << 20):
            assert snapshot._is_lexsorted(tid, ts, seq, block) == expected


def test_labels_and_intervals():
    assert [freq_label(f) for f in (1, 1.0, 0.5, 0.25, 0.125, 5, 10.0, 1000)] == [
        "1Hz",
        "1Hz",
        "0.5Hz",
        "0.25Hz",
        "0.125Hz",
        "5Hz",
        "10Hz",
        "1000Hz",
    ]
    for freq in (1, 0.5, 0.125, 3, 7, 1000):
        assert snapshot.interval_ns_for(freq) == int(1e9 / freq)
