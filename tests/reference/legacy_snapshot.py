import argparse
import glob
import json
import logging
import os
import pickle
import random
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
from pytz import timezone
from tqdm.auto import tqdm

EST = timezone("US/Eastern")

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Sync local directory with Polygon.io flat file repository")
    parser.add_argument("--data_dir", type=str, default="./data/parquet/")
    parser.add_argument("--output_dir", type=str, default="./data/snapshots/")
    parser.add_argument("--date_start", type=str, default="2008-01-01")  # Look into when Reg NMS became effective
    parser.add_argument("--date_end", type=str, default="2024-12-31")
    parser.add_argument("--freq_hz", type=float, default=1)
    parser.add_argument("--n_partitions", type=int, default=100)
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Hard cap on concurrent partition workers.",
    )
    parser.add_argument(
        "--arrow_cpu_threads",
        type=int,
        default=2,
        help="PyArrow CPU threads available to each snapshot process.",
    )
    parser.add_argument(
        "--arrow_io_threads",
        type=int,
        default=2,
        help="PyArrow I/O threads available to each snapshot process.",
    )

    return parser.parse_args()


def _clean_trade_conditions(x):
    if x is None:
        return set([1])
    else:
        try:
            return set(map(int, x.split(",")))
        except TypeError:
            return set([int(x)])


def _filter_trades(trades, rules, cols_keep=[]):
    # Get rules for trade eligibility
    updates = rules["trade"]["consolidated"].copy()

    # Modify the rules to match the intraday aggregation.
    # Form T (12) needs to be added for after-hours, Trade
    # Thru Exempt also needs to be added
    for k in ["updates_high_low", "updates_open_close"]:
        updates[k].update([12, 41])
    updates["updates_volume"].update([41])

    # Stock Option trades never affect price
    bad_conditions_price = set([35])

    # Filter the trades
    _cols_keep = ["ticker", "sip_timestamp", "sequence_number", "price", "size", "conditions"]
    _cols_keep = _cols_keep + [c for c in cols_keep if c not in _cols_keep]

    trades["conditions"] = trades.conditions.apply(_clean_trade_conditions)

    sel_has_effect = trades.conditions.isnull()
    for k in updates:
        if k == "updates_volume":
            trades[k] = 1 * (trades.conditions.apply(lambda x: x.issubset(updates[k])))
        else:
            trades[k] = 1 * (
                (trades.conditions.apply(lambda x: x.issubset(updates[k])))
                & (trades["size"] >= 1)
                & (trades.conditions.apply(lambda x: x.isdisjoint(bad_conditions_price)))
            )
            trades[k] = trades[k].replace(0, np.nan)
        sel_has_effect = sel_has_effect | (trades[k] == 1)
        _cols_keep.append(k)

    return trades.loc[sel_has_effect].drop(columns=[c for c in trades.columns if c not in _cols_keep])


def compute_aggregates(trades, conditions, col_aggregate="ts_interval", keep_conditions=False, compute_ohlc=True):
    """
    Filters trades based on Trade Conditions and determines each trade's effect on price and volume aggregations. Then
    aggregates trades by performing a groupby on `col_aggregate`.
    """
    # Compute trade filters based on conditions
    trades = _filter_trades(trades, conditions, [col_aggregate])

    # Correction for Form T trades, these should only count outside of regular market hours
    date_str = pd.to_datetime(trades["sip_timestamp"].dropna().iloc[0]).strftime("%Y-%m-%d")
    ts_open = int(
        (pd.to_datetime(date_str).tz_localize(EST) + pd.Timedelta(9.5, "h")).tz_convert("UTC").timestamp() * 1e9
    )
    ts_close = ts_open + int((16 - 9.5) * 60 * 60 * 1e9)
    _reg_hours = (trades["sip_timestamp"] >= ts_open) & (trades["sip_timestamp"] <= ts_close)
    _form_t = trades.conditions.apply(lambda x: 12 in x)
    trades["reg_hours_form_t"] = 1 * (_reg_hours & _form_t)
    trades["updates_volume"] = (trades["updates_volume"] - trades["reg_hours_form_t"]).clip(0, 1)

    # Build OHLCV columns based on trade eligibility
    trades["open"] = trades["price"] * trades["updates_open_close"]
    trades["high"] = trades["price"] * trades["updates_high_low"]
    trades["low"] = trades["price"] * trades["updates_high_low"]
    trades["close"] = trades["price"] * trades["updates_open_close"]
    # Preserve fractional-share volume. Current SIP flat files can report
    # sub-share sizes, and an integer cast would silently turn them into zero.
    trades["volume"] = trades["size"] * trades["updates_volume"]

    # NB - VWAP - The polygon.io VWAP intraday aggregates appear to be based on ALL trades
    # whether a trade's conditions say it affects prices or not. This seems wrong...
    # Compute VWAP based on all trades regardless of whether they should affect price
    trades["_vol_all"] = trades["volume"]
    trades["_dol_vol_all"] = trades["price"] * trades["volume"]

    # Only consider trades that affect BOTH price and volume when calculating VWAP
    # Note that OHLC will be NaN if not affecting price, the mean will then be the
    # trade price or NaN depending on whether it affects volume. Volume will be
    # zero when a trade does not affect the volume for the aggregate.
    trades["_vol"] = (1 * trades[["open", "high", "low", "close"]].mean(axis=1).notnull()) * trades["volume"].replace(
        0, np.nan
    )
    trades["_dol_vol"] = trades[["open", "high", "low", "close"]].mean(axis=1) * trades["volume"].replace(0, np.nan)

    # Compute a VWAP based only on odd-lot trades
    trades["_vol_ol"] = (1 * (trades["volume"] < 100)) * trades["volume"].replace(0, np.nan)
    trades["_dol_vol_ol"] = trades["price"] * (1 * (trades["volume"] < 100)) * trades["volume"].replace(0, np.nan)

    trades["n"] = 1 * trades["volume"].notnull()

    _agg = dict()
    if compute_ohlc:
        _agg.update(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }
        )

    _agg.update(
        {
            "volume": "sum",
            "n": "sum",
            "_vol_all": "sum",
            "_dol_vol_all": "sum",
            "_vol": "sum",
            "_dol_vol": "sum",
            "_vol_ol": "sum",
            "_dol_vol_ol": "sum",
            "sip_timestamp": "last",
        }
    )
    if keep_conditions:
        _agg["conditions"] = "sum"

    aggregates = (
        trades.sort_values(["ticker", "sip_timestamp", "sequence_number"]).groupby(["ticker", col_aggregate]).agg(_agg)
    )
    aggregates.columns = _agg.keys()
    aggregates["vwap_all"] = aggregates["_dol_vol_all"] / aggregates["_vol_all"]
    aggregates["vwap"] = aggregates["_dol_vol"] / aggregates["_vol"]
    aggregates["vwap_ol"] = aggregates["_dol_vol_ol"] / aggregates["_vol_ol"]
    aggregates = aggregates.drop(
        columns=["_vol_all", "_dol_vol_all", "_vol", "_dol_vol", "_vol_ol", "_dol_vol_ol"]
    ).reset_index()
    if keep_conditions:
        aggregates["conditions"] = aggregates.conditions.apply(lambda x: set(x) if isinstance(x, list) else set([x]))

    _aggregates = []
    for _, df in aggregates.groupby("ticker"):
        df = df.sort_values([col_aggregate])
        for col in ["open", "high", "low", "close", "vwap_all", "vwap", "vwap_ol"]:
            df[col] = df[col].ffill()
        _aggregates.append(df)
    aggregates = pd.concat(_aggregates)

    return aggregates, (ts_open, ts_close)


def create_snapshot(
    date,
    partition,
    col_ticker="ticker",
    col_timestamp="sip_timestamp",
    col_seq_number="sequence_number",
    col_conditions="conditions",
    col_interval="ts_interval",
    cols_quotes=["ask_price", "ask_size", "bid_price", "bid_size"],
    cols_trades=["price", "size"],
    freq_hz=1000,
    data_dir="./data/parquet",
    conditions_fn="./data/conditions_20240914.pkl",
):
    yr, mth, _ = date.split("-")
    interval_ns = int(1e9 / freq_hz)

    # LOAD AND PROCESS QUOTES
    fn_quotes = os.path.join(
        data_dir, "us_stocks_sip", "quotes_v1", str(yr), str(mth), f"{date}.parquet", f"partition={partition}"
    )
    _cols_quotes = [col_ticker, col_timestamp, col_seq_number] + cols_quotes
    quotes = pd.read_parquet(fn_quotes, columns=_cols_quotes)
    quotes[col_timestamp] = quotes[col_timestamp].astype(int)
    quotes[col_seq_number] = quotes[col_seq_number].astype(int)
    for col in cols_quotes:
        quotes[col] = quotes[col].astype(float)

    # NBBO sizes in Polygon are in round lots, convert to actual shares
    quotes["bid_size"] *= 100
    quotes["ask_size"] *= 100
    quotes[col_interval] = np.ceil(quotes[col_timestamp] / interval_ns).astype(int) * interval_ns
    quotes = (
        quotes.sort_values([col_ticker, col_timestamp, col_seq_number])
        .groupby([col_ticker, col_interval])
        .last()
        .drop(columns=[col_seq_number])
        .reset_index()
    )

    # LOAD AND PROCESS TRADES
    fn_trades = os.path.join(
        data_dir, "us_stocks_sip", "trades_v1", str(yr), str(mth), f"{date}.parquet", f"partition={partition}"
    )
    _cols_trades = [col_ticker, col_timestamp, col_seq_number, col_conditions] + cols_trades
    trades = pd.read_parquet(fn_trades, columns=_cols_trades)
    trades[col_timestamp] = trades[col_timestamp].astype(int)
    trades[col_seq_number] = trades[col_seq_number].astype(int)
    trades[col_interval] = np.ceil(trades[col_timestamp].astype(int) / interval_ns).astype(int) * interval_ns
    for col in cols_trades:
        trades[col] = trades[col].astype(float)
    # If we compute VWAP based on all trades then the time series becomes noisy due to trades
    # which are based on prices from earlier times
    with open(conditions_fn, "rb") as in_file:
        conditions = pickle.load(in_file)
    aggregates, market_hours = compute_aggregates(trades, conditions, col_interval)

    # MERGE QUOTES AND TRADES INTO SNAPSHOTS
    snapshots_all = pd.merge(
        quotes, aggregates, on=[col_ticker, col_interval], how="outer", suffixes=["_quote", "_trade"]
    ).sort_values([col_ticker, col_interval])

    _dfs = []
    for _, snapshot in snapshots_all.groupby(col_ticker):
        # Forward fill missing values from orderbook and VWAP
        for col in [
            "ask_price",
            "ask_size",
            "bid_price",
            "bid_size",
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "vwap_all",
            "vwap_ol",
        ]:
            snapshot[col] = snapshot[col].ffill()
        for col in ["sip_timestamp_quote", "sip_timestamp_trade"]:
            snapshot[col] = snapshot[col].ffill()
        # Data should start after the first valid quote and trade
        snapshot.dropna(subset=["sip_timestamp_quote", "sip_timestamp_trade"], inplace=True)
        for col in ["sip_timestamp_quote", "sip_timestamp_trade"]:
            snapshot[col] = snapshot[col].astype(int)

        # Order sizes from trades should be filled with zero
        for col in ["volume", "n"]:
            snapshot[col] = snapshot[col].fillna(0)

        _dfs.append(snapshot)

    return pd.concat(_dfs), market_hours


def load_file_info_cache(fn):
    df = pd.read_parquet(fn)
    cache = dict()
    for _, row in df.iterrows():
        date = row["date"]
        if date not in cache:
            cache[date] = dict()

        partition = row["partition"]
        assert partition not in cache[date]
        cache[date][partition] = row["size"]

    return cache


def save_file_info_cache(fn, cache):
    records = []
    for date in cache:
        for partition, size in cache[date].items():
            records.append({"date": date, "partition": partition, "size": size})

    pd.DataFrame(records).to_parquet(fn, index=False)


if __name__ == "__main__":
    args = parse_args()
    assert args.n_partitions == 100, "Values for `n_partitions` other than 100 are not supported"
    if args.max_workers < 1:
        raise ValueError("--max_workers must be at least 1")
    if args.arrow_cpu_threads < 1 or args.arrow_io_threads < 1:
        raise ValueError("PyArrow thread counts must be at least 1")
    # Pandas delegates Parquet reads/writes to PyArrow.  Its default pools are
    # sized from the host CPU count, so several multiprocessing workers can
    # otherwise create hundreds of native threads.  Bound both pools before
    # any worker processes are forked; the workers inherit these settings.
    pa.set_cpu_count(args.arrow_cpu_threads)
    pa.set_io_thread_count(args.arrow_io_threads)
    if args.date_end is None:
        logger.info("No end date provided, processing through present day")
        args.date_end = pd.to_datetime("today").strftime("%Y-%m-%d")

    def should_process(inputs):
        yr, mth, _ = inputs["date"].split("-")
        path_out = os.path.join(
            inputs["output_dir"],
            f"{inputs['freq_hz']}Hz",
            str(yr),
            str(mth),
            f"{inputs['date']}.parquet",
            f"partition={inputs['partition']}",
        )
        fn_out = os.path.join(
            path_out,
            "0.parquet",
        )

        return inputs, not os.path.exists(fn_out)

    def work(inputs, force=False):
        result = dict()
        result.update(inputs)
        # Check if work needs to be done
        yr, mth, _ = inputs["date"].split("-")
        path_out = os.path.join(
            inputs["output_dir"],
            f"{inputs['freq_hz']}Hz",
            str(yr),
            str(mth),
            f"{inputs['date']}.parquet",
            f"partition={inputs['partition']}",
        )
        fn_out = os.path.join(
            path_out,
            "0.parquet",
        )
        result["fn_out"] = fn_out
        if os.path.exists(fn_out) and not force:
            result["processed"] = False
            return result

        # Process data into a snapshot
        try:
            snapshot, _ = create_snapshot(
                inputs["date"], inputs["partition"], freq_hz=inputs["freq_hz"], data_dir=inputs["data_dir"]
            )
            os.makedirs(path_out, exist_ok=True)
            temp_out = f"{fn_out}.tmp-{os.getpid()}"
            for stale_temp in glob.glob(f"{fn_out}.tmp-*"):
                os.remove(stale_temp)
            snapshot.to_parquet(temp_out, index=False)
            os.replace(temp_out, fn_out)
            result["processed"] = True
        except Exception as error:
            temp_out = f"{fn_out}.tmp-{os.getpid()}"
            if os.path.exists(temp_out):
                os.remove(temp_out)
            result["processed"] = False
            result["error"] = {"type": str(type(error)), "msg": str(error)}
            logger.error(f'Failed to process {fn_out}: {json.dumps(result["error"])}')

        return result

    # All possible dates
    dates = set([d.strftime("%Y-%m-%d") for d in pd.bdate_range(args.date_start, args.date_end)])

    # Assemble input
    fn_cache = os.path.join(args.data_dir, "us_stocks_sip", "quotes_v1", ".cache_file_info")
    cache = load_file_info_cache(fn_cache)
    inputs = []
    bad_date_partitions = []
    for date in dates:
        base_input = {"date": date, "data_dir": args.data_dir, "output_dir": args.output_dir, "freq_hz": args.freq_hz}
        for i in range(args.n_partitions):
            rec = base_input.copy()
            rec["partition"] = i
            try:
                size = cache[date][i]
            except KeyError:
                yr, mth, _ = date.split("-")
                fn_quotes = os.path.join(
                    args.data_dir,
                    "us_stocks_sip",
                    "quotes_v1",
                    str(yr),
                    str(mth),
                    f"{date}.parquet",
                    f"partition={i}",
                    "00000000.parquet",
                )
                if not os.path.exists(fn_quotes):
                    # No data available for this date-partition
                    bad_date_partitions.append((date, i))
                    continue

                size = os.stat(fn_quotes).st_size
                if date not in cache:
                    cache[date] = dict()
                cache[date][i] = size

            rec["size"] = size
            inputs.append(rec)

    bad_dates = sorted(list(set([dp[0] for dp in bad_date_partitions])))
    n_bad_dates = len(bad_dates)
    if n_bad_dates > 0:
        logger.warning(
            f"{n_bad_dates:,} dates are missing data for at least one partition: {', '.join(bad_dates[:10])}..."
        )

    # Save updated cache
    save_file_info_cache(fn_cache, cache)

    # Check which inputs need to be processed
    inputs_filtered = []
    with Pool(min(32, args.max_workers)) as p:
        for rec, _should_proc in tqdm(
            p.imap_unordered(should_process, inputs), total=len(inputs), miniters=1, desc="Checking for output"
        ):
            if _should_proc:
                inputs_filtered.append(rec)

    # Process inputs in batches
    results = []
    pbar = tqdm(total=len(inputs_filtered), miniters=1)
    rng = random.Random(42)
    n_procs = args.max_workers
    size_max = 10e6
    batch = [e for e in inputs_filtered if e["size"] <= size_max]
    rng.shuffle(batch)
    residual = [e for e in inputs_filtered if e["size"] > size_max]
    n_procs_batch = n_procs
    logger.info(f"Processing batch of {len(batch):,} partitions using {n_procs_batch} processes")
    with Pool(n_procs_batch) as p:
        for r in p.imap_unordered(work, batch):
            results.append(r)
            pbar.update(1)

    size_max = 20e6
    batch = [e for e in residual if e["size"] <= size_max]
    rng.shuffle(batch)
    residual = [e for e in residual if e["size"] > size_max]
    n_procs_batch = min(32, n_procs)
    logger.info(f"Processing batch of {len(batch):,} partitions using {n_procs_batch} processes")
    with Pool(n_procs_batch) as p:
        for r in p.imap_unordered(work, batch):
            results.append(r)
            pbar.update(1)

    size_max = 40e6
    batch = [e for e in residual if e["size"] <= size_max]
    rng.shuffle(batch)
    residual = [e for e in residual if e["size"] > size_max]
    # This is the dominant size tier for recent SIP data.  Measured workers in
    # the adjacent 80 MB tier use roughly 1.7--2.4 GiB RSS, leaving wide
    # memory headroom for these smaller inputs.
    n_procs_batch = min(32, n_procs)
    logger.info(f"Processing batch of {len(batch):,} partitions using {n_procs_batch} processes")
    with Pool(n_procs_batch) as p:
        for r in p.imap_unordered(work, batch):
            results.append(r)
            pbar.update(1)

    size_max = 80e6
    batch = [e for e in residual if e["size"] <= size_max]
    rng.shuffle(batch)
    residual = [e for e in residual if e["size"] > size_max]
    # Recent 40--80 MB quote partitions measured 1.9 GiB RSS per worker on
    # average (2.7 GiB max), so thirty-two workers keep a wide memory margin.
    n_procs_batch = min(32, n_procs)
    logger.info(f"Processing batch of {len(batch):,} partitions using {n_procs_batch} processes")
    with Pool(n_procs_batch) as p:
        for r in p.imap_unordered(work, batch):
            results.append(r)
            pbar.update(1)

    size_max = 160e6
    batch = [e for e in residual if e["size"] <= size_max]
    rng.shuffle(batch)
    residual = [e for e in residual if e["size"] > size_max]
    # These partitions measured about 2.8 GiB RSS per worker on average and
    # 4.9 GiB at the observed maximum, so this tier uses sixteen workers.
    n_procs_batch = min(16, n_procs)
    logger.info(f"Processing batch of {len(batch):,} partitions using {n_procs_batch} processes")
    with Pool(n_procs_batch) as p:
        for r in p.imap_unordered(work, batch):
            results.append(r)
            pbar.update(1)

    rng.shuffle(residual)
    n_procs_batch = min(4, n_procs)
    logger.info(f"Processing batch of {len(residual):,} partitions using {n_procs_batch} processes")
    with Pool(n_procs_batch) as p:
        for r in p.imap_unordered(work, residual):
            results.append(r)
            pbar.update(1)

    # Provide messaging for errors
    all_errors = [r for r in results if "error" in r]
    fnf_errors = [r for r in all_errors if "FileNotFound" in r["error"]["type"]]
    errors = [r for r in all_errors if "FileNotFound" not in r["error"]["type"]]
    n_errors = len(errors)
    if n_errors > 0:
        logger.error(f"{n_errors:,} partitions were not processed successfully")

    dates_fnf_errors = sorted(list(set([e["date"] for e in fnf_errors])))
    n_fnf_dates = len(dates_fnf_errors)
    if n_fnf_dates > 0:
        logger.warning(f"{n_fnf_dates:,} dates led to FileNotFound errors: {','.join(dates_fnf_errors[:10])}...")
    if all_errors:
        sys.exit(1)
