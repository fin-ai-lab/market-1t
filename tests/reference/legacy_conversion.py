import argparse
import glob
import logging
import os
import random
import shutil
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)
logger.setLevel(logging.INFO)

SCHEMAS_BY_FEED = {
    "quotes_v1": {
        "ticker": pl.String,
        "ask_exchange": pl.Int64,
        "ask_price": pl.Float64,
        "ask_size": pl.Int64,
        "bid_exchange": pl.Int64,
        "bid_price": pl.Float64,
        "bid_size": pl.Int64,
        "conditions": pl.String,
        "indicators": pl.String,
        "participant_timestamp": pl.Int64,
        "sequence_number": pl.Int64,
        "sip_timestamp": pl.Int64,
        "tape": pl.Int64,
        "trf_timestamp": pl.Int64,
    },
    "trades_v1": {
        "ticker": pl.String,
        "conditions": pl.String,
        "correction": pl.Int64,
        "exchange": pl.Int64,
        "id": pl.Int64,
        "participant_timestamp": pl.Int64,
        "price": pl.Float64,
        "sequence_number": pl.Int64,
        "sip_timestamp": pl.Int64,
        # SIP flat files can contain fractional-share trades (for example
        # size=0.140000), so this must not be parsed as an integer.
        "size": pl.Float64,
        "tape": pl.Int64,
        "trf_id": pl.Int64,
        "trf_timestamp": pl.Int64,
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
    "trades_v1": [
        "ticker",
        "sip_timestamp",
        "sequence_number",
        "exchange",
        "price",
        "size",
        "conditions",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert gzip compressed CSV files from Polygon.io into Parquet files")
    parser.add_argument("--polygon_dir", type=str, default="./data/")
    parser.add_argument("--output_dir", type=str, default="./data/parquet/")
    parser.add_argument(
        "--product",
        type=str,
        default="us_stocks_sip",
        help="The product to sync.",
    )
    parser.add_argument(
        "--feeds",
        type=str,
        default="quotes_v1",
        help="Feeds with the product to sync",
    )
    parser.add_argument("--date_start", type=str, default="2008-01-01")  # Look into when Reg NMS became effective
    parser.add_argument("--date_end", type=str, default=None)
    parser.add_argument("--n_partitions", type=int, default=100)
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Hard cap on concurrent daily conversions. Streaming days may use up to eight workers.",
    )
    parser.add_argument(
        "--streaming_threshold_gb",
        type=float,
        default=5.0,
        help="Use bounded-memory batched quote conversion at or above this decimal-GB size.",
    )
    parser.add_argument(
        "--trade_streaming_threshold_gb",
        type=float,
        default=2.0,
        help="Use bounded-memory batched trade conversion at or above this decimal-GB size.",
    )
    parser.add_argument(
        "--streaming_block_mb",
        type=int,
        default=256,
        help="Decompressed Arrow CSV block size in MiB for bounded-memory conversion.",
    )

    args = parser.parse_args()

    return args


def write_partitioned_streaming(inputs: dict, temp_out: str):
    """Convert one CSV in bounded batches while keeping one file per partition."""
    output_root = Path(temp_out)
    output_root.mkdir(parents=True, exist_ok=False)
    selected_columns = set(COLS_BY_FEED[inputs["feed"]])
    # pl.read_csv preserves source/schema order rather than projection-list
    # order.  Match that ordering so streamed and eager outputs are identical.
    columns = [column for column in SCHEMAS_BY_FEED[inputs["feed"]] if column in selected_columns]
    arrow_types = {pl.String: pa.string(), pl.Int64: pa.int64(), pl.Float64: pa.float64()}
    column_types = {column: arrow_types[SCHEMAS_BY_FEED[inputs["feed"]][column]] for column in columns}
    writers = {}
    rows_written = 0
    try:
        with pa.input_stream(inputs["fn_in"], compression="gzip", buffer_size=1 << 20) as source:
            reader = pacsv.open_csv(
                source,
                read_options=pacsv.ReadOptions(
                    use_threads=False,
                    block_size=inputs["streaming_block_bytes"],
                ),
                convert_options=pacsv.ConvertOptions(
                    column_types=column_types,
                    include_columns=columns,
                    strings_can_be_null=True,
                ),
            )
            for batch in reader:
                if batch.num_rows == 0:
                    continue
                df = pl.from_arrow(batch)
                df = df.with_columns((pl.col("ticker").hash(42) % inputs["n_partitions"]).alias("partition"))
                for key, partition_df in df.partition_by("partition", maintain_order=True, as_dict=True).items():
                    partition = int(key[0])
                    partition_dir = output_root / f"partition={partition}"
                    writer = writers.get(partition)
                    table = partition_df.to_arrow()
                    if writer is None:
                        partition_dir.mkdir(parents=True, exist_ok=False)
                        writer = pq.ParquetWriter(
                            partition_dir / "00000000.parquet",
                            table.schema,
                            compression="zstd",
                            use_dictionary=True,
                            write_statistics=True,
                        )
                        writers[partition] = writer
                    writer.write_table(table)
                    rows_written += partition_df.height
    finally:
        for writer in writers.values():
            writer.close()

    if rows_written == 0:
        raise RuntimeError(f"Streaming conversion produced no rows for {inputs['fn_in']}")


def process_file(inputs: dict, force=False):
    result = inputs.copy()

    if os.path.exists(inputs["fn_out"]) and not force:
        result["skipped"] = True
        return result

    fn_in = inputs["fn_in"]
    temp_out = f"{inputs['fn_out']}.tmp-{os.getpid()}"
    try:
        output_path = Path(inputs["fn_out"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for stale_temp in output_path.parent.glob(f"{output_path.name}.tmp-*"):
            shutil.rmtree(stale_temp)
        if inputs["size_in"] >= inputs["streaming_threshold_bytes"]:
            logger.info(
                "Using bounded-memory streaming conversion for %s (%.2f GB compressed)",
                fn_in,
                inputs["size_in"] / 1e9,
            )
            write_partitioned_streaming(inputs, temp_out)
        else:
            df = pl.read_csv(
                fn_in,
                low_memory=True,
                columns=COLS_BY_FEED[inputs["feed"]],
                schema=SCHEMAS_BY_FEED[inputs["feed"]],
            )
            df = df.with_columns((pl.col("ticker").hash(42) % inputs["n_partitions"]).alias("partition"))
            df.write_parquet(temp_out, partition_by=["partition"])
        partition_dirs = list(Path(temp_out).glob("partition=*"))
        parquet_files = list(Path(temp_out).glob("partition=*/*.parquet"))
        if len(partition_dirs) != inputs["n_partitions"] or len(parquet_files) != inputs["n_partitions"]:
            raise RuntimeError(
                f"Expected {inputs['n_partitions']} partition directories/files; "
                f"found {len(partition_dirs)}/{len(parquet_files)}"
            )
        os.rename(temp_out, inputs["fn_out"])
    except KeyboardInterrupt as error:
        result["error"] = str(type(error))
        result["error_msg"] = str(error)
    except pl.exceptions.ComputeError as error:
        logger.error("Polars failed to process input file %s: %s", fn_in, error)
        result["error"] = str(type(error))
        result["error_msg"] = str(error)
    except Exception as error:
        logger.error(f"Unexpected {type(error)} when processing {fn_in}: {str(error)}")
        result["error"] = str(type(error))
        result["error_msg"] = str(error)

    if os.path.exists(temp_out):
        shutil.rmtree(temp_out)

    return result


def fail_if_errors(results, feed):
    failures = [result for result in results if "error" in result]
    if failures:
        logger.error("%s had %d failed daily conversions", feed, len(failures))
        for failure in failures:
            logger.error("%s: %s", failure.get("fn_in"), failure.get("error_msg"))
        raise RuntimeError(f"{feed} conversion failed for {len(failures)} day(s)")


def process_batch(batch, n_procs, results, pbar):
    """Process one size bucket without forking when only one worker is requested.

    Forking a Pool after Polars has initialized its native thread pools can leave
    the child blocked on an inherited futex.  The one-worker path is also the
    memory-safe production path, so execute it in the current process.
    """
    if n_procs == 1:
        for item in batch:
            result = process_file(item)
            results.append(result)
            pbar.update(1)
            fail_if_errors([result], item["feed"])
        return

    # Spawn fresh interpreters. Forking after Polars/Arrow has initialized
    # native thread pools can leave children blocked on inherited futexes.
    with get_context("spawn").Pool(n_procs) as pool:
        for result in pool.imap_unordered(process_file, batch):
            results.append(result)
            pbar.update(1)


def main():
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max_workers must be at least 1")
    if args.streaming_threshold_gb < 0:
        raise ValueError("--streaming_threshold_gb must be nonnegative")
    if args.trade_streaming_threshold_gb < 0:
        raise ValueError("--trade_streaming_threshold_gb must be nonnegative")
    if args.streaming_block_mb < 1:
        raise ValueError("--streaming_block_mb must be at least 1")
    if args.date_end is None:
        logger.info("No end date provided, processing through present day")
        args.date_end = pd.to_datetime("today").strftime("%Y-%m-%d")

    dates = set([d.strftime("%Y-%m-%d") for d in pd.bdate_range(args.date_start, args.date_end)])

    for feed in args.feeds.split(","):
        assert feed in COLS_BY_FEED, f"Feed `{feed}` is not supported. Must be one of: {', '.join(COLS_BY_FEED.keys())}"
        assert (
            feed in SCHEMAS_BY_FEED
        ), f"Schema for feed `{feed}` does not exist. Must be one of: {', '.join(SCHEMAS_BY_FEED.keys())}"

        # Determine days in feed to be processed
        path_feed = os.path.join(args.polygon_dir, args.product, feed)
        fns_in_all = [
            fn for fn in glob.glob(os.path.join(path_feed, "*/*/*.csv.gz")) if fn.split("/")[-1].split(".")[0] in dates
        ]
        inputs = []
        streaming_threshold_gb = (
            args.trade_streaming_threshold_gb if feed == "trades_v1" else args.streaming_threshold_gb
        )
        for fn in fns_in_all:
            fn_out = fn.replace(args.polygon_dir, args.output_dir).replace(".csv.gz", ".parquet")
            if not os.path.exists(fn_out):
                size = os.stat(fn).st_size
                inputs.append(
                    {
                        "feed": feed,
                        "fn_in": fn,
                        "fn_out": fn_out,
                        "size_in": size,
                        "n_partitions": args.n_partitions,
                        "streaming_threshold_bytes": int(streaming_threshold_gb * 1e9),
                        "streaming_block_bytes": args.streaming_block_mb * 1024 * 1024,
                    }
                )
        perc_incomp = len(inputs) / len(fns_in_all) if fns_in_all else 0.0
        if len(inputs) == 0:
            logger.info(
                f"Feed {feed} has been fully processed from {args.date_start} through {args.date_end}. No work to do."
            )
            continue
        logger.info(
            f"Feed {feed} has {len(inputs)} unprocessed files ({perc_incomp:.0%} of {len(fns_in_all)}) spanning {args.date_start} through {args.date_end}."
        )

        results = []
        pbar = tqdm(total=len(inputs), miniters=1)

        # Scaling parameters
        size_breakpoint = 1230000000  # Based on the memory of the original machine
        n_procs = min(8, args.max_workers)
        rng = random.Random(42)

        # Bounded-memory days can safely run concurrently. Keep eager days in
        # the conservative size buckets below because their RSS scales with the
        # full decompressed input and several large eager parses can exhaust RAM.
        streaming_batch = [
            item for item in inputs if item["size_in"] >= item["streaming_threshold_bytes"]
        ]
        inputs = [
            item for item in inputs if item["size_in"] < item["streaming_threshold_bytes"]
        ]
        if streaming_batch:
            streaming_workers = min(8, args.max_workers)
            rng.shuffle(streaming_batch)
            logger.info(
                "%s - Processing bounded-memory batch of %d days using %d processes",
                feed,
                len(streaming_batch),
                streaming_workers,
            )
            process_batch(streaming_batch, streaming_workers, results, pbar)
            fail_if_errors(results, feed)
        if not inputs:
            continue

        # Process first batch
        batch = [input for input in inputs if input["size_in"] <= size_breakpoint]
        rng.shuffle(batch)
        residual = [input for input in inputs if input["size_in"] > size_breakpoint]
        logger.info(f"{feed} - Processing batch of {len(batch)} days using {n_procs} processes")
        process_batch(batch, n_procs, results, pbar)
        if len(residual) == 0:
            fail_if_errors(results, feed)
            continue

        # Process second batch
        new_size_breakpoint = size_breakpoint * 1.4
        new_n_procs = min(6, args.max_workers)
        batch = [input for input in residual if input["size_in"] <= new_size_breakpoint]
        rng.shuffle(batch)
        residual = [input for input in residual if input["size_in"] > new_size_breakpoint]
        logger.info(f"{feed} - Processing batch of {len(batch)} days using {new_n_procs} processes")
        process_batch(batch, new_n_procs, results, pbar)
        if len(residual) == 0:
            fail_if_errors(results, feed)
            continue

        # Process third batch
        new_size_breakpoint = size_breakpoint * 2
        new_n_procs = max(1, min(4, args.max_workers))
        batch = [input for input in residual if input["size_in"] <= new_size_breakpoint]
        rng.shuffle(batch)
        residual = [input for input in residual if input["size_in"] > new_size_breakpoint]
        logger.info(f"{feed} - Processing batch of {len(batch)} days using {new_n_procs} processes")
        process_batch(batch, new_n_procs, results, pbar)
        if len(residual) == 0:
            fail_if_errors(results, feed)
            continue

        # Process fourth batch
        new_size_breakpoint = size_breakpoint * 2.5
        new_n_procs = min(3, args.max_workers)
        batch = [input for input in residual if input["size_in"] <= new_size_breakpoint]
        rng.shuffle(batch)
        residual = [input for input in residual if input["size_in"] > new_size_breakpoint]
        logger.info(f"{feed} - Processing batch of {len(batch)} days using {new_n_procs} processes")
        process_batch(batch, new_n_procs, results, pbar)
        if len(residual) == 0:
            fail_if_errors(results, feed)
            continue

        # Process fifth batch
        new_size_breakpoint = size_breakpoint * 4
        new_n_procs = max(1, min(2, args.max_workers))
        batch = [input for input in residual if input["size_in"] <= new_size_breakpoint]
        rng.shuffle(batch)
        residual = [input for input in residual if input["size_in"] > new_size_breakpoint]
        logger.info(f"{feed} - Processing batch of {len(batch)} days using {new_n_procs} processes")
        process_batch(batch, new_n_procs, results, pbar)
        if len(residual) == 0:
            fail_if_errors(results, feed)
            continue

        # Process remaining days using a single process
        logger.info(f"{feed} - Processing {len(residual)} remaining days using single process")
        for input in residual:
            results.append(process_file(input))
            pbar.update(1)
        fail_if_errors(results, feed)


if __name__ == "__main__":
    main()
