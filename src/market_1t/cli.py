"""Command line interface: ``market-1t {download,convert,snapshot,run}``."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from . import __version__
from .layout import FEEDS, Layout, parse_dates

logger = logging.getLogger("market_1t")

_CPUS = os.cpu_count() or 2


def _default_convert_workers() -> int:
    return max(1, min(4, _CPUS // 16))


def _add_layout_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("locations")
    group.add_argument(
        "--root",
        default=os.environ.get("MARKET_1T_ROOT"),
        help="Base directory for all data (default: $MARKET_1T_ROOT). Flat files go directly under it, "
        "Parquet under ROOT/parquet and snapshots under ROOT/snapshots.",
    )
    group.add_argument("--flatfiles-dir", help="Flat files directory (default: ROOT)")
    group.add_argument("--parquet-dir", help="Partitioned Parquet directory (default: ROOT/parquet)")
    group.add_argument("--snapshots-dir", help="Snapshot directory (default: ROOT/snapshots)")
    group.add_argument("--product", default="us_stocks_sip", help="Massive flat-file product (default: us_stocks_sip)")


def _add_date_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("dates")
    group.add_argument("--start", help="First date (YYYY-MM-DD)")
    group.add_argument("--end", help="Last date, inclusive (default: today)")
    group.add_argument("--dates", help="Comma-separated dates instead of --start/--end")


def _add_download_args(parser: argparse.ArgumentParser, workers_flag: str) -> None:
    group = parser.add_argument_group("download")
    group.add_argument(workers_flag, dest="download_workers", type=int, default=4, help="Files downloaded concurrently")
    group.add_argument("--endpoint-url", default="https://files.massive.com", help="Massive S3 endpoint")
    group.add_argument("--bucket", default="flatfiles", help="Massive S3 bucket")
    group.add_argument("--profile", help="AWS CLI profile holding the Massive S3 keys (default: environment variables)")
    group.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Times a failed download is retried, after 10 s, 20 s, 40 s, ... (default: %(default)s)",
    )


def _add_convert_args(parser: argparse.ArgumentParser, workers_flag: str) -> None:
    group = parser.add_argument_group("conversion")
    group.add_argument(
        workers_flag,
        dest="convert_workers",
        type=int,
        default=_default_convert_workers(),
        help="Flat files converted concurrently (default: %(default)s)",
    )
    group.add_argument("--parse-threads", type=int, default=min(8, _CPUS), help="CSV parsing threads per file")
    group.add_argument("--write-threads", type=int, default=min(6, _CPUS), help="Parquet writer threads per file")
    group.add_argument("--chunk-mb", type=int, default=64, help="Decompressed CSV parsed at a time (MiB)")
    group.add_argument("--buffer-mb", type=int, default=1024, help="Rows buffered for writing per file (MiB)")


def _add_snapshot_args(parser: argparse.ArgumentParser, workers_flag: str) -> None:
    group = parser.add_argument_group("snapshots")
    group.add_argument(
        "--freq-hz",
        default="1",
        help="Comma-separated sampling frequencies in Hz, e.g. 1 or 1,0.5,0.125 (default: 1). "
        "All frequencies are built from one read of the data.",
    )
    group.add_argument(
        workers_flag, dest="snapshot_workers", type=int, default=None, help="Concurrent partition workers"
    )
    group.add_argument("--threads-per-worker", type=int, default=1, help="Native threads per snapshot worker")
    group.add_argument("--conditions", help="JSON trade condition rules (default: the packaged rules)")
    group.add_argument("--overwrite", action="store_true", help="Rebuild snapshots that already exist")


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("resources")
    group.add_argument("--n-partitions", type=int, default=100, help="Ticker-hash partitions per day (default: 100)")
    group.add_argument(
        "--memory-budget-gb",
        type=float,
        default=None,
        help="Combined estimated memory of concurrent workers (default: half of the available memory)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="market-1t",
        description="Build regularly sampled (e.g. 1 Hz) US equity snapshots from Massive flat files.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only log warnings and errors")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run = commands.add_parser(
        "run",
        help="download, convert and snapshot a range of dates",
        description="Download the quote and trade flat files of the given dates, convert them to partitioned "
        "Parquet and build the snapshots. Every stage skips work that is already done.",
    )
    _add_layout_args(run)
    _add_date_args(run)
    _add_download_args(run, "--download-workers")
    run.add_argument("--skip-download", action="store_true", help="Use the flat files already on disk")
    _add_convert_args(run, "--convert-workers")
    _add_snapshot_args(run, "--snapshot-workers")
    _add_common_args(run)
    run.add_argument("--log-dir", help="Write one JSON line per converted file / processed partition here")
    run.set_defaults(handler=_run)

    download = commands.add_parser("download", help="download flat files from Massive")
    _add_layout_args(download)
    _add_date_args(download)
    download.add_argument("--feeds", default=",".join(FEEDS), help="Comma-separated feeds (default: %(default)s)")
    _add_download_args(download, "--workers")
    download.add_argument("--dry-run", action="store_true", help="Only list what would be downloaded")
    download.set_defaults(handler=_download)

    convert = commands.add_parser("convert", help="convert flat files to ticker-partitioned Parquet")
    _add_layout_args(convert)
    _add_date_args(convert)
    convert.add_argument("--feeds", default=",".join(FEEDS), help="Comma-separated feeds (default: %(default)s)")
    _add_convert_args(convert, "--workers")
    _add_common_args(convert)
    convert.add_argument("--log-file", help="Append one JSON line per converted file")
    convert.set_defaults(handler=_convert)

    snapshot = commands.add_parser("snapshot", help="build snapshots from the partitioned Parquet data")
    _add_layout_args(snapshot)
    _add_date_args(snapshot)
    _add_snapshot_args(snapshot, "--workers")
    _add_common_args(snapshot)
    snapshot.add_argument("--log-file", help="Append one JSON line per processed partition")
    snapshot.set_defaults(handler=_snapshot)
    return parser


# ---------------------------------------------------------------------------


def _layout(args) -> Layout:
    if not args.root and not (args.flatfiles_dir and args.parquet_dir and args.snapshots_dir):
        raise ValueError("--root is required (or set MARKET_1T_ROOT)")
    return Layout.from_root(
        args.root or ".",
        flatfiles=args.flatfiles_dir,
        parquet=args.parquet_dir,
        snapshots=args.snapshots_dir,
        product=args.product,
    )


def _dates(args) -> list[str]:
    dates = parse_dates(args.dates, args.start, args.end)
    if not dates:
        raise ValueError("No weekdays in the requested date range")
    return dates


def _feeds(value: str) -> list[str]:
    return [feed.strip() for feed in value.split(",") if feed.strip()]


def _memory_budget(args) -> int | None:
    from .scheduler import GiB

    return int(args.memory_budget_gb * GiB) if args.memory_budget_gb is not None else None


def _do_download(args, layout, dates, feeds) -> int:
    from .download import download

    return download(
        layout,
        dates,
        feeds,
        endpoint_url=args.endpoint_url,
        bucket=args.bucket,
        profile=args.profile,
        max_workers=args.download_workers,
        retries=args.retries,
        dry_run=getattr(args, "dry_run", False),
    )


def _do_convert(args, layout, dates, feeds, log_file) -> int:
    from .jobs import run_conversion

    return run_conversion(
        layout,
        dates,
        feeds,
        n_partitions=args.n_partitions,
        max_workers=args.convert_workers,
        memory_budget=_memory_budget(args),
        parse_threads=args.parse_threads,
        write_threads=args.write_threads,
        chunk_mb=args.chunk_mb,
        buffer_mb=args.buffer_mb,
        log_file=log_file,
    )


def _do_snapshot(args, layout, dates, log_file) -> int:
    from .jobs import parse_freqs, run_snapshots

    return run_snapshots(
        layout,
        dates,
        parse_freqs(args.freq_hz),
        n_partitions=args.n_partitions,
        max_workers=args.snapshot_workers,
        memory_budget=_memory_budget(args),
        threads_per_worker=args.threads_per_worker,
        conditions_path=args.conditions,
        overwrite=args.overwrite,
        log_file=log_file,
    )


def _download(args) -> int:
    return _do_download(args, _layout(args), _dates(args), _feeds(args.feeds))


def _convert(args) -> int:
    return _do_convert(args, _layout(args), _dates(args), _feeds(args.feeds), args.log_file)


def _snapshot(args) -> int:
    return _do_snapshot(args, _layout(args), _dates(args), args.log_file)


def _run(args) -> int:
    from .jobs import parse_freqs

    layout, dates = _layout(args), _dates(args)
    parse_freqs(args.freq_hz)  # validate before downloading anything
    log = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log = args.log_dir
    status = 0
    if not args.skip_download:
        logger.info("Stage 1/3: downloading flat files for %d dates", len(dates))
        status = max(status, _do_download(args, layout, dates, list(FEEDS)))
    logger.info("Stage 2/3: converting flat files to partitioned Parquet")
    status = max(status, _do_convert(args, layout, dates, list(FEEDS), log and os.path.join(log, "convert.jsonl")))
    logger.info("Stage 3/3: creating %s Hz snapshots", args.freq_hz)
    status = max(status, _do_snapshot(args, layout, dates, log and os.path.join(log, "snapshot.jsonl")))
    return status


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.handler(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"market-1t: error: {error}\n")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
