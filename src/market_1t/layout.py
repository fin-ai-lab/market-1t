"""Where the pipeline's files live, and date handling.

Everything lives under one root directory by default::

    {root}/{product}/{feed}/{yyyy}/{mm}/{date}.csv.gz                             flat files (as on Massive S3)
    {root}/parquet/{product}/{feed}/{yyyy}/{mm}/{date}.parquet/partition={p}/     partitioned Parquet
    {root}/snapshots/{freq}Hz/{yyyy}/{mm}/{date}.parquet/partition={p}/0.parquet  snapshots

Each of the three locations can be moved independently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, "os.PathLike[str]"]

DEFAULT_PRODUCT = "us_stocks_sip"
QUOTES = "quotes_v1"
TRADES = "trades_v1"
FEEDS = (QUOTES, TRADES)


def freq_label(freq_hz: float) -> str:
    """Directory label of a sampling frequency: ``1Hz``, ``0.5Hz``, ``0.125Hz``..."""
    return f"{float(freq_hz):g}Hz"


def _split(date: str):
    yr, mth, _ = date.split("-")
    return yr, mth


@dataclass(frozen=True)
class Layout:
    flatfiles: Path
    parquet: Path
    snapshots: Path
    product: str = DEFAULT_PRODUCT

    @classmethod
    def from_root(
        cls,
        root: PathLike,
        flatfiles: PathLike | None = None,
        parquet: PathLike | None = None,
        snapshots: PathLike | None = None,
        product: str = DEFAULT_PRODUCT,
    ) -> Layout:
        root = Path(root)
        return cls(
            flatfiles=Path(flatfiles) if flatfiles else root,
            parquet=Path(parquet) if parquet else root / "parquet",
            snapshots=Path(snapshots) if snapshots else root / "snapshots",
            product=product,
        )

    # Flat files -------------------------------------------------------------

    def s3_key(self, feed: str, date: str) -> str:
        yr, mth = _split(date)
        return f"{self.product}/{feed}/{yr}/{mth}/{date}.csv.gz"

    def flatfile(self, feed: str, date: str) -> Path:
        return self.flatfiles / self.s3_key(feed, date)

    def flatfile_for_key(self, key: str) -> Path:
        return self.flatfiles / key

    # Partitioned Parquet ----------------------------------------------------

    def parquet_day(self, feed: str, date: str) -> Path:
        yr, mth = _split(date)
        return self.parquet / self.product / feed / yr / mth / f"{date}.parquet"

    def parquet_partition(self, feed: str, date: str, partition: int) -> Path:
        return self.parquet_day(feed, date) / f"partition={partition}"

    # Snapshots --------------------------------------------------------------

    def snapshot_day(self, freq_hz: float, date: str) -> Path:
        yr, mth = _split(date)
        return self.snapshots / freq_label(freq_hz) / yr / mth / f"{date}.parquet"

    def snapshot_file(self, freq_hz: float, date: str, partition: int) -> Path:
        return self.snapshot_day(freq_hz, date) / f"partition={partition}" / "0.parquet"


def business_dates(start: str, end: str | None = None) -> list[str]:
    """Weekdays from ``start`` through ``end`` (inclusive; default: today)."""
    if end is None:
        end = pd.Timestamp("today").strftime("%Y-%m-%d")
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]


def parse_dates(dates: str | None, start: str | None, end: str | None) -> list[str]:
    """Explicit comma-separated ``dates`` or the weekdays from ``start`` to ``end``."""
    if dates:
        return sorted({pd.Timestamp(d.strip()).strftime("%Y-%m-%d") for d in dates.split(",") if d.strip()})
    if not start:
        raise ValueError("Give --dates or --start (and optionally --end)")
    return business_dates(start, end)
