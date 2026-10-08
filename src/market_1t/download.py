"""Download Massive flat files from their S3-compatible endpoint.

Flat files need a Massive plan that includes them and the *S3* access key and
secret from the Massive dashboard (they differ from the REST API key).  Pass
them as ``MASSIVE_ACCESS_KEY_ID`` / ``MASSIVE_SECRET_ACCESS_KEY`` or through an
AWS CLI profile (``--profile``).

Files are stored under the layout's flat-file directory with the same keys as
on S3 (``us_stocks_sip/trades_v1/2026/08/2026-08-03.csv.gz``).  A file is
downloaded when it is missing or its size differs from the remote object, to a
temporary name that is only renamed into place once complete.  A failed
download is retried from the start after a pause that doubles every time.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm

from .layout import Layout

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "https://files.massive.com"
DEFAULT_BUCKET = "flatfiles"
ACCESS_KEY_ENV = "MASSIVE_ACCESS_KEY_ID"
SECRET_KEY_ENV = "MASSIVE_SECRET_ACCESS_KEY"
MiB = 1024**2
DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 10.0  # seconds before the first retry; doubles after every failed attempt


@dataclass(frozen=True)
class RemoteFile:
    key: str
    size: int
    feed: str
    date: str


def make_client(endpoint_url: str = DEFAULT_ENDPOINT, profile: str | None = None, max_connections: int = 32):
    """S3 client for the Massive flat-file endpoint."""
    import boto3
    from botocore.config import Config

    key_id = os.environ.get(ACCESS_KEY_ENV)
    secret = os.environ.get(SECRET_KEY_ENV)
    if bool(key_id) != bool(secret):
        raise ValueError(f"Set both {ACCESS_KEY_ENV} and {SECRET_KEY_ENV}")
    if key_id:
        session = boto3.session.Session(aws_access_key_id=key_id, aws_secret_access_key=secret)
    else:
        session = boto3.session.Session(profile_name=profile)
    if session.get_credentials() is None:
        raise ValueError(
            f"No Massive S3 credentials: set {ACCESS_KEY_ENV} and {SECRET_KEY_ENV} to the Flat Files keys from "
            "the Massive dashboard, or pass --profile with an AWS CLI profile holding them"
        )
    config = Config(
        signature_version="s3v4",
        retries={"max_attempts": 10, "mode": "standard"},
        max_pool_connections=max_connections,
    )
    return session.client("s3", endpoint_url=endpoint_url, config=config)


def list_remote_files(
    client, bucket: str, product: str, feeds: Sequence[str], dates: Sequence[str]
) -> list[RemoteFile]:
    """Flat files of ``feeds`` for ``dates`` (days without trading have none)."""
    wanted = set(dates)
    months = sorted({date[:7] for date in dates})
    paginator = client.get_paginator("list_objects_v2")
    files = []
    for feed in feeds:
        for month in months:
            yr, mth = month.split("-")
            for page in paginator.paginate(Bucket=bucket, Prefix=f"{product}/{feed}/{yr}/{mth}/"):
                for obj in page.get("Contents", []):
                    name = obj["Key"].rsplit("/", 1)[-1]
                    date = name.split(".", 1)[0]
                    if name.endswith(".csv.gz") and date in wanted:
                        files.append(RemoteFile(key=obj["Key"], size=int(obj["Size"]), feed=feed, date=date))
    return files


def _is_complete(path: Path, size: int) -> bool:
    try:
        return path.stat().st_size == size
    except FileNotFoundError:
        return False


def plan_downloads(files: Sequence[RemoteFile], layout: Layout) -> list[RemoteFile]:
    """Files that are missing locally or whose size differs from the remote object."""
    return [f for f in files if not _is_complete(layout.flatfile_for_key(f.key), f.size)]


def download_file(client, bucket: str, remote: RemoteFile, path: Path, transfer_config=None, callback=None) -> None:
    """Download one object to ``path``; the file only appears once complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in path.parent.glob(f"{path.name}.part-*"):
        stale.unlink(missing_ok=True)
    temp = path.with_name(f"{path.name}.part-{os.getpid()}")
    try:
        client.download_file(bucket, remote.key, str(temp), Config=transfer_config, Callback=callback)
        size = temp.stat().st_size
        if size != remote.size:
            raise OSError(f"{remote.key}: downloaded {size} bytes, expected {remote.size}")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class _AttemptProgress:
    """Forwards transfer progress and remembers how much one attempt reported."""

    def __init__(self, callback):
        self.callback = callback
        self.received = 0
        self.lock = threading.Lock()  # transfer threads report concurrently

    def __call__(self, n: int) -> None:
        with self.lock:
            self.received += n
        if self.callback:
            self.callback(n)

    def take_back(self) -> None:
        if self.callback and self.received:
            self.callback(-self.received)


def download_file_with_retries(
    client,
    bucket: str,
    remote: RemoteFile,
    path: Path,
    transfer_config=None,
    callback=None,
    *,
    retries: int = DEFAULT_RETRIES,
    delay: float = DEFAULT_RETRY_DELAY,
) -> None:
    """``download_file``, retried from the start up to ``retries`` times.

    Botocore retries throttling and server errors itself, but not everything
    that turns out to be transient: the flat-file endpoint occasionally answers
    one of the ranged GETs of a multipart download with 403 Forbidden, which
    would otherwise fail the whole file.  The pause before a retry is ``delay``
    seconds and doubles every time.  Progress reported by a failed attempt is
    taken back (reported as negative) so that progress totals stay correct.
    """
    for attempt in range(retries + 1):
        progress = _AttemptProgress(callback)
        try:
            download_file(client, bucket, remote, path, transfer_config, progress)
            return
        except Exception as error:  # noqa: BLE001 - retried, then raised
            progress.take_back()
            if attempt == retries:
                raise
            pause = delay * 2**attempt
            logger.warning(
                "Download of %s failed (%s); retry %d of %d in %.0fs", remote.key, error, attempt + 1, retries, pause
            )
            time.sleep(pause)


def download(
    layout: Layout,
    dates: Sequence[str],
    feeds: Sequence[str],
    *,
    endpoint_url: str = DEFAULT_ENDPOINT,
    bucket: str = DEFAULT_BUCKET,
    profile: str | None = None,
    max_workers: int = 4,
    per_file_connections: int = 8,
    retries: int = DEFAULT_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    dry_run: bool = False,
    client=None,
) -> int:
    """Download the flat files of ``feeds`` for ``dates``; returns an exit status."""
    if retries < 0:
        raise ValueError("retries must be at least 0")
    if client is None:
        client = make_client(endpoint_url, profile, max_connections=max_workers * per_file_connections + 8)
    files = list_remote_files(client, bucket, layout.product, feeds, dates)
    todo = plan_downloads(files, layout)
    days = {f.date for f in files}
    total = sum(f.size for f in todo)
    logger.info(
        "%d flat files for %d of %d requested days (%s); %d to download (%.1f GB)",
        len(files),
        len(days),
        len(dates),
        ", ".join(feeds),
        len(todo),
        total / 1e9,
    )
    without = sorted(set(dates) - days)
    if without:
        shown = ", ".join(without[:10]) + ("..." if len(without) > 10 else "")
        logger.info("No files for %d days (market holidays or not yet published): %s", len(without), shown)
    if dry_run or not todo:
        for f in todo:
            logger.info("would download %s (%.2f GB)", f.key, f.size / 1e9)
        return 0

    from boto3.s3.transfer import TransferConfig

    transfer_config = TransferConfig(
        multipart_threshold=64 * MiB,
        multipart_chunksize=64 * MiB,
        max_concurrency=per_file_connections,
        use_threads=True,
    )
    failures = []
    with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="download") as progress:
        with ThreadPoolExecutor(max(1, max_workers)) as pool:
            futures = {
                pool.submit(
                    download_file_with_retries,
                    client,
                    bucket,
                    remote,
                    layout.flatfile_for_key(remote.key),
                    transfer_config,
                    progress.update,
                    retries=retries,
                    delay=retry_delay,
                ): remote
                for remote in sorted(todo, key=lambda f: -f.size)
            }
            for future in as_completed(futures):
                remote = futures[future]
                try:
                    future.result()
                except Exception as error:  # noqa: BLE001 - reported below
                    failures.append((remote, error))
                    logger.error("Failed to download %s: %s", remote.key, error)
    if failures:
        logger.error("%d of %d downloads failed", len(failures), len(todo))
        return 1
    logger.info("Downloaded %d files (%.1f GB)", len(todo), total / 1e9)
    return 0
