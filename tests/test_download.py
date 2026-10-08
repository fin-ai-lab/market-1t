"""Flat-file download logic, against an in-memory stand-in for the S3 endpoint."""

from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from market_1t import download as dl
from market_1t.layout import Layout


class FakeS3:
    """Minimal stand-in for the boto3 S3 client used by market_1t.download.

    ``failures`` maps keys to the number of download attempts that fail with
    403 Forbidden halfway through, like a transient error from the endpoint.
    """

    def __init__(self, objects, page_size=2, corrupt=(), failures=None):
        self.objects = dict(objects)
        self.page_size = page_size
        self.corrupt = set(corrupt)
        self.failures = dict(failures or {})
        self.attempts = []
        self.downloaded = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket, Prefix):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        if not keys:
            yield {"KeyCount": 0}
        for i in range(0, len(keys), self.page_size):
            yield {"Contents": [{"Key": k, "Size": len(self.objects[k])} for k in keys[i : i + self.page_size]]}

    def download_file(self, bucket, key, filename, Config=None, Callback=None):
        self.attempts.append(key)
        data = self.objects[key]
        if self.failures.get(key, 0) > 0:
            self.failures[key] -= 1
            with open(filename, "wb") as out:
                out.write(data[: len(data) // 2])
            if Callback:
                Callback(len(data) // 2)
            raise ClientError({"Error": {"Code": "403", "Message": "Forbidden"}}, "GetObject")
        if key in self.corrupt:
            data = data[:-1]
        with open(filename, "wb") as out:
            out.write(data)
        if Callback:
            Callback(len(data))
        self.downloaded.append(key)


def objects_for(dates, feeds=("quotes_v1", "trades_v1")):
    out = {}
    for date in dates:
        yr, mth, _ = date.split("-")
        for feed in feeds:
            out[f"us_stocks_sip/{feed}/{yr}/{mth}/{date}.csv.gz"] = f"{feed} {date}".encode() * 10
    # Other feeds and months must be ignored.
    out["us_stocks_sip/day_aggs_v1/2026/07/2026-07-31.csv.gz"] = b"x"
    out["us_stocks_sip/trades_v1/2026/06/2026-06-30.csv.gz"] = b"y"
    return out


DATES = ["2026-07-30", "2026-07-31", "2026-08-03"]


def test_listing_filters_dates_and_feeds():
    s3 = FakeS3(objects_for(DATES))
    files = dl.list_remote_files(s3, "flatfiles", "us_stocks_sip", ["quotes_v1", "trades_v1"], DATES + ["2026-08-04"])
    assert sorted((f.feed, f.date) for f in files) == sorted(
        (feed, d) for d in DATES for feed in ("quotes_v1", "trades_v1")
    )


def test_download_skips_complete_files_and_replaces_partial_ones(tmp_path):
    objects = objects_for(DATES)
    layout = Layout.from_root(tmp_path)
    s3 = FakeS3(objects)
    assert dl.download(layout, DATES, ["quotes_v1", "trades_v1"], client=s3) == 0
    assert len(s3.downloaded) == 6
    for date in DATES:
        for feed in ("quotes_v1", "trades_v1"):
            assert layout.flatfile(feed, date).read_bytes() == objects[layout.s3_key(feed, date)]
    assert not layout.flatfile_for_key("us_stocks_sip/day_aggs_v1/2026/07/2026-07-31.csv.gz").exists()

    # A truncated local file is replaced; complete files are left alone.
    truncated = layout.flatfile("trades_v1", "2026-07-31")
    truncated.write_bytes(b"partial")
    s3 = FakeS3(objects)
    assert dl.download(layout, DATES, ["quotes_v1", "trades_v1"], client=s3) == 0
    assert s3.downloaded == ["us_stocks_sip/trades_v1/2026/07/2026-07-31.csv.gz"]
    assert truncated.read_bytes() == objects["us_stocks_sip/trades_v1/2026/07/2026-07-31.csv.gz"]
    assert not list(tmp_path.rglob("*.part-*"))


def test_dry_run_downloads_nothing(tmp_path):
    s3 = FakeS3(objects_for(DATES))
    assert dl.download(Layout.from_root(tmp_path), DATES, ["trades_v1"], client=s3, dry_run=True) == 0
    assert s3.downloaded == []
    assert not list(tmp_path.rglob("*.csv.gz"))


def test_size_mismatch_is_an_error(tmp_path):
    objects = objects_for(DATES[:1], feeds=("trades_v1",))
    key = "us_stocks_sip/trades_v1/2026/07/2026-07-30.csv.gz"
    s3 = FakeS3(objects, corrupt=[key])
    layout = Layout.from_root(tmp_path)
    assert dl.download(layout, DATES[:1], ["trades_v1"], client=s3, retry_delay=0) == 1
    assert s3.attempts == [key] * (1 + dl.DEFAULT_RETRIES)
    assert not layout.flatfile_for_key(key).exists()
    assert not list(tmp_path.rglob("*.part-*"))


def test_transient_errors_are_retried_with_growing_pauses(tmp_path, monkeypatch):
    objects = objects_for(DATES[:1])
    key = "us_stocks_sip/quotes_v1/2026/07/2026-07-30.csv.gz"
    s3 = FakeS3(objects, failures={key: 2})
    pauses = []
    monkeypatch.setattr(dl, "time", SimpleNamespace(sleep=pauses.append))
    layout = Layout.from_root(tmp_path)
    assert dl.download(layout, DATES[:1], ["quotes_v1", "trades_v1"], client=s3) == 0
    assert s3.attempts.count(key) == 3
    assert pauses == [dl.DEFAULT_RETRY_DELAY, 2 * dl.DEFAULT_RETRY_DELAY]
    for feed in ("quotes_v1", "trades_v1"):
        assert layout.flatfile(feed, DATES[0]).read_bytes() == objects[layout.s3_key(feed, DATES[0])]
    assert not list(tmp_path.rglob("*.part-*"))


def test_retries_are_bounded(tmp_path):
    objects = objects_for(DATES[:1], feeds=("trades_v1",))
    key = "us_stocks_sip/trades_v1/2026/07/2026-07-30.csv.gz"
    s3 = FakeS3(objects, failures={key: 99})
    layout = Layout.from_root(tmp_path)
    assert dl.download(layout, DATES[:1], ["trades_v1"], client=s3, retries=2, retry_delay=0) == 1
    assert s3.attempts == [key] * 3
    assert not layout.flatfile_for_key(key).exists()
    assert not list(tmp_path.rglob("*.part-*"))


def test_failed_attempts_take_back_their_progress(tmp_path):
    objects = objects_for(DATES[:1], feeds=("trades_v1",))
    key = "us_stocks_sip/trades_v1/2026/07/2026-07-30.csv.gz"
    remote = dl.RemoteFile(key=key, size=len(objects[key]), feed="trades_v1", date=DATES[0])
    reported = []
    s3 = FakeS3(objects, failures={key: 1})
    dl.download_file_with_retries(s3, "flatfiles", remote, tmp_path / "file.csv.gz", callback=reported.append, delay=0)
    assert sum(reported) == len(objects[key])
    assert min(reported) < 0
    assert (tmp_path / "file.csv.gz").read_bytes() == objects[key]


def test_retries_must_not_be_negative(tmp_path):
    with pytest.raises(ValueError):
        dl.download(Layout.from_root(tmp_path), DATES, ["trades_v1"], client=FakeS3({}), retries=-1)


def test_credentials_must_be_complete(monkeypatch):
    monkeypatch.setenv(dl.ACCESS_KEY_ENV, "key")
    monkeypatch.delenv(dl.SECRET_KEY_ENV, raising=False)
    with pytest.raises(ValueError):
        dl.make_client()


def test_client_from_environment(monkeypatch):
    monkeypatch.setenv(dl.ACCESS_KEY_ENV, "key")
    monkeypatch.setenv(dl.SECRET_KEY_ENV, "secret")
    client = dl.make_client()
    assert client.meta.endpoint_url == dl.DEFAULT_ENDPOINT
