"""End to end through the command line, on synthetic flat files."""

import filecmp
import json

import numpy as np
import pandas as pd
import pytest
from flatfiles import REAL_CONDITIONS, quotes_csv, trades_csv, write_gzip
from test_download import FakeS3

from market_1t import download as dl
from market_1t.cli import main
from market_1t.layout import Layout

N_PARTITIONS = "4"
THREADS = ["--parse-threads", "2", "--write-threads", "2"]


def flatfile_texts(date, seed):
    """Quote and trade CSV text for ``date`` (timestamps within its session)."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(f"{date} 04:00", tz="US/Eastern").value
    span = 16 * 3600 * 10**9
    trades = trades_csv(rng, 12_000, start=start, span=span, conditions=REAL_CONDITIONS)
    return quotes_csv(rng, 30_000, start=start, span=span), trades


def write_flatfiles(layout, date, seed):
    quotes, trades = flatfile_texts(date, seed)
    write_gzip(str(layout.flatfile("quotes_v1", date)), quotes)
    write_gzip(str(layout.flatfile("trades_v1", date)), trades)


def assert_matches_reference(layout, dates, freqs, legacy_snapshot, legacy_conditions, tmp_path):
    n = 0
    for date in dates:
        for partition in range(int(N_PARTITIONS)):
            for freq in freqs:
                reference, _ = legacy_snapshot.create_snapshot(
                    date, partition, freq_hz=freq, data_dir=str(layout.parquet), conditions_fn=legacy_conditions
                )
                ref_fn = tmp_path / "reference.parquet"
                reference.to_parquet(ref_fn, index=False)
                assert filecmp.cmp(ref_fn, layout.snapshot_file(freq, date, partition), shallow=False), (
                    date,
                    partition,
                    freq,
                )
                n += 1
    return n


def test_convert_then_snapshot(tmp_path, legacy_snapshot, legacy_conditions):
    layout = Layout.from_root(tmp_path / "data")
    dates = ["2026-08-03", "2026-08-04"]
    for seed, date in enumerate(dates):
        write_flatfiles(layout, date, seed)
    common = ["--root", str(layout.flatfiles), "--dates", ",".join(dates), "--n-partitions", N_PARTITIONS]
    assert main(["convert", *common, "--workers", "2", *THREADS]) == 0
    assert all(layout.parquet_partition("trades_v1", d, p).is_dir() for d in dates for p in range(4))

    log = tmp_path / "snapshot.jsonl"
    assert main(["snapshot", *common, "--freq-hz", "1,0.125", "--workers", "2", "--log-file", str(log)]) == 0
    assert assert_matches_reference(layout, dates, [1.0, 0.125], legacy_snapshot, legacy_conditions, tmp_path) == 16
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 8 and all(r["processed"] for r in records)

    # Re-running finds nothing to do; adding a frequency only builds that one.
    assert main(["snapshot", *common, "--freq-hz", "1,0.125", "--workers", "2"]) == 0
    before = layout.snapshot_file(1.0, dates[0], 0).stat().st_mtime_ns
    assert main(["snapshot", *common, "--freq-hz", "1,0.5", "--workers", "2"]) == 0
    assert layout.snapshot_file(1.0, dates[0], 0).stat().st_mtime_ns == before
    assert assert_matches_reference(layout, dates, [0.5], legacy_snapshot, legacy_conditions, tmp_path) == 8


def test_run_downloads_converts_and_snapshots(tmp_path, monkeypatch, legacy_snapshot, legacy_conditions):
    dates = ["2026-08-05"]
    objects = {}
    source = Layout.from_root(tmp_path / "remote")
    for date in dates:
        quotes, trades = flatfile_texts(date, 11)
        for feed, text in (("quotes_v1", quotes), ("trades_v1", trades)):
            path = tmp_path / "blob.gz"
            write_gzip(str(path), text)
            objects[source.s3_key(feed, date)] = path.read_bytes()
    fake = FakeS3(objects)
    monkeypatch.setattr(dl, "make_client", lambda *args, **kwargs: fake)

    root = tmp_path / "data"
    argv = [
        "run",
        "--root",
        str(root),
        "--start",
        "2026-08-05",
        "--end",
        "2026-08-05",
        "--freq-hz",
        "1",
        "--n-partitions",
        N_PARTITIONS,
        "--convert-workers",
        "1",
        "--snapshot-workers",
        "2",
        *THREADS,
    ]
    assert main(argv) == 0
    assert len(fake.downloaded) == 2
    layout = Layout.from_root(root)
    assert assert_matches_reference(layout, dates, [1.0], legacy_snapshot, legacy_conditions, tmp_path) == 4

    # Everything is already done on a second run.
    fake.downloaded.clear()
    assert main(argv) == 0
    assert fake.downloaded == []


def test_missing_root_is_a_usage_error(monkeypatch, capsys):
    monkeypatch.delenv("MARKET_1T_ROOT", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        main(["snapshot", "--dates", "2026-08-03"])
    assert exit_info.value.code == 2
    assert "--root is required" in capsys.readouterr().err


def test_bad_frequency_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        main(["snapshot", "--root", str(tmp_path), "--dates", "2026-08-03", "--freq-hz", "0"])
    assert exit_info.value.code == 2
