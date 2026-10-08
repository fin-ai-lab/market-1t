# Market-1T

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](https://openreview.net/forum?id=OEokq7iASc)
[![arXiv](https://img.shields.io/badge/arXiv-2610.09048-b31b1b.svg)](https://arxiv.org/abs/2610.09048)
[![CI](https://github.com/fin-ai-lab/market-1t/actions/workflows/ci.yml/badge.svg)](https://github.com/fin-ai-lab/market-1t/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

One trillion 1-second aggregates of US equities, built from [Massive](https://massive.com) (formerly Polygon.io) flat files.

`market-1t` turns the consolidated (SIP) quote and trade flat files that Massive publishes for every trading day into
regularly sampled market snapshots. By default the interval is one second, but any frequency works, e.g.
`--freq-hz 1,0.5,0.125`. Each snapshot row combines the prevailing NBBO with OHLCV and VWAP aggregates of the trades in
that interval. The pipeline:

1. **downloads** the flat files from Massive's S3 endpoint,
2. **converts** each day's gzip CSVs into Parquet, split into 100 partitions by ticker hash, and
3. **builds snapshots** for any number of frequencies from a single pass over each partition.

Every stage is resumable and memory-bounded. A day of 2026 data (about 12 GB of compressed flat files) goes from flat
files to 1 Hz snapshots in about 1.5 minutes on a 64-core machine.

## Output

One Parquet dataset per day and frequency, partitioned by ticker hash:

```
{root}/snapshots/1Hz/2026/08/2026-08-03.parquet/partition={0..99}/0.parquet
```

Each partition has one row per ticker and interval in which the ticker had a quote update or a trade. Rows start once
the ticker has had both a quote and a trade, and inactive intervals are omitted, so this is not a dense grid.

| Column | Type | Description |
|---|---|---|
| `ticker` | string | Stock symbol |
| `ts_interval` | int64 | End of the interval (ns since the epoch, UTC); intervals are `(ts_interval - Δ, ts_interval]` |
| `sip_timestamp_quote` | int64 | SIP timestamp of the latest quote up to the interval |
| `ask_price`, `bid_price` | double | NBBO at the end of the interval (forward filled) |
| `ask_size`, `bid_size` | double | NBBO sizes in shares |
| `open`, `high`, `low`, `close` | double | Prices of the trades eligible to set them (forward filled) |
| `volume` | double | Eligible volume in the interval (0 when no trades) |
| `n` | double | Number of trades in the interval that update any aggregate |
| `sip_timestamp_trade` | int64 | SIP timestamp of the latest eligible trade up to the interval |
| `vwap_all` | double | VWAP of all volume-eligible trades |
| `vwap` | double | VWAP of trades that update both price and volume |
| `vwap_ol` | double | VWAP of odd-lot trades (< 100 shares) |

[docs/methodology.md](docs/methodology.md) defines every column: which trade conditions count, Form T handling,
forward filling, and the numerical details.

## Requirements

- A Massive plan that includes **flat files for US stocks**, with the flat-file **S3 Access Key ID and Secret Access
  Key** from the Massive dashboard. These are not the REST API key.
- Linux or macOS with [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 and the pinned dependencies.
- Disk space. For a 2026 trading day, the flat files take about 12 GB, the Parquet about 4.5 GB and the 1 Hz snapshots
  about 1.7 GB; earlier years are much smaller. The flat files can be deleted once converted.
- Memory. Each converted file needs about 3 GB, and each snapshot worker 0.5–2.5 GB. Workers are scheduled within a
  memory budget (half of the available memory by default), so smaller machines run fewer at a time.

## Installation

```bash
git clone https://github.com/fin-ai-lab/market-1t.git
cd market-1t
uv sync
uv run market-1t --help
```

`uv sync` installs the exact dependency versions pinned in `uv.lock`.

## Quick start

```bash
export MASSIVE_ACCESS_KEY_ID=...      # Flat-file S3 "Access Key ID" from the Massive dashboard
export MASSIVE_SECRET_ACCESS_KEY=...  # Flat-file S3 "Secret Access Key"

uv run market-1t run --root ./data --start 2026-08-03 --end 2026-08-07 --freq-hz 1
```

This downloads the quote and trade files of those days into `./data/us_stocks_sip/`, converts them to
`./data/parquet/` and writes the 1 Hz snapshots to `./data/snapshots/1Hz/`. Rerunning the command skips everything that
is already done, so interrupted runs can simply be restarted. Days without trading (weekends, holidays) are skipped
automatically.

## Usage

Each stage has its own command. `--root` (or `$MARKET_1T_ROOT`) is the base directory, and `--flatfiles-dir`,
`--parquet-dir` and `--snapshots-dir` move individual locations. Dates are given with `--start`/`--end` (weekdays,
inclusive) or as a list with `--dates`.

```bash
# 1. Download flat files (quotes_v1 and trades_v1 by default); --dry-run lists what would be fetched
uv run market-1t download --root ./data --start 2026-08-01 --end 2026-08-31

# 2. Convert them to ticker-partitioned Parquet
uv run market-1t convert --root ./data --start 2026-08-01 --end 2026-08-31

# 3. Build snapshots; several frequencies share one pass over the data
uv run market-1t snapshot --root ./data --start 2026-08-01 --end 2026-08-31 --freq-hz 1,0.5,0.25,0.125
```

A snapshot frequency can be added later without converting again: `snapshot` only builds outputs that are missing.
A failed download is retried from the start up to three times, after 10, 20 and 40 seconds (`--retries`).
`--help` on each command lists the tuning options: worker counts, threads, `--memory-budget-gb`, and `--log-file` for
one JSON line per processed file or partition.

### Data layout

```
{root}/us_stocks_sip/{quotes_v1,trades_v1}/YYYY/MM/DATE.csv.gz          flat files (same keys as on S3)
{root}/parquet/us_stocks_sip/{quotes_v1,trades_v1}/YYYY/MM/DATE.parquet/partition={p}/00000000.parquet
{root}/snapshots/{freq}Hz/YYYY/MM/DATE.parquet/partition={p}/0.parquet
```

A ticker always lands in the same partition (`polars.col("ticker").hash(42) % 100`).

### Reading the snapshots

```python
import polars as pl
from market_1t.convert import ticker_partitions

day = "data/snapshots/1Hz/2026/08/2026-08-03.parquet"
snapshots = pl.scan_parquet(f"{day}/*/0.parquet")  # the whole day

partition = ticker_partitions(["AAPL"], 100)[0]  # or just one ticker's partition
aapl = pl.read_parquet(f"{day}/partition={partition}/0.parquet").filter(pl.col("ticker") == "AAPL")
```

## Performance

Measured on a 64-core Linux server with hard-disk storage:

| Task | Time |
|---|---|
| Convert one 10 GB quotes file (561M rows) | ~45 s, <3.3 GB RSS |
| Convert a month of quotes and trades | limited by disk reads (~0.5 GB/s of compressed input) |
| Snapshot one 2026 day at 1 Hz and 0.125 Hz (100 partitions) | ~25 s with 32 workers |
| One new day from flat files to snapshots | ~90 s |

## Reproducibility

The outputs are byte-identical to those of the original pandas implementation, which is kept in `tests/reference/`.
The tests compare the two on synthetic edge cases, and `tests/validate_against_reference.py` compares them on real days.
This was checked on 17 days from 2008 to 2026 at 1 Hz and 0.125 Hz.

`uv.lock` pins the exact versions the outputs were validated with (NumPy 1.26.4, pandas 2.2.2, Polars 1.6.0,
PyArrow 17.0.0), and `uv sync` installs them. `pyproject.toml` declares the compatible ranges. Tickers are assigned to
partitions with Polars' hash, which changed in Polars 1.26, so the range stops there and the converter refuses to run
with a Polars that would move tickers.

## Development

```bash
uv sync
uv run pytest                    # ~30 s
uv run ruff check && uv run ruff format --check
```

## Citation

If you use Market-1T or the data it produces in your research, please cite our paper:

```bibtex
@inproceedings{merchant2026towards,
  title={Towards Financial World Modeling},
  author={Merchant, Humzah and Guthrie, Alec and Mahns, Simon and Balestriero, Randall and Levy, Bradford},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
  url={https://arxiv.org/abs/2610.09048}
}
```

## License

MIT, see [LICENSE](LICENSE).
