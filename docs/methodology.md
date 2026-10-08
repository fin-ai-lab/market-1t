# Methodology

This page defines how the snapshots are computed from Massive's `us_stocks_sip` flat files: `quotes_v1` holds the
NBBO quotes and `trades_v1` the trades of the consolidated tape. The definitions follow the original implementation
exactly, including the numerical details listed at the end; `tests/reference/legacy_snapshot.py` is that implementation.

## Intervals

For a sampling frequency `f` (Hz) the interval length is `Δ = int(1e9 / f)` nanoseconds, e.g. 1 s for 1 Hz. Every
quote and trade is assigned to the interval that ends at

```
ts_interval = ceil(sip_timestamp / Δ) · Δ
```

so an interval covers `(ts_interval − Δ, ts_interval]` of SIP time (UTC nanoseconds). Output directories are named by
the shortest form of the frequency: `1Hz`, `0.5Hz`, `0.125Hz`, `10Hz`.

## Quotes

Quotes are ordered by `(ticker, sip_timestamp, sequence_number)`. Each interval keeps the **last** quote per ticker,
which is the NBBO in force at the end of the interval: `ask_price`, `bid_price` and the sizes. Sizes are converted from
round lots to shares (×100).

## Trades

### Eligibility

Each trade carries a set of condition codes. Following the CTA and UTP consolidated processing rules, a trade may
update an aggregate only if **all** of its codes belong to that aggregate's eligible set. A trade without conditions is
a regular sale (code 1).

| Aggregate | Eligible when |
|---|---|
| high / low | all codes in the high/low set, no Stock Option code (35), and size ≥ 1 share |
| open / close | all codes in the open/close set, no Stock Option code (35), and size ≥ 1 share |
| volume | all codes in the volume set, and not a Form T trade during regular hours |

The sets are in [`src/market_1t/trade_conditions.json`](../src/market_1t/trade_conditions.json) (captured 2024-09-14)
and are extended at load time:

- **Form T (12):** extended-hours trades reported late. They count for prices and volume, except that a Form T trade
  inside the regular session (09:30–16:00 US/Eastern, both ends inclusive) does not count for volume.
- **Trade Thru Exempt (41):** counts for prices and volume.
- **Stock Option (35):** never counts for prices.

A trade's eligible volume is its size if it is volume-eligible and 0 otherwise. Trades eligible for nothing are
ignored.

### Aggregates per interval

Trades are ordered by `(ticker, sip_timestamp, sequence_number)`. For each ticker and interval:

| Column | Definition |
|---|---|
| `open` | price of the first open/close-eligible trade |
| `close` | price of the last open/close-eligible trade |
| `high`, `low` | maximum / minimum price of the high/low-eligible trades |
| `volume` | sum of eligible volume |
| `n` | number of trades that update any aggregate (trades without a size are not counted) |
| `sip_timestamp_trade` | SIP timestamp of the last such trade |
| `vwap_all` | `Σ price · volume / Σ volume` over the trades with eligible volume |
| `vwap` | the same, restricted to trades that are also eligible for a price aggregate |
| `vwap_ol` | the same, restricted to odd lots (eligible volume below 100 shares) |

A VWAP whose denominator is zero (no qualifying trade) is missing for that interval and is then forward filled like the
prices.

## Snapshot rows

Quote intervals and trade intervals are merged per `(ticker, ts_interval)`, so an interval appears if the ticker had
a quote update, a trade, or both. Within each ticker, in time order:

- quote columns (`ask_*`, `bid_*`, `sip_timestamp_quote`), prices (`open`, `high`, `low`, `close`) and the VWAPs are
  forward filled;
- `volume` and `n` are 0 in intervals without trades;
- rows before the ticker's first quote **and** first trade are dropped.

Inactive intervals are not materialised. To get a dense grid, reindex each ticker on the intervals of interest and
forward fill.

## Numerical details

These properties of the original implementation are part of the data and are reproduced exactly:

- **Interval assignment in float64.** `ceil(sip_timestamp / Δ)` divides in float64, so a timestamp within about
  128 ns of an interval boundary can fall into the neighbouring interval.
- **Timestamp precision.** `sip_timestamp_quote` and `sip_timestamp_trade` are rounded to float64 precision (multiples
  of 256 ns for current timestamps) whenever some interval of the partition lacks a quote or a trade, which in
  practice is always. `n` is stored as a double for the same reason.
- **Compensated sums.** Volume and the VWAP numerators and denominators are Kahan-compensated sums in trade order.
- **VWAP price.** The price `vwap` uses for a trade is the mean of its eligible open/high/low/close entries, evaluated
  as `(((o + h) + l) + c) / count`. This equals the trade price up to the last bit.
- **Missing values.** First/last/max/min/sum skip missing values, and missing floats are stored as Parquet nulls.
- **Session for Form T.** The regular session used for the Form T rule starts at 09:30 US/Eastern on the UTC calendar
  date of the partition's first eligible trade in file order.
- **Null tickers.** Rows without a ticker are ignored.

## Parquet conversion

The flat files are converted to one Parquet dataset per day and feed with 100 partitions. The partition is
`polars.col("ticker").hash(42) % 100`, so a ticker always lands in the same partition. Rows keep their file order
within a partition. Only empty, unquoted CSV fields are missing values, so tickers such as `NA` or `NULL` are kept.
Blank lines are skipped.
