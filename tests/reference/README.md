# Reference implementation

`legacy_snapshot.py` and `legacy_conversion.py` are the original pandas/Polars
implementations of the snapshot and conversion steps. Only their default paths
(now under `./data`) and comments about the machine they ran on were changed.
The tests run them next to `market_1t` and require identical output
(byte-identical snapshot files, identical Parquet data), and
`tests/validate_against_reference.py` does the same on real data.

They are not part of the package. `legacy_snapshot.create_snapshot` expects the
trade condition rules as a pickle (`{"trade": {"consolidated": {...}}}`); the
tests write one from `src/market_1t/trade_conditions.json`.
