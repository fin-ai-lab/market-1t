"""Compare market-1t snapshots with the reference implementation on real data.

Builds the snapshots of the given dates twice -- with the original pandas code
(``tests/reference/legacy_snapshot.py``) and with ``market-1t snapshot`` -- from
the same partitioned Parquet data, and compares every file byte for byte::

    uv run python tests/validate_against_reference.py --parquet-dir ./data/parquet \\
        --dates 2019-11-29,2026-07-01 --freq-hz 1,0.125 --workdir /tmp/validation

The reference run is slow (it is the code being replaced).
"""

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from multiprocessing import get_context
from pathlib import Path

from market_1t.layout import Layout

REFERENCE = Path(__file__).parent / "reference" / "legacy_snapshot.py"
_LEGACY = None


def _legacy():
    global _LEGACY
    if _LEGACY is None:
        spec = importlib.util.spec_from_file_location("legacy_snapshot", REFERENCE)
        _LEGACY = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_LEGACY)
    return _LEGACY


def _init_worker():
    import pyarrow as pa

    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)


def reference_task(task):
    date, partition, freq, parquet_dir, out_root, conditions_fn = task
    fn_out = Layout.from_root(".", snapshots=out_root).snapshot_file(freq, date, partition)
    if fn_out.exists():
        return task, None
    try:
        snapshot, _ = _legacy().create_snapshot(
            date, partition, freq_hz=freq, data_dir=parquet_dir, conditions_fn=conditions_fn
        )
    except Exception as error:  # noqa: BLE001 - market-1t must fail the same way
        return task, f"{type(error).__name__}: {error}"
    fn_out.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(f"{fn_out}.tmp", index=False)
    os.replace(f"{fn_out}.tmp", fn_out)
    return task, None


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_files(root):
    return {str(p.relative_to(root)): p for p in Path(root).rglob("0.parquet")}


def _legacy_conditions(workdir):
    rules = json.loads(resources.files("market_1t").joinpath("trade_conditions.json").read_text())
    consolidated = {key: set(rules[key]) for key in ("updates_high_low", "updates_open_close", "updates_volume")}
    path = Path(workdir) / "conditions.pkl"
    with open(path, "wb") as out:
        pickle.dump({"bbo": {}, "nbbo": {}, "trade": {"consolidated": consolidated}}, out)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parquet-dir", required=True, help="Partitioned Parquet directory (ROOT/parquet)")
    parser.add_argument("--dates", required=True)
    parser.add_argument("--freq-hz", default="1,0.125")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--reference-workers", type=int, default=16)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    freqs = [float(f) for f in args.freq_hz.split(",")]
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    reference_root, new_root = workdir / "reference", workdir / "new"
    conditions_fn = _legacy_conditions(workdir)
    layout = Layout.from_root(".", parquet=args.parquet_dir)

    tasks = []
    for date in dates:
        for partition in range(100):
            quotes = layout.parquet_partition("quotes_v1", date, partition)
            if quotes.is_dir():
                size = sum(entry.stat().st_size for entry in os.scandir(quotes))
                for freq in freqs:
                    tasks.append((size, (date, partition, freq, args.parquet_dir, str(reference_root), conditions_fn)))
    tasks = [task for _, task in sorted(tasks, key=lambda item: -item[0])]

    started = time.time()
    failed = 0
    with get_context("spawn").Pool(args.reference_workers, initializer=_init_worker, maxtasksperchild=1) as pool:
        for _, error in pool.imap_unordered(reference_task, tasks):
            failed += bool(error)
    print(f"reference: {len(tasks)} partition-frequencies in {time.time() - started:.0f}s ({failed} failed)")

    started = time.time()
    command = [
        sys.executable,
        "-m",
        "market_1t",
        "-q",
        "snapshot",
        "--root",
        str(workdir),
        "--parquet-dir",
        args.parquet_dir,
        "--snapshots-dir",
        str(new_root),
        "--dates",
        ",".join(dates),
        "--freq-hz",
        args.freq_hz,
        "--workers",
        str(args.workers),
    ]
    status = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    print(f"market-1t: finished in {time.time() - started:.0f}s (exit status {status})")

    ref, new = _snapshot_files(reference_root), _snapshot_files(new_root)
    common = sorted(set(ref) & set(new))
    with ThreadPoolExecutor(32) as pool:
        same = list(pool.map(lambda key: _md5(ref[key]) == _md5(new[key]), common))
    mismatched = [key for key, ok in zip(common, same, strict=True) if not ok]
    only_ref, only_new = sorted(set(ref) - set(new)), sorted(set(new) - set(ref))
    print(f"byte-identical files: {len(common) - len(mismatched)}/{len(common)}")
    for label, keys in (("differ", mismatched), ("only reference", only_ref), ("only market-1t", only_new)):
        for key in keys[:20]:
            print(f"  {label}: {key}")
    # A partition the reference cannot process must fail in market-1t too, so
    # a file present on one side only is a difference.
    ok = not (mismatched or only_ref or only_new)
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
