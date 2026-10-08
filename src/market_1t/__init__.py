"""Market-1T: regularly sampled US equity snapshots built from Massive flat files."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("market-1t")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"
