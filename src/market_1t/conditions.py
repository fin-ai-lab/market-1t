"""Trade condition rules: which trades may update prices and volume.

Every trade carries a set of condition codes (Massive's unified codes).  A
trade updates an aggregate only if all its codes belong to that aggregate's
eligible set, following the CTA/UTP consolidated processing rules.  The sets
ship with the package (``trade_conditions.json``); a JSON file with the same
keys can be passed instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources

FORM_T = 12  # Form T: extended-hours trades reported late
TRADE_THRU_EXEMPT = 41
STOCK_OPTION = 35  # never updates prices
BAD_CONDITIONS_PRICE = frozenset([STOCK_OPTION])

# Eligibility flag bits.
HIGH_LOW, OPEN_CLOSE, VOLUME, HAS_FORM_T = 1, 2, 4, 8

_KEYS = ("updates_high_low", "updates_open_close", "updates_volume")


def parse_conditions(value: str | None) -> set:
    """Codes of a ``conditions`` field; a missing field means a regular sale (1)."""
    if value is None:
        return set([1])
    try:
        return set(map(int, value.split(",")))
    except TypeError:
        return set([int(value)])


@dataclass(frozen=True)
class TradeRules:
    """Condition codes allowed to update high/low, open/close and volume."""

    high_low: frozenset
    open_close: frozenset
    volume: frozenset

    @classmethod
    def from_dict(cls, rules: dict) -> TradeRules:
        missing = [key for key in _KEYS if key not in rules]
        if missing:
            raise ValueError(f"Trade rules lack {missing}")
        # Intraday aggregation also counts Form T (12) and Trade Thru Exempt
        # (41) trades for prices, and Trade Thru Exempt trades for volume.
        return cls(
            high_low=frozenset(rules["updates_high_low"]) | {FORM_T, TRADE_THRU_EXEMPT},
            open_close=frozenset(rules["updates_open_close"]) | {FORM_T, TRADE_THRU_EXEMPT},
            volume=frozenset(rules["updates_volume"]) | {TRADE_THRU_EXEMPT},
        )

    @classmethod
    def load(cls, path: str | None = None) -> TradeRules:
        """Rules from a JSON file, or the packaged defaults."""
        if path is None:
            text = resources.files("market_1t").joinpath("trade_conditions.json").read_text()
        else:
            with open(path) as handle:
                text = handle.read()
        return cls.from_dict(json.loads(text))

    def bits(self, value: str | None) -> int:
        """Eligibility flags of one ``conditions`` field."""
        conditions = parse_conditions(value)
        bits = 0
        if conditions.issubset(self.high_low) and conditions.isdisjoint(BAD_CONDITIONS_PRICE):
            bits |= HIGH_LOW
        if conditions.issubset(self.open_close) and conditions.isdisjoint(BAD_CONDITIONS_PRICE):
            bits |= OPEN_CLOSE
        if conditions.issubset(self.volume):
            bits |= VOLUME
        if FORM_T in conditions:
            bits |= HAS_FORM_T
        return bits
