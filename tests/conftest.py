import importlib.util
import json
import pickle
from importlib import resources
from pathlib import Path

import pytest

from market_1t.conditions import TradeRules

REFERENCE = Path(__file__).parent / "reference"


def _load_reference(name: str):
    spec = importlib.util.spec_from_file_location(f"reference_{name}", REFERENCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def legacy_snapshot():
    """The original pandas snapshot implementation."""
    return _load_reference("legacy_snapshot")


@pytest.fixture(scope="session")
def legacy_conversion():
    """The original conversion implementation."""
    return _load_reference("legacy_conversion")


@pytest.fixture(scope="session")
def legacy_conditions(tmp_path_factory) -> str:
    """The packaged trade rules in the pickle layout the original code reads."""
    rules = json.loads(resources.files("market_1t").joinpath("trade_conditions.json").read_text())
    consolidated = {key: set(rules[key]) for key in ("updates_high_low", "updates_open_close", "updates_volume")}
    path = tmp_path_factory.mktemp("conditions") / "conditions.pkl"
    with open(path, "wb") as out:
        pickle.dump({"bbo": {}, "nbbo": {}, "trade": {"consolidated": consolidated}}, out)
    return str(path)


@pytest.fixture(scope="session")
def rules() -> TradeRules:
    return TradeRules.load()
