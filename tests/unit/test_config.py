from datetime import date

import pytest
from pydantic import ValidationError

from quant_research.config import load_settings


def test_loads_all_configs(settings):
    assert settings.instruments["NQ"].tick_value == 5.0
    assert settings.instruments["MNQ"].tick_value == 0.5
    assert settings.contract.symbol == settings.backtest.instrument
    assert {"opening_range_breakout", "vwap_reversion"} <= set(settings.strategies)


def test_overrides_and_env(monkeypatch, tmp_path):
    monkeypatch.setenv("NQQR_LAKE_ROOT", str(tmp_path))
    s = load_settings(start_date=date(2025, 6, 2))
    assert s.pipeline.lake_root == str(tmp_path)
    assert s.pipeline.start_date == date(2025, 6, 2)


def test_rejects_inverted_dates():
    with pytest.raises(ValidationError):
        load_settings(start_date=date(2025, 6, 2), end_date=date(2025, 6, 1))
