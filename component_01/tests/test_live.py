"""Tests for the live-data path, with FAKE Binance responses (no internet).
Run:  python -m pytest"""
import os
import subprocess
import sys

import pandas as pd
import pytest

from src import config
from src.live import binance_rest, store
from src.live.update import compare_prefix

FIVE_MIN_US = 5 * 60 * 1_000_000
START = pd.Timestamp("2026-10-01")


def kline(open_us: int, close_price: float = 100.0, step_us: int = FIVE_MIN_US) -> list:
    """One API kline array, microsecond times, as the API sends it (strings for prices)."""
    return [open_us, "100.0", "101.0", "99.0", str(close_price), "5.0", open_us + step_us - 1,
            "500.0", 7, "2.5", "250.0", "0"]


class FakeClient:
    """Serves klines from a fixed list, honouring startTime and limit like the API."""

    def __init__(self, rows: list[list]):
        self.rows, self.requests, self.used_weight = rows, 0, None

    def get_json(self, path, params=None):
        self.requests += 1
        start = params["startTime"]
        selected = [r for r in self.rows if r[0] >= start]
        return selected[:params["limit"]]


def start_us(ts: pd.Timestamp) -> int:
    return int(ts.value // 1000)


def test_parse_matches_the_frozen_format_in_both_time_units():
    micro = binance_rest.parse_klines([kline(start_us(START))])
    milli = binance_rest.parse_klines([[start_us(START) // 1000, "1", "1", "1", "1", "1",
                                        start_us(START) // 1000 + 299_999, "1", 1, "1", "1", "0"]])
    assert list(micro.columns) == config.RAW_COLUMNS
    assert micro["open_time"].iloc[0] == START == milli["open_time"].iloc[0]
    assert micro["close_time"].iloc[0] == START + pd.Timedelta(minutes=5) - pd.Timedelta(microseconds=1)
    assert str(micro["close_time"].dtype) == "datetime64[ns]" and micro["close"].dtype == "float64"
    assert micro["num_trades"].dtype == "int64"


def test_only_closed_candles_are_kept():
    rows = [kline(start_us(START) + k * FIVE_MIN_US) for k in range(4)]
    now = START + pd.Timedelta(minutes=17)          # the 4th candle (15:00-20:00) is still open
    candles = binance_rest.fetch_closed(FakeClient(rows), "BTCUSDT", "5m", START, now)
    assert len(candles) == 3
    assert (candles["close_time"] < now).all()


def test_pages_of_1000_are_followed_to_the_end():
    rows = [kline(start_us(START) + k * FIVE_MIN_US) for k in range(2_500)]
    now = START + pd.Timedelta(days=30)
    client = FakeClient(rows)
    candles = binance_rest.fetch_closed(client, "BTCUSDT", "5m", START, now)
    assert len(candles) == 2_500 and client.requests == 3
    assert candles["open_time"].is_monotonic_increasing and not candles["open_time"].duplicated().any()


def test_merge_is_repeatable_and_refuses_changed_history():
    first = binance_rest.parse_klines([kline(start_us(START) + k * FIVE_MIN_US) for k in range(3)])
    again = binance_rest.parse_klines([kline(start_us(START) + k * FIVE_MIN_US) for k in range(1, 5)])
    merged, added = store.merge(first, again)
    assert len(merged) == 5 and added == 2
    changed = binance_rest.parse_klines([kline(start_us(START), close_price=999.0)])
    with pytest.raises(ValueError):
        store.merge(first, changed)


def test_check_reports_gaps_and_continuity():
    step = config.INTERVAL_STEP["5m"]
    rows = [kline(start_us(START) + k * FIVE_MIN_US) for k in (0, 1, 3)]   # one candle missing
    facts = store.check(binance_rest.parse_klines(rows), START - step, step)
    assert facts["continuous_start"] is True and facts["gaps"] == 1 and facts["candles"] == 3
    with pytest.raises(ValueError):                  # overlapping the frozen data is refused
        store.check(binance_rest.parse_klines(rows), START, step)


def test_workspace_variable_redirects_only_data_models_results(tmp_path):
    code = ("from src import config; "
            "print(config.DATA_DIR, config.MODELS_DIR, config.RESULTS_DIR, sep='|')")
    env = {**os.environ, config.WORKSPACE_ENV: str(tmp_path)}
    redirected = subprocess.run([sys.executable, "-c", code], cwd=config.ROOT_DIR, env=env,
                                capture_output=True, text=True).stdout.strip().split("|")
    assert redirected == [str(tmp_path / "data"), str(tmp_path / "models"), str(tmp_path / "results")]
    env.pop(config.WORKSPACE_ENV)
    normal = subprocess.run([sys.executable, "-c", code], cwd=config.ROOT_DIR, env=env,
                            capture_output=True, text=True).stdout.strip().split("|")
    assert normal == [str(config.ROOT_DIR / "data"), str(config.ROOT_DIR / "models"),
                      str(config.ROOT_DIR / "results")]


def test_reproduction_check_detects_a_changed_past_value():
    frozen = pd.DataFrame({"t": [1, 2, 3], "label": ["a", "b", "c"], "x": [0.1, 0.2, 0.3]})
    extended = pd.concat([frozen, pd.DataFrame({"t": [4], "label": ["d"], "x": [0.4]})])
    assert compare_prefix(frozen, extended, "t", ["label"], ["x"])[0]
    altered = extended.copy()
    altered.loc[1, "x"] = 0.2001
    assert not compare_prefix(frozen, altered, "t", ["label"], ["x"])[0]
    relabelled = extended.copy()
    relabelled.loc[0, "label"] = "z"
    assert not compare_prefix(frozen, relabelled, "t", ["label"], ["x"])[0]
