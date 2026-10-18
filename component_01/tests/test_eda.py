"""Tests for the EDA data-quality counts. Run:  python -m pytest"""
import pandas as pd

from src.eda.explore import quality_row

HOUR = pd.Timedelta(hours=1)
ALMOST = pd.Timedelta(milliseconds=1)


def make(open_times: list[pd.Timestamp], volumes: list[float]) -> pd.DataFrame:
    open_time = pd.DatetimeIndex(open_times)
    return pd.DataFrame({"open_time": open_time, "close_time": open_time + HOUR - ALMOST,
                         "volume": volumes})


def test_clean_series_has_no_problems():
    start = pd.Timestamp("2024-01-01")
    row = quality_row(make([start + k * HOUR for k in range(5)], [1.0] * 5), HOUR)
    assert (row["gaps"], row["odd_length_candles"], row["zero_volume_candles"],
            row["duplicate_times"], row["missing_values"]) == (0, 0, 0, 0, 0)
    assert row["candles"] == 5 and row["first"] == "2024-01-01"


def test_gaps_short_candles_and_zero_volume_are_counted():
    start = pd.Timestamp("2024-01-01")
    times = [start, start + HOUR, start + 4 * HOUR, start + 5 * HOUR]   # 2 missing hours = 1 gap
    candles = make(times, [1.0, 0.0, 1.0, 0.0])
    candles.loc[2, "close_time"] = candles.loc[2, "open_time"] + pd.Timedelta(minutes=20)
    row = quality_row(candles, HOUR)
    assert (row["gaps"], row["odd_length_candles"], row["zero_volume_candles"]) == (1, 1, 2)
