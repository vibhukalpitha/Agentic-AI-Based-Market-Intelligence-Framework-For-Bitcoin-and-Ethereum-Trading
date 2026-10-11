"""Tests for M1 timestamp parsing. Run:  python -m pytest

Binance SPOT files use milliseconds before 2025-01-01 and microseconds from
2025-01-01. Reading a microsecond value as milliseconds would place a candle
tens of thousands of years in the future, so the unit is detected per row.
"""
import pandas as pd

from src.data.download import parse_timestamps


def test_milliseconds_and_microseconds_are_both_read_correctly():
    raw = pd.Series([
        1502928000000,          # 2017-08-17 00:00:00 in milliseconds
        1735689599999,          # 2024-12-31 23:59:59.999 in milliseconds
        1735689600000000,       # 2025-01-01 00:00:00 in microseconds
    ])
    parsed = parse_timestamps(raw)
    assert list(parsed) == [pd.Timestamp("2017-08-17"),
                            pd.Timestamp("2024-12-31 23:59:59.999"),
                            pd.Timestamp("2025-01-01")]
    assert parsed.is_monotonic_increasing


def test_microsecond_precision_is_kept():
    # A 5-minute candle's close time in the microsecond era ends in .999999
    parsed = parse_timestamps(pd.Series([1735689899999999]))
    assert parsed.iloc[0] == pd.Timestamp("2025-01-01 00:04:59.999999")


def test_a_file_mixing_both_units_stays_in_order():
    raw = pd.Series([1735689300000, 1735689600000000, 1735689900000000])
    parsed = parse_timestamps(raw)
    assert list(parsed.diff().dropna()) == [pd.Timedelta(minutes=5)] * 2
