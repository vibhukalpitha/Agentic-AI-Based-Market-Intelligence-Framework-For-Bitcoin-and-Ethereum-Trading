"""Streaming strict-extremum swings, emitted only at candle completion."""

from __future__ import annotations

import csv
import math
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator


INTERVALS = {"15m": timedelta(minutes=15), "1h": timedelta(hours=1),
             "4h": timedelta(hours=4)}
REQUIRED_COLUMNS = ("timestamp_utc", "available_at_utc", "open", "high", "low",
                    "close", "atr_14_sma")
SWING_COLUMNS = ("swing_id", "symbol", "source_timeframe", "swing_type",
                 "swing_price", "swing_time_utc", "confirmed_at_utc",
                 "atr_at_confirmation")


def utc(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("Timezone-aware timestamp required")
    return timestamp.astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    available_at: datetime
    open: float
    high: float
    low: float
    close: float
    atr: float | None


@dataclass(frozen=True)
class Swing:
    swing_id: str
    symbol: str
    source_timeframe: str
    swing_type: str
    swing_price: float
    swing_time: datetime
    confirmed_at: datetime
    atr_at_confirmation: float | None

    def record(self) -> dict:
        return dict(zip(SWING_COLUMNS, (
            self.swing_id, self.symbol, self.source_timeframe, self.swing_type,
            self.swing_price, iso(self.swing_time), iso(self.confirmed_at),
            self.atr_at_confirmation,
        )))


def check_header(path: Path) -> None:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        fields = csv.DictReader(handle).fieldnames
    if not fields or len(fields) != len(set(fields)):
        raise ValueError(f"{path}: empty or duplicate CSV columns")
    missing = set(REQUIRED_COLUMNS) - set(fields)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")


def read_candles(path: Path, timeframe: str, stats: Counter) -> Iterator[Candle]:
    """Validate required fields while streaming. Optional mark/index fields are unused."""
    check_header(path)
    interval = INTERVALS[timeframe]
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    previous = None
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for line, row in enumerate(csv.DictReader(handle), 2):
            try:
                timestamp = utc(row["timestamp_utc"])
                available = utc(row["available_at_utc"])
                if (timestamp - epoch) % interval != timedelta(0):
                    raise ValueError("candle opening is off the UTC interval grid")
                if available != timestamp + interval:
                    raise ValueError("availability must equal candle opening plus interval")
                if previous is not None and timestamp - previous != interval:
                    raise ValueError("duplicate, unordered or missing candle timestamp")
                prices = [float(row[name]) for name in ("open", "high", "low", "close")]
                if not all(positive(value) for value in prices):
                    raise ValueError("OHLC must be finite and positive")
                opening, high, low, close = prices
                if high < max(opening, low, close) or low > min(opening, close):
                    raise ValueError("inconsistent OHLC")
                raw_atr = row["atr_14_sma"].strip()
                if not raw_atr:
                    atr = None
                    stats["missing_atr_rows"] += 1
                else:
                    try:
                        atr = float(raw_atr)
                    except ValueError:
                        atr = None
                    if not positive(atr):
                        stats["invalid_atr_rows"] += 1
                        atr = None
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                raise ValueError(f"{path}, CSV line {line}: {exc}") from exc
            previous = timestamp
            stats["input_rows"] += 1
            stats["timestamp_rows_checked"] += 1
            stats["ohlc_rows_checked"] += 1
            yield Candle(timestamp, available, opening, high, low, close, atr)


def detect_swings(candles: Iterable[Candle], symbol: str, timeframe: str,
                  left_bars: int = 3, right_bars: int = 3) -> Iterator[Swing]:
    """Strict unique extrema; a same-candle high/low pair emits high then low.

    The bounded window is never flushed at EOF, so trailing unconfirmed pivots
    cannot produce events. ATR comes from the confirming candle, not the pivot.
    """
    if left_bars < 1 or right_bars < 1:
        raise ValueError("left_bars and right_bars must be positive")
    window: deque[Candle] = deque(maxlen=left_bars + right_bars + 1)
    for candle in candles:
        window.append(candle)
        if len(window) != window.maxlen:
            continue
        pivot = window[left_bars]
        others = [bar for index, bar in enumerate(window) if index != left_bars]
        for kind, price, qualifies in (
            ("high", pivot.high, all(pivot.high > bar.high for bar in others)),
            ("low", pivot.low, all(pivot.low < bar.low for bar in others)),
        ):
            if qualifies:
                swing_id = f"{symbol}_{timeframe}_{pivot.timestamp.strftime('%Y%m%dT%H%M%SZ')}_{kind}"
                yield Swing(swing_id, symbol, timeframe, kind, price,
                            pivot.timestamp, candle.available_at, candle.atr)
