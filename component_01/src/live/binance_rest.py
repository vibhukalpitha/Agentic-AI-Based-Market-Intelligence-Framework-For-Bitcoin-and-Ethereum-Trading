"""Live data - fetch CLOSED candles from the Binance spot REST API.

Endpoint: GET /api/v3/klines on https://data-api.binance.vision, Binance's
host for public market data (no account, no API key), with api.binance.com
as a fallback. Source: Binance Spot API docs, "General API Information" and
"Market Data endpoints" (checked 21 Oct 2026): weight 2 per request, at most
1000 candles per request, times in milliseconds unless the header
X-MBX-TIME-UNIT: MICROSECOND is sent, HTTP 429/418 carry a Retry-After header.

WHY microseconds: the frozen Binance Vision archives use microseconds from
2025 (a Daily candle closes at 23:59:59.999999). Asking the API for the same
unit makes live candles identical in form to the frozen ones.

WHY only closed candles: the newest candle the API returns can still be
forming (today's Daily candle, for instance). Using it would put information
from the future into the present record. A candle is kept only if its close
time lies before Binance's own server time (not this computer's clock).

Python's standard urllib is used, so no new library is needed.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from src import config
from src.data.download import parse_timestamps

BASE_URLS = ["https://data-api.binance.vision", "https://api.binance.com"]
HEADERS = {"X-MBX-TIME-UNIT": "MICROSECOND", "User-Agent": "component01-regime-engine"}
MAX_LIMIT = 1000          # the API's maximum candles per request
PAUSE_SECONDS = 0.25      # politeness pause between requests
MAX_RETRIES = 5
TIMEOUT_SECONDS = 30


class BinanceClient:
    """Minimal read-only client. Tests replace it with a fake that has get_json."""

    def __init__(self, base_urls: list[str] = BASE_URLS):
        self.base_urls = base_urls
        self.used_weight = None

    def get_json(self, path: str, params: dict | None = None):
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        last_error = None
        for attempt in range(MAX_RETRIES):
            for base in self.base_urls:
                request = urllib.request.Request(base + path + query, headers=HEADERS)
                try:
                    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                        self.used_weight = response.headers.get("X-MBX-USED-WEIGHT-1M")
                        time.sleep(PAUSE_SECONDS)
                        return json.load(response)
                except urllib.error.HTTPError as error:
                    if error.code in (418, 429):
                        # Binance requires backing off; Retry-After is in seconds.
                        wait = int(error.headers.get("Retry-After", "60"))
                        print(f"  Binance asked to wait {wait} s (HTTP {error.code})")
                        time.sleep(wait)
                        last_error = error
                        break
                    last_error = error                       # try the next host
                except (urllib.error.URLError, TimeoutError) as error:
                    last_error = error                       # network problem: next host
            time.sleep(2 ** attempt)
        raise ConnectionError(f"Binance request failed after {MAX_RETRIES} attempts: {last_error}")


def server_time(client) -> pd.Timestamp:
    """Binance's clock, so that 'closed' never depends on this computer's clock."""
    raw = client.get_json("/api/v3/time")["serverTime"]
    return parse_timestamps(pd.Series([raw])).iloc[0]


def parse_klines(rows: list[list]) -> pd.DataFrame:
    """API kline arrays -> the same 11 columns and types as the frozen Parquet files."""
    frame = pd.DataFrame(rows, columns=config.BINANCE_CSV_COLUMNS).drop(columns="ignore")
    frame["open_time"] = parse_timestamps(frame["open_time"])
    frame["close_time"] = parse_timestamps(frame["close_time"])
    frame["num_trades"] = frame["num_trades"].astype("int64")
    floats = [c for c in config.RAW_COLUMNS if c not in ("open_time", "close_time", "num_trades")]
    frame[floats] = frame[floats].astype("float64")
    return frame[config.RAW_COLUMNS]


def fetch_closed(client, symbol: str, interval: str, start: pd.Timestamp,
                 now: pd.Timestamp) -> pd.DataFrame:
    """Every candle that opened at or after `start` and had closed by `now`."""
    step = config.INTERVAL_STEP[interval]
    parts, cursor = [], start
    while cursor + step <= now:
        rows = client.get_json("/api/v3/klines", {
            "symbol": symbol, "interval": interval, "limit": MAX_LIMIT,
            "startTime": int(cursor.value // 1000)})          # microseconds, as sent back
        if not rows:
            break
        page = parse_klines(rows)
        parts.append(page)
        if len(rows) < MAX_LIMIT:
            break
        cursor = page["open_time"].iloc[-1] + step
    if not parts:
        return pd.DataFrame(columns=config.RAW_COLUMNS)
    candles = pd.concat(parts, ignore_index=True)
    closed = candles[(candles["open_time"] >= start) & (candles["close_time"] < now)]
    return closed.drop_duplicates("open_time").reset_index(drop=True)
