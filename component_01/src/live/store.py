"""Live data - check and store the candles fetched after the frozen dataset.

Live candles are kept apart from the frozen data: data/live/candles/ holds only
candles that open AFTER the last frozen candle. data/parquet/ is only read.

Rules (the same as for the frozen data):
  - no duplicate candles, times strictly increasing
  - gaps are REPORTED, never filled (filling would invent prices)
  - storing is repeatable: a candle fetched twice is stored once; if Binance
    ever sends DIFFERENT values for a candle already stored, storing stops with
    an error instead of silently overwriting history
"""
import pandas as pd

from src import config

LIVE_DIR = config.DATA_DIR / "live"
CANDLES_DIR = LIVE_DIR / "candles"


def frozen_path(symbol: str, interval: str):
    return config.PARQUET_DIR / f"{symbol}_{interval}.parquet"


def live_path(symbol: str, interval: str):
    return CANDLES_DIR / f"{symbol}_{interval}.parquet"


def frozen_last_open(symbol: str, interval: str) -> pd.Timestamp:
    return pd.read_parquet(frozen_path(symbol, interval), columns=["open_time"])["open_time"].max()


def load_live(symbol: str, interval: str) -> pd.DataFrame:
    path = live_path(symbol, interval)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=config.RAW_COLUMNS)


def merge(existing: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Join stored and newly fetched candles; return (merged, number truly new).
    A candle present in both must have identical values in every column."""
    if existing.empty:
        merged = new.copy()
    elif new.empty:
        merged = existing.copy()
    else:
        both = existing.merge(new, on="open_time", suffixes=("_old", "_new"))
        for column in config.RAW_COLUMNS[1:]:
            if not (both[f"{column}_old"] == both[f"{column}_new"]).all():
                raise ValueError(f"Binance sent different '{column}' values for candles "
                                 "already stored; nothing was overwritten")
        merged = pd.concat([existing, new[~new["open_time"].isin(existing["open_time"])]])
    merged = merged.sort_values("open_time").reset_index(drop=True)
    for column in merged.columns:
        merged[column] = merged[column].astype(new[column].dtype if not new.empty
                                               else existing[column].dtype)
    return merged[config.RAW_COLUMNS], len(merged) - len(existing)


def check(live: pd.DataFrame, after: pd.Timestamp, step: pd.Timedelta) -> dict:
    """Facts about the live series that follows the frozen candle `after`."""
    if live.empty:
        return {"candles": 0, "first": None, "last": None, "gaps": 0, "continuous_start": None}
    times = live["open_time"]
    # The same hard checks as src/data/validate.py applies to the frozen files.
    # (Its fixed row-count check belongs to the frozen inventory only.)
    problems = []
    if list(live.columns) != config.RAW_COLUMNS:
        problems.append("columns differ from schema")
    if times.duplicated().any() or not times.is_monotonic_increasing:
        problems.append("duplicated or out of order")
    if live.isna().any().any():
        problems.append("null values")
    if ((live["high"] < live[["open", "close"]].max(axis=1))
            | (live["low"] > live[["open", "close"]].min(axis=1))).any():
        problems.append("high/low inconsistent")
    if (live[["open", "high", "low", "close"]] <= 0).any().any():
        problems.append("non-positive price")
    if times.iloc[0] <= after:
        problems.append("overlaps the frozen dataset")
    if problems:
        raise ValueError("live candles failed checks: " + "; ".join(problems))
    length = live["close_time"] - times
    return {"candles": int(len(live)), "first": str(times.iloc[0]), "last": str(times.iloc[-1]),
            "continuous_start": bool(times.iloc[0] == after + step),
            "gaps": int((times.diff().dropna() != step).sum()),
            "zero_volume_candles": int((live["volume"] == 0).sum()),
            "odd_length_candles": int(((length - step).abs() > pd.Timedelta(seconds=1)).sum())}


def save(symbol: str, interval: str, candles: pd.DataFrame) -> None:
    CANDLES_DIR.mkdir(parents=True, exist_ok=True)
    candles.to_parquet(live_path(symbol, interval), index=False)


def combined(symbol: str, interval: str) -> pd.DataFrame:
    """Frozen candles followed by the live ones: the full history the pipeline needs
    (features look back 365 days)."""
    frozen = pd.read_parquet(frozen_path(symbol, interval))
    live = load_live(symbol, interval)
    if live.empty:
        return frozen
    return pd.concat([frozen, live.astype(frozen.dtypes.to_dict())], ignore_index=True)
