"""Component 01 regime engine as an HTTP service (port 8001) for the dashboard.

Run (from inside component_01):
    python -W ignore -m uvicorn src.api.service:app --host 127.0.0.1 --port 8001

GET /api/health                          which data the service reads, and its date range
GET /api/regime/{symbol}?as_of=<UTC>     the output record at that moment (default: latest)

Every record is built by src/output/snapshot.build_snapshot, the same function that
writes results/sample_output_<symbol>.json, so at 30 Sep 2026 the two are identical.
The record also carries `dtw_similarity`: the M7 chart data of
src/figures/similarity_charts, as numbers instead of PNG files.

DATA. When the live workspace (data/live/workspace, filled by src.live.update) exists,
it is read, so dates after 30 Sep 2026 are covered; otherwise the frozen data. The
service only reads: nothing is fetched, trained or written. Inputs are cached and
reloaded when a new live update changes the workspace's regime files.
"""
import os
import threading
from pathlib import Path

# Must be set before src.config is imported: config reads it once, at import.
_LIVE_WORKSPACE = Path(__file__).resolve().parents[2] / "data" / "live" / "workspace"
if not os.environ.get("COMPONENT01_WORKSPACE") and (_LIVE_WORKSPACE / "data" / "regimes").is_dir():
    os.environ["COMPONENT01_WORKSPACE"] = str(_LIVE_WORKSPACE)

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src import config
from src.explain import explainer
from src.figures import similarity_charts
from src.output import snapshot
from src.regime import describe
from src.similarity import dtw_match

app = FastAPI(title="Component 01 - Market Regime Engine")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"],
                   allow_headers=["*"])

# Dashboard feature key -> (feature column, unit, scale applied to the stored value)
CHART_FEATURES = {"rsi": ("rsi", "pts", 1.0),
                  "sma_dist": ("sma_dist", "%", 100.0),
                  "macd_hist": ("macd_hist_norm", "% of price", 100.0),
                  "atr_vol": ("atr_rel", "log ratio", 1.0)}
DISTRIBUTION_BINS = 30

_cache: dict[str, dict] = {}
_lock = threading.Lock()


def _data_version() -> float:
    """Latest change to the regime files; a live update changes it."""
    return max((p.stat().st_mtime for p in config.REGIMES_DIR.glob("*.parquet")), default=0.0)


def _inputs(symbol: str) -> dict:
    """Everything build_snapshot reads for one asset, loaded once per data version."""
    version = _data_version()
    with _lock:
        cached = _cache.get(symbol)
        if cached is None or cached["version"] != version:
            labels = snapshot.load_all_labels(symbol)
            cached = {
                "version": version,
                "labels": labels,
                "profiles": {i: describe.load_profiles(symbol, i) for i in config.INTERVALS},
                "stability": snapshot.load_stability(symbol),
                "archetypes": explainer.load_archetypes(symbol),
                "similarity": dtw_match.load_daily(symbol),
                "features": pd.read_parquet(config.FEATURES_DIR / f"{symbol}_1d_features.parquet"),
                # a full record needs a closed candle at every timeframe
                "first": max(frame["known_time"].iloc[0] for frame in labels.values()),
                "last": max(frame["known_time"].iloc[-1] for frame in labels.values()),
            }
            _cache[symbol] = cached
        return cached


def _symbol(raw: str) -> str:
    clean = raw.replace("/", "").replace("-", "").upper()
    if not clean.endswith("USDT"):
        clean += "USDT"
    if clean not in config.SYMBOLS:
        raise HTTPException(404, f"unknown symbol {raw}; expected one of {config.SYMBOLS}")
    return clean


def _iso(time) -> str:
    return snapshot._iso(pd.Timestamp(time))


def similarity_charts_data(data: dict, symbol: str, block: dict | None) -> dict | None:
    """The four M7 charts of src/figures/similarity_charts as numbers, for the
    stretch the record's similar_history block describes."""
    if not block:
        return None
    regimes, x = data["similarity"]
    features = data["features"]
    days = regimes["open_time"]
    window = block["window_days"]
    end = int(np.flatnonzero(days.dt.date.astype(str).to_numpy() == block["window_end"])[0])
    ends = [int(np.flatnonzero(days.dt.date.astype(str).to_numpy() == m["window_end"])[0])
            for m in block["matches"]]
    candidate_ends, distances = similarity_charts.all_distances(x, end, window)

    # percentile from the unrounded distance, as in similarity_charts.analyse
    exact = {int(e): float(d) for e, d in zip(candidate_ends, distances)}
    matches = [dict(m, similarity_percentile=round(similarity_charts.percentile(exact[e], distances), 2))
               for e, m in zip(ends, block["matches"])]

    # 1. where in history: one close per month, plus today's and each match's last day
    marks = {end: (0, f"Today's stretch ({block['window_end']})")}
    marks.update({e: (k + 1, f"Match {k + 1} ({m['window_end']})") for k, (e, m) in enumerate(zip(ends, matches))})
    close = features.set_index("open_time")["close"].iloc[:end + 1]
    monthly = set(close.groupby(close.index.to_period("M")).tail(1).index)
    timeline = []
    for row in sorted({days.iloc[e] for e in marks} | {t for t in monthly}):
        point = {"date": str(pd.Timestamp(row).date()), "price": round(float(close.loc[row]), 2)}
        position = int(np.flatnonzero(days.to_numpy() == np.datetime64(row))[0])
        if position in marks:
            point["match"], point["label"] = marks[position]
        timeline.append(point)

    # 2. four key features over the 10 days, today beside each match
    def stretch(column: str, last: int, scale: float) -> list[float]:
        values = features[column].iloc[last - window + 1:last + 1].to_numpy() * scale
        return [round(float(v), 3) for v in values]

    patterns = {}
    for key, (column, unit, scale) in CHART_FEATURES.items():
        patterns[key] = {"name": similarity_charts.SHOWN_FEATURES[column], "unit": unit,
                         "days": [f"Day {d}" for d in range(1, window + 1)],
                         "today": stretch(column, end, scale)}
        for k, e in enumerate(ends):
            patterns[key][f"match{k + 1}"] = stretch(column, e, scale)

    # 3. how close: histogram of the distance to every past stretch
    counts, edges = np.histogram(distances, bins=DISTRIBUTION_BINS)
    centres = (edges[:-1] + edges[1:]) / 2
    median = float(np.median(distances))
    median_bin = int(np.argmin(np.abs(centres - median)))
    distribution = [dict({"dist": round(float(c), 3), "density": int(n)},
                         **({"is_median": True} if b == median_bin else {}))
                    for b, (c, n) in enumerate(zip(centres, counts))]

    # 4. how DTW paired the days of today's stretch and the best match
    pairs, distance = similarity_charts.dtw_path(x[end - window + 1:end + 1],
                                                 x[ends[0] - window + 1:ends[0] + 1])
    if abs(distance - matches[0]["distance"]) > 1e-3:
        raise ValueError("traced DTW path does not reproduce the match distance")

    return {"symbol": symbol, "date": block["window_end"], "window_days": window,
            "regime_now": block["regime_now"], "past_stretches_compared": int(len(distances)),
            "median_distance": round(median, 3), "matches": matches,
            "scope": "Supporting evidence for the present regime only; nothing about what "
                     "followed any stretch is assessed.",
            "vector_charts_data": {"timeline": timeline, "patterns": patterns,
                                   "distribution": distribution,
                                   "dtw_path_pairs": [{"day_today": i + 1, "day_match": j + 1}
                                                      for i, j in pairs]}}


@app.get("/api/health")
def health():
    ranges = {s: {"from": _iso(_inputs(s)["first"]), "to": _iso(_inputs(s)["last"])}
              for s in config.SYMBOLS}
    return {"status": "ok", "component": "market_regime_engine",
            "data_source": "live workspace" if os.environ.get("COMPONENT01_WORKSPACE") else "frozen data",
            "coverage": ranges}


@app.get("/api/regime/{symbol}")
def regime(symbol: str, as_of: str | None = None):
    symbol = _symbol(symbol)
    data = _inputs(symbol)
    requested = None
    if as_of:
        try:
            requested = pd.Timestamp(as_of)
        except ValueError:
            raise HTTPException(422, f"as_of is not a date/time: {as_of}")
        if requested.tzinfo is not None:
            requested = requested.tz_convert("UTC").tz_localize(None)
    if requested is not None and requested < data["first"]:
        raise HTTPException(422, f"no regime record before {_iso(data['first'])}: the 15m and "
                                 "5m data start on 1 Jan 2021, so a full five-timeframe "
                                 "record cannot be built earlier")
    # A moment after the last closed candle gets the latest record.
    target = None if requested is None or requested >= data["last"] else requested

    record = snapshot.build_snapshot(symbol, target, labels=data["labels"],
                                     profiles=data["profiles"], stability=data["stability"],
                                     archetypes=data["archetypes"], similarity=data["similarity"])
    record["requested_as_of"] = _iso(requested) if requested is not None else None
    record["data_end"] = _iso(data["last"])
    record["data_source"] = ("live workspace (Binance REST, frozen models)"
                             if os.environ.get("COMPONENT01_WORKSPACE") else "frozen data")
    record["dtw_similarity"] = similarity_charts_data(data, symbol, record["similar_history"])
    return record
