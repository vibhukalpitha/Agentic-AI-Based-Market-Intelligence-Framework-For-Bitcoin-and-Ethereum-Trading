"""M7 - Historical similarity: the past stretches most like the last few days.

Run:  python -m src.similarity.dtw_match      (checks that similarity is meaningful)

The last SIMILARITY_WINDOW_DAYS Daily candles, described by the nine model
features, are compared with every earlier stretch of the same length. The
closest ones are reported with their dates and the regime the market was in
then. If stretches that look like today carried the same regime label as
today, that supports the label.

Distance: dynamic time warping (DTW), written out explicitly. Two stretches
can show the same shape slightly out of step (a move one day earlier or
later). A day-by-day comparison would call them different; DTW lets a day in
one stretch be matched with a neighbouring day in the other and takes the
cheapest such matching.

Scope - SUPPORTING EVIDENCE ONLY. This module reports when a similar stretch
occurred and what the regime was at that time. It deliberately reports
nothing about what happened AFTER a past stretch: retrieving past episodes as
evidence for what may come next belongs to another component of the project.

No look-ahead: a stretch can only be matched with stretches that ended before
it began.
"""
import json
import pickle

import numpy as np
import pandas as pd

from src import config


def dtw_distances(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """DTW distance from one stretch to many, each of shape (days, features).

    query has shape (W, F); candidates has shape (N, W, F). Returns N distances.

    Step 1: cost[n, i, j] = straight-line distance between day i of the query
            and day j of candidate n.
    Step 2: total[n, i, j] = cheapest total cost of matching the first i+1
            query days with the first j+1 candidate days. Each cell extends
            the cheapest of three neighbours: both stretches move on a day
            (diagonal), or only one of them does (up, left).
    The answer is the bottom-right cell, divided by the stretch length so that
    the number reads as "average distance per day".
    """
    cost = np.sqrt(((query[None, :, None, :] - candidates[:, None, :, :]) ** 2).sum(axis=3))
    days = query.shape[0]
    total = np.full_like(cost, np.inf)
    total[:, 0, 0] = cost[:, 0, 0]
    for i in range(days):
        for j in range(days):
            if i == 0 and j == 0:
                continue
            best_previous = np.minimum.reduce([
                total[:, i - 1, j - 1] if i > 0 and j > 0 else np.full(len(cost), np.inf),
                total[:, i - 1, j] if i > 0 else np.full(len(cost), np.inf),
                total[:, i, j - 1] if j > 0 else np.full(len(cost), np.inf)])
            total[:, i, j] = cost[:, i, j] + best_previous
    return total[:, -1, -1] / days


def find_matches(x: np.ndarray, end: int, window: int, count: int) -> list[tuple[int, float]]:
    """The `count` past stretches most like the stretch ending at row `end`.

    x holds the standardised features, one row per day. Returns
    (end row of the match, distance) pairs, closest first. A match must end
    before the query stretch begins, and reported matches do not overlap
    each other (otherwise the same episode would be listed several times).
    """
    start = end - window + 1
    last_candidate_end = start - 1
    if start < 0 or last_candidate_end < window - 1:
        return []
    candidate_ends = np.arange(window - 1, last_candidate_end + 1)
    candidates = np.stack([x[e - window + 1:e + 1] for e in candidate_ends])
    distances = dtw_distances(x[start:end + 1], candidates)

    matches: list[tuple[int, float]] = []
    for position in np.argsort(distances, kind="stable"):
        candidate_end = int(candidate_ends[position])
        if all(abs(candidate_end - chosen) >= window for chosen, _ in matches):
            matches.append((candidate_end, float(distances[position])))
        if len(matches) == count:
            break
    return matches


def load_daily(symbol: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Daily candles with their regime, and the features standardised with the
    Daily model's training-period mean and spread."""
    features = pd.read_parquet(config.FEATURES_DIR / f"{symbol}_1d_features.parquet")
    regimes = pd.read_parquet(config.REGIMES_DIR / f"{symbol}_1d_regimes.parquet",
                              columns=["open_time", "regime"])
    with open(config.MODELS_DIR / f"{symbol}_1d_hmm.pkl", "rb") as file:
        model = pickle.load(file)
    return regimes, model.standardise(features)


def similar_history(regimes: pd.DataFrame, x: np.ndarray, date: pd.Timestamp) -> dict | None:
    """The `similar_history` block of the output record for the stretch ending
    on Daily candle `date`. None if there is not enough history."""
    positions = np.flatnonzero(regimes["open_time"].to_numpy() == np.datetime64(date))
    if len(positions) == 0:
        return None
    end = int(positions[0])
    window = config.SIMILARITY_WINDOW_DAYS
    found = find_matches(x, end, window, config.SIMILARITY_MATCHES)
    if not found:
        return None
    days = regimes["open_time"]
    regime_now = regimes["regime"].iloc[end]
    matches = [{"window_start": str(days.iloc[e - window + 1].date()),
                "window_end": str(days.iloc[e].date()),
                "distance": round(distance, 3),
                "regime_then": regimes["regime"].iloc[e]} for e, distance in found]
    return {
        "window_days": window,
        "window_end": str(pd.Timestamp(date).date()),
        "regime_now": regime_now,
        "matches": matches,
        "matches_in_same_regime": sum(m["regime_then"] == regime_now for m in matches),
    }


def check(symbol: str) -> dict:
    """Is similarity meaningful? For one day in every seven of the test period,
    find the closest past stretch and see whether it ended in the same regime
    as the query day. Compare with chance: how often a RANDOM earlier stretch
    would end in the same regime."""
    regimes, x = load_daily(symbol)
    window = config.SIMILARITY_WINDOW_DAYS
    label = regimes["regime"].to_numpy()
    test_rows = np.flatnonzero(regimes["open_time"] > pd.Timestamp(config.TRAIN_END))[::7]

    same, chance, distances = [], [], []
    for end in test_rows:
        found = find_matches(x, int(end), window, 1)
        if not found:
            continue
        match_end, distance = found[0]
        same.append(label[match_end] == label[end])
        # every stretch that was allowed as a match
        allowed = label[window - 1:end - window + 1]
        chance.append((allowed == label[end]).mean())
        distances.append(distance)
    return {"symbol": symbol, "window_days": window, "query_days": len(same),
            "closest_match_same_regime": float(np.mean(same)),
            "random_past_stretch_same_regime": float(np.mean(chance)),
            "median_distance": float(np.median(distances))}


def main() -> None:
    results = [check(symbol) for symbol in config.SYMBOLS]
    print("Is the closest past stretch in the same regime more often than a random one? "
          "(test-period days, one in seven)\n")
    print(pd.DataFrame(results).to_string(index=False, formatters={
        "closest_match_same_regime": "{:.1%}".format,
        "random_past_stretch_same_regime": "{:.1%}".format,
        "median_distance": "{:.3f}".format}))
    with open(config.RESULTS_DIR / "similarity_check.json", "w") as file:
        json.dump(results, file, indent=2)
    print("\nSaved: results/similarity_check.json")


if __name__ == "__main__":
    main()
