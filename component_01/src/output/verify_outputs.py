"""Verify the output record against the underlying data at many past moments.

Run:  python -m src.output.verify_outputs            (300 random moments per asset)

For each moment the record from src/output/snapshot.py is rebuilt and every
block is checked against the saved data it should come from, or recomputed
independently:

  timeframes        regime and confidence equal the saved regime file at the
                    last closed candle; probabilities sum to 1; the regime
                    is the most probable one; the candle closed before `as_of`
  fusion            Alignment and Confidence Scores recomputed from the
                    formula; the agreeing list matches the regimes
  propagation       each flip leads into the current regime; the depth is
                    within range and matches the stated direction
  stability         equals the saved stability score of the last closed day
  explanation       the archetype matches the timeframes it describes; the
                    base rate equals the training-period figure
  similar_history   every match ended before the current stretch began, and
                    its stated regime is the saved regime on that day
  no future data    for some moments, the record is rebuilt with every
                    candle after `as_of` deleted, and must not change

It only reads. A failure prints what disagreed.
"""
import json

import numpy as np
import pandas as pd

from src import config
from src.explain import archetypes, explainer
from src.output import snapshot
from src.similarity import dtw_match

MOMENTS_PER_ASSET = 300
FUTURE_CHECKS_PER_ASSET = 25


def check_record(record: dict, as_of: pd.Timestamp, inputs: dict) -> list[str]:
    """Every problem found in one record (empty list = correct)."""
    problems = []
    labels, stability_table, (archetype_table, rates) = (
        inputs["labels"], inputs["stability"], inputs["archetypes"])
    regimes_daily, _ = inputs["similarity"]

    # ---- timeframes ----
    for interval, state in record["timeframes"].items():
        frame = labels[interval]
        position = int(np.searchsorted(frame["known_time"].to_numpy(),
                                       np.datetime64(as_of), side="right")) - 1
        row = frame.iloc[position]
        if state["regime"] != row["regime"]:
            problems.append(f"{interval}: regime {state['regime']} != saved {row['regime']}")
        if abs(state["confidence"] - row["confidence"]) > 1e-4:
            problems.append(f"{interval}: confidence differs from the saved file")
        probabilities = state["probabilities"]
        if abs(sum(probabilities.values()) - 1) > 1e-3:
            problems.append(f"{interval}: probabilities do not sum to 1")
        if max(probabilities, key=probabilities.get) != state["regime"]:
            problems.append(f"{interval}: regime is not the most probable one")
        if pd.Timestamp(state["last_closed_candle"].rstrip("Z")) > as_of:
            problems.append(f"{interval}: uses a candle that had not closed")

    # ---- fusion: recompute the formula ----
    fusion = record["fusion"]
    votes = {name: 0.0 for name in config.REGIMES}
    total = 0.0
    for interval, state in record["timeframes"].items():
        vote = config.TIMEFRAME_WEIGHTS[interval] * state["confidence"]
        votes[state["regime"]] += vote
        total += vote
    dominant = max(votes, key=votes.get)
    agreeing = [i for i, s in record["timeframes"].items() if s["regime"] == dominant]
    alignment = 100 * votes[dominant] / total
    mean_conf = np.mean([record["timeframes"][i]["confidence"] for i in agreeing])
    if fusion["dominant_regime"] != dominant:
        problems.append("fusion: dominant regime does not match the votes")
    if abs(fusion["alignment_score"] - alignment) > 0.02:
        problems.append(f"fusion: alignment {fusion['alignment_score']} != {alignment:.2f}")
    if abs(fusion["confidence_score"] - alignment * mean_conf) > 0.05:
        problems.append("fusion: confidence score does not match the formula")
    if fusion["timeframes_agreeing"] != agreeing:
        problems.append("fusion: agreeing list is wrong")

    # ---- propagation ----
    for interval, flip in record["propagation"].items():
        if flip is None:
            continue
        if flip["to_regime"] != record["timeframes"][interval]["regime"]:
            problems.append(f"propagation {interval}: flip does not lead into the current regime")
        if flip["from_regime"] == flip["to_regime"]:
            problems.append(f"propagation {interval}: from and to are the same")
        if not 0 <= flip["support_depth"] <= flip["faster_timeframes"]:
            problems.append(f"propagation {interval}: depth out of range")
        if flip["faster_timeframes"] and (flip["direction"] == "bottom-up") != (flip["support_depth"] >= 1):
            problems.append(f"propagation {interval}: direction does not match depth")
        if pd.Timestamp(flip["flip_candle_open"].rstrip("Z")) > as_of:
            problems.append(f"propagation {interval}: flip lies in the future")

    # ---- stability ----
    block = record["stability"]
    daily_position = int(np.searchsorted(labels["1d"]["known_time"].to_numpy(),
                                         np.datetime64(as_of), side="right")) - 1
    daily_date = labels["1d"]["open_time"].iloc[daily_position]
    if block is not None:
        saved = stability_table.loc[daily_date]
        if abs(block["probability_regime_persists"] - saved["delivered"]) > 1e-4:
            problems.append("stability: score differs from the saved score")
        if not 0 <= block["probability_regime_persists"] <= 1:
            problems.append("stability: not a probability")
        if block["regime"] != record["timeframes"]["1d"]["regime"]:
            problems.append("stability: refers to a different regime than Daily")
    elif daily_date in stability_table.index:
        problems.append("stability: missing although a saved score exists")

    # ---- explanation ----
    block = record["explanation"]
    if block is not None:
        faster = block["faster_timeframes_at_close"]
        name, n_agree, _ = archetypes.classify(
            np.array([record["timeframes"]["1d"]["regime"]]),
            np.array([[faster[i] for i in config.INTERVALS[1:]]]))
        if name[0] != block["archetype"] or n_agree[0] != block["faster_timeframes_agreeing"]:
            problems.append("explanation: archetype does not match the timeframes described")
        if abs(block["base_rate_regime_change"] - rates["archetypes"][block["archetype"]]["change_rate"]) > 1e-4:
            problems.append("explanation: base rate differs from the training-period figure")
        if archetype_table.loc[daily_date, "archetype"] != block["archetype"]:
            problems.append("explanation: archetype differs from the saved archetype of that day")
        if f"{block['base_rate_regime_change']:.0%}" not in block["text"]:
            problems.append("explanation: text does not quote the base rate")
        for word in ("price", "buy", "sell", "will rise", "will fall", "next regime"):
            if word in block["text"].lower():
                problems.append(f"explanation: text contains '{word}'")

    # ---- similar history ----
    block = record["similar_history"]
    if block is not None:
        regime_by_day = regimes_daily.set_index("open_time")["regime"]
        window_start = pd.Timestamp(block["window_end"]) - pd.Timedelta(days=block["window_days"] - 1)
        for match in block["matches"]:
            if pd.Timestamp(match["window_end"]) >= window_start:
                problems.append("similar_history: a match overlaps or follows the current stretch")
            if regime_by_day.loc[pd.Timestamp(match["window_end"])] != match["regime_then"]:
                problems.append("similar_history: stated regime differs from the saved regime")
    return problems


def main() -> None:
    rng = np.random.default_rng(config.RANDOM_SEED)
    summary = {}
    for symbol in config.SYMBOLS:
        inputs = {"labels": snapshot.load_all_labels(symbol),
                  "profiles": None,
                  "stability": snapshot.load_stability(symbol),
                  "archetypes": explainer.load_archetypes(symbol),
                  "similarity": dtw_match.load_daily(symbol)}
        inputs["profiles"] = {i: snapshot.describe.load_profiles(symbol, i) for i in config.INTERVALS}
        start = max(frame["known_time"].iloc[0] for frame in inputs["labels"].values()) + pd.Timedelta(days=30)
        end = inputs["labels"]["5m"]["known_time"].iloc[-1]
        moments = pd.to_datetime(rng.integers(start.value, end.value, MOMENTS_PER_ASSET))

        failures, blocks_present = [], {"stability": 0, "explanation": 0, "similar_history": 0}
        for k, as_of in enumerate(moments):
            record = snapshot.build_snapshot(symbol, as_of, inputs["labels"], inputs["profiles"],
                                             inputs["stability"], inputs["archetypes"],
                                             inputs["similarity"])
            for name in blocks_present:
                blocks_present[name] += record[name] is not None
            problems = check_record(record, as_of, inputs)
            if k < FUTURE_CHECKS_PER_ASSET:
                past = {i: f[f["known_time"] <= as_of].reset_index(drop=True)
                        for i, f in inputs["labels"].items()}
                if snapshot.build_snapshot(symbol, as_of, past, inputs["profiles"], inputs["stability"],
                                           inputs["archetypes"], inputs["similarity"]) != record:
                    problems.append("record changes when the future is deleted")
            if problems:
                failures.append({"as_of": str(as_of), "problems": problems})

        summary[symbol] = {"moments": MOMENTS_PER_ASSET, "future_checks": FUTURE_CHECKS_PER_ASSET,
                           "records_with_problems": len(failures), "blocks_present": blocks_present,
                           "first_failures": failures[:5]}
        print(f"{symbol}: {MOMENTS_PER_ASSET} moments checked "
              f"({moments.min():%Y-%m-%d} -> {moments.max():%Y-%m-%d}); "
              f"records with problems: {len(failures)}; blocks present: {blocks_present}")
        for failure in failures[:5]:
            print("   ", failure["as_of"], failure["problems"])

    with open(config.RESULTS_DIR / "output_verification.json", "w") as file:
        json.dump(summary, file, indent=2)
    print("Saved: results/output_verification.json")


if __name__ == "__main__":
    main()
