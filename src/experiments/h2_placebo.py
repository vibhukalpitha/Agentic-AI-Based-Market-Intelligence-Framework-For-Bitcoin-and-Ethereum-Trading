"""Post-hoc control for H2 (a placebo test).

Run:  python -m src.experiments.h2_placebo

STATUS: this control was NOT part of the pre-registered H2 design. It was
added on 10 Oct 2026, AFTER the H2 test-period result was known, to check
whether that result could be an artefact. It must be reported as post-hoc.

The question. H2 found that flips "supported from below" are more durable.
But a market can spend months in an era where one regime is common at every
timeframe. In such an era a flip into that regime is more likely to last AND
more likely to look supported, without any change having climbed up.

The control. Measure "support" from the faster timeframes as they stood
PLACEBO_LAG_DAYS before each flip. A regime seen 30 days earlier cannot be a
change that climbed up into this flip, so any effect left is the era effect.

Reading. Only the part of the real effect that EXCEEDS the placebo can be
credited to propagation. The script reports real minus placebo with a 95%
interval (both computed on the same resampled flips, so they are compared
like with like).
"""
import json

import numpy as np
import pandas as pd

from src import config
from src.experiments import h2_propagation as h2
from src.experiments.stats import matched_auc
from src.fusion import propagation

PLACEBO_LAG_DAYS = 30
PERIOD = "test"


def placebo_depth(events: pd.DataFrame, labels: dict[str, pd.DataFrame], interval: str) -> np.ndarray:
    """Support depth computed from faster-timeframe labels PLACEBO_LAG_DAYS earlier."""
    faster = config.INTERVALS[config.INTERVALS.index(interval) + 1:]
    earlier = (events["open_time"] - pd.Timedelta(days=PLACEBO_LAG_DAYS)).to_numpy()
    depth, _ = propagation.support_depth(
        events["to_regime"].to_numpy(),
        [propagation.labels_as_of(labels[i], earlier) for i in faster])
    return depth


def compare(events: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Real effect, placebo effect, and their difference with a paired 95% interval."""
    durable = events["durable"].to_numpy()
    strata = events["to_regime"].to_numpy()
    real = -events["depth"].to_numpy(dtype=float)        # negated: "deeper" becomes "lower"
    fake = -events["placebo_depth"].to_numpy(dtype=float)

    groups = [(np.flatnonzero((strata == s) & durable), np.flatnonzero((strata == s) & ~durable))
              for s in np.unique(strata)]
    groups = [(d, s) for d, s in groups if len(d) and len(s)]
    draws = np.empty((config.H1_BOOTSTRAPS, 2))
    for b in range(config.H1_BOOTSTRAPS):
        pick = np.concatenate([np.concatenate([rng.choice(d, len(d)), rng.choice(s, len(s))])
                               for d, s in groups])
        draws[b, 0], _, _ = matched_auc(real[pick], durable[pick], strata[pick])
        draws[b, 1], _, _ = matched_auc(fake[pick], durable[pick], strata[pick])
    difference = draws[:, 0] - draws[:, 1]

    real_effect, _, _ = matched_auc(real, durable, strata)
    fake_effect, _, _ = matched_auc(fake, durable, strata)
    return {
        "flips": int(len(events)),
        "real_effect": float(real_effect),
        "placebo_effect": float(fake_effect),
        "placebo_ci_low": float(np.percentile(draws[:, 1], 2.5)),
        "placebo_ci_high": float(np.percentile(draws[:, 1], 97.5)),
        "real_minus_placebo": float(real_effect - fake_effect),
        "difference_ci_low": float(np.percentile(difference, 2.5)),
        "difference_ci_high": float(np.percentile(difference, 97.5)),
    }


def main() -> None:
    rng = np.random.default_rng(config.RANDOM_SEED)
    results = {"status": "post-hoc control, added after the H2 test result was known",
               "placebo_lag_days": PLACEBO_LAG_DAYS, "period": PERIOD}
    rows = []
    for symbol in config.SYMBOLS:
        labels = {i: propagation.load_labels(symbol, i) for i in config.INTERVALS}
        results[symbol] = {}
        for interval in [config.H2_HEADLINE_INTERVAL] + config.H2_SECONDARY_INTERVALS:
            events = h2.load_events(symbol, interval, PERIOD, config.H2_DURABLE_CANDLES)
            events["placebo_depth"] = placebo_depth(events, labels, interval)
            results[symbol][interval] = compare(events, rng)
            rows.append({"symbol": symbol, "interval": interval, **results[symbol][interval]})

    table = pd.DataFrame(rows)
    table["exceeds_placebo"] = table["difference_ci_low"] > 0
    print("H2 placebo control (post-hoc). An effect of 0.5 means no difference.")
    print("Only where the difference interval lies above 0 does the real effect exceed the era effect.\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    with open(config.RESULTS_DIR / "h2_placebo_test.json", "w") as file:
        json.dump(results, file, indent=2)
    print("\nSaved: h2_placebo_test.json")


if __name__ == "__main__":
    main()
