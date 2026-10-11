"""Shared statistics for the H1 and H2 experiments.

One effect size is used everywhere so results can be compared:

    effect = probability that a value from group A is LOWER than a value from
             group B, counting ties as half.   0.5 = no difference.

It is the Mann-Whitney U statistic divided by the number of pairs. It is
always computed inside matching groups ("strata") and then averaged, so the
two sides are compared like with like.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

from src import config

_MAX_CELLS = 2_000_000   # memory cap for one batch of label shuffles


def _probability_lower(events: np.ndarray, controls: np.ndarray) -> float:
    """P(event < control) + 0.5 * P(event == control), from ranks.

    Rank all values together. If events tended to be lower, their ranks are
    small. (sum of event ranks) - n(n+1)/2 counts the pairs where the event is
    the HIGHER one, so one minus that share is the probability it is lower.
    """
    n_e, n_c = len(events), len(controls)
    ranks = rankdata(np.concatenate([events, controls]))
    higher_pairs = ranks[:n_e].sum() - n_e * (n_e + 1) / 2
    return 1.0 - higher_pairs / (n_e * n_c)


def matched_auc(values: np.ndarray, is_event: np.ndarray, strata: np.ndarray) -> tuple[float, int, int]:
    """Effect size across matching groups, weighting each group by its number
    of events. Groups lacking either side are skipped.
    Returns (effect, events used, controls used)."""
    total, n_events, n_controls = 0.0, 0, 0
    for stratum in np.unique(strata):
        inside = strata == stratum
        events, controls = values[inside & is_event], values[inside & ~is_event]
        if len(events) == 0 or len(controls) == 0:
            continue
        total += len(events) * _probability_lower(events, controls)
        n_events += len(events)
        n_controls += len(controls)
    return (total / n_events if n_events else np.nan), n_events, n_controls


def permutation_p_value(values: np.ndarray, is_event: np.ndarray, strata: np.ndarray,
                        rng: np.random.Generator,
                        n_permutations: int = config.H1_PERMUTATIONS) -> float:
    """One-sided p-value: how often shuffled labels give an effect at least as
    large as the real one.

    Labels are shuffled only INSIDE each matching group, so the make-up of the
    two sides never changes. If the measure had nothing to do with the
    outcome, the real labelling would look like a typical shuffle.
    """
    observed, n_events, _ = matched_auc(values, is_event, strata)
    if n_events == 0:
        return np.nan
    shuffled_total = np.zeros(n_permutations)
    for stratum in np.unique(strata):
        inside = strata == stratum
        n_e, n_c = int(is_event[inside].sum()), int((~is_event[inside]).sum())
        if n_e == 0 or n_c == 0:
            continue
        ranks = rankdata(values[inside])
        # Shuffling labels = picking n_e of the ranks at random to be "events".
        # Done in batches so large groups do not exhaust memory.
        batch = max(1, _MAX_CELLS // len(ranks))
        for start in range(0, n_permutations, batch):
            rows = min(batch, n_permutations - start)
            picks = rng.permuted(np.tile(ranks, (rows, 1)), axis=1)[:, :n_e]
            higher_pairs = picks.sum(axis=1) - n_e * (n_e + 1) / 2
            shuffled_total[start:start + rows] += n_e * (1.0 - higher_pairs / (n_e * n_c))
    shuffled = shuffled_total / n_events
    return float((1 + (shuffled >= observed - 1e-12).sum()) / (n_permutations + 1))


def bootstrap_interval(values: np.ndarray, is_event: np.ndarray, strata: np.ndarray,
                       rng: np.random.Generator,
                       n_bootstraps: int = config.H1_BOOTSTRAPS) -> tuple[float, float]:
    """95% interval for the effect size, resampling rows inside each group."""
    groups = [(np.flatnonzero((strata == s) & is_event), np.flatnonzero((strata == s) & ~is_event))
              for s in np.unique(strata)]
    groups = [(e, c) for e, c in groups if len(e) and len(c)]
    if not groups:
        return np.nan, np.nan
    estimates = np.empty(n_bootstraps)
    for b in range(n_bootstraps):
        total, n_events = 0.0, 0
        for events, controls in groups:
            e = values[rng.choice(events, len(events))]
            c = values[rng.choice(controls, len(controls))]
            total += len(e) * _probability_lower(e, c)
            n_events += len(e)
        estimates[b] = total / n_events
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def logistic_coefficients(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Logistic regression fitted by maximum likelihood, written out.

    Model: P(y = 1) = 1 / (1 + exp(-(b0 + b . x))). The minimised function is
    the negative log-likelihood. Returns b0 followed by b.
    """
    design = np.column_stack([np.ones(len(x)), x])

    def negative_log_likelihood(beta: np.ndarray) -> float:
        z = design @ beta
        return float(np.sum(np.logaddexp(0.0, z) - y * z))

    return minimize(negative_log_likelihood, np.zeros(design.shape[1]), method="BFGS").x


def logistic_with_intervals(x: np.ndarray, y: np.ndarray, names: list[str],
                            rng: np.random.Generator, n_bootstraps: int) -> dict:
    """Standardised logistic coefficients with bootstrap 95% intervals.

    Predictors are standardised, so coefficients can be compared with each
    other. Rows are resampled within each outcome class.
    """
    if y.sum() < 10 or (1 - y).sum() < 10:
        return {"note": "too few rows for a joint model"}
    spread = x.std(axis=0)
    if (spread == 0).any():
        return {"note": "a predictor does not vary"}
    x = (x - x.mean(axis=0)) / spread
    fitted = logistic_coefficients(x, y)[1:]
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = np.empty((n_bootstraps, len(names)))
    for b in range(n_bootstraps):
        pick = np.concatenate([rng.choice(positives, len(positives)),
                               rng.choice(negatives, len(negatives))])
        draws[b] = logistic_coefficients(x[pick], y[pick])[1:]
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    return {name: {"coefficient": float(fitted[k]), "ci_low": float(low[k]), "ci_high": float(high[k])}
            for k, name in enumerate(names)}
