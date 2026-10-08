"""M3 - Hidden Markov Model wrapper: fitting and leak-free regime probabilities.

A Hidden Markov Model assumes the market is always in one of a few hidden
states. Each state produces feature values from its own Gaussian distribution,
and the market moves between states according to a transition matrix. Training
(Baum-Welch) finds the Gaussians and the transition matrix that best explain
the data, without ever being told what the states mean.

The most important function here is filtered_probabilities. hmmlearn's own
predict() and predict_proba() read the WHOLE series, so their answer for day t
depends on days after t. Using them would make every "early warning" result in
this project invalid. filtered_probabilities uses days 0..t only.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import multivariate_normal

from src import config


@dataclass
class RegimeModel:
    """Everything needed to reproduce the regime labels of one asset + timeframe."""
    symbol: str
    interval: str
    hmm: GaussianHMM
    feature_mean: np.ndarray      # training-period mean of each feature
    feature_std: np.ndarray       # training-period standard deviation
    train_end: str
    seed: int                     # the random start that gave the best fit
    state_to_regime: dict[int, str] = field(default_factory=dict)

    def standardise(self, features: pd.DataFrame) -> np.ndarray:
        """Scale with the TRAINING mean and std, also for later data.

        Using statistics of the whole history would leak the future into the
        past. WHY scale at all: rsi and adx run 0-100 while log_return is
        about 0.03; unscaled, the large-number features would dominate.
        """
        values = features[config.FEATURE_COLUMNS].to_numpy(dtype=float)
        return (values - self.feature_mean) / self.feature_std


def split_train(features: pd.DataFrame, train_end: str) -> pd.Series:
    """Boolean mask: True for candles that had fully closed by the end of train_end."""
    return features["close_time"] < pd.Timestamp(train_end) + pd.Timedelta(days=1)


def fit_hmm(x_train: np.ndarray, n_states: int, covariance_type: str) -> tuple[GaussianHMM, int]:
    """Fit from HMM_N_RESTARTS random starts; return the best model and its seed.

    Baum-Welch climbs to the nearest local optimum, so one run can land on a
    poor solution. Several fixed seeds make the result both better and repeatable.
    """
    best_model, best_seed, best_score = None, -1, -np.inf
    for seed in range(config.RANDOM_SEED, config.RANDOM_SEED + config.HMM_N_RESTARTS):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=config.HMM_MAX_ITER,
            tol=config.HMM_TOLERANCE,
            random_state=seed,
        )
        try:
            model.fit(x_train)
            score = model.score(x_train)
        except ValueError:
            continue                      # a degenerate start; try the next seed
        if np.isfinite(score) and score > best_score:
            best_model, best_seed, best_score = model, seed, score
    if best_model is None:
        raise RuntimeError("no HMM fit succeeded")
    return best_model, best_seed


def convergence(model: GaussianHMM) -> tuple[bool, float]:
    """(reached the tolerance before the iteration limit?, last log-likelihood gain).

    hmmlearn's own `converged` flag is also True when training simply ran out of
    iterations, so it cannot be trusted on its own. The last gain shows how
    close a run that hit the limit actually was.
    """
    monitor = model.monitor_
    history = list(monitor.history)
    last_gain = history[-1] - history[-2] if len(history) > 1 else float("nan")
    return bool(monitor.iter < monitor.n_iter and abs(last_gain) < monitor.tol), float(last_gain)


def emission_log_density(model: GaussianHMM, x: np.ndarray) -> np.ndarray:
    """log p(features on day t | state k), shape (days, states).

    Each state is a multivariate Gaussian; this is just its density formula.
    """
    return np.column_stack([
        multivariate_normal.logpdf(x, mean=model.means_[k], cov=model.covars_[k],
                                   allow_singular=True)
        for k in range(model.n_components)
    ])


def filtered_probabilities(model: GaussianHMM, x: np.ndarray) -> np.ndarray:
    """P(state on day t | data from day 0 to day t), shape (days, states).

    This is the forward algorithm, written out:
      1. prior    = yesterday's probabilities pushed through the transition matrix
                    (on the first day: the model's start probabilities)
      2. evidence = how well each state explains today's features
      3. today    = prior * evidence, rescaled to sum to 1
    Nothing from day t+1 or later is ever touched.
    """
    log_density = emission_log_density(model, x)
    probabilities = np.empty_like(log_density)
    prior = model.startprob_
    for t in range(len(x)):
        # Subtracting the row maximum avoids exp() underflow; the rescaling
        # in the next lines makes the constant cancel out.
        evidence = np.exp(log_density[t] - log_density[t].max())
        posterior = prior * evidence
        posterior /= posterior.sum()
        probabilities[t] = posterior
        prior = posterior @ model.transmat_
    return probabilities


def confirmed_states(states: np.ndarray, persistence: int) -> np.ndarray:
    """Remove flicker: a new state is accepted only after it has held for
    `persistence` consecutive candles. Until then the previous state is kept.

    The change is recorded on the candle where it becomes confirmed, not
    back-dated to where it began, so this too uses no future information.
    persistence = 1 returns the input unchanged.
    """
    confirmed = np.empty_like(states)
    current = states[0]
    run_length = 0
    for t, state in enumerate(states):
        run_length = run_length + 1 if t > 0 and state == states[t - 1] else 1
        if state != current and run_length >= persistence:
            current = state
        confirmed[t] = current
    return confirmed


def count_transitions(states: np.ndarray) -> int:
    """Number of candles whose state differs from the previous candle's."""
    return int((states[1:] != states[:-1]).sum())
