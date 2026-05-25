"""
alpha_init.py
-------------
Alpha initialization strategies for sensitivity analysis.

Each strategy returns a float64 array of shape (N,) with values in (0, 1).
All strategies accept the same signature to allow drop-in substitution.

Strategies
----------
uniform         : U(0, 1)  — current default
constant        : alpha_i = c for all i
beta_symmetric  : Beta(a, a), symmetric around 0.5
beta_skewed     : Beta(a, b), a != b, asymmetric
income_monotone : alpha_i = f(p_i), monotone in income percentile
bimodal         : mixture of two Beta distributions
issp_proxy      : synthetic proxy of ISSP-like distribution (survey-calibrated)
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def initialize_alpha(
    strategy: str,
    percentiles: np.ndarray,
    rng: Generator,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Initialize alpha_i values for all agents.

    Parameters
    ----------
    strategy : str
        One of the strategy keys listed in STRATEGIES.
    percentiles : np.ndarray, shape (N,)
        Income percentiles p_i in (0, 1), computed from sorted incomes.
        Required by income-dependent strategies; ignored by others.
    rng : numpy.random.Generator
        Seeded RNG passed from the caller (preserves reproducibility).
    params : dict, optional
        Strategy-specific keyword arguments (see each strategy below).

    Returns
    -------
    np.ndarray, dtype float64, shape (N,)
        Alpha values clipped to (eps, 1 - eps) to avoid boundary degeneracy.
    """
    if params is None:
        params = {}

    fn = _STRATEGIES.get(strategy)
    if fn is None:
        raise ValueError(
            f"Unknown alpha initialization strategy '{strategy}'. "
            f"Valid options: {sorted(_STRATEGIES)}"
        )

    alpha = fn(percentiles=percentiles, rng=rng, **params)
    return _clip(alpha)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _uniform(percentiles: np.ndarray, rng: Generator, **_) -> np.ndarray:
    """
    U(0, 1): current model default.
    No income gradient. Mean = 0.5, Var = 1/12.
    """
    return rng.uniform(0.0, 1.0, len(percentiles))


def _constant(
    percentiles: np.ndarray,
    rng: Generator,
    value: float = 0.5,
    **_,
) -> np.ndarray:
    """
    alpha_i = value for all i.
    Equivalent to burn-in starting point when value = 0.5.
    params: value (float, default 0.5)
    """
    return np.full(len(percentiles), float(value))


def _beta_symmetric(
    percentiles: np.ndarray,
    rng: Generator,
    concentration: float = 2.0,
    **_,
) -> np.ndarray:
    """
    Beta(a, a): symmetric around 0.5, unimodal if a > 1, U-shaped if a < 1.
    Increasing concentration sharpens the peak around 0.5.
    params: concentration (float, default 2.0)
    """
    a = float(concentration)
    return rng.beta(a, a, len(percentiles))


def _beta_skewed(
    percentiles: np.ndarray,
    rng: Generator,
    a: float = 3.0,
    b: float = 1.5,
    **_,
) -> np.ndarray:
    """
    Beta(a, b): asymmetric. Default (3, 1.5) skews toward high support,
    consistent with Italian ISSP 2019 data (strong pro-redistribution majority).
    params: a (float), b (float)
    """
    return rng.beta(float(a), float(b), len(percentiles))


def _income_monotone(
    percentiles: np.ndarray,
    rng: Generator,
    slope: float = -0.4,
    intercept: float = 0.7,
    noise_sd: float = 0.05,
    **_,
) -> np.ndarray:
    """
    alpha_i = intercept + slope * p_i + epsilon_i.
    Default: negative slope — lower income -> higher redistributive support.
    
    params:
        slope     (float, default -0.4)
        intercept (float, default  0.7)
        noise_sd  (float, default  0.05): std of additive Gaussian noise
    """
    alpha = intercept + slope * percentiles
    if noise_sd > 0:
        alpha = alpha + rng.normal(0.0, noise_sd, len(percentiles))
    return alpha


def _bimodal(
    percentiles: np.ndarray,
    rng: Generator,
    low_mean: float = 0.25,
    high_mean: float = 0.75,
    concentration: float = 8.0,
    mix_weight: float = 0.5,
    **_,
) -> np.ndarray:
    """
    Mixture of two Beta distributions symmetric around low_mean and high_mean.
    Represents a polarized electorate.
    params:
        low_mean      (float, default 0.25)
        high_mean     (float, default 0.75)
        concentration (float, default 8.0): controls peak sharpness: higher = more mass near means, lower = more spread
        mix_weight    (float, default 0.5): fraction assigned to low_mean component
    """
    N = len(percentiles)
    n_low = int(round(mix_weight * N))
    n_high = N - n_low

    a_low, b_low = _beta_params_from_mean(low_mean, concentration)
    a_high, b_high = _beta_params_from_mean(high_mean, concentration)

    low_draw = rng.beta(a_low, b_low, n_low)
    high_draw = rng.beta(a_high, b_high, n_high)

    alpha = np.concatenate([low_draw, high_draw])
    rng.shuffle(alpha)
    return alpha


def _issp_proxy(
    percentiles: np.ndarray,
    rng: Generator,
    quintile_means: tuple[float, ...] = (0.80, 0.73, 0.68, 0.60, 0.48),
    within_sd: float = 0.12,
    **_,
) -> np.ndarray:
    """
    Synthetic proxy of ISSP 2019 Italy DM_4b distribution.
    Assigns quintile-specific mean support, with Gaussian noise within quintiles.

    Default quintile_means are illustrative (Q1 poorest -> Q5 richest).
    TODO: They should be replaced with empirical means from the actual ISSP dataset
    once available.

    params:
        quintile_means : tuple of 5 floats (Q1..Q5 means), default as above
        within_sd      : float, within-quintile std (default 0.12)
    """
    N = len(percentiles)
    alpha = np.empty(N, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, 6)  # quintile boundaries

    for q, mean_q in enumerate(quintile_means):
        mask = (percentiles >= edges[q]) & (percentiles < edges[q + 1])
        if q == 4:  # include right edge for top quintile
            mask = percentiles >= edges[q]
        n_q = mask.sum()
        if n_q == 0:
            continue
        a, b = _beta_params_from_mean(mean_q, _concentration_from_mean_sd(mean_q, within_sd))
        alpha[mask] = rng.beta(a, b, n_q)

    return alpha


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_STRATEGIES: dict[str, Any] = {
    "uniform": _uniform,
    "constant": _constant,
    "beta_symmetric": _beta_symmetric,
    "beta_skewed": _beta_skewed,
    "income_monotone": _income_monotone,
    "bimodal": _bimodal,
    "issp_proxy": _issp_proxy, #TODO: replace with actual ISSP quintile means once available
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPS = np.finfo(np.float64).eps * 4


def _clip(alpha: np.ndarray) -> np.ndarray:
    return np.clip(alpha, _EPS, 1.0 - _EPS).astype(np.float64)


def _beta_params_from_mean(mean: float, concentration: float) -> tuple[float, float]:
    """Convert (mean, concentration) to Beta(a, b) parameters."""
    mean = float(np.clip(mean, 1e-6, 1.0 - 1e-6))
    c = float(concentration)
    return mean * c, (1.0 - mean) * c


def _concentration_from_mean_sd(mean: float, sd: float) -> float:
    """Estimate Beta concentration from mean and std via method of moments."""
    mean = float(np.clip(mean, 1e-6, 1.0 - 1e-6))
    var = min(float(sd) ** 2, mean * (1.0 - mean) * 0.99)
    return mean * (1.0 - mean) / var - 1.0