"""Income-distribution generation and inter-year assignment for the ABM.

Pipeline role
-------------
Each simulation year the model needs a cross-sectional income distribution
that matches a target mean and Gini coefficient.  This module provides:

1. **Generation** — ``generate_income_distribution`` draws a lognormal sample
   calibrated to (mean, Gini) via either a closed-form sigma inversion
   (``"theoretical"``) or a deterministic quantile construction (``"exact"``).
2. **Assignment** — ``assign_incomes_deterministic`` maps last-year ranks to the
   new marginal distribution (rank-preserving transport).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

from .stats import compute_gini


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _gini_from_lognormal_sigma(sigma, mean_target, z):
    """Compute the sample Gini for a lognormal draw with given sigma and mean."""
    mu = np.log(mean_target) - 0.5 * sigma**2
    x = np.exp(mu + sigma * z)
    # Rescale so the sample mean equals mean_target exactly.
    x *= mean_target / x.mean()
    return compute_gini(x)


# ---------------------------------------------------------------------------
# Public functions — generation
# ---------------------------------------------------------------------------

def sample_lognormal_with_exact_mean_gini(n, mean_target, gini_target, rng):
    """Sample a lognormal distribution whose sample mean and Gini match the targets exactly.

    Parameters
    ----------
    n : int
        Number of samples.
    mean_target : float
        Desired sample mean.
    gini_target : float
        Desired sample Gini coefficient.
    rng : np.random.Generator
        Random number generator used only to shuffle the deterministic draw.

    Returns
    -------
    np.ndarray
        Array of length ``n`` with sample mean ``mean_target`` and sample
        Gini ``gini_target`` (up to floating-point tolerance).

    Notes
    -----
    Constructs a deterministic grid of standard normal quantiles via the
    mid-point rule, then inverts the Gini-sigma relationship numerically
    (Brent's method) to find the lognormal sigma that reproduces ``gini_target``
    on that fixed grid.  The resulting incomes are shuffled to break the
    rank-preserving order before returning.
    """
    # Deterministic normal quantiles for a fixed grid of size n.
    u = (np.arange(n) + 0.5) / n
    z = norm.ppf(u)
    # Find sigma such that the sample Gini on the fixed grid equals gini_target.
    sigma = brentq(lambda s: _gini_from_lognormal_sigma(s, mean_target, z) - gini_target, 1e-3, 5.0)
    mu = np.log(mean_target) - 0.5 * sigma**2
    x = np.exp(mu + sigma * z)
    # Rescale to enforce exact sample mean.
    x *= mean_target / x.mean()
    rng.shuffle(x)
    return x


def generate_income_distribution(
    n_agents: int,
    mean_income: float,
    target_gini: float,
    method: str = "theoretical",
    rng: np.random.Generator | None = None,
    dtype_income: np.dtype | type = np.float64,
) -> dict:
    """Generate a lognormal income distribution matching target mean and Gini.

    The income of each agent is assumed to follow a lognormal distribution
    X ~ LogNormal(μ, σ²), where log X ~ Normal(μ, σ²).  Given a target mean
    ``mean_income`` and a target Gini ``target_gini``, two calibration
    strategies are available:

    1. ``method="theoretical"``
       Uses the closed-form relationship between the Gini and σ for a
       lognormal:

           G* = 2 · Φ(σ / √2) − 1,

       where Φ is the standard-normal CDF.  Inverts analytically to obtain σ,
       computes μ = ln(m) − σ²/2, draws n samples, and rescales so the
       sample mean equals ``mean_income``.  The sample Gini approximates G*
       with sampling noise.

    2. ``method="exact"``
       Calls ``sample_lognormal_with_exact_mean_gini`` to construct a
       deterministic sample whose sample mean and Gini match the targets up
       to numerical tolerance.

    Parameters
    ----------
    n_agents : int
        Number of agents (sample size).
    mean_income : float
        Target average income.
    target_gini : float
        Target Gini coefficient for the income distribution.
    method : {"theoretical", "exact"}, optional
        Calibration strategy (see above).  Default is ``"theoretical"``.
    rng : np.random.Generator or None, optional
        Random number generator.  If ``None``, a new generator is created
        from system entropy.
    dtype_income : np.dtype or type, optional
        Data type used to store generated incomes.

    Returns
    -------
    dict
        Keys:

        ``"incomes"``
            ``np.ndarray`` of shape ``(n_agents,)`` — generated incomes.
        ``"actual_mean"``
            ``float`` — sample mean of the generated incomes.
        ``"actual_gini"``
            ``float`` — sample Gini of the generated incomes.

    Notes
    -----
    The lognormal is the standard distributional assumption for individual
    incomes within a country.  The closed-form
    Gini–sigma relationship follows directly from the properties of the
    lognormal CDF.
    """
    if rng is None:
        rng = np.random.default_rng()

    if method == "theoretical":
        # Invert G* = 2Φ(σ/√2) − 1 to obtain σ.
        p = np.clip((target_gini + 1.0) / 2.0, 1e-12, 1 - 1e-12)
        sigma = np.sqrt(2.0) * norm.ppf(p)
        mu = np.log(mean_income) - 0.5 * (sigma ** 2)
        incomes = rng.lognormal(mean=mu, sigma=sigma, size=n_agents)
        # Rescale sample to enforce exact mean_income.
        incomes *= mean_income / incomes.mean()
    elif method == "exact":
        incomes = sample_lognormal_with_exact_mean_gini(
            n=n_agents,
            mean_target=mean_income,
            gini_target=target_gini,
            rng=rng,
        )
    else:
        raise ValueError("method must be 'theoretical' or 'exact'.")

    return {
        "incomes": incomes.astype(dtype_income),
        "actual_mean": float(incomes.mean()),
        "actual_gini": float(compute_gini(incomes)),
    }


# ---------------------------------------------------------------------------
# Private helpers — ranking
# ---------------------------------------------------------------------------

def absolute_ranks_1_to_n(y: np.ndarray) -> np.ndarray:
    """Compute stable absolute ranks (1 … n) for a 1-D array.

    Parameters
    ----------
    y : np.ndarray
        1-D array of values to rank (e.g. incomes).

    Returns
    -------
    np.ndarray
        Integer ranks from 1 to ``n``.  Ties are broken by the original
        position in ``y`` (stable sort), which guarantees deterministic
        results for repeated equal values.

    Notes
    -----
    Stable ranking preserves the relative order of equal values, making
    the rank mapping deterministic for agents with identical incomes.  This
    is important for the income-mobility copula, which maps rank positions
    across years.
    """
    y = np.asarray(y)
    # Stable argsort gives deterministic tie-breaking.
    order = np.argsort(y, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, y.size + 1)
    return ranks


# ---------------------------------------------------------------------------
# Public functions — annual income update pipeline
# ---------------------------------------------------------------------------

def assign_incomes_deterministic(
    df_prev: pd.DataFrame,
    year: int,
    data: dict,
    *,
    rng: np.random.Generator,
    region: str | None = None,
    dtype_income: np.dtype | type = np.float64,
) -> pd.DataFrame:
    """Deterministically assign new-year incomes to agents given the target marginal.

    Updates the income marginal distribution (target mean and Gini) and maps
    it to agents by last-year rank (stable for ties).  Does **not** apply
    any additional mobility beyond the deterministic rank mapping.

    Parameters
    ----------
    df_prev : pd.DataFrame
        Previous-year agent data; must contain ``"income"`` and ``"region"``
        columns.
    year : int
        Year for which new incomes are being assigned.
    data : dict
        Nested dict ``data[year][region]`` with keys ``"mean_income"`` and
        ``"gini"``.
    rng : np.random.Generator
        Random generator used to construct the target income draw.
    region : str or None, optional
        Region name.  Defaults to the region found in ``df_prev``.
    dtype_income : np.dtype or type, optional
        Data type for the income column.

    Returns
    -------
    pd.DataFrame
        Copy of ``df_prev`` with deterministically assigned incomes for
        ``year``.

    Notes
    -----
    Deterministic rank mapping (rank-preserving transport) is the zero-mobility
    baseline: agent *i* in rank position *r* at year *t−1* receives the income
    at rank position *r* in the year-*t* marginal distribution.
    """
    n = len(df_prev)
    if region is None:
        region = str(df_prev["region"].iat[0])

    mean2 = data[year][region]["mean_income"]
    gini2 = data[year][region]["gini"]

    # Draw a target sample with exact mean and Gini, then sort for rank assignment.
    dist = generate_income_distribution(
        n_agents=n,
        mean_income=mean2,
        target_gini=gini2,
        method="exact",
        rng=rng,
        dtype_income=dtype_income,
    )
    y2_sorted = np.sort(dist["incomes"])

    # Map sorted year-2 incomes to agents by stable year-1 rank.
    y1 = df_prev["income"].to_numpy(dtype=float, copy=False)
    ranks1 = absolute_ranks_1_to_n(y1)
    order = np.argsort(ranks1, kind="mergesort")
    Y2 = np.empty_like(y2_sorted)
    Y2[order] = y2_sorted

    df_new = df_prev.copy()
    df_new["income"] = np.asarray(Y2, dtype=dtype_income)
    return df_new


