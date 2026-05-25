"""Unified private signal with exogenous climate events and geometric
income-elastic memory of damage.

Pipeline overview
-----------------
The climate channel runs in three stages, mirroring the old (v.2025)
three-function decomposition pattern:

  1. PRE-LOOP : `generate_climate_events` produces a binary series
                {E(t)}_t in {0, 1}, shared across all simulations and
                scenarios.  E(t) = 1 marks an exogenous extreme-event
                year.

  2. IN-LOOP  : `select_hit_agents` picks (when E(t)=1) a subset
                S(t) of size floor(N * f) from the population, with
                income-elastic selection probabilities
                pi_i propto (y_bar / y_i)^gamma.  Agents in S(t) get
                H_i(t) = 1; all others get H_i(t) = 0.

  3. IN-LOOP  : `update_climate_memory` propagates a per-agent log-damage
                stock M_i(t) with geometric decay rate delta_i(t),
                income-elastic so that poorer agents have longer memory:
                  delta_i(t) = delta_0 ^ ((y_i(t) / y_bar(t))^epsilon)
                  M_i(t)     = delta_i(t) * M_i(t-1)
                               + H_i(t) * log(1 / (1 - D_i(t)))
                with D_i(t) the income-elastic damage from
                Gilli et al. (2024).

The perceived income used in the private signal is the multiplicative
correction
  y_i^own(t) = exp(-M_i(t)) * y_i(t),
and the signal compares its log-growth against an aspiration benchmark.

Parsimony — single income-elasticity epsilon
--------------------------------------------
By design choice the same epsilon parameter governs three concurrent
income-elastic effects on the climate channel:
  (a) vulnerability:   D_i = D_base * (y_bar/y_i)^epsilon_vulnerability
                       (Gilli et al. 2024 calibration)
  (b) persistence:     delta_i = delta_0^((y_i/y_bar)^epsilon_persistence)
  (c) exposure:        pi_i propto (y_bar/y_i)^epsilon_exposure
                       (calibrated to Hallegatte et al. 2017 ratio)
This is a parsimony choice: epsilon = 0.36 (Gilli et al. 2024) is the
only empirical estimate available; using it for all three avoids
introducing free parameters not anchored in evidence.

Benchmark treatment (unchanged from previous design)
----------------------------------------------------
- "gdp", "mean_income", "median_income": macro aggregates, raw incomes,
  no climate correction.
- "local_mean", "local_median": numerator depends on the
  `local_climate_visibility` parameter:
    * "current_event" (option iii, default): numerator uses
      (1 - H_j(t) D_j(t)) * y_j(t).  Agent i sees neighbours' present
      shock but not the cumulative memory of past adapted shocks of
      others.
    * "none" (option iv): numerator uses raw y_j(t); the climate
      channel operates exclusively through own-side perceived income.
      Maximum separability and parsimony.
  Denominator is always raw past income y_j(t-1).

References
----------
- Aggregate damage:        Kalkuhl & Wenz (2020) 
- Income-elastic damage:   Gilli et al. (2024) 
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Lognormal utility (unchanged)
# ---------------------------------------------------------------------------

def sigma_log_from_gini(gini: float) -> float:
    """Return sigma_log of a lognormal distribution from its Gini coefficient.

    Closed-form inversion of the lognormal Gini formula:
        G = 2 * Phi(sigma / sqrt(2)) - 1
        =>  sigma = sqrt(2) * Phi^{-1}((G + 1) / 2)

    Parameters
    ----------
    gini : float
        Gini coefficient in (0, 1).

    Returns
    -------
    float
        Standard deviation of the underlying log-normal.
    """
    return float(np.sqrt(2) * norm.ppf((gini + 1) / 2))


# ---------------------------------------------------------------------------
# Aggregate and individual damage (unchanged from previous version)
# ---------------------------------------------------------------------------

def compute_damage_base(
    T: float,
    T0: float,
    *,
    a: float = 0.035,
    b: float = 0.0009,
) -> float:
    """Aggregate Kalkuhl & Wenz (2020) quadratic damage.

        D_base(t) = a * tau + b * tau^2,   tau = max(T(t) - T0, 0).

    Parameters
    ----------
    T : float
        Temperature anomaly at time t (degrees C above pre-industrial).
    T0 : float
        Reference baseline temperature anomaly.
    a, b : float
        KW2020 coefficients.

    Returns
    -------
    float
        D_base in [0, ~0.15] for tau in [0, ~3 C].
    """
    tau = max(T - T0, 0.0)
    return a * tau + b * tau ** 2


def compute_individual_damage(
    D_base: float,
    income: np.ndarray,
    *,
    epsilon: float = 0.36,
    D_max: float = 0.5,    # safety clipping for individual damage fraction: D_i(t) = min(D_base(t) * (y_i / y_mean)^(-elasticity), D_max)
) -> np.ndarray:
    """Income-elastic individual damage D_i(t) (Gilli et al. 2024).

        D_i(t) = min( D_base(t) * (y_bar / y_i)^epsilon,  D_max )

    Poor agents (y_i < y_bar) suffer amplified damage; rich agents
    suffer attenuated damage.

    Parameters
    ----------
    D_base : float
        Aggregate damage at time t.
    income : np.ndarray, shape (N,)
        Raw agent incomes at time t.
    epsilon : float
        Income elasticity of vulnerability (default 0.36, Gilli et al.).
    D_max : float
        Safety clipping (default 0.5).

    Returns
    -------
    np.ndarray, shape (N,), dtype float64
        D_i in [0, D_max].
    """
    y_bar = float(np.mean(income))
    # Floor to avoid division blow-up for near-zero incomes.
    r_i = np.maximum(income / y_bar, 1e-10)
    D_i = D_base * (r_i ** (-epsilon))
    return np.minimum(D_i, D_max)


# ---------------------------------------------------------------------------
# 1. Exogenous event series
# ---------------------------------------------------------------------------

def generate_climate_events(
    years: list[int],
    temperature: dict[int, float],
    T0: float,
    rate: float,
    rng: np.random.Generator,
) -> dict[int, int]:
    """Build the exogenous climate-event series E(t).

    Event probability follows a Poisson-saliency mapping on the temperature
    anomaly directly:

        tau(t)    = max(T(t) - T0, 0)
        p(t)      = 1 - exp(-rate * tau(t))
        E(t)     ~ Bern(p(t))

    Generated once outside the simulation loop with a fixed RNG seed,
    ensuring reproducibility and scenario invariance (temperature is
    scenario-invariant in MAPS model).

    Parameters
    ----------
    years : list[int]
        Full simulation year range.
    temperature : dict[int, float]
        Exogenous temperature anomaly series {year: T(t)} in °C.
    T0 : float
        Reference baseline temperature anomaly in °C.
    rate : float
        Poisson-saliency rate c > 0.  Controls how fast p(t) saturates
        with warming.  At tau=1°C: p ≈ 1 - exp(-c).
        for c = 0.4, p ≈ 0.33 (one extreme event every 3 years)
        for c = 0.5, p ≈ 0.39 (one extreme event every 2.5 years)
        for c = 1.0, p ≈ 0.63 (one extreme event every 1.6 years) 
        for c = 2.0, p ≈ 0.86 (one extreme event every 1.15 years)   
    rng : np.random.Generator
        Random generator seeded before the simulation loop.

    Returns
    -------
    events : dict[int, int]
        Mapping year -> E(t) in {0, 1}.
    """
    events: dict[int, int] = {}
    for y in years:
        tau = max(float(temperature[y]) - T0, 0.0)
        p_event = 1.0 - np.exp(-rate * tau)
        events[y] = int(rng.random() < p_event)
    return events


# ---------------------------------------------------------------------------
# 2. Hit allocation — income-elastic selection
# ---------------------------------------------------------------------------

def select_hit_agents(
    income: np.ndarray,
    *,
    fraction: float,
    gamma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select which agents are hit by the current-year climate event.

    Given an active event (caller's responsibility: only call when
    E(t) = 1), draws floor(N * fraction) agents WITHOUT REPLACEMENT
    from the population, with selection weights

        w_i = (y_bar / y_i)^gamma  =  (y_i / y_bar)^(-gamma).

    The probability of agent i being chosen is pi_i = w_i / sum(w_j).

    Limit cases
    -----------
      gamma = 0       : uniform selection (no income elasticity).
      gamma > 0       : poor agents over-represented among the hit.

    Parameters
    ----------
    income : np.ndarray, shape (N,)
        Raw agent incomes at time t.
    fraction : float in (0, 1]
        Target population share f to be hit.  The actual number of hits
        is `int(floor(N * fraction))`, clipped to the range [1, N] when
        fraction > 0.
    gamma : float >= 0
        Income elasticity of exposure.  See module docstring for the
        parsimony choice gamma = epsilon.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, shape (N,), dtype int8
        H_i(t) in {0, 1}: 1 iff agent i is among the selected.

    Raises
    ------
    ValueError
        If `fraction` is outside [0, 1] or `gamma` is negative.
    """
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must lie in [0, 1]; got {fraction}.")
    if gamma < 0.0:
        raise ValueError(f"gamma must be non-negative; got {gamma}.")

    N = len(income)
    if fraction == 0.0 or N == 0:
        return np.zeros(N, dtype=np.int8)

    # Number of agents to select (at least 1 if fraction > 0 and N >= 1).
    n_hit = int(np.floor(N * fraction))
    n_hit = max(1, min(n_hit, N))

    # Build selection weights.  For gamma = 0 this collapses to uniform.
    y_bar = float(np.mean(income))
    r_i = np.maximum(income / y_bar, 1e-10)  # avoid division blow-up
    weights = r_i ** (-gamma)
    probs = weights / weights.sum()

    # Draw without replacement.  rng.choice supports `p=` and `replace=False`.
    chosen = rng.choice(N, size=n_hit, replace=False, p=probs)

    H = np.zeros(N, dtype=np.int8)
    H[chosen] = 1
    return H


# ---------------------------------------------------------------------------
# 3. Memory stack with income-elastic geometric decay
# ---------------------------------------------------------------------------

def compute_individual_delta(
    income: np.ndarray,
    *,
    delta_base: float,
    epsilon: float,
) -> np.ndarray:
    """Income-elastic geometric persistence delta_i(t).

        delta_i(t) = delta_0 ^ ((y_i(t) / y_bar(t))^epsilon)

    Properties
    ----------
      epsilon = 0   : delta_i = delta_0 for all agents (no differentiation).
      epsilon > 0   : poor agents (y_i < y_bar) get delta_i > delta_0
                      (longer memory half-life);
                      rich agents (y_i > y_bar) get delta_i < delta_0
                      (faster recovery).
      delta_i in (0, 1) is guaranteed for any delta_0 in (0, 1) and
      epsilon >= 0.

    The underlying interpretation (Hallegatte et al. 2017, pp. 4 and
    51): poor households have less ability to cope and recover from
    shocks, hence climate-event salience persists longer in their
    perception.

    Parameters
    ----------
    income : np.ndarray, shape (N,)
        Raw agent incomes at time t.
    delta_base : float in (0, 1)
        Baseline persistence delta_0 (median-income agent).  For
        delta_0 = 0.5 the median agent has a 1-year memory half-life.
    epsilon : float >= 0
        Income elasticity (the same epsilon used for vulnerability).

    Returns
    -------
    np.ndarray, shape (N,), dtype float64
        Per-agent delta_i in (0, 1).
    """
    if not (0.0 < delta_base < 1.0):
        raise ValueError(f"delta_base must lie in (0, 1); got {delta_base}.")
    y_bar = float(np.mean(income))
    r_i = np.maximum(income / y_bar, 1e-10)
    exponent = r_i ** epsilon            # (y_i / y_bar)^epsilon
    return delta_base ** exponent        # delta_0 ^ exponent  in (0, 1)


def update_climate_memory(
    M_prev: np.ndarray,
    hit: np.ndarray,
    D_i: np.ndarray,
    delta_i: np.ndarray,
    *,
    M_max: float = 3.0,  # corresponds to tilde_D_max = 1 - exp(-M_max) ≈ 0.95
) -> np.ndarray:
    """Update the agent-specific log-damage memory stock M_i(t).

    Recursion
    ---------
        M_i(t) = delta_i(t) * M_i(t-1)
                 + H_i(t) * log(1 / (1 - D_i(t))).

    The first term is geometric decay of past damage; the second is
    today's contribution (zero unless the agent is hit).  No cap on
    sum-of-shocks beyond the soft cap M_max applied at the end of the
    function (corresponding to a maximum perceived damage fraction
    `tilde_D_max = 1 - exp(-M_max)`).

    Parameters
    ----------
    M_prev : np.ndarray, shape (N,)
        M_i at the previous period (M_i(t-1)).
    hit : np.ndarray, shape (N,), int or bool
        H_i(t) in {0, 1}.
    D_i : np.ndarray, shape (N,)
        Income-elastic damage D_i(t) in [0, D_max].
    delta_i : np.ndarray, shape (N,)
        Income-elastic persistence delta_i(t) in (0, 1).
    M_max : float
        Upper bound on M_i (soft cap on cumulative perceived damage).

    Returns
    -------
    np.ndarray, shape (N,), dtype float64
        Updated memory stock M_i(t) in [0, M_max].
    """
    # log(1 / (1 - D_i)) = -log(1 - D_i).  Clip 1-D_i away from zero
    # for numerical safety (D_i should be < 1 by construction since
    # D_max < 1, but defensive coding).
    one_minus_D = np.maximum(1.0 - D_i, 1e-10)
    shock_log = -np.log(one_minus_D)              # >= 0 by construction

    # Geometric decay + new shock (zero where hit == 0).
    M_t = delta_i * M_prev + hit.astype(np.float64) * shock_log

    # Soft cap on cumulative perceived damage.
    return np.minimum(M_t, M_max)


def compute_perceived_incomes(
    income: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """Convert raw incomes to perceived incomes given the memory stock.

        y_i^own(t) = exp(-M_i(t)) * y_i(t).

    Equivalent to applying a cumulative damage fraction
    `tilde_D_i(t) = 1 - exp(-M_i(t))` to raw income.

    Parameters
    ----------
    income : np.ndarray, shape (N,)
        Raw agent incomes at the same time index as M.
    M : np.ndarray, shape (N,)
        Memory stock at that time.

    Returns
    -------
    np.ndarray, shape (N,), dtype float64
        Perceived incomes; floor-clipped at 1e-10 to remain strictly
        positive (needed before downstream log operations).
    """
    return np.maximum(np.exp(-M) * income, 1e-10)


# ---------------------------------------------------------------------------
# Local benchmark helper (numerator / denominator chosen by caller)
# ---------------------------------------------------------------------------

def _compute_local_benchmark(
    income_num: np.ndarray,
    income_den: np.ndarray,
    neighbor_lists: list[np.ndarray],
    statistic: str,
) -> np.ndarray:
    """Per-agent local benchmark log-growth from neighbour incomes.

    For each agent i:
        B_i = log( stat(income_num[N_i]) / stat(income_den[N_i]) ).

    The caller (compute_private_signal_unified) decides what
    `income_num` and `income_den` represent according to the chosen
    `local_climate_visibility` mode:
      "current_event" -> income_num = (1 - H_j(t) D_j(t)) * y_j(t),
                         income_den = y_j(t-1)  (raw)
      "none"          -> income_num = y_j(t)    (raw),
                         income_den = y_j(t-1)  (raw)

    Parameters
    ----------
    income_num, income_den : np.ndarray, shape (N,)
        Numerator and denominator income vectors.
    neighbor_lists : list of np.ndarray
        neighbor_lists[i] holds the position indices of i's neighbours
        (excluding i itself, by NetworkX convention).
    statistic : {"mean", "median"}

    Returns
    -------
    np.ndarray, shape (N,), dtype float64
        Per-agent local benchmark log-growth.  Agents with no
        neighbours receive 0.
    """
    stat_fn = np.median if statistic == "median" else np.mean
    N = len(neighbor_lists)
    benchmark = np.zeros(N, dtype=np.float64)

    for i, neighbors in enumerate(neighbor_lists):
        if len(neighbors) == 0:
            continue
        s_t   = stat_fn(income_num[neighbors])
        s_tm1 = stat_fn(income_den[neighbors])
        if s_tm1 > 0.0:
            benchmark[i] = np.log(s_t / s_tm1)

    return benchmark


# ---------------------------------------------------------------------------
# Unified private signal — perceived-income form
# ---------------------------------------------------------------------------

def compute_private_signal_unified(
    y_own_t: np.ndarray,
    y_own_tm1: np.ndarray,
    income_t: np.ndarray,
    income_tm1: np.ndarray,
    *,
    benchmark: str = "gdp",
    gdp_growth_t: float = 0.0,
    neighbor_lists: list[np.ndarray] | None = None,
    local_climate_visibility: str = "current_event",
    hit_t: np.ndarray | None = None,
    D_i_t: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the unified private signal s_i^priv in {-1, 0, +1}.

    Signal formula (multiplicative perceived-income form)
    -----------------------------------------------------
        s_i^priv(t) = sgn{ log[y_i^own(t) / y_i^own(t-1)]  -  B_i(t) }

    where y_i^own(t) = exp(-M_i(t)) * y_i(t) is the perceived income
    incorporating cumulative geometric memory of past climate shocks.

    The memory mechanism is fully encapsulated in y_own_t / y_own_tm1
    (computed upstream); this function performs ONLY the comparison
    against the benchmark.

    Benchmark options (B_i)
    -----------------------
    "gdp":
        B_i = log(1 + g_pc(t)).  External macroeconomic benchmark.
        Identical for all agents.  No climate correction (MAPS model
        does not feed climate damage back into GDP).

    "mean_income":
        B_i = log( mean(y(t)) / mean(y(t-1)) ).
        Global mean of RAW incomes; macro aggregate analogue.

    "median_income":
        B_i = log( median(y(t)) / median(y(t-1)) ).
        Global median of RAW incomes.

    "local_mean", "local_median":
        Per-agent benchmark over the neighbourhood N_i.  The numerator
        depends on `local_climate_visibility` (see below); the
        denominator is always RAW previous-year incomes y_j(t-1).

    Local climate visibility (controls the local numerator only)
    ------------------------------------------------------------
    "current_event"  (option iii):
        B_i = log( stat_j[ (1 - H_j(t) D_j(t)) y_j(t) ]  /  stat_j[ y_j(t-1) ] )
        Strict egocentric availability.  Agent i
        sees neighbours' CONCURRENT events (j was hit this year), but
        does NOT internalise the cumulative memory of j's past shocks.
        Information visible to i: a current disaster striking the
        neighbourhood.  Requires `hit_t` and `D_i_t`.

    "none"  (option iv):
        B_i = log( stat_j[ y_j(t) ]  /  stat_j[ y_j(t-1) ] )
        No climate effect in the local benchmark.  The climate channel
        operates EXCLUSIVELY through the own-side perceived income.
        Maximum separability and parsimony.  `hit_t` and `D_i_t` are
        ignored.

    Parameters
    ----------
    y_own_t, y_own_tm1 : np.ndarray, shape (N,)
        Perceived incomes at t and t-1.  Must be strictly positive.
        Used only on the OWN side (log-growth).
    income_t, income_tm1 : np.ndarray, shape (N,)
        RAW incomes at t and t-1.  Used by global benchmarks
        ("mean_income", "median_income") and by the local benchmark
        denominator.  income_t is also used by the local numerator
        in BOTH modes.
    benchmark : str
        One of: "gdp", "mean_income", "median_income", "local_mean",
        "local_median".
    gdp_growth_t : float
        Real GDP per capita growth (decimal); used only for "gdp".
    neighbor_lists : list of np.ndarray or None
        Required for local benchmarks.
    local_climate_visibility : {"current_event", "none"}
        Selects how climate damage enters the local benchmark
        numerator.  Ignored for non-local benchmarks.
    hit_t : np.ndarray or None
        H_i(t) in {0, 1}.  Required when local benchmark is selected
        AND visibility="current_event".  Ignored otherwise.
    D_i_t : np.ndarray or None
        D_i(t) (per-period income-elastic damage).  Required when local
        benchmark is selected AND visibility="current_event".  Ignored
        otherwise.

    Returns
    -------
    np.ndarray, shape (N,), dtype int8
        s_i^priv in {-1, 0, +1}.

    Raises
    ------
    ValueError
        If `benchmark` is unrecognised; if local benchmark is requested
        without `neighbor_lists`; if `local_climate_visibility` is not
        recognised; or if visibility="current_event" is requested for a
        local benchmark without `hit_t`/`D_i_t`.
    """
    _VALID_BM = {"gdp", "mean_income", "median_income", "local_mean", "local_median"}
    _VALID_VIS = {"current_event", "none"}

    if benchmark not in _VALID_BM:
        raise ValueError(f"Unknown benchmark '{benchmark}'; expected one of {_VALID_BM}.")
    if local_climate_visibility not in _VALID_VIS:
        raise ValueError(
            f"Unknown local_climate_visibility '{local_climate_visibility}'; "
            f"expected one of {_VALID_VIS}."
        )

    is_local = benchmark in ("local_mean", "local_median")
    if is_local and neighbor_lists is None:
        raise ValueError(
            f"benchmark='{benchmark}' requires neighbor_lists to be provided."
        )
    if is_local and local_climate_visibility == "current_event":
        # Option (iii) needs the current-year event info to build the
        # numerator (1 - H_j(t) D_j(t)) y_j(t).
        if hit_t is None or D_i_t is None:
            raise ValueError(
                "local_climate_visibility='current_event' requires "
                "both hit_t and D_i_t to be provided."
            )

    # --- Perceived-income log-growth (own side) ---
    # The own-side ALWAYS uses the full perceived income (cumulative
    # memory): regardless of the benchmark choice, the agent feels its
    # own past adapted shocks.
    log_growth = np.log(y_own_t / y_own_tm1)

    # --- Benchmark growth ---
    if benchmark == "gdp":
        benchmark_growth = np.full_like(y_own_t, np.log(1.0 + gdp_growth_t))

    elif benchmark == "mean_income":
        benchmark_growth = np.full_like(
            y_own_t,
            np.log(np.mean(income_t) / np.mean(income_tm1)),
        )

    elif benchmark == "median_income":
        benchmark_growth = np.full_like(
            y_own_t,
            np.log(np.median(income_t) / np.median(income_tm1)),
        )

    elif is_local:
        # Build the local numerator according to visibility mode.
        # Denominator is always raw past income.
        if local_climate_visibility == "current_event":
            # Option (iii): only the current-year event affects what
            # i sees of j's income.  Floor-clip for log safety
            # (1 - D_i_t < 1 by construction, but defensive).
            local_num = np.maximum(
                (1.0 - hit_t.astype(np.float64) * D_i_t) * income_t,
                1e-10,
            )
        else:  # local_climate_visibility == "none"
            # Option (iv): climate has no effect on the local benchmark.
            local_num = income_t

        statistic = "mean" if benchmark == "local_mean" else "median"
        benchmark_growth = _compute_local_benchmark(
            local_num, income_tm1, neighbor_lists, statistic=statistic,
        )

    # --- Sign of the residual ---
    return np.sign(log_growth - benchmark_growth).astype(np.int8)