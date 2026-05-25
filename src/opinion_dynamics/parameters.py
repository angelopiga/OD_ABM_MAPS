"""Global configuration parameters for the Opinion Dynamics ABM.

All tuneable constants are defined here as module-level names so that notebooks
and scripts can import them directly.  The module also attempts to load external
CSV time-series (income, Gini, emissions, temperature); if the files are absent
or malformed it falls back to hard-coded baseline values.

Sections
--------
Paths & External Data Source
Simulation Scope & Reproducibility
Population & Geography
Alpha Initialization
Opinion Dynamics (unified private signal + social signal)
Climate / Environment Parameters
Network Formation
Macro Time-Series Inputs
Numeric Types
"""
from pathlib import Path
import numpy as np
from .external_data import load_all_external_timeseries, load_temperature_timeseries

# -----------------------------
# Paths & External Data Source
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# -----------------------------
# Scenario Selection
# -----------------------------

ACTIVE_SCENARIO: list[str] = ["Green_growth"]  # options: "Baseline", "Degrowth", "Green_growth"
BENCHMARK_SCENARIO: str | None = "Baseline"           # Reference scenario label used when PLOT_DIFF_FROM_BASELINE=True

# -----------------------------
# Simulation Scope & Reproducibility
# -----------------------------
YEARS_FULL = list(range(2022, 2050 + 1))
YEARS = YEARS_FULL.copy()
N_SIMULATIONS = 10
SEED = 123
SAVE_RESULTS = True
PLOT_UNCERTAINTY_BANDS = "std"                # None | "std" | "sem"
PLOT_DIFF_FROM_BASELINE: bool = BENCHMARK_SCENARIO is not None
SHOW_BASELINE_RESULTS: bool = False           # If True (and PLOT_DIFF_FROM_BASELINE=True), include Baseline
                                              # in the centered-quintile spread panel of
                                              # plot_multi_scenario_opinion_quintiles.

# -----------------------------
# Population & Geography
# -----------------------------
POPULATION_SIZE = 3000
REGION_NAMES = ["Italy"]

# -----------------------------
# Alpha (opinions) initialization
# -----------------------------
# Options: "uniform", "constant", "beta_symmetric", "beta_skewed",
#          "income_monotone", "bimodal", "issp_proxy"
ALPHA_INIT_STRATEGY: str = "uniform"
# parameters for the chosen strategy; e.g. a,b for beta distribution; ignored for "uniform" and "constant"
ALPHA_INIT_PARAMS: dict = {"a": 3.0, "b": 1.5} 
# -----------------------------
# Opinion Dynamics — private signal (income+climate damage) + social signal (neighbour opinions)
# -----------------------------
# p_i(t) = w_priv * s_i^priv + (1 - w_priv) * s_i^soc
OPINION_W_PRIV = 0.5    # weight on private channel; w_soc = 1 - w_priv
OPINION_NU = 0.05       # Bernoulli step size; 10 concordant signals for 0.5 -> 1.0

# private signal configuration
# benchmark options: "gdp", "mean_income", "median_income", "local_mean", "local_median"
PRIVATE_SIGNAL_BENCHMARK = "local_mean"
LOCAL_CLIMATE_VISIBILITY: str = "current_event"   # "none" | "current_event"

# -----------------------------
# Climate / Environment Parameters
# -----------------------------

# climate event parameters
T0 = 1.0                             # baseline temperature anomaly (°C above pre-industrial)
CLIMATE_EVENT_RATE: float = 0.5      # Poisson-saliency rate c; p_event(t) = 1 - exp(-c * D_base(t))

# climate damage parameters: lobal damage fraction D(t) = τa * τ²^b
CLIMATE_A:float = 0.035              # linear damage coefficient (τ term)
CLIMATE_B:float = 0.0009             # quadratic damage coefficient (τ² term)

# damage parameters for private signal (D_base) and social signal (D_soc)
CLIMATE_HIT_FRACTION: float = 0.05   # fraction of population directly hit by climate events
MEMORY_DELTA_BASE: float    = 0.5       # base geometric decay rate delta_0

# elasticities for income-based vulnerability, persistence, and exposure to climate events; 
CLIMATE_ELASTICITY_VULNERABILITY: float = 0.36   # D_i ∝ (ȳ/y_i)^ε; Gilli et al. 2024
CLIMATE_ELASTICITY_PERSISTENCE: float   = 0   # δ_i = δ_0^((y_i/ȳ)^ε)
CLIMATE_ELASTICITY_EXPOSURE: float      = 0   # π_i ∝ (ȳ/y_i)^ε 

# -----------------------------
# Network Formation
# -----------------------------
NETWORK_AVG_DEGREE = 10
NETWORK_ALGORITHM = "topk"           # "topk" | "sampled"
NETWORK_BETA_INCOME = 1.0            # income-homophily strength
NETWORK_INCOME_NORMALIZATION = "rank"# "rank" | None (absolute income); "rank" recommended to avoid extreme degree heterogeneity from absolute income differences.
NETWORK_RANK_BINS: int = POPULATION_SIZE // (3 * NETWORK_AVG_DEGREE) # number of bins for rank-based normalization; ignored if NETWORK_INCOME_NORMALIZATION != "rank"; smaller bins yield more extreme homophily and degree heterogeneity, larger bins yield smoother homophily and more homogeneous degrees; default is 3x avg degree, which yields a good balance of homophily and connectivity in the baseline scenario.
NETWORK_NOISE_STRENGTH_SD = 0.5      # standard deviation of Gaussian noise added to homophily scores before top-k selection; 0.0 disables noise and yields deterministic network

# Watts–Strogatz rewiring (optional hegemonic channel)
NETWORK_WS_REWIRE_P = 0.2 # Per-edge rewiring probability; 0.0 disables.
# Preferential-attachment exponent γ for rewiring targets:
NETWORK_WS_TARGET_EXPONENT = 0.0 #   γ = 0 : uniform WS (Watts & Strogatz 1998); γ > 0 : income-targeted, P(k) ∝ y_k^γ (hegemonic channel)

# -----------------------------
# Macro Time-Series Inputs
# -----------------------------

# EUROGREEN scenario-dependent series.
# Raises on missing or malformed data — no fallback.
YEARS, REGION_AVG_INCOMES, GINIS, EMISSIONS, GDP, INFLATION, ALL_INDICATORS = (
    load_all_external_timeseries(
        data_dir=DATA_DIR,
        base_years=YEARS_FULL,
        scenario=ACTIVE_SCENARIO[0],
    )
)

# Temperature: exogenous IPCC/CMIP6 series.
# TEMPERATURE_SCENARIO selects the SSP pathway.
#
# "SSP1-1.9": 1.6 °C above pre-industrial by 2050, very strong mitigation
# "SSP1-2.6": 1.7 °C above pre-industrial by 2050, strong mitigation
# "SSP2-4.5": 2.0 °C above pre-industrial by 2050, intermediate emissions
# "SSP3-7.0": 2.1 °C above pre-industrial by 2050, high emissions
# "SSP5-8.5": 2.4 °C above pre-industrial by 2050, very high emissions / worst-case scenario

TEMPERATURE_SCENARIO: str = "SSP5-8.5"

# Past-temperature treatment method:
#
# "raw_obs"     : observed annual anomalies, including year-to-year variability
# "smooth_obs"  : smoothed observed anomalies
# "forced_resp" : forced-response component, excluding short-run variability

EXTRAPOLATION_PAST_T: str = "forced_resp"

TEMPERATURE: dict[int, float] = load_temperature_timeseries(
    data_dir=DATA_DIR,
    base_years=YEARS_FULL,
    scenario=TEMPERATURE_SCENARIO,
    extrapolation_past_t=EXTRAPOLATION_PAST_T,
)

# -----------------------------
# Numeric Types
# -----------------------------
DTYPE_INCOME = np.dtype("float32")