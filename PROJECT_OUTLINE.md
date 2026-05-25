# PROJECT SCHEMA: Opinion Dynamics ABM

> Reference document — module tree, simulation pipeline, data-flow diagram, model equations, and parameter reference.  
> Last updated: 2026-05-07.  
> Author: Angelo Piga (Departament of Economics and Management, University of Pisa)

Agent-based model developed for **Task 4.3** of the [MAPS Horizon Project](https://mapsresearch.eu/)
(*Models, Assessment & Policies for Sustainability*).
The model is described in full in the companion report `ABM_Opinion_Dynamics_MAPS.pdf`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Module Tree](#2-module-tree)
3. [Simulation Pipeline](#3-simulation-pipeline)
4. [Data Passing Between Phases](#4-data-passing-between-phases)
5. [Model Theory and Code Correspondence](#5-model-theory-and-code-correspondence)
6. [Parameter Reference](#6-parameter-reference)

---

## 1. Overview

**Package**: `opinion_dynamics` (`src/opinion_dynamics/`)  
**Entry point**: `notebooks/main.py`  
**Language**: Python 3.10+

Simulates the evolution of climate-policy support (`alpha ∈ [0, 1]`) in a heterogeneous population under alternative macro-economic scenarios (Baseline, Degrowth, Green_growth). Each agent receives a **unified private signal** `s_priv` that combines an income channel (aspiration vs. benchmark growth) and a climate-damage channel (income-elastic memory stock `M_i`). Agents also observe a **social signal** `xi_soc` from network neighbours. Opinion updating follows a Bernoulli step rule.

**Programmatic use (single scenario):**
```python
from opinion_dynamics.abm import run_abm
results = run_abm(scenario="Degrowth")
# results keys: "df_simulation_stats", "df_agents_panel", "df_year_summary"
```

**Multi-scenario run (from entry point):**
```python
from opinion_dynamics.abm import run_all_scenarios
all_results = run_all_scenarios()   # uses ACTIVE_SCENARIO from parameters.py
```

---

## 2. Module Tree

```
OD_ABM_MAPS/
│
├── notebooks/
│   └── main.py                          # Entry point; runs scenarios, calls preprocessing and plotting
│
├── src/
│   └── opinion_dynamics/
│       ├── __init__.py
│       │
│       ├── parameters.py                # All tuneable constants; loads external data at import time
│       ├── external_data.py             # CSV loaders for income/Gini/GDP/emissions/temperature series
│       │
│       ├── abm.py                       # Top-level simulation loop; orchestrates all sub-modules
│       ├── agents.py                    # Agent DataFrame initialisation (income, region)
│       ├── alpha_init.py                # Alpha initialisation strategies (uniform, beta, ISSP proxy, …)
│       ├── income.py                    # Lognormal calibration; rank-preserving inter-year transport
│       ├── network.py                   # Income-assortative social network (topk / WS rewiring)
│       ├── utilityfun_satisfaction.py   # Climate-event pipeline; unified private signal s_priv
│       ├── utilityfun_opinion_update.py # Social signal; two-channel combiner; Bernoulli update
│       │
│       ├── stats.py                     # Gini, income assortativity, diagnostics row builder
│       ├── utils.py                     # Seeded RNG streams; scenario slug; state management
│       │
│       ├── preprocessing.py             # Post-simulation aggregation (called once before plotting)
│       ├── plot_utils.py                # Low-level plotting helpers; heatmap; quintile opinion plot
│       └── plotting.py                  # High-level figures: scenario overview, signal diagnostics,
│                                        #   channel contributions, multi-scenario comparisons
│
├── data/
│   └── raw/
│       ├── italy_data.csv               # Long-format CSV (income, Gini, GDP, emissions; all scenarios)
│       └── temperature_scenarios.csv    # IPCC/CMIP6 temperature anomaly series
│
├── results/
│   ├── figures/                         # Generated PNGs (written when SAVE_RESULTS = True)
│   └── data/                            # Agent panels (.parquet) + parameter snapshots (.json)
│
└── pyproject.toml
```

### Module roles at a glance

| Module | Role | Category |
|---|---|---|
| `parameters` | Global config; loads external time-series at import time | Configuration |
| `external_data` | Parses scenario CSVs; computes GDP per capita | I/O |
| `agents` | Samples initial agent DataFrame (income, region) | Initialisation |
| `alpha_init` | Seven strategies for opinion initialisation | Initialisation |
| `income` | Lognormal fitting; rank-preserving annual income transport | Economics |
| `network` | Income-assortative graph (topk / WS); returns CSR matrices | Network |
| `utilityfun_satisfaction` | Climate-event generation; individual damage; memory update; `s_priv` | Signal computation |
| `utilityfun_opinion_update` | Social signal `xi_soc`; two-channel combiner; Bernoulli step | Opinion dynamics |
| `stats` | Gini; income assortativity; stats-row builder | Statistics |
| `utils` | Seeded RNG streams; `_to_state`; `_attach_outputs`; scenario slug | Utilities |
| `abm` | Outer/inner loops; assembles output DataFrames | Orchestration |
| `preprocessing` | Pre-aggregates panel data before plotting | Post-processing |
| `plot_utils` | Low-level helpers; heatmap; quintile opinion plot | Visualisation |
| `plotting` | High-level multi-panel figures; multi-scenario comparisons | Visualisation |
| `main` (notebook) | Entry point; drives runs, preprocessing, and plotting | Entry point |

---

## 3. Simulation Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│  CONFIGURATION  [parameters.py, loaded at import time]               │
│  external_data.py → REGION_AVG_INCOMES, GINIS, GDP, INFLATION,       │
│                     EMISSIONS, TEMPERATURE (dicts keyed by year)     │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────┐
│  PRE-SIMULATION  [utilityfun_satisfaction.py]                         │
│  generate_climate_events(years, temperature, T0, rate, rng)          │
│  → events: dict[int, int]   {year → E(t) ∈ {0, 1}}                  │
│  (one draw shared across simulations; injected into run_abm via arg) │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────┐
│  OUTER LOOP: for sim in range(N_SIMULATIONS)                         │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │  INITIALISATION  (year = YEARS[0])                           │    │
│  │  agents.py: generate_agents_for_year() → df_agents           │    │
│  │  network.py: build_network() → (G, A_bin, A_w)               │    │
│  │  alpha_init.py: initialize_alpha(strategy, percentiles, rng) │    │
│  │  M_climate = zeros(N)      [climate memory; persistent]      │    │
│  │  percentiles = agent rank  [fixed for simulation lifetime]   │    │
│  └──────────────────────────────┬───────────────────────────────┘    │
│                                 │                                    │
│  ┌──────────────────────────────▼───────────────────────────────┐    │
│  │  INNER LOOP: for year in YEARS[1:]                           │    │
│  │                                                              │    │
│  │  1. INCOME UPDATE  [income.py]                               │    │
│  │     assign_incomes_deterministic()                           │    │
│  │     → df_agents["income"]  (rank-preserving transport)       │    │
│  │                                                              │    │
│  │  2. CLIMATE DAMAGE  [utilityfun_satisfaction.py]             │    │
│  │     compute_damage_base(T, T0)          → D_base(t)          │    │
│  │     compute_individual_damage(D_base)   → D_i(t)             │    │
│  │     If E(t) = 1:                                             │    │
│  │       select_hit_agents(income, fraction, gamma) → hit       │    │
│  │     compute_individual_delta(income)    → delta_i(t)         │    │
│  │     update_climate_memory(M_prev, hit, D_i, delta_i)         │    │
│  │     → M_climate(new)                                         │    │
│  │                                                              │    │
│  │  3. PERCEIVED INCOMES + PRIVATE SIGNAL                       │    │
│  │     [utilityfun_satisfaction.py]                             │    │
│  │     compute_perceived_incomes(income, M_climate)             │    │
│  │     → y_own_t  (perceived income; climate-downweighted)      │    │
│  │     compute_private_signal_unified(y_own_t, y_own_tm1,       │    │
│  │       income_t, income_tm1, benchmark, gdp_growth, …)        │    │
│  │     → s_priv ∈ {-1, 0, +1}                                   │    │
│  │                                                              │    │
│  │  4. SOCIAL SIGNAL  [utilityfun_opinion_update.py]            │    │
│  │     compute_social_signal(alpha, neighbor_lists, rng)        │    │
│  │     → xi_soc ∈ {-1, 0, +1}                                   │    │
│  │                                                              │    │
│  │  5. SIGNAL COMBINATION  [utilityfun_opinion_update.py]       │    │
│  │     combine_signals_two_channel(s_priv, xi_soc, w_priv)      │    │
│  │     → p ∈ [-1, +1]                                           │    │
│  │                                                              │    │
│  │  6. OPINION UPDATE  [utilityfun_opinion_update.py]           │    │
│  │     bernoulli_update(alpha, p, nu, rng)                      │    │
│  │     → alpha(new) ∈ [0, 1]                                    │    │
│  │                                                              │    │
│  │  7. SNAPSHOT  [utils.py, stats.py via abm.py]                │    │
│  │     _attach_outputs() → append to df_agents_panel            │    │
│  │     _make_stats_row() → append to df_simulation_stats        │    │
│  └──────────────────────────────────────────────────────────────┘    │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────┐
│  OUTPUT ASSEMBLY  [abm.py]                                            │
│  df_agents_panel      (N_years × N_sims × N_agents rows)             │
│  df_simulation_stats  (N_years × N_sims rows)                        │
│  df_year_summary      (N_years rows; cross-sim aggregates)           │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────┐
│  POST-PROCESSING  [preprocessing.py]                                  │
│  precompute(df_agents_panel, years, n_quantiles)                     │
│  → precomputed dict (shared summary frames for all plot calls)       │
└─────────────────────────────┬────────────────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────────────────┐
│  VISUALISATION  [plotting.py, plot_utils.py]                          │
│  plot_scenario_overview()          — macro drivers + quintile opinion │
│  plot_signal_diagnostics()         — heatmap, s_priv, D_i by quintile│
│  plot_channel_contributions()      — annual and cumulative Δα push   │
│  plot_multi_scenario_opinion()     — aggregate support by scenario    │
│  plot_multi_scenario_opinion_quintiles() — quintile curves            │
│  plot_multi_scenario_bc()          — bimodality coefficient BC(t)    │
│  plot_opinion_heatmap()            — α density heatmap [plot_utils]  │
│  plot_opinion_by_income_quintile() — quintile mean α [plot_utils]    │
└──────────────────────────────────────────────────────────────────────┘
```

### Key state variables (persistent across years, within one simulation)

| Variable | Shape | Domain | Description |
|---|---|---|---|
| `alpha` | `(N,)` | `[0, 1]` | Agent opinion (pro-climate-policy support) |
| `M_climate` | `(N,)` | `[0, M_max]` | Agent log-damage memory stock |
| `percentiles` | `(N,)` | `(0, 1)` | Agent income percentile ranks (fixed at initialisation) |
| `df_agents["income"]` | `(N,)` | `ℝ⁺` | Current-year income (updated by rank-preserving transport) |

### Per-period intermediates (recomputed each year)

| Variable | Domain | Description |
|---|---|---|
| `D_base` | `ℝ⁺` | Aggregate Kalkuhl & Wenz quadratic damage |
| `D_i` | `[0, D_max]` | Income-elastic individual damage |
| `hit` | `{0, 1}^N` | Binary climate-event hit mask (non-zero only when `E(t)=1`) |
| `delta_i` | `[0, 1]^N` | Income-elastic geometric persistence of memory |
| `y_own_t` | `ℝ⁺` | Perceived income (raw income deflated by memory stock) |
| `s_priv` | `{-1, 0, +1}^N` | Unified private signal (income + climate channel) |
| `xi_soc` | `{-1, 0, +1}^N` | Social influence signal |
| `p` | `[-1, +1]^N` | Combined update probability |

---


### df_agents_panel columns

| Column | Type | Description |
|---|---|---|
| `year` | int | Simulation year |
| `simulation_id` | int | Monte Carlo run index |
| `agent_id` | int | Agent identifier (0 … N−1) |
| `region` | str | Region name |
| `income` | float32 | Realised income after transport |
| `alpha` | float | Opinion (pro-climate-policy support) |
| `s_priv` | int | Unified private signal ∈ {−1, 0, +1} |
| `p` | float | Combined update probability |
| `M_climate` | float | Climate memory stock |
| `hit` | int | Climate-event hit indicator ∈ {0, 1} |

---

## 5. Model Theory and Code Correspondence

> Full mathematical derivations are in the companion report `ABM_Opinion_Dynamics.pdf`. This section provides a structured map from theory to code.

### 5.1 Agent state

Each agent `i` is characterised at every year `t` by:

| Symbol | Domain | Description | Code variable |
|---|---|---|---|
| `y_i(t)` | ℝ⁺ | Nominal income | `df_agents["income"]` |
| `α_i(t)` | [0, 1] | Opinion (policy support) | `alpha` array |
| `M_i(t)` | [0, M_max] | Cumulative perceived-damage memory stock | `M_climate` array |
| `p_i` | (0, 1) | Income percentile rank (fixed) | `percentiles` array |

### 5.2 Income distribution and rank-preserving transport

Annual incomes are drawn from a LogNormal calibrated to macro targets `(m(t), G(t))`:

```
σ_log(t) = √2 · Φ⁻¹((G(t)+1)/2)          [sigma_log_from_gini]
y_i(t)   = ỹ(t) · exp(σ_log(t) · Φ⁻¹(p_i)) [assign_incomes_deterministic]
```

The agent's percentile rank `p_i` is fixed for the lifetime of a simulation (zero-mobility baseline).

| Theory | Code function | Module |
|---|---|---|
| LogNormal calibration from `(m, G)` | `sample_lognormal_with_exact_mean_gini` | `income.py` |
| Rank-preserving transport (eq. 3 in report) | `assign_incomes_deterministic` | `income.py` |
| `σ_log` from Gini (eq. 1) | `sigma_log_from_gini` | `utilityfun_satisfaction.py` |

### 5.3 Social network

The network is built once at initialisation using an income-assortative kernel:

```
W_ij = exp(−β_inc · |s_i − s_j|)           [build_weight_matrix]
```

where `s_i` are rank-normalised incomes. Each agent selects the `K` highest-weight candidates (top-K); unilateral links accepted with probability 0.5. An optional Watts–Strogatz rewiring fraction (`p_rewire`) creates long-range connections.

| Theory | Code function | Module |
|---|---|---|
| Weight matrix W (eq. 4) | `build_weight_matrix` | `network.py` |
| Top-K edge selection + symmetrization | `_edges_from_W` | `network.py` |
| WS rewiring | `_ws_rewire_edges` | `network.py` |
| Full network constructor | `build_network` | `network.py` |

### 5.4 Climate damage channel (three stages)

**Stage 1 — Exogenous event series** (computed once before the simulation loop):

```
τ(t)       = max(T(t) − T₀, 0)
p_event(t) = 1 − exp(−c · τ(t))
E(t)       ~ Bernoulli(p_event(t))          [generate_climate_events]
```

**Stage 2 — Hit selection** (when `E(t) = 1`):

```
π_i(t) ∝ (ȳ(t)/y_i(t))^μ                  [select_hit_agents]
H_i(t) = 1[i ∈ S(t)],  |S(t)| = N · f
```

**Stage 3 — Individual damage and memory recursion**:

```
D_base(t) = a·τ + b·τ²                     [compute_damage_base]
D_i(t)    = D_base(t) · (ȳ/y_i)^ε         [compute_individual_damage]
δ_i(t)    = δ₀^((y_i/ȳ)^ω)                [compute_individual_delta]

(1 − D̃_i(t)) = (1 − D̃_i(t−1))^δ_i · (1 − D_i)^H_i   [update_climate_memory]

y_i^perc(t) = (1 − D̃_i(t)) · y_i(t)       [compute_perceived_incomes]
```

| Theory | Code function | Module |
|---|---|---|
| Event series {E(t)} (eq. 6) | `generate_climate_events` | `utilityfun_satisfaction.py` |
| Hit selection (eq. 7) | `select_hit_agents` | `utilityfun_satisfaction.py` |
| Aggregate damage D_base (eq. 10) | `compute_damage_base` | `utilityfun_satisfaction.py` |
| Individual damage D_i (eq. 9) | `compute_individual_damage` | `utilityfun_satisfaction.py` |
| Income-elastic persistence δ_i (eq. 12) | `compute_individual_delta` | `utilityfun_satisfaction.py` |
| Memory recursion (eq. 11) | `update_climate_memory` | `utilityfun_satisfaction.py` |
| Perceived income (eq. 8) | `compute_perceived_incomes` | `utilityfun_satisfaction.py` |

### 5.5 Private signal

The private signal compares the agent's perceived log-income growth against a benchmark `B_i(t)`:

```
s_i^priv(t) = sgn(log(y_i^perc(t)/y_i^perc(t−1)) − B_i(t))  ∈ {−1, 0, +1}
```

Five benchmark options are available via `PRIVATE_SIGNAL_BENCHMARK`:

| Option | Formula | Scope |
|---|---|---|
| `"gdp"` | `log(1 + g_pc(t))` | Global (GDP per capita growth) |
| `"mean_income"` | `log(ȳ(t)/ȳ(t−1))` | Global mean income |
| `"median_income"` | `log(ỹ(t)/ỹ(t−1))` | Global median income |
| `"local_mean"` *(default)* | log-growth of mean neighbour perceived income | Local network |
| `"local_median"` | log-growth of median neighbour perceived income | Local network |

For local benchmarks, `LOCAL_CLIMATE_VISIBILITY` controls whether neighbours' current-period damage enters the numerator (`"current_event"`) or only raw incomes are used (`"none"`).

| Theory | Code function | Module |
|---|---|---|
| Private signal (eq. 5) | `compute_private_signal_unified` | `utilityfun_satisfaction.py` |
| Local benchmark computation | `_compute_local_benchmark` | `utilityfun_satisfaction.py` |

### 5.6 Social signal

Agent `i` draws one neighbour `j` from `N_i` and computes:

```
s_i^soc(t) = sgn(α_j − α_i)  with prob. |α_j(t) − α_i(t)|
           = 0                with prob. 1 − |α_j(t) − α_i(t)|
```

| Theory | Code function | Module |
|---|---|---|
| Social signal (eq. 13) | `compute_social_signal` | `utilityfun_opinion_update.py` |

### 5.7 Aggregation and opinion update

```
p_i(t) = w_priv · s_i^priv(t) + (1 − w_priv) · s_i^soc(t)  ∈ [−1, +1]

α_i(t+1) = min(α_i + ν, 1)  with prob. max(p_i, 0)
          = max(α_i − ν, 0)  with prob. max(−p_i, 0)
          = α_i              otherwise
```

| Theory | Code function | Module |
|---|---|---|
| Two-channel mix (eq. 14) | `combine_signals_two_channel` | `utilityfun_opinion_update.py` |
| Bernoulli opinion step (eq. 15) | `bernoulli_update` | `utilityfun_opinion_update.py` |

### 5.8 Alpha initialisation

The default initialisation is `α_i(0) ~ Uniform(0, 1)`. Alternatives are selected via `ALPHA_INIT_STRATEGY`:

| Strategy key | Distribution | Code function |
|---|---|---|
| `"uniform"` | U(0, 1) | `_uniform` |
| `"constant"` | α_i = c | `_constant` |
| `"beta_symmetric"` | Beta(a, a) | `_beta_symmetric` |
| `"beta_skewed"` | Beta(a, b) | `_beta_skewed` |
| `"income_monotone"` | α_i = c₀ + c₁·p_i + ε_i | `_income_monotone` |
| `"bimodal"` | Mixture of two Beta | `_bimodal` |
| `"issp_proxy"` | Synthetic ISSP 2019 Italy proxy | `_issp_proxy` |

All strategies are in `alpha_init.py` and dispatched by `initialize_alpha(strategy, percentiles, rng, params)`.

---

## 6. Parameter Reference

All parameters live in `src/opinion_dynamics/parameters.py` and are imported directly where needed. Parameter names in `parameters.py` are the authoritative code-side identifiers; the PDF column gives the corresponding symbol or section reference.

### 6.1 Scenario and run control

| Parameter | Default | Description | PDF ref |
|---|---|---|---|
| `ACTIVE_SCENARIO` | `["Green_growth","Degrowth"]` | Scenario(s) to simulate | §12 |
| `BENCHMARK_SCENARIO` | `"Baseline"` | Reference scenario for difference plots | §12 |
| `N_SIMULATIONS` | `10` | Number of Monte Carlo replications | §12 |
| `SEED` | `123` | Global random seed for event series and initialisation | — |
| `YEARS` | `2022–2050` | Simulation years (from CSV coverage) | §3 |

### 6.2 Output control

| Parameter | Default | Description |
|---|---|---|
| `SAVE_RESULTS` | `True` | Write figures and data to `results/` |
| `PLOT_UNCERTAINTY_BANDS` | `"std"` | Shaded bands: `None` / `"std"` / `"sem"` |
| `PLOT_DIFF_FROM_BASELINE` | `True` (if `BENCHMARK_SCENARIO` set) | Show Δα relative to baseline in multi-scenario plots |
| `SHOW_BASELINE_RESULTS` | `False` | Include baseline in quintile spread panel |

### 6.3 Population

| Parameter | Default | Description | PDF ref |
|---|---|---|---|
| `POPULATION_SIZE` | `3000` | Number of agents N | §3, §11 |
| `REGION_NAMES` | `["Italy"]` | Region labels (single region) | §3 |

### 6.4 Alpha initialisation

| Parameter | Default | Options | PDF ref |
|---|---|---|---|
| `ALPHA_INIT_STRATEGY` | `"uniform"` | `"uniform"`, `"constant"`, `"beta_symmetric"`, `"beta_skewed"`, `"income_monotone"`, `"bimodal"`, `"issp_proxy"` | §6 |
| `ALPHA_INIT_PARAMS` | `{"a": 3.0, "b": 1.5}` | Strategy-specific keyword args (passed through to the chosen initialiser) | §6 |

### 6.5 Opinion dynamics

| Parameter | Symbol | Default | Description | PDF ref |
|---|---|---|---|---|
| `OPINION_W_PRIV` | w_priv | `0.5` | Weight on private signal; social weight = 1 − w_priv | eq. 14, §11 |
| `OPINION_NU` | ν | `0.05` | Bernoulli step size; ~10 concordant signals to move 0→1 | eq. 15, §11 |
| `PRIVATE_SIGNAL_BENCHMARK` | B(t) | `"local_mean"` | Benchmark for private signal comparison | §7.1, §11 |
| `LOCAL_CLIMATE_VISIBILITY` | — | `"current_event"` | How neighbours' climate damage enters local benchmark | §7.2 |

### 6.6 Climate / environment

| Parameter | Symbol | Default | Description | PDF ref |
|---|---|---|---|---|
| `T0` | T₀ | `1.0` | Baseline temperature anomaly (°C above pre-industrial) | eq. 6, §11 |
| `CLIMATE_EVENT_RATE` | c | `0.5` | Poisson-saliency rate; p_event = 1 − exp(−c·τ) | eq. 6, §11 |
| `CLIMATE_A` | a | `0.035` | Kalkuhl–Wenz linear damage coefficient | eq. 10, §11 |
| `CLIMATE_B` | b | `0.0009` | Kalkuhl–Wenz quadratic damage coefficient | eq. 10, §11 |
| `CLIMATE_HIT_FRACTION` | f | `0.05` | Fraction of population directly hit per event year | eq. 7, §11 |
| `MEMORY_DELTA_BASE` | δ₀ | `0.5` | Base geometric decay rate of damage memory | eq. 12, §11 |
| `CLIMATE_ELASTICITY_VULNERABILITY` | ε | `0.36` | Gilli et al. income-elastic individual damage exponent | eq. 9, §11 |
| `CLIMATE_ELASTICITY_PERSISTENCE` | ω | `0` | Income-elastic memory persistence exponent; 0 = uniform decay | eq. 12, §11 |
| `CLIMATE_ELASTICITY_EXPOSURE` | μ | `0` | Income-elastic hit-probability exponent; 0 = uniform selection | eq. 7, §11 |

### 6.7 Network formation

| Parameter | Symbol | Default | Description | PDF ref |
|---|---|---|---|---|
| `NETWORK_AVG_DEGREE` | K̄ | `10` | Target average node degree | §5, §11 |
| `NETWORK_BETA_INCOME` | β_inc | `1.0` | Income-homophily strength in weight matrix | eq. 4, §11 |
| `NETWORK_ALGORITHM` | — | `"topk"` | Edge-selection algorithm: `"topk"` or `"sampled"` | §5 |
| `NETWORK_INCOME_NORMALIZATION` | — | `"rank"` | Income normalisation before kernel: `"rank"` or `None` | §5 |
| `NETWORK_RANK_BINS` | — | `N/(3K̄)` | Bins for rank discretisation (reduces degree heterogeneity) | §5 |
| `NETWORK_NOISE_STRENGTH_SD` | — | `0.5` | Gaussian noise σ added to homophily scores before top-K; 0 = deterministic | §5 |
| `NETWORK_WS_REWIRE_P` | p_rewire | `0.2` | Watts–Strogatz per-edge rewiring probability; 0 disables | §5 |
| `NETWORK_WS_TARGET_EXPONENT` | γ | `0.0` | Rewiring target exponent; 0 = uniform WS; > 0 = income-targeted | §5 |

### 6.8 External time-series (loaded at import)

| Parameter | Type | Description |
|---|---|---|
| `REGION_AVG_INCOMES` | `dict[int, float]` | Mean real disposable income per capita by year |
| `GINIS` | `dict[int, float]` | Gini coefficient of income distribution by year |
| `GDP` | `dict[int, float]` | Real GDP by year |
| `INFLATION` | `dict[int, float]` | Inflation rate by year |
| `EMISSIONS` | `dict[int, float]` | Total emissions by year |
| `TEMPERATURE` | `dict[int, float]` | Temperature anomaly (°C) by year |
| `TEMPERATURE_SCENARIO` | `str` | IPCC pathway: `"SSP1-1.9"` / `"SSP1-2.6"` / `"SSP2-4.5"` / `"SSP3-7.0"` / `"SSP5-8.5"` |
| `EXTRAPOLATION_PAST_T` | `str` | Past-temperature treatment: `"raw_obs"` / `"smooth_obs"` / `"forced_resp"` |

Socio-economic series are loaded from `data/raw/italy_data.csv` (scenario: first entry in `ACTIVE_SCENARIO`). Temperature is loaded from `data/raw/temperature_scenarios.csv` (scenario: `TEMPERATURE_SCENARIO`).

---
