"""Reference-income ABM — unified private signal with exogenous climate
events and geometric income-elastic memory of damage.

Pipeline (year loop)
--------------------
  1. Income assignment: Rank-Preserving Transport (RPT).
  2. Climate-event lookup E(t) from the pre-generated series.
  3. If E(t) = 1: select hit subset S(t) with income-elastic weights
     (parsimony: gamma = epsilon).
  4. Compute D_i(t), delta_i(t), update M_i(t) by geometric recursion.
  5. Build perceived incomes y_i^own(t) = exp(-M_i(t)) * y_i(t).
  6. Compute private signal: sgn{ log[y_i^own(t)/y_i^own(t-1)] - B_i(t) }.
  7. Compute social signal (gap-gated conformity), combine, Bernoulli update.

Architectural notes
-------------------
- M_i is carried across iterations as a separate numpy array (not via
  df_state), parallel to income_tm1 in earlier versions.
- The event series E(t) is generated once before the simulation loop
  (and shared across all simulations and scenarios).
- The third RNG returned by `rngs_for` is used as a dedicated stream
  for hit-agent draws, isolating climate stochasticity from social draws.
"""
import numpy as np
import pandas as pd

from .parameters import (
    DTYPE_INCOME, EMISSIONS, GINIS,
    N_SIMULATIONS, NETWORK_INCOME_NORMALIZATION,
    NETWORK_BETA_INCOME, NETWORK_RANK_BINS,
    NETWORK_NOISE_STRENGTH_SD,
    NETWORK_ALGORITHM, NETWORK_WS_REWIRE_P, NETWORK_WS_TARGET_EXPONENT,
    POPULATION_SIZE, REGION_AVG_INCOMES, REGION_NAMES,
    SEED, NETWORK_AVG_DEGREE, TEMPERATURE, YEARS,
    OPINION_W_PRIV, OPINION_NU,
    CLIMATE_A, CLIMATE_B,
    T0,
    CLIMATE_ELASTICITY_VULNERABILITY,
    CLIMATE_ELASTICITY_PERSISTENCE,
    CLIMATE_ELASTICITY_EXPOSURE,
    PRIVATE_SIGNAL_BENCHMARK,
    CLIMATE_EVENT_RATE,
    CLIMATE_HIT_FRACTION,
    MEMORY_DELTA_BASE,
    LOCAL_CLIMATE_VISIBILITY,
    ALL_INDICATORS,
    GDP,
    ACTIVE_SCENARIO,
    ALPHA_INIT_STRATEGY,
    ALPHA_INIT_PARAMS,
    PLOT_DIFF_FROM_BASELINE,
    BENCHMARK_SCENARIO,
    YEARS_FULL,
    DATA_DIR,
)
from .agents import generate_agents_for_year
from .income import assign_incomes_deterministic
from .network import build_network
from .utilityfun_satisfaction import (
    compute_individual_damage,
    compute_individual_delta,
    compute_perceived_incomes,
    compute_private_signal_unified,
    generate_climate_events,
    compute_damage_base,
    select_hit_agents,
    update_climate_memory,
)
from .utilityfun_opinion_update import (
    compute_social_signal,
    combine_signals_two_channel,
    bernoulli_update,
)
from .alpha_init import initialize_alpha
from .stats import compute_gini, income_assortativity, _make_stats_row
from .utils import rngs_for, _to_state, _attach_outputs

from opinion_dynamics.external_data import load_all_external_timeseries, _gdp_per_capita

# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_abm(
    scenario: str | None = None,
    _timeseries: dict | None = None,
    _events: dict[int, int] | None = None,           
    _D_base_series: dict[int, float] | None = None,  
) -> dict:
    """Run the ABM over all simulations and years for one scenario.

    Parameters
    ----------
    scenario : str or None
        Scenario label attached to all output frames.  When None, falls
        back to ACTIVE_SCENARIO from parameters.py.
    _timeseries : dict or None
        Optional override for scenario-dependent time series; used by
        run_all_scenarios() for multi-scenario runs.

    Returns
    -------
    dict with keys:
        df_simulation_stats : per-(year, sim) summary stats.
        df_agents_panel     : per-(year, sim, agent) panel.
        df_year_summary     : per-year cross-sim summaries.
    """
    if isinstance(scenario, list):
        scenario = scenario[0] if scenario else None
    if scenario is None:
        scenario = ACTIVE_SCENARIO[0] if isinstance(ACTIVE_SCENARIO, list) else ACTIVE_SCENARIO

    if _timeseries is None:
        _years       = YEARS
        _incomes     = REGION_AVG_INCOMES
        _ginis       = GINIS
        _emissions   = EMISSIONS
        _temperature = TEMPERATURE
        _gdp         = GDP
        _all_ind     = ALL_INDICATORS
    else:
        _years       = _timeseries["years"]
        _incomes     = _timeseries["incomes"]
        _ginis       = _timeseries["ginis"]
        _emissions   = _timeseries["emissions"]
        _temperature = _timeseries["temperature"]
        _gdp         = _timeseries["gdp"]
        _all_ind     = _timeseries["all_indicators"]

    # Build data[year]["Italy"] inline — no macroeconomics module required.
    data = {
        y: {"Italy": {
            "population": POPULATION_SIZE,
            "mean_income": float(_incomes[y]),
            "gini":        float(_ginis[y]),
        }}
        for y in _years
    }

    _population = _all_ind.get("Population total", {})
    gdp_pc = _gdp_per_capita(_gdp, _population, _years)

    first_year = _years[0]

    # ------------------------------------------------------------------
    # EXOGENOUS DRIVERS — deterministic, scenario-invariant
    # ------------------------------------------------------------------
    # Pre-loop: T(t) → E(t) ~ Bern(1 − exp(−c·tau(t))) [events, scenario-invariant].
    #           T(t) → D_base(t) = a*tau + b*tau^2      [damage magnitude, KW 2020].
    # The two series are independent: E(t) does not pass through D_base.

    # Climate event series and damage magnitudes are scenario-invariant.
    # Computed once externally (run_all_scenarios) and injected here to
    # guarantee identical realisations across scenarios.
    if _events is None:
        events = generate_climate_events(
            years=_years, temperature=_temperature,
            T0=T0, rate=CLIMATE_EVENT_RATE,
            rng=np.random.default_rng(SEED),
        )
        # print event years for diagnostics  
    else:
        events = _events
    ev = [key for key, value in events.items() if value == 1]
    # print("event years:\n", ev)

    if _D_base_series is None:
        D_base_series = {
            y: compute_damage_base(float(_temperature[y]), T0, a=CLIMATE_A, b=CLIMATE_B)
            for y in _years
        }
    else:
        D_base_series = _D_base_series

    shared_network_kwargs = dict(
        beta_income=NETWORK_BETA_INCOME,
        income_normalization=NETWORK_INCOME_NORMALIZATION,
        rank_bins=NETWORK_RANK_BINS,
        noise_strength_sd=NETWORK_NOISE_STRENGTH_SD,
        algorithm=NETWORK_ALGORITHM,
        ws_rewire_p=NETWORK_WS_REWIRE_P,
        ws_target_exponent=NETWORK_WS_TARGET_EXPONENT,
    )

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    snaps_by_year = {Y: [] for Y in _years}
    stats_by_year = {Y: [] for Y in _years}

    for sim in range(N_SIMULATIONS):
        rng_income, rng_network, _ = rngs_for(first_year, sim, SEED)
        df_agents = generate_agents_for_year(
            year=first_year, data=data, method="exact",
            rng=rng_income, dtype_income=DTYPE_INCOME,
        )
        G, _, _ = build_network(
            df_agents, first_year, avg_degree=NETWORK_AVG_DEGREE,
            rng=rng_network, **shared_network_kwargs,
        )
        avg_degree = float(np.mean([d for _, d in G.degree()]))

        incomes_sorted_idx = np.argsort(df_agents["income"].to_numpy())
        percentiles = np.empty(len(df_agents), dtype=np.float64)
        percentiles[incomes_sorted_idx] = (np.arange(len(df_agents)) + 0.5) / len(df_agents)

        agent_ids = df_agents["agent_id"].to_numpy()
        id_to_pos = {aid: pos for pos, aid in enumerate(agent_ids)}
        neighbor_lists = [
            np.array([id_to_pos[nb] for nb in G.neighbors(aid)], dtype=np.int64)
            for aid in agent_ids
        ]

        s_priv   = np.zeros(len(df_agents), dtype=np.int8)
        p        = np.zeros(len(df_agents), dtype=np.float64)
        M_prev   = np.zeros(len(df_agents), dtype=np.float64)
        hit_zero = np.zeros(len(df_agents), dtype=np.int8)

        alpha = initialize_alpha(
            strategy=ALPHA_INIT_STRATEGY,
            percentiles=percentiles,
            rng=rng_income,
            params=ALPHA_INIT_PARAMS,
        )

        df_agents["alpha"] = alpha
        df_state = _to_state(df_agents)
        income_tm1 = df_agents["income"].to_numpy(dtype=np.float64, copy=True)

        _attach_outputs(
            df_agents,
            alpha=alpha, s_priv=s_priv, p=p,
            M_climate=M_prev, hit=hit_zero,
            year=first_year, sim=sim,
        )
        snaps_by_year[first_year].append(df_agents.copy())
        _r_assort = income_assortativity(neighbor_lists, income_tm1)
        stats_by_year[first_year].append(
            _make_stats_row(
                first_year, sim, alpha, df_agents, avg_degree=avg_degree,
                income_assortativity_r=_r_assort,
                M_climate=M_prev, hit=hit_zero, event_flag=events.get(first_year, 0),
            )
        )

        for _, Y in enumerate(_years[1:]):
            rng_income, _, rng_hits = rngs_for(Y, sim, SEED)
            df_agents = assign_incomes_deterministic(
                df_state, Y, data, rng=rng_income, dtype_income=DTYPE_INCOME,
            )
            income_t = df_agents["income"].to_numpy(dtype=np.float64, copy=True)

            # GDP per capita growth (consumed only by benchmark="gdp").
            pc_t   = gdp_pc.get(Y)
            pc_tm1 = gdp_pc.get(Y - 1)
            if pc_t is not None and pc_tm1 is not None and pc_tm1 != 0.0:
                gdp_growth_t = (pc_t - pc_tm1) / pc_tm1
            else:
                gdp_growth_t = 0.0

            D_base_t = D_base_series[Y]

            # --- Climate channel ---
            E_t = events.get(Y, 0)
            if E_t == 1:
                hit_t = select_hit_agents(
                    income_t,
                    fraction=CLIMATE_HIT_FRACTION,
                    gamma=CLIMATE_ELASTICITY_EXPOSURE,
                    rng=rng_hits,
                )
            else:
                hit_t = np.zeros(len(df_agents), dtype=np.int8)

            D_i_t = compute_individual_damage(
                D_base_t, income_t,
                epsilon=CLIMATE_ELASTICITY_VULNERABILITY,
            )
            delta_i_t = compute_individual_delta(
                income_t,
                delta_base=MEMORY_DELTA_BASE,
                epsilon=CLIMATE_ELASTICITY_PERSISTENCE,
            )
            M_t = update_climate_memory(
                M_prev=M_prev,
                hit=hit_t,
                D_i=D_i_t,
                delta_i=delta_i_t
            )

            y_own_t   = compute_perceived_incomes(income_t,   M_t)
            y_own_tm1 = compute_perceived_incomes(income_tm1, M_prev)

            # --- Unified private signal ---
            s_priv = compute_private_signal_unified(
                y_own_t=y_own_t,
                y_own_tm1=y_own_tm1,
                income_t=income_t,
                income_tm1=income_tm1,
                benchmark=PRIVATE_SIGNAL_BENCHMARK,
                gdp_growth_t=gdp_growth_t,
                neighbor_lists=neighbor_lists,
                local_climate_visibility=LOCAL_CLIMATE_VISIBILITY,
                hit_t=hit_t,
                D_i_t=D_i_t,
            )

            # --- Social signal (gap-gated conformity) ---
            xi_soc = compute_social_signal(alpha, neighbor_lists, rng_income)

            # --- Flat two-channel mix + Bernoulli update ---
            p     = combine_signals_two_channel(s_priv, xi_soc, OPINION_W_PRIV)
            alpha = bernoulli_update(alpha, p, OPINION_NU, rng_income)

            income_tm1 = income_t.copy()
            M_prev = M_t.copy()
            df_agents["alpha"] = alpha
            df_state = _to_state(df_agents)

            _attach_outputs(
                df_agents,
                alpha=alpha, s_priv=s_priv, p=p,
                M_climate=M_t, hit=hit_t,
                year=Y, sim=sim,
            )

            snaps_by_year[Y].append(df_agents.copy())
            _r_assort = income_assortativity(neighbor_lists, income_t)
            stats_by_year[Y].append(
                _make_stats_row(
                    Y, sim, alpha, df_agents, avg_degree=avg_degree,
                    income_assortativity_r=_r_assort,
                    M_climate=M_t, hit=hit_t, event_flag=E_t,
                )
            )

        print(f"[run_abm] Simulation {sim + 1}/{N_SIMULATIONS}")

    # Concatenate per-year snapshots into a single panel.
    agents_snapshots = []
    all_stats = []
    for Y in _years:
        agents_snapshots.extend(snaps_by_year[Y])
        all_stats.extend(stats_by_year[Y])

    df_simulation_stats = pd.DataFrame(all_stats).sort_values(["year", "simulation_id"])
    df_agents_panel = (
        pd.concat(agents_snapshots, ignore_index=True)
        if agents_snapshots
        else pd.DataFrame(columns=[
            "agent_id", "region", "income", "alpha",
            "s_priv", "p", "M_climate", "climate_hit",
            "year", "simulation_id",
        ])
    )
    df_year_summary = df_simulation_stats.groupby("year", as_index=False).agg(
        n_sims=("simulation_id", "nunique"),
        S_mean=("S", "mean"),
        S_sd=("S", "std"),
        S_std_mean=("S_std", "mean"),
        alpha_q25_mean=("alpha_q25", "mean"),
        alpha_q75_mean=("alpha_q75", "mean"),
        mean_income_mean=("mean_income", "mean"),
        gini_mean=("gini", "mean"),
        avg_degree_mean=("avg_degree", "mean"),
        income_assortativity_mean=("income_assortativity", "mean"),
        M_mean=("M_mean", "mean"),
        hit_frac_mean=("hit_frac", "mean"),
        event_flag=("event_flag", "max"),
    )

    for df in (df_simulation_stats, df_agents_panel, df_year_summary):
        df.insert(0, "scenario", scenario)

    return {
        "df_simulation_stats": df_simulation_stats,
        "df_agents_panel":     df_agents_panel,
        "df_year_summary":     df_year_summary,
    }


def run_all_scenarios(
    scenarios: list[str] = ACTIVE_SCENARIO,
) -> dict[str, pd.DataFrame]:
    """Run the ABM for each scenario and concatenate output DataFrames.

    Each scenario's macroeconomic time series are loaded explicitly via
    ``_timeseries``, avoiding any mutation of module-level state.  The
    climate event series is identical across scenarios.

    Parameters
    ----------
    scenarios : list[str]
        Scenario labels to run (default: ``ACTIVE_SCENARIO`` from parameters.py).

    Returns
    -------
    dict with keys ``"df_agents_panel"``, ``"df_simulation_stats"``,
    ``"df_year_summary"`` — each a concatenation over all scenarios.
    """

    # When diff mode is active, Baseline must be run regardless of whether it
    # appears in the caller-supplied scenario list.
    if PLOT_DIFF_FROM_BASELINE and BENCHMARK_SCENARIO not in list(scenarios):
        scenarios = [BENCHMARK_SCENARIO] + list(scenarios)

    panels:    list[pd.DataFrame] = []
    stats:     list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []

    # Compute scenario-invariant climate series once before the scenario loop.
    _shared_events = generate_climate_events(
        years=YEARS_FULL,
        temperature=TEMPERATURE,
        T0=T0,
        rate=CLIMATE_EVENT_RATE,
        rng=np.random.default_rng(SEED),
    )
    _shared_D_base = {
        y: compute_damage_base(
            float(TEMPERATURE[y]), T0,
            a=CLIMATE_A, b=CLIMATE_B,
        )
        for y in YEARS_FULL
    }

    for scen in scenarios:
        print(f"\n[run_all_scenarios] Starting scenario: {scen}")

        years, incomes, ginis, emissions, gdp, inflation, all_indicators = (
            load_all_external_timeseries(
                data_dir=DATA_DIR,
                base_years=YEARS_FULL,
                scenario=scen,
            )
        )
        timeseries = {
            "years":          years,
            "incomes":        incomes,
            "ginis":          ginis,
            "emissions":      emissions,
            "temperature":    TEMPERATURE,
            "gdp":            gdp,
            "all_indicators": all_indicators,
        }

        results = run_abm(
            scenario=scen,
            _timeseries=timeseries,
            _events=_shared_events,          # ← aggiungere
            _D_base_series=_shared_D_base,   # ← aggiungere
        )
        panels.append(results["df_agents_panel"])
        stats.append(results["df_simulation_stats"])
        summaries.append(results["df_year_summary"])

    _event_years = [y for y in YEARS_FULL if _shared_events.get(y, 0) == 1]

    return {
        "df_agents_panel":     pd.concat(panels,    ignore_index=True),
        "df_simulation_stats": pd.concat(stats,     ignore_index=True),
        "df_year_summary":     pd.concat(summaries, ignore_index=True),
        "event_years":         _event_years,
    }