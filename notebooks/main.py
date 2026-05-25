# main.py — entry point for the unified two-channel ABM with exogenous
# climate events and geometric memory of damage.
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
plt.ion()

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()
    if not (PROJECT_ROOT / "pyproject.toml").exists() and PROJECT_ROOT.name == "notebooks":
        PROJECT_ROOT = PROJECT_ROOT.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from opinion_dynamics.abm import run_abm, run_all_scenarios
from opinion_dynamics.parameters import (
    ACTIVE_SCENARIO, SAVE_RESULTS, YEARS, PLOT_DIFF_FROM_BASELINE,
    SEED, N_SIMULATIONS, POPULATION_SIZE,
    ALPHA_INIT_STRATEGY, ALPHA_INIT_PARAMS,
    OPINION_W_PRIV, OPINION_NU,
    TEMPERATURE_SCENARIO, CLIMATE_EVENT_RATE, NETWORK_AVG_DEGREE,
)
from opinion_dynamics.stats import print_diagnostics_table
from opinion_dynamics.utils import scenario_slug
from opinion_dynamics.preprocessing import precompute
from opinion_dynamics.plotting import (
    plot_scenario_overview,
    plot_signal_diagnostics,
    plot_channel_contributions,
    plot_multi_scenario_opinion_quintiles,
    plot_multi_scenario_opinion,
    plot_multi_scenario_bc,
)


if __name__ == "__main__":

    figures_dir = PROJECT_ROOT / "results" / "figures"
    data_dir    = PROJECT_ROOT / "results" / "data"
    if SAVE_RESULTS:
        figures_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)

    if len(ACTIVE_SCENARIO) == 1:
        results = run_abm()
        print_diagnostics_table(results)
        suffix = scenario_slug(ACTIVE_SCENARIO[0])
        df_agents_panel = results["df_agents_panel"]

        print("df_agents_panel columns:", df_agents_panel.columns.tolist())

        pre = precompute(df_agents_panel, years=YEARS)

        if SAVE_RESULTS:
            df_agents_panel.to_parquet(data_dir / f"agents_panel_{suffix}.parquet")
            params_snapshot = {
                "scenario": ACTIVE_SCENARIO,
                "seed": SEED,
                "n_simulations": N_SIMULATIONS,
                "population_size": POPULATION_SIZE,
                "years": YEARS,
                "alpha_init_strategy": ALPHA_INIT_STRATEGY,
                "alpha_init_params": ALPHA_INIT_PARAMS,
                "opinion_w_priv": OPINION_W_PRIV,
                "opinion_nu": OPINION_NU,
                "temperature_scenario": TEMPERATURE_SCENARIO,
                "climate_event_rate": CLIMATE_EVENT_RATE,
                "network_avg_degree": NETWORK_AVG_DEGREE,
            }
            (data_dir / f"parameters_{suffix}.json").write_text(
                json.dumps(params_snapshot, indent=2)
            )

        ev = pre.get("event_years", [])
        if ev:
            print(f"[main] Climate event years detected (E(t)=1): {ev}")
        else:
            print("[main] No climate event years detected (CLIMATE_EVENT_YEARS empty?)")

        fig, _ = plot_scenario_overview(
            df_agents_panel, years=YEARS, precomputed=pre, inflations={},
            event_years=ev,
        )
        if SAVE_RESULTS:
            fig.savefig(figures_dir / f"scenario_overview_{suffix}.png", dpi=300)

        fig, _ = plot_signal_diagnostics(
            df_agents_panel, years=YEARS, precomputed=pre,
            event_years=ev,
        )
        if SAVE_RESULTS:
            fig.savefig(figures_dir / f"signal_diagnostics_{suffix}.png", dpi=300)

        fig, _ = plot_channel_contributions(
            df_agents_panel, years=YEARS, precomputed=pre,
            event_years=ev,
        )
        if SAVE_RESULTS:
            fig.savefig(figures_dir / f"channel_contributions_{suffix}.png", dpi=300)

    else:
        multi    = run_all_scenarios(ACTIVE_SCENARIO)
        df_multi = multi["df_agents_panel"]
        ev_multi = multi.get("event_years", [])

        print("Multi-scenario panel shape:", df_multi.shape)
        print("Scenarios present:", df_multi["scenario"].unique().tolist())
        if ev_multi:
            print(f"[main] Climate event years (multi-scenario): {ev_multi}")

        if SAVE_RESULTS:
            df_multi.to_parquet(data_dir / "agents_panel_multi.parquet")
            params_snapshot = {
                "scenarios": ACTIVE_SCENARIO,
                "seed": SEED,
                "n_simulations": N_SIMULATIONS,
                "population_size": POPULATION_SIZE,
                "years": YEARS,
                "alpha_init_strategy": ALPHA_INIT_STRATEGY,
                "alpha_init_params": ALPHA_INIT_PARAMS,
                "opinion_w_priv": OPINION_W_PRIV,
                "opinion_nu": OPINION_NU,
                "temperature_scenario": TEMPERATURE_SCENARIO,
                "climate_event_rate": CLIMATE_EVENT_RATE,
                "network_avg_degree": NETWORK_AVG_DEGREE,
            }
            (data_dir / "parameters_multi.json").write_text(
                json.dumps(params_snapshot, indent=2)
            )

        fig, _ = plot_multi_scenario_opinion(df_multi, years=YEARS,
                                             diff_from_baseline=PLOT_DIFF_FROM_BASELINE,
                                             event_years=ev_multi)
        if SAVE_RESULTS:
            fig.savefig(figures_dir / "comparison_opinion.png", dpi=300)

        fig, _ = plot_multi_scenario_opinion_quintiles(df_multi,
                                                       years=YEARS,
                                                       diff_from_baseline=PLOT_DIFF_FROM_BASELINE,
                                                       event_years=ev_multi,
                                                       spread_indicators=["eta2"]) # spread_indicators=["eta2", "rho_s"] | spread_indicators=["eta2"] | spread_indicators=None
        if SAVE_RESULTS:
            fig.savefig(figures_dir / "comparison_opinion_q.png", dpi=300)
        print("Scenarios in df_multi:", df_multi["scenario"].unique())
        fig, _ = plot_multi_scenario_bc(df_multi, years=YEARS, event_years=ev_multi,
                                 exclude_scenarios=["Baseline"])
        if SAVE_RESULTS:
            fig.savefig(figures_dir / "comparison_bc.png", dpi=300)

    plt.ioff()
    plt.show()