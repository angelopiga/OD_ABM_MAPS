"""Pre-aggregation of agent-panel data for downstream plotting.

Overview
--------
This module provides a single public function, ``precompute``, which
transforms the raw per-agent simulation panel produced by ``run_abm`` /
``run_all_scenarios`` into a collection of ready-to-plot summary frames.
All aggregations are performed **once** and cached in a dictionary, so
that every plotting function in ``plotting.py`` can consume pre-computed
results instead of recomputing them independently.

Aggregation pipeline
--------------------
1. **Filter & label** — restrict the panel to the requested years and
   assign each agent to an income quintile (or n-quantile) via
   ``plot_utils._assign_income_quintiles``.
2. **Opinion dynamics** — compute mean ``alpha`` (climate-concern index)
   and mean combined signal ``p`` per (year, quintile, simulation), then
   average across simulations.
3. **Private signal** — compute mean ``s_priv`` per (year, quintile) and
   the fractions of +1 / 0 / −1 signals per year.
4. **Climate damage channel** — derive instantaneous individual damage
   ``D_i(t)`` from the Kalkuhl-Wenz (2020) damage function and aggregate
   per (year, quintile).
5. **Climate memory channel** — aggregate the geometric memory stock
   ``M_i(t)`` and the implied cumulative perceived damage
   ``1 − exp(−M_i)`` per (year, quintile).
6. **Hit rate** — aggregate the fraction of climate-hit agents per
   (year, quintile).
7. **Event years** — derive the list of years with at least one active
   climate event (E(t) = 1) from the ``climate_hit`` indicator.

Dependencies
------------
- ``plot_utils.py``       — quintile assignment helper.
- ``utilityfun_satisfaction.py`` — ``compute_damage_base``,
  ``compute_individual_damage``.
- ``parameters.py``       — ``CLIMATE_A``, ``CLIMATE_B``, ``T0``,
  ``TEMPERATURE``, ``CLIMATE_ELASTICITY_VULNERABILITY``.

Typical usage
-------------
>>> cache = precompute(df_agents_panel, years=list(range(2000, 2051)))
>>> plot_opinion_dynamics(cache, ...)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from opinion_dynamics.plot_utils import _assign_income_quintiles

from opinion_dynamics.utilityfun_satisfaction import (
    compute_individual_damage, compute_damage_base,
)
from opinion_dynamics.parameters import (
    CLIMATE_A, CLIMATE_B, T0, TEMPERATURE,
    CLIMATE_ELASTICITY_VULNERABILITY,
)


def precompute(
    df_agents_panel: pd.DataFrame,
    years: list[int],
    n_quantiles: int = 5,
) -> dict:
    """Compute all shared summary frames from df_agents_panel exactly once.

    Parameters
    ----------
    df_agents_panel : pd.DataFrame
        Full agent panel produced by ``run_abm``.
    years : list[int]
        Years to include in the aggregations.
    n_quantiles : int, optional
        Number of income quantiles (default 5 = quintiles).

    Returns
    -------
    dict with keys:
        df_filt              — filtered + quintile-labelled panel.
        agg_year_alpha       — mean alpha per year across simulations.
        agg_year_q_alpha     — mean alpha per (year, quintile).
        agg_year_q_srel      — mean s_priv per (year, quintile).
        agg_year_q_p         — mean combined signal p per (year, quintile).
        spriv_fractions      — fractions of s_priv = +1/0/-1 per year.
        agg_year_q_Di        — mean instantaneous damage D_i per (year, quintile).
        agg_year_q_M         — mean memory stock M_i per (year, quintile).
        agg_year_q_tildeD    — mean cumulative perceived damage per (year, quintile).
        agg_year_q_hit       — mean climate-hit rate per (year, quintile).
        event_years          — years with E(t)=1 (active climate events).
    """


    df = df_agents_panel[df_agents_panel["year"].isin(years)].copy()
    df["quintile"] = _assign_income_quintiles(df, n_quantiles)

    def _q_agg(col, name):
        per_sim = (
            df.groupby(["year", "simulation_id", "quintile"])[col]
            .mean()
            .reset_index(name=name)
        )
        return (
            per_sim.groupby(["year", "quintile"])[name]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
        )

    # --- alpha ---
    per_sim_alpha = (
        df.groupby(["year", "simulation_id", "quintile"])["alpha"]
        .mean()
        .reset_index(name="alpha_mean")
    )
    agg_year_q_alpha = (
        per_sim_alpha.groupby(["year", "quintile"])["alpha_mean"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    agg_year_alpha = (
        per_sim_alpha.groupby(["year", "simulation_id"])["alpha_mean"]
        .mean()
        .reset_index(name="alpha_mean")
        .groupby("year")["alpha_mean"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )

    # --- s_priv and p ---
    agg_year_q_srel = _q_agg("s_priv", "s_priv_mean")
    agg_year_q_p    = _q_agg("p",     "p_mean")

    # --- s_priv fractions {+1, 0, -1} per year ---
    frac_records = []
    for y in years:
        vals = df.loc[df["year"] == y, "s_priv"].values
        n = len(vals) if len(vals) else 1
        frac_records.append({
            "year":      y,
            "frac_pos":  float((vals > 0).sum() / n),
            "frac_zero": float((vals == 0).sum() / n),
            "frac_neg":  float((vals < 0).sum() / n),
        })
    spriv_fractions = pd.DataFrame(frac_records)

    # --- Instantaneous individual damage D_i per quintile ---
    Di_col = np.zeros(len(df), dtype=np.float64)
    for y in years:
        T_y    = float(TEMPERATURE[y])
        D_base = compute_damage_base(T_y, T0, a=CLIMATE_A, b=CLIMATE_B)
        mask   = df["year"] == y
        inc    = df.loc[mask, "income"].values.astype(np.float64)
        Di_col[mask.values] = compute_individual_damage(
            D_base, inc,
            epsilon=CLIMATE_ELASTICITY_VULNERABILITY,            
        )
    df["_Di"] = Di_col

    per_sim_di = (
        df.groupby(["year", "simulation_id", "quintile"])["_Di"]
        .mean().reset_index(name="Di_mean")
    )
    agg_year_q_Di = (
        per_sim_di.groupby(["year", "quintile"])["Di_mean"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )

    # --- Memory stock M_i per quintile ---
    if "M_climate" in df.columns:
        agg_year_q_M = _q_agg("M_climate", "M_mean")
        df["_tildeD"] = 1.0 - np.exp(-df["M_climate"].values)
        agg_year_q_tildeD = _q_agg("_tildeD", "tildeD_mean")
    else:
        agg_year_q_M      = pd.DataFrame()
        agg_year_q_tildeD = pd.DataFrame()

    # --- Climate-hit rate per quintile ---
    if "climate_hit" in df.columns:
        agg_year_q_hit = _q_agg("climate_hit", "hit_rate")
    else:
        agg_year_q_hit = pd.DataFrame()

    # --- Years with active climate events ---
    event_years: list[int] = []
    if "climate_hit" in df_agents_panel.columns:
        per_year_any_hit = (
            df_agents_panel.groupby("year")["climate_hit"].max().reset_index()
        )
        event_years = per_year_any_hit.loc[
            per_year_any_hit["climate_hit"] > 0, "year"
        ].tolist()

    return {
        "df_filt":           df,
        "agg_year_alpha":    agg_year_alpha,
        "agg_year_q_alpha":  agg_year_q_alpha,
        "agg_year_q_srel":   agg_year_q_srel,
        "agg_year_q_p":      agg_year_q_p,
        "spriv_fractions":   spriv_fractions,
        "agg_year_q_Di":     agg_year_q_Di,
        "agg_year_q_M":      agg_year_q_M,
        "agg_year_q_tildeD": agg_year_q_tildeD,
        "agg_year_q_hit":    agg_year_q_hit,
        "event_years":       event_years,
    }
