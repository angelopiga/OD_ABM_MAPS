"""plotting.py — plotting functions for the unified two-channel architecture.

Public functions
----------------
plot_scenario_overview      — macro drivers + opinion by quintile
plot_signal_diagnostics     — opinion heatmap + s_priv fractions + D_i by quintile
plot_channel_contributions  — push_priv + push_soc (residual), annual and cumulative
plot_multi_scenario_opinion_quintiles — multi-scenario opinion by quintile
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

from opinion_dynamics.plot_utils import (
    _UNSET,
    _resolve_bands,
    _band_err,
    _title,
    _year_ticks,
    _apply_year_axes,
    _assign_income_quintiles,
    _scenario_color_map,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_event_vlines(axes, event_years, *, color="#888888", alpha=0.35, linewidth=0.9, zorder=0):
    """Draw a thin vertical line at each climate-event year on every axis in *axes*."""
    if not event_years:
        return
    for ax in axes:
        for yr in event_years:
            ax.axvline(yr, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)


# ---------------------------------------------------------------------------
# plot_scenario_overview
# ---------------------------------------------------------------------------

def plot_scenario_overview(
    df_agents_panel: pd.DataFrame,
    *,
    avg_incomes: dict | None = None,
    ginis: dict | None = None,
    temperatures: dict | None = None,
    gdps: dict | None = None,
    inflations: dict | None = None,
    n_quantiles: int = 5,
    years: list[int] | None = None,
    bands=_UNSET,
    figsize: tuple[float, float] = (9, 6.5),
    title: str = "Scenario overview",
    precomputed: dict | None = None,
    event_years: list[int] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """Macro drivers (top) and mean opinion by income quintile (bottom)."""
    bands = _resolve_bands(bands)

    if years is None:
        years = sorted(df_agents_panel["year"].unique())

    if event_years is None and precomputed is not None:
        event_years = precomputed.get("event_years", [])

    try:
        from opinion_dynamics.parameters import REGION_AVG_INCOMES, GINIS, TEMPERATURE, GDP, INFLATION
        if avg_incomes  is None: avg_incomes  = REGION_AVG_INCOMES
        if ginis        is None: ginis        = GINIS
        if temperatures is None: temperatures = TEMPERATURE
        if gdps         is None: gdps         = GDP
        if inflations   is None: inflations   = INFLATION
    except Exception:
        pass

    def _pct_change(d):
        if d is None:
            return None
        s = pd.Series(
            {int(k): float(np.nanmean(v)) if hasattr(v, "__len__") else float(v)
             for k, v in d.items()}
        ).sort_index()
        s = s[s.index.isin(years)]
        base = float(s.iloc[0]) if len(s) and s.iloc[0] != 0 else 1.0
        return (s / base - 1) * 100

    def _as_pct_level(d):
        if d is None:
            return None
        s = pd.Series(
            {int(k): float(np.nanmean(v)) if hasattr(v, "__len__") else float(v)
             for k, v in d.items()}
        ).sort_index()
        s = s[s.index.isin(years)]
        return s * 100

    def _as_abs(d):
        """Return a Series of absolute values for the years in scope."""
        if d is None:
            return None
        s = pd.Series(
            {int(k): float(np.nanmean(v)) if hasattr(v, "__len__") else float(v)
             for k, v in d.items()}
        ).sort_index()
        return s[s.index.isin(years)]

    # Temperature is plotted separately on a right y-axis with absolute °C values.
    temp_abs = _as_abs(temperatures)

    macro = {
        r"Avg. income": (_pct_change(avg_incomes),  "#1b7837"),
        r"Gini":        (_pct_change(ginis),         "#d73027"),
        r"GDP real":    (_pct_change(gdps),           "#7b2d8b"),
    }
    inflation_s    = _as_pct_level(inflations)
    inflation_color = "#e08214"
    temp_color = "#4575b4"

    if precomputed is not None:
        stats = precomputed["agg_year_q_alpha"]
        agg   = precomputed["agg_year_alpha"]
    else:
        df = df_agents_panel[df_agents_panel["year"].isin(years)].copy()
        df["quintile"] = _assign_income_quintiles(df, n_quantiles)
        per_sim = (
            df.groupby(["year", "simulation_id", "quintile"])["alpha"]
            .mean().reset_index(name="alpha_mean")
        )
        stats = (
            per_sim.groupby(["year", "quintile"])["alpha_mean"]
            .agg(mean="mean", std="std", count="count").reset_index()
        )
        agg_per_sim = (
            df.groupby(["year", "simulation_id"])["alpha"]
            .mean().reset_index(name="alpha_mean")
        )
        agg = (
            agg_per_sim.groupby("year")["alpha_mean"]
            .agg(mean="mean", std="std", count="count").reset_index()
        )

    colors = plt.cm.coolwarm(np.linspace(0.05, 0.95, n_quantiles))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [1, 2.5], "hspace": 0.22},
    )
    fig.suptitle(_title(title), fontsize=12)

    for label, (s, c) in macro.items():
        if s is not None and len(s):
            ax1.plot(s.index, s.values, label=label, color=c, linewidth=1.6)
    ax1.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax1.set_ylabel(r"$\Delta\,(\%\ \mathrm{vs}\ t_0)$", fontsize=9)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    # Temperature on separate right y-axis (absolute °C).
    ax1r = ax1.twinx()
    if temp_abs is not None and len(temp_abs):
        ax1r.plot(temp_abs.index, temp_abs.values,
                  label="Temperature (°C)", color=temp_color,
                  linewidth=1.6, linestyle="-")
    ax1r.set_ylabel("Temperature (°C)", fontsize=8, color=temp_color)
    ax1r.tick_params(axis="y", labelcolor=temp_color, labelsize=7)

    if inflation_s is not None and len(inflation_s):
        ax1r.plot(inflation_s.index, inflation_s.values,
                  label=r"Inflation (pp)", color=inflation_color,
                  linewidth=1.6, linestyle="--")

    _apply_year_axes((ax1, ax2), years, fontsize=10)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2,
               fontsize=7.5, loc="best", framealpha=0.7, ncol=5)

    for q in range(1, n_quantiles + 1):
        s = stats[stats["quintile"] == q].sort_values("year")
        label = (f"$Q_1$ (poorest)" if q == 1
                 else f"$Q_{n_quantiles}$ (richest)" if q == n_quantiles
                 else f"$Q_{q}$")
        color = colors[q - 1]
        ax2.plot(s["year"], s["mean"], label=label, color=color, linewidth=1.6)
        err = _band_err(s["std"], s["count"], bands)
        if err is not None:
            ax2.fill_between(s["year"], s["mean"] - err, s["mean"] + err,
                             color=color, alpha=0.15)

    ax2.plot(agg["year"], agg["mean"],
             label=r"Aggregate $\bar{\alpha}$", color="black",
             linestyle="--", linewidth=1.6, zorder=5)
    agg_err = _band_err(agg["std"], agg["count"], bands)
    if agg_err is not None:
        ax2.fill_between(agg["year"], agg["mean"] - agg_err, agg["mean"] + agg_err,
                         color="black", alpha=0.10, zorder=4)

    ax2.axhline(0.5, color="grey", linestyle="--", linewidth=0.8,
                label=r"Neutral ($\alpha = 0.5$)")
    ax2.set_ylabel(r"Mean opinion $\alpha$", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.legend(fontsize=8, loc="best", framealpha=0.7)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)

    _add_event_vlines((ax1, ax2), event_years or [])

    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# plot_signal_diagnostics
# ---------------------------------------------------------------------------

def plot_signal_diagnostics(
    df_agents_panel: pd.DataFrame,
    *,
    n_quantiles: int = 5,
    years: list[int] | None = None,
    n_alpha_bins: int = 50,
    bands=_UNSET,
    figsize: tuple[float, float] = (10, 9),
    title: str = "Signal diagnostics",
    precomputed: dict | None = None,
    event_years: list[int] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """Three-panel diagnostic: opinion heatmap, s_priv fractions, D_i by quintile.

    Panel 1 — Opinion heatmap: full cross-sectional distribution of alpha
    over time.  Reveals bimodality, polarization, and convergence patterns
    that quintile means cannot capture.

    Panel 2 — Stacked area of s_priv fractions: for each year, the share
    of agents receiving s_priv = +1, 0, -1.  Shows the balance of positive
    vs negative private signals and the prevalence of zero signals.

    Panel 3 — Mean individual damage D_i by income quintile: shows the
    income-elastic differentiation of climate damage.  D_i = D_base *
    (y_bar / y_i)^epsilon; poor agents (Q1) suffer amplified damage.
    """
    bands = _resolve_bands(bands)

    if years is None:
        years = sorted(df_agents_panel["year"].unique())

    if event_years is None and precomputed is not None:
        event_years = precomputed.get("event_years", [])

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [2, 1.5, 1.5], "hspace": 0.25},
    )
    fig.suptitle(_title(title), fontsize=12)

    # ---- Panel 1: Opinion heatmap ----
    df_filt = df_agents_panel[df_agents_panel["year"].isin(years)]
    alpha_bins = np.linspace(0.0, 1.0, n_alpha_bins + 1)
    bin_centers = 0.5 * (alpha_bins[:-1] + alpha_bins[1:])
    year_arr = np.array(years)

    density = np.zeros((n_alpha_bins, len(years)))
    for j, y in enumerate(years):
        vals = df_filt.loc[df_filt["year"] == y, "alpha"].values
        if len(vals):
            counts, _ = np.histogram(vals, bins=alpha_bins)
            density[:, j] = counts / counts.sum()

    im = ax1.pcolormesh(
        year_arr, bin_centers, density,
        cmap="YlOrRd", shading="nearest", rasterized=True,
    )
    # overlay aggregate mean
    agg_alpha = df_filt.groupby("year")["alpha"].mean()
    ax1.plot(agg_alpha.index, agg_alpha.values,
             color="black", linewidth=1.5, linestyle="-", label=r"$\bar{\alpha}$")
    ax1.set_ylabel(r"$\alpha$", fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.7)
    # fraction=0.015 keeps the colorbar narrow so it does not shrink ax1
    plt.colorbar(im, ax=ax1, fraction=0.015, pad=0.01, aspect=30, label="Density")

    # ---- Panel 2: Stacked area of s_priv fractions ----
    if precomputed is not None:
        frac_df = precomputed["spriv_fractions"]
    else:
        records = []
        for y in years:
            vals = df_filt.loc[df_filt["year"] == y, "s_rel"].values
            n = len(vals) if len(vals) else 1
            records.append({
                "year": y,
                "frac_pos":  (vals > 0).sum() / n,
                "frac_zero": (vals == 0).sum() / n,
                "frac_neg":  (vals < 0).sum() / n,
            })
        frac_df = pd.DataFrame(records)

    yy = frac_df["year"].values
    ax2.fill_between(yy, 0, frac_df["frac_pos"],
                     color="#2166ac", alpha=0.7, label=r"$s^{\mathrm{priv}} = +1$")
    ax2.fill_between(yy, frac_df["frac_pos"],
                     frac_df["frac_pos"] + frac_df["frac_zero"],
                     color="#d9d9d9", alpha=0.7, label=r"$s^{\mathrm{priv}} = 0$")
    ax2.fill_between(yy, frac_df["frac_pos"] + frac_df["frac_zero"], 1.0,
                     color="#b2182b", alpha=0.7, label=r"$s^{\mathrm{priv}} = -1$")
    ax2.set_ylabel("Fraction of agents", fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax2.legend(fontsize=8, loc="center right", framealpha=0.7)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)

    # ---- Panel 3: Mean memory M_i(t) by income quintile ----
    # M_climate is the geometric memory stock (state variable of the climate
    # channel).  Peaks at event years, decays geometrically between events;
    # income-elastic persistence makes Q1 (poorest) recover more slowly.
    colors = plt.cm.coolwarm(np.linspace(0.05, 0.95, n_quantiles))

    if precomputed is not None and "agg_year_q_M" in precomputed:
        stats_M = precomputed["agg_year_q_M"]
    else:
        df_tmp = df_filt.copy()
        df_tmp["quintile"] = _assign_income_quintiles(df_tmp, n_quantiles)
        per_sim_M = (
            df_tmp.groupby(["year", "simulation_id", "quintile"])["M_climate"]
            .mean().reset_index(name="M_mean")
        )
        stats_M = (
            per_sim_M.groupby(["year", "quintile"])["M_mean"]
            .agg(mean="mean", std="std", count="count").reset_index()
        )

    for q in range(1, n_quantiles + 1):
        s = stats_M[stats_M["quintile"] == q].sort_values("year")
        label = "Q1 (poorest)" if q == 1 else f"Q{q} (richest)" if q == n_quantiles else f"Q{q}"
        ax3.plot(s["year"], s["mean"], label=label, color=colors[q - 1], linewidth=1.6)
        err = _band_err(s["std"], s["count"], bands)
        if err is not None:
            ax3.fill_between(s["year"], s["mean"] - err, s["mean"] + err,
                             color=colors[q - 1], alpha=0.12)

    ax3.set_ylabel(r"Mean memory $\bar{M}_i$", fontsize=10)
    ax3.set_ylim(bottom=0)
    ax3.legend(fontsize=8, loc="best", framealpha=0.7)
    ax3.grid(axis="y", linestyle=":", alpha=0.4)

    _apply_year_axes((ax1, ax2, ax3), years, fontsize=10)
    _add_event_vlines((ax1, ax2, ax3), event_years or [])
    fig.tight_layout()
    return fig, (ax1, ax2, ax3)


# ---------------------------------------------------------------------------
# plot_channel_contributions
# ---------------------------------------------------------------------------

def plot_channel_contributions(
    df_agents_panel: pd.DataFrame,
    *,
    n_quantiles: int = 5,
    years: list[int] | None = None,
    w_priv: float | None = None,
    nu: float | None = None,
    figsize: tuple[float, float] = (9, 6.5),
    title: str = "Channel contributions to opinion change",
    precomputed: dict | None = None,
    event_years: list[int] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Annual push (top) and cumulative push (bottom) by channel and quintile.

    Private push:  push_priv(q, t) = nu * w_priv * <s_priv>_q(t)
    Social push:   push_soc(q, t) = <Delta_alpha>_q(t) - push_priv(q, t)

    The cumulative panel shows the running sum of each push from t_0.
    By construction, cumulative push_priv + push_soc = alpha_q(T) - alpha_q(0):
    it decomposes the total opinion shift into the contribution of each channel.
    """
    if years is None:
        years = sorted(df_agents_panel["year"].unique())

    if event_years is None and precomputed is not None:
        event_years = precomputed.get("event_years", [])

    try:
        from opinion_dynamics.parameters import OPINION_W_PRIV, OPINION_NU
        if w_priv is None: w_priv = OPINION_W_PRIV
        if nu     is None: nu     = OPINION_NU
    except Exception:
        pass
    if w_priv is None: w_priv = 0.5
    if nu     is None: nu     = 0.05

    scale_priv = nu * w_priv

    # --- private push ---
    if precomputed is not None:
        df_filt = precomputed["df_filt"]
        src = precomputed["agg_year_q_srel"][["year", "quintile", "mean"]].copy()
        src["push"] = src["mean"] * scale_priv
        src = src.sort_values(["quintile", "year"])
        src["cumsum"] = src.groupby("quintile")["push"].cumsum()
        push_priv = src
    else:
        df_filt = df_agents_panel[df_agents_panel["year"].isin(years)].copy()
        df_filt["quintile"] = _assign_income_quintiles(df_filt, n_quantiles)
        per_sim = (
            df_filt.groupby(["year", "simulation_id", "quintile"])["s_rel"]
            .mean().reset_index(name="val")
        )
        agg = (
            per_sim.groupby(["year", "quintile"])["val"]
            .mean().reset_index(name="push")
        )
        agg["push"] *= scale_priv
        agg = agg.sort_values(["quintile", "year"])
        agg["cumsum"] = agg.groupby("quintile")["push"].cumsum()
        push_priv = agg

    # --- social push: residual from realized Delta_alpha ---
    df_sorted = df_filt.sort_values(["simulation_id", "agent_id", "year"])
    df_sorted["delta_alpha"] = df_sorted.groupby(
        ["simulation_id", "agent_id"]
    )["alpha"].diff()
    df_delta = df_sorted.dropna(subset=["delta_alpha"])

    realized = (
        df_delta.groupby(["year", "simulation_id", "quintile"])["delta_alpha"]
        .mean().reset_index(name="val")
        .groupby(["year", "quintile"])["val"].mean()
        .reset_index(name="push")
    )
    realized = realized.sort_values(["quintile", "year"])

    push_soc = realized.merge(
        push_priv[["year", "quintile", "push"]].rename(columns={"push": "p_priv"}),
        on=["year", "quintile"],
    )
    push_soc["push"] = push_soc["push"] - push_soc["p_priv"]
    push_soc = push_soc.sort_values(["quintile", "year"])
    push_soc["cumsum"] = push_soc.groupby("quintile")["push"].cumsum()

    # --- plot ---
    colors = plt.cm.coolwarm(np.linspace(0.05, 0.95, n_quantiles))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"hspace": 0.22},
    )
    fig.suptitle(_title(title), fontsize=12)

    for q in range(1, n_quantiles + 1):
        c = colors[q - 1]
        label = ("$Q_1$ (poorest)" if q == 1
                 else f"$Q_{n_quantiles}$ (richest)" if q == n_quantiles
                 else f"$Q_{q}$")
        r  = push_priv[push_priv["quintile"] == q].sort_values("year")
        so = push_soc [push_soc ["quintile"] == q].sort_values("year")

        ax1.plot(r["year"],  r["push"],  color=c, linewidth=1.6, linestyle="-",  label=label)
        ax1.plot(so["year"], so["push"], color=c, linewidth=0.7, linestyle="-")
        ax2.plot(r["year"],  r["cumsum"],  color=c, linewidth=1.6, linestyle="-")
        ax2.plot(so["year"], so["cumsum"], color=c, linewidth=0.7, linestyle="-")

    ax1.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax1.set_ylabel(r"$\Delta\alpha$ / year", fontsize=10)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    ax2.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    _apply_year_axes((ax1, ax2), years, fontsize=10)
    ax2.set_ylabel(r"Cumulative $\sum \Delta\alpha$", fontsize=10)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)

    _add_event_vlines((ax1, ax2), event_years or [])

    channel_handles = [
        Line2D([0], [0], color="grey", linestyle="-",  linewidth=1.4,
               label=r"$s^{\mathrm{priv}}$ (private)"),
        Line2D([0], [0], color="grey", linestyle="-",  linewidth=0.7,
               label=r"$\xi^{\mathrm{soc}}$ (social, residual)"),
    ]
    q_handles, q_labels = ax1.get_legend_handles_labels()
    ax1.legend(
        q_handles + channel_handles,
        q_labels + [h.get_label() for h in channel_handles],
        fontsize=7.5, loc="lower left", framealpha=0.7, ncol=2,
    )
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# Spread indicator helpers (η² and ρ_S)
# ---------------------------------------------------------------------------

def _compute_eta2_per_sim(
    df_scen: pd.DataFrame,
    years: list[int],
    n_quantiles: int,
) -> pd.DataFrame:
    df_filt = df_scen[df_scen["year"].isin(years)]

    # grand mean and SS_total per (year, sim)
    grand = (
        df_filt.groupby(["year", "simulation_id"])["alpha"]
        .agg(alpha_mean="mean",
             ss_total=lambda x: float(np.sum((x.values - x.values.mean()) ** 2)))
        .reset_index()
    )

    # quintile mean and count per (year, sim, quintile)
    q_stats = (
        df_filt.groupby(["year", "simulation_id", "quintile"])["alpha"]
        .agg(alpha_q_mean="mean", n_q="count")
        .reset_index()
    )

    merged = q_stats.merge(grand[["year", "simulation_id", "alpha_mean"]],
                           on=["year", "simulation_id"])
    merged["ss_q"] = merged["n_q"] * (merged["alpha_q_mean"] - merged["alpha_mean"]) ** 2

    ss_between = (
        merged.groupby(["year", "simulation_id"])["ss_q"]
        .sum().reset_index(name="ss_between")
    )
    result = ss_between.merge(
        grand[["year", "simulation_id", "ss_total"]], on=["year", "simulation_id"]
    )
    result["eta2"] = np.where(
        result["ss_total"] > 0,
        result["ss_between"] / result["ss_total"],
        np.nan,
    )
    return result[["year", "simulation_id", "eta2"]]


def _compute_rho_s_per_sim(
    df_scen: pd.DataFrame,
    years: list[int],
) -> pd.DataFrame:
    """Compute Spearman rank correlation ρ_S(t) between income and alpha per simulation.

    Uses scipy.stats.spearmanr (mid-rank averaging for ties; correct for
    alpha grids with discrete step ν=0.05).
    Returns DataFrame with columns [year, simulation_id, rho_s].
    """
    from scipy.stats import spearmanr

    records = []
    df_filt = df_scen[df_scen["year"].isin(years)]
    for (year, sim_id), grp in df_filt.groupby(["year", "simulation_id"]):
        if len(grp) < 3:
            records.append({"year": year, "simulation_id": sim_id, "rho_s": np.nan})
            continue
        rho, _ = spearmanr(grp["income"].values, grp["alpha"].values)
        records.append({"year": year, "simulation_id": sim_id, "rho_s": float(rho)})
    return pd.DataFrame(records)


def _aggregate_indicator(df_ind: pd.DataFrame, col: str) -> pd.DataFrame:
    """Aggregate an indicator column across simulations by year."""
    return (
        df_ind.groupby("year")[col]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
        .sort_values("year")
    )


def _compute_bc_per_sim(
    df_scen: pd.DataFrame,
    years: list[int],
) -> pd.DataFrame:
    """Compute bimodality coefficient BC(t) per simulation.

    BC = (γ₁² + 1) / (γ₂ + 3·(N−1)²/((N−2)(N−3)))
    where γ₁ = skewness, γ₂ = excess kurtosis.
    Threshold for bimodality: BC > 5/9 ≈ 0.555 (Freeman & Dale 2013).

    Returns DataFrame with columns [year, simulation_id, bc].
    """
    from scipy.stats import skew, kurtosis

    records = []
    df_filt = df_scen[df_scen["year"].isin(years)]
    for (year, sim_id), grp in df_filt.groupby(["year", "simulation_id"]):
        alpha = grp["alpha"].values
        n = len(alpha)
        if n < 4:
            records.append({"year": year, "simulation_id": sim_id, "bc": np.nan})
            continue
        g1 = float(skew(alpha))
        g2 = float(kurtosis(alpha, fisher=True))  # excess kurtosis
        denom = g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        bc = (g1 ** 2 + 1.0) / denom if denom != 0.0 else np.nan
        records.append({"year": year, "simulation_id": sim_id, "bc": float(bc)})
    return pd.DataFrame(records)


def plot_multi_scenario_bc(
    df_multi: pd.DataFrame,
    *,
    years: list[int] | None = None,
    bands=_UNSET,
    event_years: list[int] | None = None,
    exclude_scenarios: list[str] | None = None,   # ← add
    figsize: tuple[float, float] = (7, 3.8),
    title: str = "Bimodality coefficient BC(t) — scenario comparison",
) -> tuple[plt.Figure, plt.Axes]:
    """BC(t) time-series for each scenario with bimodality threshold.

    BC > 5/9 ≈ 0.555 indicates a bimodal distribution (Freeman & Dale 2013,
    doi:10.3389/fpsyg.2013.00700).  Computed per simulation then averaged.

    Parameters
    ----------
    df_multi : DataFrame
        Concatenated multi-scenario agents panel with columns
        ``scenario``, ``year``, ``simulation_id``, ``alpha``.
    years : list[int] | None
    bands : None | "std" | "sem"
    event_years : list[int] | None
        Climate-event years for vertical band overlay.
    figsize : (width, height) in inches.
    title : str
    """
    bands = _resolve_bands(bands)

    if years is None:
        years = sorted(df_multi["year"].unique())

    scenarios = sorted(df_multi["scenario"].unique())
    try:
        from opinion_dynamics.parameters import BENCHMARK_SCENARIO as _base, SHOW_BASELINE_RESULTS as _show
    except Exception:
        _base = "Baseline"; _show = True
    if _base is not None and not _show:
        scenarios = [s for s in scenarios if s != _base]

    if exclude_scenarios:
        # Do not exclude the baseline scenario when SHOW_BASELINE_RESULTS is True.
        _effective_excl = [s for s in exclude_scenarios if not (_show and s == _base)]
        scenarios = [s for s in scenarios if s not in _effective_excl]
    cmap = _scenario_color_map(scenarios)

    df_filt = df_multi[df_multi["year"].isin(years)].copy()

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=12)

    for scen in scenarios:
        color = cmap[scen]
        df_scen = df_filt[df_filt["scenario"] == scen]
        bc_df = _compute_bc_per_sim(df_scen, years)
        bc_agg = _aggregate_indicator(bc_df, "bc")
        ax.plot(bc_agg["year"], bc_agg["mean"],
                label=scen, color=color, linewidth=2.0)
        err = _band_err(bc_agg["std"], bc_agg["count"], bands)
        if err is not None:
            ax.fill_between(bc_agg["year"],
                            bc_agg["mean"] - err, bc_agg["mean"] + err,
                            color=color, alpha=0.12)

    ax.axhline(5.0 / 9.0, color="black", linestyle="--", linewidth=1.0,
               label=r"Bimodality threshold ($\frac{5}{9} \approx 0.556$)")
    ax.set_ylabel(r"$BC(t)$", fontsize=10)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_xticks(_year_ticks(years))
    ax.legend(fontsize=9, loc="best", framealpha=0.7)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    _add_event_vlines([ax], event_years or [])

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# plot_multi_scenario_opinion
# ---------------------------------------------------------------------------

def plot_multi_scenario_opinion_quintiles(
    df_multi: pd.DataFrame,
    *,
    years: list[int] | None = None,
    n_quantiles: int = 5,
    bands=_UNSET,
    diff_from_baseline: bool | None = None,
    show_baseline_spread: bool | None = None,
    spread_indicators: list[str] | None = None,
    event_years: list[int] | None = None,
    figsize: tuple[float, float] = (9, 5.5),
    figsize_diff: tuple[float, float] = (9, 8.0),
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Aggregate + quintile opinion curves for each scenario on a single axis.

    Color = scenario.  Linestyle = quintile rank.
    Thick line = scenario aggregate.  Thin lines = per-quintile means.

    Parameters
    ----------
    diff_from_baseline : bool | None
        If True, switch to a two-panel layout:
          Panel 1 — spread indicators η²(t) and/or ρ_S(t) (see below).
          Panel 2 — centered quintile spread ᾱ_q(t) − ᾱ(t) per scenario,
                    computed within each scenario independently.
        If None, reads ``PLOT_DIFF_FROM_BASELINE`` from parameters.py.
    show_baseline_spread : bool | None
        Controls whether the Baseline scenario is included in the spread
        panel when diff_from_baseline=True.  If None, reads
        ``SHOW_BASELINE_RESULTS`` from parameters.py.
    spread_indicators : list[str] | None
        Which spread indicators to show in the top panel (diff mode only).
        Accepted values: ``"eta2"`` (between-quintile variance share η²),
        ``"rho_s"`` (Spearman rank correlation income–alpha ρ_S).
        Both active by default: ``["eta2", "rho_s"]``.
        Pass ``[]`` or ``None`` to suppress the top panel entirely.
        When both are active, η² is plotted on the left y-axis (solid lines)
        and ρ_S on the right y-axis (dashed lines).
    figsize : figure size for the standard (non-diff) layout.
    figsize_diff : figure size for the two-panel diff layout.
    """
    try:
        from opinion_dynamics.parameters import (
            PLOT_DIFF_FROM_BASELINE, BENCHMARK_SCENARIO,
            SHOW_BASELINE_RESULTS,
        )
    except Exception:
        PLOT_DIFF_FROM_BASELINE = False
        BENCHMARK_SCENARIO = "Baseline"
        SHOW_BASELINE_RESULTS = True

    if diff_from_baseline is None:
        diff_from_baseline = PLOT_DIFF_FROM_BASELINE
    if show_baseline_spread is None:
        show_baseline_spread = SHOW_BASELINE_RESULTS
    if spread_indicators is None:
        spread_indicators = ["eta2", "rho_s"]

    bands = _resolve_bands(bands)

    if years is None:
        years = sorted(df_multi["year"].unique())

    if title is None:
        title = (
            "Opinion by quintile — difference from Baseline"
            if diff_from_baseline
            else "Opinion dynamics by quintile — scenario comparison"
        )

    scenarios = sorted(df_multi["scenario"].unique())
    cmap = _scenario_color_map(scenarios)

    Q_STYLES = ["-", (0, (6, 2)), "--", "-.", ":"]

    # Assign quintiles once per scenario (within-scenario, as before).
    df_filt = df_multi[df_multi["year"].isin(years)].copy()
    per_scenario_dfs = {}
    for scen in scenarios:
        tmp = df_filt[df_filt["scenario"] == scen].copy()
        tmp["quintile"] = _assign_income_quintiles(tmp, n_quantiles)
        per_scenario_dfs[scen] = tmp

    # ------------------------------------------------------------------
    # Diff mode: two-panel layout (spread indicators + quintile spread)
    # ------------------------------------------------------------------
    if diff_from_baseline:
        show_indicators = bool(spread_indicators)

        if show_indicators:
            fig, (ax_ind, ax_spread) = plt.subplots(
                2, 1, figsize=figsize_diff, sharex=True,
                gridspec_kw={"hspace": 0.28},
            )
        else:
            fig, ax_spread = plt.subplots(1, 1, figsize=figsize_diff)
            ax_ind = None

        fig.suptitle(title, fontsize=12, y=0.965)

        scenarios_to_plot = [s for s in scenarios if s != BENCHMARK_SCENARIO]

        # ---- Panel 1: spread indicators η²(t) and/or ρ_S(t) ---------------
        if show_indicators:
            want_eta2 = "eta2" in spread_indicators
            want_rho  = "rho_s" in spread_indicators
            both      = want_eta2 and want_rho

            ax_rho = ax_ind.twinx() if both else (ax_ind if want_rho else None)
            ax_eta = ax_ind if want_eta2 else None

            spread_legend_handles = []

            for scen in (scenarios if show_baseline_spread else scenarios_to_plot):
                color = cmap[scen]
                df_scen = per_scenario_dfs[scen]  # already has 'quintile' column

                if want_eta2:
                    eta2_df = _compute_eta2_per_sim(df_scen, years, n_quantiles)
                    eta2_agg = _aggregate_indicator(eta2_df, "eta2")
                    ax_eta.plot(eta2_agg["year"], eta2_agg["mean"],
                                color=color, linewidth=1.8, linestyle="-", zorder=3)
                    err = _band_err(eta2_agg["std"], eta2_agg["count"], bands)
                    if err is not None:
                        ax_eta.fill_between(eta2_agg["year"],
                                            eta2_agg["mean"] - err,
                                            eta2_agg["mean"] + err,
                                            color=color, alpha=0.12, zorder=2)

                if want_rho:
                    rho_df  = _compute_rho_s_per_sim(df_scen, years)
                    rho_agg = _aggregate_indicator(rho_df, "rho_s")
                    ax_rho.plot(rho_agg["year"], rho_agg["mean"],
                                color=color, linewidth=1.8, linestyle="--", zorder=3)
                    err = _band_err(rho_agg["std"], rho_agg["count"], bands)
                    if err is not None:
                        ax_rho.fill_between(rho_agg["year"],
                                            rho_agg["mean"] - err,
                                            rho_agg["mean"] + err,
                                            color=color, alpha=0.09, zorder=2)

                spread_legend_handles.append(
                    Line2D([0], [0], color=color, linewidth=1.8, linestyle="-",
                           label=scen + (" (base)" if scen == BENCHMARK_SCENARIO else ""))
                )

            if want_eta2:
                ax_eta.set_ylabel(r"$\eta^2(t)$", fontsize=10)
                ax_eta.set_ylim(bottom=0.0)
                ax_eta.grid(axis="y", linestyle=":", alpha=0.4)

            if want_rho:
                ax_rho.set_ylabel(r"$\rho_S(t)$", fontsize=10,
                                  color="dimgrey" if both else "black")
                ax_rho.tick_params(axis="y",
                                   labelcolor="dimgrey" if both else "black",
                                   labelsize=8)
                ax_rho.axhline(0.0, color="grey", linestyle=":", linewidth=0.7)
                if not want_eta2:
                    ax_ind.grid(axis="y", linestyle=":", alpha=0.4)

            # Legend: scenario colors + linestyle guide
            style_handles = []
            if want_eta2:
                style_handles.append(Line2D([0], [0], color="grey", linewidth=1.4,
                                            linestyle="-", label=r"$\eta^2$ (left)"))
            if want_rho:
                style_handles.append(Line2D([0], [0], color="grey", linewidth=1.4,
                                            linestyle="--", label=r"$\rho_S$ (right)" if both
                                            else r"$\rho_S$"))
            ax_ind.legend(handles=spread_legend_handles + style_handles,
                          fontsize=8, loc="best", framealpha=0.7)
        # ---- Panel 2: centered quintile spread ᾱ_q − ᾱ per scenario ----
        # Computed within each scenario independently: the spread is
        # informative of within-scenario heterogeneity regardless of the
        # scenario level relative to Baseline.
        spread_scenarios = (
            scenarios if show_baseline_spread else scenarios_to_plot
        )

        for scen in spread_scenarios:
            color = cmap[scen]
            df_scen = per_scenario_dfs[scen]

            # Per-(year, sim, quintile) mean alpha
            per_sim_q = (
                df_scen.groupby(["year", "simulation_id", "quintile"])["alpha"]
                .mean().reset_index(name="alpha_q")
            )
            # Per-(year, sim) aggregate alpha (scenario mean)
            per_sim_scen = (
                df_scen.groupby(["year", "simulation_id"])["alpha"]
                .mean().reset_index(name="alpha_agg")
            )
            # Center: deviation of each quintile from its scenario mean
            merged = per_sim_q.merge(per_sim_scen, on=["year", "simulation_id"])
            merged["centered"] = merged["alpha_q"] - merged["alpha_agg"]

            stats_c = (
                merged.groupby(["year", "quintile"])["centered"]
                .agg(mean="mean", std="std", count="count").reset_index()
            )

            for q in range(1, n_quantiles + 1):
                sq = stats_c[stats_c["quintile"] == q].sort_values("year")
                ls = Q_STYLES[(q - 1) % len(Q_STYLES)]
                lw = 0.9 if scen == BENCHMARK_SCENARIO else 1.3
                alpha_line = 0.55 if scen == BENCHMARK_SCENARIO else 0.85
                ax_spread.plot(sq["year"], sq["mean"],
                               color=color, linewidth=lw, linestyle=ls,
                               alpha=alpha_line, zorder=2)
                err = _band_err(sq["std"], sq["count"], bands)
                if err is not None:
                    ax_spread.fill_between(sq["year"],
                                           sq["mean"] - err, sq["mean"] + err,
                                           color=color, alpha=0.06, zorder=1)

        ax_spread.axhline(0.0, color="grey", linestyle=":", linewidth=0.8)
        ax_spread.set_ylabel(
            r"$\bar{\alpha}_q - \bar{\alpha}$ (quintile deviation)",
            fontsize=9,
        )
        ax_spread.grid(axis="y", linestyle=":", alpha=0.4)

        # Shared legend for panel 2
        spread_scenario_handles = [
            Line2D([0], [0], color=cmap[s], linewidth=1.8, linestyle="-",
                   label=s + (" (baseline)" if s == BENCHMARK_SCENARIO else ""))
            for s in spread_scenarios
        ]
        q_labels = ["Q1 (poorest)", "Q2", "Q3", "Q4", f"Q{n_quantiles} (richest)"]
        quintile_handles = [
            Line2D([0], [0], color="grey", linewidth=1.0,
                   linestyle=Q_STYLES[(q) % len(Q_STYLES)],
                   label=q_labels[q])
            for q in range(n_quantiles)
        ]
        ax_spread.legend(
            handles=spread_scenario_handles + quintile_handles,
            fontsize=8, loc="best", framealpha=0.7, ncol=2,
        )

        ax_spread.set_xlabel("Year", fontsize=10)
        ax_spread.set_xticks(_year_ticks(years))
        if show_indicators:
            ax_ind.set_xlabel("Year", fontsize=10)
            ax_ind.set_xticks(_year_ticks(years))
            ax_ind.tick_params(labelbottom=True)
            _add_event_vlines([ax_ind, ax_spread], event_years or [])
        else:
            _add_event_vlines([ax_spread], event_years or [])
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        return fig, (ax_ind, ax_spread) if show_indicators else (None, ax_spread)

    # ------------------------------------------------------------------
    # Standard mode: single-panel, identical to previous behaviour
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=12)

    for scen in scenarios:
        color = cmap[scen]
        df_scen = per_scenario_dfs[scen]

        per_sim_q = (
            df_scen.groupby(["year", "simulation_id", "quintile"])["alpha"]
            .mean().reset_index(name="alpha_mean")
        )
        stats_q = (
            per_sim_q.groupby(["year", "quintile"])["alpha_mean"]
            .agg(mean="mean", std="std", count="count").reset_index()
        )

        for q in range(1, n_quantiles + 1):
            sq = stats_q[stats_q["quintile"] == q].sort_values("year")
            ls = Q_STYLES[(q - 1) % len(Q_STYLES)]
            ax.plot(sq["year"], sq["mean"],
                    color=color, linewidth=1.0, linestyle=ls, alpha=0.75, zorder=2)
            err = _band_err(sq["std"], sq["count"], bands)
            if err is not None:
                ax.fill_between(sq["year"], sq["mean"] - err, sq["mean"] + err,
                                color=color, alpha=0.06, zorder=1)

        per_sim_agg = (
            df_scen.groupby(["year", "simulation_id"])["alpha"]
            .mean().reset_index(name="alpha_mean")
        )
        agg = (
            per_sim_agg.groupby("year")["alpha_mean"]
            .agg(mean="mean", std="std", count="count")
            .reset_index().sort_values("year")
        )
        ax.plot(agg["year"], agg["mean"],
                color=color, linewidth=2.2, linestyle="-", zorder=4, label=scen)
        err_agg = _band_err(agg["std"], agg["count"], bands)
        if err_agg is not None:
            ax.fill_between(agg["year"], agg["mean"] - err_agg, agg["mean"] + err_agg,
                            color=color, alpha=0.13, zorder=3)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_ylabel(r"Mean opinion $\bar{\alpha}$", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    scenario_handles = [
        Line2D([0], [0], color=cmap[s], linewidth=2.2, linestyle="-", label=f"{s} (agg.)")
        for s in scenarios
    ]
    q_labels = ["Q1 (poorest)", "Q2", "Q3", "Q4", f"Q{n_quantiles} (richest)"]
    quintile_handles = [
        Line2D([0], [0], color="grey", linewidth=1.0,
               linestyle=Q_STYLES[(q) % len(Q_STYLES)],
               label=q_labels[q])
        for q in range(n_quantiles)
    ]
    neutral_handle = Line2D([0], [0], color="grey", linewidth=0.8,
                            linestyle="--", label=r"Neutral ($\alpha=0.5$)")

    ax.legend(handles=scenario_handles + quintile_handles + [neutral_handle],
              fontsize=8, loc="best", framealpha=0.7, ncol=2)

    ax.set_xlabel("Year", fontsize=10)
    ax.set_xticks(_year_ticks(years))
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    _add_event_vlines([ax], event_years or [])

    fig.tight_layout()
    return fig, ax


def _compute_per_sim_agg(df_multi, years, scenarios):
    """Return per-(scenario, year, simulation_id) mean alpha as a DataFrame.

    Used internally by both multi-scenario plot functions.
    """
    return (
        df_multi[df_multi["year"].isin(years)]
        .groupby(["scenario", "year", "simulation_id"])["alpha"]
        .mean()
        .reset_index(name="alpha_mean")
    )


def _diff_agg_from_baseline(per_sim, baseline_scenario, scenarios):
    """Compute simulation-matched differences delta = scenario - baseline.

    For each non-baseline scenario, the difference is computed per
    (year, simulation_id) before aggregating across simulations.  This
    exploits the shared-seed structure to reduce variance in the estimate
    relative to differencing post-aggregation.

    Parameters
    ----------
    per_sim : DataFrame
        Output of ``_compute_per_sim_agg``.
    baseline_scenario : str
        Label of the reference scenario.
    scenarios : list[str]
        All scenario labels present in ``per_sim``.

    Returns
    -------
    diff_stats : dict[str, DataFrame]
        Keys: non-baseline scenario labels.
        Values: DataFrame with columns [year, mean, std, count].
    """
    base = (
        per_sim[per_sim["scenario"] == baseline_scenario]
        [["year", "simulation_id", "alpha_mean"]]
        .rename(columns={"alpha_mean": "alpha_base"})
    )
    result = {}
    for scen in scenarios:
        if scen == baseline_scenario:
            continue
        scen_ps = per_sim[per_sim["scenario"] == scen][["year", "simulation_id", "alpha_mean"]]
        merged = scen_ps.merge(base, on=["year", "simulation_id"], how="inner")
        merged["delta"] = merged["alpha_mean"] - merged["alpha_base"]
        agg = (
            merged.groupby("year")["delta"]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
            .sort_values("year")
        )
        result[scen] = agg
    return result


def plot_multi_scenario_opinion(
    df_multi: pd.DataFrame,
    *,
    years: list[int] | None = None,
    bands=_UNSET,
    diff_from_baseline: bool | None = None,
    event_years: list[int] | None = None,
    figsize: tuple[float, float] = (7, 4.5),
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Aggregate support curve ᾱ(t) for each scenario on a single axis.

    Replicates the bottom panel of :func:`plot_scenario_overview` for an
    arbitrary number of scenarios simultaneously: one line per scenario
    (aggregate mean opinion only, no quintile breakdown), using the same
    axis style as the single-scenario plot.

    The input DataFrame must contain a ``scenario`` column — as produced by
    :func:`~opinion_dynamics.abm.run_abm_reference` when called with an
    explicit ``scenario`` argument, or by :func:`run_all_scenarios` in
    ``main.py``.

    Parameters
    ----------
    df_multi : DataFrame
        Concatenated multi-scenario agents panel.  Required columns:
        ``scenario``, ``year``, ``simulation_id``, ``alpha``.
    years : list[int] | None
        Years to display; defaults to all years in the panel.
    bands : None | "std" | "sem"
        Cross-simulation uncertainty bands.
        None = no bands, "std" = ±1σ, "sem" = ±SEM.
        Defaults to ``PLOT_UNCERTAINTY_BANDS`` from parameters.py.
    diff_from_baseline : bool | None
        If True, plot delta = ᾱ(scenario) - ᾱ(Baseline) using
        simulation-matched differences.  If None, reads
        ``PLOT_DIFF_FROM_BASELINE`` from parameters.py.
    figsize : (width, height) in inches.
    title : str | None
        Defaults to mode-appropriate string.

    Returns
    -------
    fig, ax
    """
    try:
        from opinion_dynamics.parameters import PLOT_DIFF_FROM_BASELINE, BENCHMARK_SCENARIO
    except Exception:
        PLOT_DIFF_FROM_BASELINE = False
        BENCHMARK_SCENARIO = "Baseline"

    if diff_from_baseline is None:
        diff_from_baseline = PLOT_DIFF_FROM_BASELINE

    bands = _resolve_bands(bands)

    if years is None:
        years = sorted(df_multi["year"].unique())

    scenarios = sorted(df_multi["scenario"].unique())
    cmap = _scenario_color_map(scenarios)

    if title is None:
        title = (
            "Opinion dynamics — difference from Baseline"
            if diff_from_baseline
            else "Opinion dynamics — scenario comparison"
        )

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title, fontsize=12)

    if diff_from_baseline:
        per_sim = _compute_per_sim_agg(df_multi, years, scenarios)
        diff_stats = _diff_agg_from_baseline(per_sim, BENCHMARK_SCENARIO, scenarios)

        scenarios_to_plot = [s for s in scenarios if s != BENCHMARK_SCENARIO]
        for scen in scenarios_to_plot:
            agg = diff_stats[scen]
            color = cmap[scen]
            ax.plot(agg["year"], agg["mean"],
                    label=scen, color=color, linewidth=2.0)
            err = _band_err(agg["std"], agg["count"], bands)
            if err is not None:
                ax.fill_between(agg["year"],
                                agg["mean"] - err, agg["mean"] + err,
                                color=color, alpha=0.12)

        ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8,
                   label=f"Baseline ({BENCHMARK_SCENARIO})")
        ax.set_ylabel(r"$\Delta\bar{\alpha} = \bar{\alpha}_{\mathrm{scen}} - \bar{\alpha}_{\mathrm{base}}$",
                      fontsize=10)

    else:
        per_sim = _compute_per_sim_agg(df_multi, years, scenarios)

        for scen in scenarios:
            agg = (
                per_sim[per_sim["scenario"] == scen]
                .groupby("year")["alpha_mean"]
                .agg(mean="mean", std="std", count="count")
                .reset_index()
                .sort_values("year")
            )
            color = cmap[scen]
            ax.plot(agg["year"], agg["mean"],
                    label=scen, color=color, linewidth=2.0)
            err = _band_err(agg["std"], agg["count"], bands)
            if err is not None:
                ax.fill_between(agg["year"],
                                agg["mean"] - err, agg["mean"] + err,
                                color=color, alpha=0.12)

        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8,
                   label=r"Neutral ($\alpha = 0.5$)")
        ax.set_ylabel(r"Mean opinion $\bar{\alpha}$", fontsize=10)

    ax.set_xlabel("Year", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xticks(_year_ticks(years))
    ax.legend(fontsize=9, loc="best", framealpha=0.7)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    _add_event_vlines([ax], event_years or [])

    fig.tight_layout()
    return fig, ax