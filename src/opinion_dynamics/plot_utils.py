"""Plotting utilities for the Opinion Dynamics ABM.

Self-contained module.  Each public function takes a df_agents_panel
DataFrame and returns (fig, ax).

Expected columns in df_agents_panel
-------------------------------------
year          : int
simulation_id : int
agent_id      : int
alpha         : float   opinion in [0, 1]
income        : float
s_rel         : int     private signal ∈ {-1, 0, +1}
M_climate     : float   climate memory ∈ [0, M_max]
climate_hit   : int     H_i(t) ∈ {0, 1}

Public API
----------
GROUP A — Opinion output
  plot_opinion_heatmap            — year × alpha heatmap of the opinion distribution
  plot_opinion_by_income_quintile — mean opinion over time, one line per income quintile
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_UNSET = object()  # sentinel for "load from parameters.py"


def _resolve_bands(bands):
    """Return the effective band mode: None, 'std', or 'sem'."""
    if bands is not _UNSET:
        return bands
    try:
        from .parameters import PLOT_UNCERTAINTY_BANDS
        return PLOT_UNCERTAINTY_BANDS
    except Exception:
        return None


def _band_err(std: pd.Series, count: pd.Series, bands) -> pd.Series | None:
    """Return the error series for fill_between, or None if bands is None."""
    if bands is None:
        return None
    if bands == "sem":
        return std / np.sqrt(count)
    return std  # "std"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _scenario_label() -> str:
    """Return ACTIVE_SCENARIO from parameters, or empty string on failure."""
    try:
        from .parameters import ACTIVE_SCENARIO
        return str(ACTIVE_SCENARIO).strip()
    except Exception:
        return ""


def _title(base: str) -> str:
    """Append '| Scenario: <n>' to a title when a scenario is active."""
    label = _scenario_label()
    return f"{base} | Scenario: {label}" if label else base


def _year_ticks(years: list[int], step: int = 5) -> list[int]:
    """Return years that are exact multiples of step."""
    return [yr for yr in years if yr % step == 0]


def _apply_year_axis(ax: plt.Axes, years: list[int], *, fontsize: int = 10) -> None:
    """Show year ticks and label explicitly, even on shared x-axes."""
    ax.set_xlabel("Year", fontsize=fontsize)
    ax.set_xticks(_year_ticks(years))
    ax.tick_params(axis="x", labelbottom=True)


def _apply_year_axes(axes, years: list[int], *, fontsize: int = 10) -> None:
    """Apply explicit year ticks/labels to one or more axes."""
    for ax in axes:
        _apply_year_axis(ax, years, fontsize=fontsize)


def _density_matrix(
    df: pd.DataFrame,
    years: list[int],
    bins: int,
) -> np.ndarray:
    """Return a (n_years × bins) density matrix of alpha values."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    density = np.zeros((len(years), bins))
    for i, yr in enumerate(years):
        vals = df.loc[df["year"] == yr, "alpha"].dropna().to_numpy()
        if vals.size:
            counts, _ = np.histogram(vals, bins=edges)
            total = counts.sum()
            if total:
                density[i] = counts / total
    return density


def _assign_income_quintiles(df: pd.DataFrame, n: int) -> pd.Series:
    """Assign income quintiles independently within each simulation.

    Quintile labels are assigned per-simulation to reflect each
    simulation's own income ranking, avoiding the label-misalignment
    bug that arises when rankings from a single simulation are mapped
    by agent_id across all simulations.
    """
    first_year = df["year"].min()
    ref = df[df["year"] == first_year][["agent_id", "simulation_id", "income"]].copy()
    ref["quintile"] = (
        ref.groupby("simulation_id")["income"]
        .transform(lambda x: pd.qcut(x, n, labels=False, duplicates="drop"))
        .astype(int) + 1
    )
    key = ref.set_index(["agent_id", "simulation_id"])["quintile"]
    return df.set_index(["agent_id", "simulation_id"]).index.map(key.get)


def _srel_fractions(df: pd.DataFrame, years: list[int]) -> np.ndarray:
    """Return a (n_years, 3) array of s_rel fractions.

    Columns: [fraction(+1), fraction(0), fraction(-1)].
    """
    out = np.zeros((len(years), 3))
    for i, yr in enumerate(years):
        s = df.loc[df["year"] == yr, "s_rel"].dropna()
        if s.size:
            out[i, 0] = (s == 1).mean()
            out[i, 1] = (s == 0).mean()
            out[i, 2] = (s == -1).mean()
    return out


def _add_event_markers(
    ax: plt.Axes,
    climate_events: dict[int, int],
    years: list[int],
    *,
    color: str = "#d73027",
    alpha: float = 0.12,
    label: str = "Climate event",
) -> None:
    """Overlay semi-transparent vertical spans on years where E(t) = 1.

    Parameters
    ----------
    ax : matplotlib Axes
    climate_events : dict[int, int]
        Mapping year -> E(t) in {0, 1}.
    years : list[int]
        Years currently displayed on the x-axis.
    color, alpha : str, float
        Visual style of the event bands.
    label : str
        Legend label (applied only to the first band to avoid duplicates).
    """
    first = True
    for y in sorted(years):
        if climate_events.get(y, 0) == 1:
            ax.axvspan(y - 0.4, y + 0.4, color=color, alpha=alpha,
                       zorder=0, label=label if first else None)
            first = False


# ---------------------------------------------------------------------------
# GROUP A — Opinion output
# ---------------------------------------------------------------------------

def plot_opinion_heatmap(
    df_agents_panel: pd.DataFrame,
    *,
    bins: int = 50,
    years: list[int] | None = None,
    cmap: str = "YlOrRd",
    figsize: tuple[float, float] = (9, 4),
    title: str = "Opinion distribution over time",
) -> tuple[plt.Figure, plt.Axes]:
    """Heatmap of the opinion (α) distribution across years."""
    if years is None:
        years = sorted(df_agents_panel["year"].unique())

    density = _density_matrix(df_agents_panel, years, bins)

    year_arr    = np.array(years, dtype=float)
    year_edges  = np.concatenate([[year_arr[0] - 0.5], year_arr + 0.5])
    alpha_edges = np.linspace(0.0, 1.0, bins + 1)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(year_edges, alpha_edges, density.T, cmap=cmap, shading="flat")

    ax.set_xticks(_year_ticks(years))
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Opinion α", fontsize=12)
    ax.set_title(_title(title), fontsize=13)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Fraction of agents", fontsize=10)

    fig.tight_layout()
    return fig, ax


def plot_opinion_by_income_quintile(
    df_agents_panel: pd.DataFrame,
    *,
    n_quantiles: int = 5,
    years: list[int] | None = None,
    bands=_UNSET,
    figsize: tuple[float, float] = (8, 4.5),
    title: str = "Mean opinion by income quintile",
) -> tuple[plt.Figure, plt.Axes]:
    """Mean opinion α over time, one line per income quintile."""
    bands = _resolve_bands(bands)

    if years is None:
        years = sorted(df_agents_panel["year"].unique())

    df = df_agents_panel[df_agents_panel["year"].isin(years)].copy()
    df["quintile"] = _assign_income_quintiles(df, n_quantiles)

    per_sim = (
        df.groupby(["year", "simulation_id", "quintile"])["alpha"]
        .mean()
        .reset_index(name="alpha_mean")
    )
    stats = (
        per_sim.groupby(["year", "quintile"])["alpha_mean"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )

    colors = plt.cm.coolwarm(np.linspace(0.05, 0.95, n_quantiles))

    fig, ax = plt.subplots(figsize=figsize)

    for q in range(1, n_quantiles + 1):
        s = stats[stats["quintile"] == q].sort_values("year")
        label = "Q1 (poorest)" if q == 1 else f"Q{q} (richest)" if q == n_quantiles else f"Q{q}"
        color = colors[q - 1]
        ax.plot(s["year"], s["mean"], label=label, color=color, linewidth=1.8)
        err = _band_err(s["std"], s["count"], bands)
        if err is not None:
            ax.fill_between(s["year"], s["mean"] - err, s["mean"] + err,
                            color=color, alpha=0.15)

    agg_per_sim = (
        df.groupby(["year", "simulation_id"])["alpha"]
        .mean()
        .reset_index(name="alpha_mean")
    )
    agg_stats = (
        agg_per_sim.groupby("year")["alpha_mean"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    ax.plot(agg_stats["year"], agg_stats["mean"],
            label="Aggregate", color="black", linewidth=2.5, zorder=5)
    agg_err = _band_err(agg_stats["std"], agg_stats["count"], bands)
    if agg_err is not None:
        ax.fill_between(agg_stats["year"],
                        agg_stats["mean"] - agg_err,
                        agg_stats["mean"] + agg_err,
                        color="black", alpha=0.10, zorder=4)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="Neutral (α=0.5)")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Mean opinion α", fontsize=12)
    ax.set_title(_title(title), fontsize=13)
    ax.set_xticks(_year_ticks(years))
    ax.legend(fontsize=9, loc="best")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Multi-scenario comparison
# ---------------------------------------------------------------------------

# Canonical colors for the three named scenarios; colorblind-tolerant palette.
_SCENARIO_COLORS: dict[str, str] = {
    "Baseline":     "#1b7837",
    "Degrowth":     "#d73027",
    "Green_growth": "#4575b4",
}
_FALLBACK_COLORS: list[str] = ["#8c510a", "#762a83", "#01665e", "#bf812d"]


def _scenario_color_map(scenarios: list[str]) -> dict[str, str]:
    """Map each scenario label to a plot color."""
    cmap: dict[str, str] = {}
    fallback_iter = iter(_FALLBACK_COLORS)
    for scen in scenarios:
        cmap[scen] = _SCENARIO_COLORS.get(scen, next(fallback_iter, "#333333"))
    return cmap
