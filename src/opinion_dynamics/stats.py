"""Summary statistics and diagnostics for the Opinion Dynamics ABM."""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_gini(x: np.ndarray) -> float:
    """Compute the Gini coefficient of a non-negative array.

    Parameters
    ----------
    x : np.ndarray
        1-D array of non-negative values (e.g. incomes).  Need not be sorted.

    Returns
    -------
    float
        Gini coefficient in [0, 1].  Returns 0 for a perfectly equal
        distribution and approaches 1 as inequality increases.

    Notes
    -----
    Uses the standard sorted-array formula:

        G = (2 / n) * Σ_i [ rank_i * x_i ] / Σ_i x_i  −  (n + 1) / n

    where rank_i = 1, 2, …, n after sorting x in ascending order.
    """
    x = np.asarray(x, dtype=float)
    x = np.sort(x)
    n = x.size
    return (2.0 / n) * np.sum((np.arange(1, n + 1) * x)) / np.sum(x) - (n + 1) / n


def income_assortativity(
    neighbor_lists: list[np.ndarray],
    income: np.ndarray,
) -> float:
    """Compute Newman's scalar assortativity for income on an undirected graph.

    Pearson correlation of income values at the two endpoints of every edge.
    Each undirected edge is counted once (j > i).

    Parameters
    ----------
    neighbor_lists : list of np.ndarray
        Per-agent arrays of neighbor indices (0-based positional).
    income : np.ndarray, shape (N,)
        Current income vector.

    Returns
    -------
    float
        Pearson r across edges.  np.nan if < 2 edges or zero variance.

    References
    ----------
    Newman, M.E.J. (2003). Mixing patterns in networks.
    Physical Review E, 67(2), 026126.
    """
    src_list = []
    tgt_list = []
    for i, nbs in enumerate(neighbor_lists):
        for j in nbs:
            if j > i:
                src_list.append(i)
                tgt_list.append(j)

    if len(src_list) < 2:
        return np.nan

    x = income[np.array(src_list, dtype=np.intp)]
    y = income[np.array(tgt_list, dtype=np.intp)]

    mx, my = x.mean(), y.mean()
    dx, dy = x - mx, y - my
    sx = np.sqrt((dx ** 2).mean())
    sy = np.sqrt((dy ** 2).mean())

    if sx < 1e-15 or sy < 1e-15:
        return np.nan

    return float((dx * dy).mean() / (sx * sy))


def print_diagnostics_table(results: dict) -> None:
    """Print a year-by-year table of network and climate diagnostics.

    Columns
    -------
    Year       : simulation year.
    Assort r   : Newman income assortativity (mean ± std across simulations).
    E          : exogenous climate event indicator (1 = event, 0 = quiet).
    hit        : fraction of agents struck by the year's event (mean).
    <M>        : population-mean memory stock M_i(t) (mean).

    Parameters
    ----------
    results : dict
        Output of ``run_abm`` or ``run_all_scenarios``, containing key
        ``"df_simulation_stats"``.
    """
    df = results["df_simulation_stats"]

    has_climate = "M_mean" in df.columns

    agg_dict: dict = {
        "income_assortativity": ["mean", "std"],
    }
    if has_climate:
        agg_dict.update({
            "M_mean":     ["mean"],
            "hit_frac":   ["mean"],
            "event_flag": ["max"],
        })

    table = df.groupby("year").agg(agg_dict).reset_index()
    table.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in table.columns
    ]

    print()
    print("=" * 80)
    print("  Network and climate diagnostics (averaged over simulations)")
    print("=" * 80)

    parts = [f"  {'Year':<6}", f"  {'Assort r':>10}", f"  {'± std':>8}"]
    if has_climate:
        parts += [f"  {'E':>3}", f"  {'hit':>7}", f"  {'<M>':>8}"]
    print("".join(parts))
    print("-" * 80)

    for _, row in table.iterrows():
        year   = int(row["year"])
        r_mean = row["income_assortativity_mean"]
        r_std  = row["income_assortativity_std"]
        line = f"  {year:<6}  {r_mean:>10.4f}  {r_std:>8.4f}"

        if has_climate:
            E_y  = int(row["event_flag_max"])
            hit  = row["hit_frac_mean"]
            Mbar = row["M_mean_mean"]
            line += f"  {E_y:>3}  {hit:>7.3f}  {Mbar:>8.4f}"

        print(line)

    print("=" * 80)
    print("  Assort r : Newman income assortativity (Pearson r across edges)")
    if has_climate:
        print("  E        : exogenous climate event indicator (1 = event year)")
        print("  hit      : fraction of agents struck by this year's event")
        print("  <M>      : population mean of the memory stock M_i(t)")
    print()


def _make_stats_row(
    year, sim, alpha, df_agents, avg_degree=0.0,
    income_assortativity_r=np.nan,
    M_climate=None,
    hit=None,
    event_flag=0,
) -> dict:
    """Build a per-(year, sim) row of summary statistics."""
    return {
        "year":                year,
        "simulation_id":       sim,
        "S":                   float(alpha.mean()),
        "S_std":               float(alpha.std()),
        "alpha_q25":           float(np.percentile(alpha, 25)),
        "alpha_q75":           float(np.percentile(alpha, 75)),
        "mean_income":         float(df_agents["income"].mean()),
        "gini":                float(compute_gini(df_agents["income"].to_numpy())),
        "avg_degree":          float(avg_degree),
        "income_assortativity": float(income_assortativity_r),
        "M_mean":              float(np.mean(M_climate)) if M_climate is not None else 0.0,
        "hit_frac":            float(np.mean(hit))       if hit       is not None else 0.0,
        "event_flag":          int(event_flag),
    }


def _make_stats_row(
    year, sim, alpha, df_agents, avg_degree=0.0,
    income_assortativity_r=np.nan,
    M_climate=None,
    hit=None,
    event_flag=0,
) -> dict:
    """Build a per-(year, sim) row of summary statistics."""
    return {
        "year":                year,
        "simulation_id":       sim,
        "S":                   float(alpha.mean()),
        "S_std":               float(alpha.std()),
        "alpha_q25":           float(np.percentile(alpha, 25)),
        "alpha_q75":           float(np.percentile(alpha, 75)),
        "mean_income":         float(df_agents["income"].mean()),
        "gini":                float(compute_gini(df_agents["income"].to_numpy())),
        "avg_degree":          float(avg_degree),
        "income_assortativity": float(income_assortativity_r),
        "M_mean":              float(np.mean(M_climate)) if M_climate is not None else 0.0,
        "hit_frac":            float(np.mean(hit))       if hit       is not None else 0.0,
        "event_flag":          int(event_flag),
    }
