"""Reproducibility utilities for the Opinion Dynamics ABM."""
import numpy as np
import pandas as pd


def rngs_for(year: int, sim: int, base_seed: int, n_streams: int = 3) -> tuple[np.random.Generator, ...]:
    """Create independent RNG streams for a given (year, sim, base_seed) triple.

    Uses :class:`numpy.random.SeedSequence` to derive ``n_streams`` child
    sequences from a single entropy source, ensuring statistical independence
    between streams while remaining fully reproducible.

    Parameters
    ----------
    year : int
        Simulation year; mixed into the seed to make streams year-specific.
    sim : int
        Simulation index; mixed into the seed to make streams sim-specific.
    base_seed : int
        Global base seed shared across all calls (typically ``SEED`` from
        ``parameters.py``).
    n_streams : int, optional
        Number of independent RNG streams to create.  Default is 3.

    Returns
    -------
    tuple[np.random.Generator, ...]
        Tuple of ``n_streams`` independent generators.
    """
    ss = np.random.SeedSequence([int(base_seed), int(year), int(sim)])
    children = ss.spawn(int(n_streams))
    return tuple(np.random.default_rng(c) for c in children)


def scenario_slug(name: str) -> str:
    """Return a filename-safe lower-case slug for a scenario label.

    Parameters
    ----------
    name : str
        Scenario label (e.g. ``"Greengrowth"``).

    Returns
    -------
    str
        Lower-case alphanumeric slug with underscores (e.g. ``"greengrowth"``).
    """
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name).strip())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "scenario"


# Columns produced within a period that should not be carried over to
# the next iteration's state DataFrame.
_STATE_DROP = ["alpha", "s_priv", "p", "M_climate", "climate_hit", "year", "simulation_id"]


def _to_state(df: pd.DataFrame) -> pd.DataFrame:
    """Strip per-period output columns from an agent DataFrame."""
    return df.drop(columns=_STATE_DROP, errors="ignore")


def _attach_outputs(
    df: pd.DataFrame,
    *,
    alpha, s_priv, p, M_climate, hit, year, sim,
) -> pd.DataFrame:
    """Attach per-period outputs to the agent DataFrame for the panel."""
    df["alpha"]         = alpha
    df["s_priv"]         = s_priv
    df["p"]             = p
    df["M_climate"]     = M_climate
    df["climate_hit"]   = hit.astype(np.int8)
    df["year"]          = year
    df["simulation_id"] = sim
    return df
