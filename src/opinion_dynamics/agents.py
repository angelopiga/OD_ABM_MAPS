"""Agent initialisation for the ABM.

Public functions
----------------
generate_agents_for_year
    Creates the agent DataFrame for a given year from regional macro data.
"""
import numpy as np
import pandas as pd

from .income import generate_income_distribution


def generate_agents_for_year(
    year: int,
    data: dict,
    method: str = "exact",
    rng: np.random.Generator | None = None,
    dtype_income: np.dtype | type = np.float64,
) -> pd.DataFrame:
    """Generate a DataFrame of agents for a given year from the Italy data entry.

    Reads population size, mean income, and target Gini from
    ``data[year]["Italy"]`` and generates individual incomes via
    ``generate_income_distribution``.

    Parameters
    ----------
    year : int
        Year for which agents are generated (key into ``data``).
    data : dict
        Nested dict; ``data[year]["Italy"]`` must contain ``"population"``,
        ``"mean_income"``, and ``"gini"``.
    method : {"theoretical", "exact"}, optional
        Income-generation method passed to ``generate_income_distribution``.
        Default is ``"exact"``.
    rng : np.random.Generator or None, optional
        RNG passed to the income-generation routine.  If ``None``, a new
        generator is initialised from system entropy.
    dtype_income : np.dtype or type, optional
        Data type used to store generated incomes.

    Returns
    -------
    df_agents : pd.DataFrame
        One row per agent; columns ``["agent_id", "region", "income"]``.

    Notes
    -----
    ``agent_id`` values are globally unique integers assigned sequentially
    (0-based).  ``region`` is always ``"Italy"`` and is retained as a label
    for downstream access to ``data[year]["Italy"]`` in
    ``assign_incomes_deterministic``.
    """
    if rng is None:
        rng = np.random.default_rng()

    params = data[year]["Italy"]
    result = generate_income_distribution(
        n_agents=params["population"],
        mean_income=params["mean_income"],
        target_gini=params["gini"],
        method=method,
        rng=rng,
        dtype_income=dtype_income,
    )
    incomes = result["incomes"]
    n = len(incomes)

    df_agents = pd.DataFrame({
        "agent_id": range(n),
        "region":   "Italy",
        "income":   incomes,
    })
    return df_agents
