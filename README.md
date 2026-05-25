**Author**: Angelo Piga 
**Affiliation**: Departament of Economics and Management, University of Pisa
**Contact**: angelo.piga@ec.unipi.it  
**ORCID**: 0000-0002-7409-2978


# Opinion Dynamics ABM

**Assessing Public Support on Climate and Redistributive Policies**

Agent-based model developed for **Task 4.3** of the [MAPS Horizon Project](https://mapsresearch.eu/)
(*Models, Assessment & Policies for Sustainability*).
The model is described in full in the companion report `ABM_Opinion_Dynamics_MAPS.pdf`.

---

## What this model does

Simulates the evolution of public support for climate and redistributive policy scenarios in Italy
(2022–2050, annual time steps) for a heterogeneous population of N agents. Exogenous
macroeconomic trajectories are provided by the MAPS integrated assessment model (MAPSM).

Agents form opinions under two mechanisms:

- **Private signal**: perceived income growth (net of cumulative climate-shock memory) compared
  against a benchmark (aggregate GDP growth or local peer-income average).
- **Social signal**: conformity pressure from income-similar network neighbours.

Scenarios currently available: **Baseline**, **Degrowth**, **Green_growth**.

---

## Requirements and installation

**Python 3.10 or newer** is required. To check your version, open a terminal and type:

```bash
python --version
```

### Step-by-step installation

**1. Open a terminal and go to the project folder:**

```bash
cd path/to/OD_ABM_Report
```

**2. (Recommended) Create a virtual environment** to keep the project's packages isolated from the rest of your system:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows
```

You will see `(.venv)` appear at the start of your prompt when the environment is active.

**3. Install the package and all dependencies:**

```bash
pip install -e .
```

The `-e` flag installs in *editable* mode: changes you make to files under `src/` take effect immediately without reinstalling.

> **What `pyproject.toml` does.**  This file, located at the project root, serves two purposes.
> First, it lists all required packages and their minimum versions (`numpy`, `pandas`, `scipy`,
> `networkx`, `matplotlib`, `pyarrow`), so `pip install -e .` can fetch and install them all in
> one step — no need to install each library manually.  Second, it registers the `opinion_dynamics`
> package with Python, making `from opinion_dynamics.abm import run_abm` work correctly from any
> script or notebook, regardless of where it is located on your machine.

---

## Quick start

**Run all active scenarios** (configured in `parameters.py`):

```bash
python notebooks/main.py
```

**Run a single scenario programmatically**:

```python
from opinion_dynamics.abm import run_abm
results = run_abm(scenario="Degrowth")
# results["df_agents_panel"]      — per-(year, sim, agent) snapshots
# results["df_simulation_stats"]  — per-(year, sim) aggregates
# results["df_year_summary"]      — per-year cross-sim summary
```

**Run multiple scenarios and compare**:

```python
from opinion_dynamics.abm import run_all_scenarios
all_results = run_all_scenarios()   # runs all scenarios listed in ACTIVE_SCENARIO
```

Figures and data are saved to `results/` when `SAVE_RESULTS = True` in `parameters.py`.

---

## Configuration

All tuneable parameters are in `src/opinion_dynamics/parameters.py`. The most commonly
changed settings:

| Parameter | What it controls |
|---|---|
| `ACTIVE_SCENARIO` | Which scenarios to run |
| `N_SIMULATIONS` | Number of Monte Carlo replications |
| `POPULATION_SIZE` | Number of agents |
| `PRIVATE_SIGNAL_BENCHMARK` | Income-comparison benchmark (`"local_mean"`, `"gdp"`, …) |
| `TEMPERATURE_SCENARIO` | IPCC SSP pathway for climate events |
| `ALPHA_INIT_STRATEGY` | Initial opinion distribution strategy |
| `OPINION_W_PRIV` | Weight on private vs. social signal |

See `PROJECT_OUTLINE.md §6` for the full parameter reference, including symbols and PDF equation numbers.

---

## Repository layout

```
OD_ABM_Report/
├── notebooks/main.py          Entry point — run this to produce all figures and data
├── src/opinion_dynamics/      Python package (simulation + plotting)
├── data/raw/                  Input CSV time-series (MAPSM scenarios + IPCC temperature)
├── results/
│   ├── figures/               Output figures (PNG), written when SAVE_RESULTS = True
│   └── data/                  Agent panels (Parquet) + parameter snapshots (JSON)
├── ABM_Opinion_Dynamics.pdf   Companion report (full model description)
├── PROJECT_OUTLINE.md         Technical reference: module tree, pipeline, parameters
└── pyproject.toml             Package metadata and dependencies
```

For the full module tree, simulation pipeline, data-flow diagram, model-to-code mapping, and
parameter reference, see **[PROJECT_OUTLINE.md](PROJECT_OUTLINE.md)**.

---

## Output figures

Running `main.py` produces the following figures automatically:

| Figure | Function | When |
|---|---|---|
| Scenario overview | `plot_scenario_overview` | Each active scenario |
| Signal diagnostics | `plot_signal_diagnostics` | Each active scenario |
| Channel contributions | `plot_channel_contributions` | Each active scenario |
| Aggregate opinion comparison | `plot_multi_scenario_opinion` | Multi-scenario only |
| Opinion by quintile comparison | `plot_multi_scenario_opinion_quintiles` | Multi-scenario only |
| Bimodality coefficient | `plot_multi_scenario_bc` | Multi-scenario only |

---

## Project context

This code is a deliverable of **MAPS Horizon Project**, Work Package 4, Task 4.3:
*Using agent-based modelling to incorporate behavioural and lifestyle changes*.
The MAPS project is funded by the European Union's Horizon Europe programme.

**Author**: Angelo Piga  
**Contact**: angelo.piga@ec.unipi.it
