"""Loaders for external CSV time-series data (income, Gini, emissions, temperature).

Supported input format — long-format unified CSV (semicolon-delimited, IT locale):

    Indicator;Scenario;Year;Median;Std

where decimal separator may be a comma (e.g. ``1,234`` → ``1.234``).

All indicators present in the file are loaded into the ``all_indicators`` return
value of :func:`load_all_external_timeseries`.  The four series required by the
ABM core (income, Gini, emissions, temperature) are also returned individually
for backward compatibility.

Indicator mapping (ABM core)
-----------------------------
- Income    : ``mean yd real per capita``
- Gini      : ``GINI yd``              (already a fraction, scale = 1.0)
- Emissions : ``Total emissions``
- Temperature: not present in CSV → linear fallback 1.24 °C (2015) → 2.00 °C (2050)

Additional indicators exposed as-is (no unit conversion):
    ``CPI``, ``GDP real``, ``Inflation``, ``Population total``,
    ``mean yd d real``, ``mean yd hhs real``, ``yd hh gini``,
    ``yd national real per person``

Note: ``yd rq real per capita_region,qu`` is excluded because it contains
multiple rows per (scenario, year) — one per quintile/region combination —
and is therefore structurally incompatible with the scalar time-series format
expected by this loader.

Scenario name mapping (parameters.py → CSV)
--------------------------------------------
- ``"Baseline"``    → ``"Baseline"``
- ``"Degrowth"``    → ``"Degrowth"``
- ``"Green_growth"``→ ``"Green growth"``  (underscore → space)
- any other string  → attempted as-is, then fallback to ``"Baseline"``
"""
import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Public constants: names used in the unified CSV file.
# ---------------------------------------------------------------------------
_INDICATOR_INCOME      = "mean yd real per capita"
_INDICATOR_GINI        = "GINI yd"
_INDICATOR_EMISSIONS   = "Total emissions"
_INDICATOR_GDP         = "GDP real"
_INDICATOR_INFLATION   = "Inflation"

# Canonical file name searched inside data_dir.
_UNIFIED_FILENAME = "italy_data.csv"

# Mapping from internal scenario identifiers (parameters.py) to CSV labels.
_SCENARIO_MAP: dict[str, str] = {
    "Baseline":     "Baseline",
    "Degrowth":     "Degrowth",
    "Green_growth": "Green growth",
    "GreenGrowth":  "Green growth",   # legacy alias
}


def _normalise_float(raw: str) -> float:
    """Parse a float that may use a comma as decimal separator (IT locale)."""
    return float(raw.strip().replace(",", "."))


def load_all_external_timeseries(
    *,
    data_dir: Path,
    base_years: list[int],
    scenario: str,
) -> tuple[
    list[int],
    dict[int, float],
    dict[int, float],
    dict[int, float],
    dict[int, float],
    dict[int, float],
    dict[str, dict[int, float]],
]:
    """Load all external time series from a unified long-format CSV.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``italy_data.csv``.
    base_years : list[int]
        Candidate years to include.
    scenario : str
        Internal scenario identifier.

    Returns
    -------
    years : list[int]
    incomes : dict[int, float]
        ``mean yd real per capita``
    ginis : dict[int, float]
        ``GINI yd``
    emissions : dict[int, float]
        ``Total emissions``
    gdp : dict[int, float]
        ``GDP real``
    inflation : dict[int, float]
        ``Inflation``
    all_indicators : dict[str, dict[int, float]]

    Raises
    ------
    FileNotFoundError
        If the unified CSV is not found in ``data_dir``.
    ValueError
        If a required indicator is absent, or no years survive filtering.
    """
    csv_path = data_dir / _UNIFIED_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Required CSV not found: {csv_path}")

    # Resolve the CSV scenario label.
    csv_scenario = _SCENARIO_MAP.get(scenario, scenario)

    # Parse the long-format file into:
    #   raw[(indicator, scenario)][year] = median_value
    raw: dict[tuple[str, str], dict[int, float]] = {}

    with csv_path.open("r", newline="", encoding="utf-8-sig") as fh:
        first_line = fh.readline()
        fh.seek(0)
        delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
        reader = csv.DictReader(fh, delimiter=delimiter)
        required_cols = {"Indicator", "Scenario", "Year", "Median"}
        if reader.fieldnames is None or not required_cols.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV must contain columns {required_cols}; "
                f"found: {reader.fieldnames}"
            )
        for row in reader:
            indicator = row["Indicator"].strip()
            scen      = row["Scenario"].strip()
            year_raw  = row["Year"].strip()
            val_raw   = row["Median"].strip()
            if not year_raw or not val_raw:
                continue
            key = (indicator, scen)
            raw.setdefault(key, {})[int(year_raw)] = _normalise_float(val_raw)

    def _get_series(indicator: str) -> dict[int, float]:
        """Extract {year: value} for *indicator* and the resolved scenario."""
        key = (indicator, csv_scenario)
        if key not in raw:
            fallback_key = (indicator, "Baseline")
            if fallback_key not in raw:
                available = [k for k in raw if k[0] == indicator]
                raise ValueError(
                    f"Indicator '{indicator}' not found for scenario "
                    f"'{csv_scenario}' (or 'Baseline') in {csv_path}. "
                    f"Available scenario keys: {available}"
                )
            key = fallback_key
        series = raw[key]
        if not series:
            raise ValueError(f"No usable rows for indicator '{indicator}' in {csv_path}")
        return series

    income_raw    = _get_series(_INDICATOR_INCOME)
    gini_raw      = _get_series(_INDICATOR_GINI)
    emissions_raw = _get_series(_INDICATOR_EMISSIONS)
    gdp_raw       = _get_series(_INDICATOR_GDP)
    inflation_raw = _get_series(_INDICATOR_INFLATION)

    # Compute the intersection of covered years across all required series.
    series_years = {
        "income":    set(income_raw.keys()),
        "gini":      set(gini_raw.keys()),
        "emissions": set(emissions_raw.keys()),
        "gdp":       set(gdp_raw.keys()),
        "inflation": set(inflation_raw.keys()),
    }
    common_start = max(min(yrs) for yrs in series_years.values())
    common_end   = min(max(yrs) for yrs in series_years.values())
    if common_start > common_end:
        raise ValueError(
            f"No overlapping year window across series: "
            f"common_start={common_start}, common_end={common_end}"
        )
    common_years_set = set.intersection(*series_years.values())
    years = [
        year for year in base_years
        if (common_start <= year <= common_end) and (year in common_years_set)
    ]
    if not years:
        raise ValueError(
            "After truncation/intersection, no years remain. "
            "Check base_years and CSV year coverage."
        )

    def _to_scalar(series: dict[int, float]) -> dict[int, float]:
        return {year: series[year] for year in years}

    incomes   = _to_scalar(income_raw)
    ginis     = _to_scalar(gini_raw)
    emissions = _to_scalar(emissions_raw)
    gdp       = _to_scalar(gdp_raw)
    inflation = _to_scalar(inflation_raw)

    # Build all_indicators.
    _MULTI_ROW_INDICATORS = {"yd rq real per capita_region,qu"}
    all_indicators: dict[str, dict[int, float]] = {}
    for (indicator, scen), series in raw.items():
        if scen != csv_scenario:
            continue
        if indicator in _MULTI_ROW_INDICATORS:
            continue
        available_years = {y: v for y, v in series.items() if y in years}
        if not available_years:
            continue
        all_indicators[indicator] = available_years

    return years, incomes, ginis, emissions, gdp, inflation, all_indicators


_TEMPERATURE_FILENAME = "temperature_scenarios.csv"

_TEMPERATURE_SCENARIOS = {
    "SSP1-1.9",
    "SSP1-2.6",
    "SSP2-4.5",
    "SSP3-7.0",
    "SSP5-8.5",
}

_TEMPERATURE_PAST_METHODS = {
    "raw_obs",
    "smooth_obs",
    "forced_resp",
}

# Temperature loader (scenario-invariant; separate from MAPS MODEL data)`
def load_temperature_timeseries(
    *,
    data_dir: Path,
    base_years: list[int],
    scenario: str,
    extrapolation_past_t: str,
) -> dict[int, float]:
    """Load an IPCC/CMIP6 temperature anomaly series from the temperature CSV.

    Expected CSV format: two header rows, followed by one row per year.

    Example structure:

        ,SSP1-1.9,SSP1-1.9,SSP1-1.9,SSP1-2.6,SSP1-2.6,SSP1-2.6,...
        Year,raw_obs,smooth_obs,forced_resp,raw_obs,smooth_obs,forced_resp,...
        2010,1.06,1.003,1.00,1.06,1.003,1.00,...
        ...

    The first header row identifies the IPCC/CMIP6 scenario.
    The second header row identifies the past-temperature treatment method.

    Parameters
    ----------
    data_dir : Path
        Directory containing ``temperature_scenarios_data_multiheader.csv``.

    base_years : list[int]
        Candidate years to include.

    scenario : str
        IPCC/CMIP6 scenario label. Valid values are:

        - ``"SSP1-1.9"``
        - ``"SSP1-2.6"``
        - ``"SSP2-4.5"``
        - ``"SSP3-7.0"``
        - ``"SSP5-8.5"``

    extrapolation_past_t : str
        Method used for the past-temperature segment. Valid values are:

        - ``"raw_obs"``      : observed annual anomalies, including year-to-year variability
        - ``"smooth_obs"``   : smoothed observed anomalies
        - ``"forced_resp"``  : forced-response component, excluding short-run variability

    Returns
    -------
    dict[int, float]
        ``{year: temperature_anomaly}``, where the anomaly is measured in
        °C above the pre-industrial baseline, for years in ``base_years``
        that are present in the CSV.

    Raises
    ------
    FileNotFoundError
        If the temperature CSV is not found in ``data_dir``.

    ValueError
        If the requested scenario/method combination is absent, or if no
        years overlap with ``base_years``.
    """
    csv_path = data_dir / _TEMPERATURE_FILENAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Temperature CSV not found: {csv_path}")

    scenario = scenario.strip()
    extrapolation_past_t = extrapolation_past_t.strip()

    if scenario not in _TEMPERATURE_SCENARIOS:
        raise ValueError(
            f"Unknown temperature scenario '{scenario}'. "
            f"Valid options are: {sorted(_TEMPERATURE_SCENARIOS)}"
        )

    if extrapolation_past_t not in _TEMPERATURE_PAST_METHODS:
        raise ValueError(
            f"Unknown extrapolation_past_t '{extrapolation_past_t}'. "
            f"Valid options are: {sorted(_TEMPERATURE_PAST_METHODS)}"
        )

    with csv_path.open("r", newline="", encoding="utf-8-sig") as fh:
        first_line = fh.readline()
        fh.seek(0)

        # The multi-header temperature CSV is normally comma-delimited,
        # but this keeps the loader robust to semicolon-delimited exports.
        delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
        reader = csv.reader(fh, delimiter=delimiter)

        try:
            scenario_header = next(reader)
            method_header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Temperature CSV is empty or malformed: {csv_path}") from exc

        if len(scenario_header) != len(method_header):
            raise ValueError(
                "Temperature CSV has inconsistent header lengths: "
                f"{len(scenario_header)} scenario columns vs "
                f"{len(method_header)} method columns."
            )

        # Find the column matching both requested scenario and requested method.
        target_col: int | None = None
        for col_idx in range(1, len(scenario_header)):
            col_scenario = scenario_header[col_idx].strip()
            col_method = method_header[col_idx].strip()

            if col_scenario == scenario and col_method == extrapolation_past_t:
                target_col = col_idx
                break

        if target_col is None:
            available = [
                (scenario_header[i].strip(), method_header[i].strip())
                for i in range(1, len(scenario_header))
            ]
            raise ValueError(
                f"No temperature column found for scenario='{scenario}' and "
                f"extrapolation_past_t='{extrapolation_past_t}' in {csv_path}. "
                f"Available pairs are: {available}"
            )

        raw: dict[int, float] = {}

        for row in reader:
            if not row:
                continue

            year_raw = row[0].strip()
            if not year_raw:
                continue

            if target_col >= len(row):
                continue

            value_raw = row[target_col].strip()
            if not value_raw:
                continue

            year = int(year_raw)
            raw[year] = _normalise_float(value_raw)

    if not raw:
        raise ValueError(
            f"No usable temperature data found for scenario='{scenario}' and "
            f"extrapolation_past_t='{extrapolation_past_t}' in {csv_path}."
        )

    result = {year: raw[year] for year in base_years if year in raw}

    if not result:
        raise ValueError(
            f"No temperature years overlap with base_years for "
            f"scenario='{scenario}' and "
            f"extrapolation_past_t='{extrapolation_past_t}'."
        )

    return result

def _gdp_per_capita(
    gdp: dict,
    population: dict,
    years: list,
) -> dict:
    """Pre-compute real GDP per capita for each year."""
    result = {}
    for y in years:
        g = gdp.get(y)
        p = population.get(y)
        if g is not None and p is not None:
            gv = g[0] if isinstance(g, (list, tuple)) else float(g)
            pv = p[0] if isinstance(p, (list, tuple)) else float(p)
            if pv != 0.0:
                result[y] = gv / pv
    return result
