"""
Historical market regime windows — each scenario is a date range.

To add or change a scenario, edit SCENARIOS below.
Keys (e.g. "bull_run") are used in code; labels are in config.SCENARIO_LABELS.
"""

SCENARIOS = {
    "recession": {"start": "2008-09-01", "end": "2009-06-01"},
    "bull_run": {"start": "2017-01-01", "end": "2018-01-01"},
    "rate_hike": {"start": "2022-01-01", "end": "2023-01-01"},
    "covid_crash": {"start": "2020-02-01", "end": "2020-04-30"},
    "recovery": {"start": "2020-05-01", "end": "2021-06-01"},
}


def get_scenario_dates(name: str) -> tuple[str, str]:
    """Return (start_date, end_date) strings for a scenario name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Choose from {list(SCENARIOS)}")
    scenario = SCENARIOS[name]
    return scenario["start"], scenario["end"]
