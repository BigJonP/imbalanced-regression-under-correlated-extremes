"""European zonal load (OPSD, CC-BY-4.0) with ERA5 daily max temperature (Open-Meteo, CC-BY-4.0).

Zones are countries; y is the daily peak of hourly actual load (UTC days with at
least 20 valid hours). Temperature is ERA5 at the capital, the covariate for the
GBM baseline and the heat-wave event definition.
"""

import json
from pathlib import Path

import pandas as pd

OPSD_URL = (
    "https://data.open-power-system-data.org/time_series/2020-10-06/"
    "time_series_60min_singleindex.csv"
)
OPENMETEO_URL = (
    "https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
    "&start_date={start}&end_date={end}&daily=temperature_2m_max&timezone=UTC"
)
START, END = "2015-01-01", "2020-09-30"
ZONES = {  # capital coordinates for ERA5
    "AT": (48.21, 16.37), "BE": (50.85, 4.35), "DE": (52.52, 13.40),
    "ES": (40.42, -3.70), "FR": (48.86, 2.35), "GB": (51.51, -0.13),
    "IT": (41.89, 12.48), "NL": (52.37, 4.90), "PL": (52.23, 21.01),
    "PT": (38.72, -9.14),
}
_LOAD_SUFFIX = "_load_actual_entsoe_transparency"
_ALIASES = {"GB": ["GB_UKM" + _LOAD_SUFFIX, "GB_GBN" + _LOAD_SUFFIX]}


def resolve_load_columns(csv_path) -> dict[str, str]:
    header = pd.read_csv(csv_path, nrows=0).columns
    resolved = {}
    for zone in ZONES:
        for cand in [zone + _LOAD_SUFFIX, *_ALIASES.get(zone, [])]:
            if cand in header:
                resolved[zone] = cand
                break
    return resolved


def read_openmeteo_temp(path) -> pd.Series:
    daily = json.loads(Path(path).read_text(encoding="utf-8"))["daily"]
    index = pd.DatetimeIndex(pd.to_datetime(daily["time"]), name="date")
    return pd.Series(daily["temperature_2m_max"], index=index, name="temp_max")


def build_load_panel(opsd_csv, temp_files: dict[str, str], min_hours=20) -> pd.DataFrame:
    """Long panel [unit, date, y, temp_max] with y = daily peak load in MW."""
    columns = resolve_load_columns(opsd_csv)
    df = pd.read_csv(opsd_csv, usecols=["utc_timestamp", *columns.values()], parse_dates=["utc_timestamp"])
    df = df[(df["utc_timestamp"] >= START) & (df["utc_timestamp"] <= f"{END} 23:59")]
    df["date"] = df["utc_timestamp"].dt.normalize().dt.tz_localize(None)

    frames = []
    for zone, col in columns.items():
        g = df.groupby("date")[col]
        daily = pd.DataFrame({"y": g.max(), "hours": g.count()})
        daily = daily[(daily["hours"] >= min_hours) & (daily["y"] > 0)].drop(columns="hours")
        temp = read_openmeteo_temp(temp_files[zone])
        daily = daily.join(temp, how="inner").reset_index()
        daily["unit"] = zone
        frames.append(daily[["unit", "date", "y", "temp_max"]])
    return pd.concat(frames, ignore_index=True).sort_values(["unit", "date"], ignore_index=True)
