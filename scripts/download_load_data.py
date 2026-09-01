#!/usr/bin/env python3
"""Download OPSD hourly load and Open-Meteo ERA5 temperature, build the load panel.

Idempotent; raw files are kept under data/raw/, the panel lands in
data/processed/load_panel.parquet with a checksum manifest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dire.data import opsd
from dire.data.diagnostics import design_effect, intraclass_correlation
from dire.data.events import episode_labels, heatwave_flags
from dire.data.io import PROCESSED_DIR, RAW_DIR, download, write_checksum_manifest

import numpy as np


def main() -> int:
    opsd_csv = RAW_DIR / "opsd" / "time_series_60min_singleindex.csv"
    print(f"OPSD -> {opsd_csv} ...", flush=True)
    download(opsd.OPSD_URL, opsd_csv, timeout=600)

    temp_files = {}
    for zone, (lat, lon) in opsd.ZONES.items():
        dest = RAW_DIR / "openmeteo" / f"{zone}.json"
        url = opsd.OPENMETEO_URL.format(lat=lat, lon=lon, start=opsd.START, end=opsd.END)
        download(url, dest)
        temp_files[zone] = dest
    print(f"temperature: {len(temp_files)} zones", flush=True)

    panel = opsd.build_load_panel(opsd_csv, temp_files)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "load_panel.parquet"
    panel.to_parquet(out)
    write_checksum_manifest(
        [opsd_csv, *temp_files.values(), out], PROCESSED_DIR / "load_manifest.json"
    )

    log_y = np.log(panel["y"])
    icc = intraclass_correlation(log_y, panel["date"])
    deff = design_effect(log_y, panel["date"])
    cross_temp = panel.groupby("date")["temp_max"].mean()
    eps = episode_labels(heatwave_flags(cross_temp))
    n_eps = eps.dropna().nunique()
    print(f"panel: {len(panel)} rows, {panel['unit'].nunique()} zones, "
          f"{panel['date'].nunique()} days")
    print(f"ICC(log peak | day) = {icc:.3f}, deff = {deff:.1f}")
    print(f"heat-wave episodes (q=0.95, full sample): {n_eps}")
    hot = cross_temp[heatwave_flags(cross_temp)]
    print("hottest flagged dates:", ", ".join(str(d.date()) for d in hot.nlargest(5).index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
