#!/usr/bin/env python3
"""Build the S&P 500 volatility panel from the Stooq bulk archive.

The archive itself must be fetched once in a browser (Stooq bot-checks scripted
access): https://stooq.com/db/h/ -> "Daily, US, ASCII" -> save as
data/raw/stooq/d_us_txt.zip. Everything after that is scripted. Pass --refresh
to re-fetch the committed universe snapshot from its ODC-PDDL source.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from dire.data import sp500
from dire.data.diagnostics import design_effect, intraclass_correlation
from dire.data.io import PROCESSED_DIR, RAW_DIR, download, write_checksum_manifest

ZIP_PATH = RAW_DIR / "stooq" / "d_us_txt.zip"


def main() -> int:
    if "--refresh" in sys.argv:
        sp500.UNIVERSE_FILE.unlink(missing_ok=True)
        download(sp500.CONSTITUENTS_URL, sp500.UNIVERSE_FILE)
        print(f"universe snapshot refreshed -> {sp500.UNIVERSE_FILE}")

    if not ZIP_PATH.exists():
        print(
            "Missing Stooq archive. One manual step (browser, ~15 min):\n"
            "  1. open https://stooq.com/db/h/\n"
            '  2. download "Daily, US, ASCII" (d_us_txt.zip)\n'
            f"  3. save it as {ZIP_PATH}\n"
            "then rerun this script."
        )
        return 1

    tickers = sp500.load_universe()
    panel = sp500.build_panel(ZIP_PATH, tickers)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "sp500_vol_panel.parquet"
    panel.to_parquet(out)
    write_checksum_manifest([ZIP_PATH, out], PROCESSED_DIR / "sp500_manifest.json")

    log_y = np.log(panel["y"])
    skipped = panel.attrs["skipped_tickers"]
    print(f"panel: {len(panel)} rows, {panel['unit'].nunique()} tickers "
          f"({len(skipped)} skipped: {', '.join(skipped[:8])}{'...' if len(skipped) > 8 else ''})")
    print(f"ICC(log vol | day) = {intraclass_correlation(log_y, panel['date']):.3f}, "
          f"deff = {design_effect(log_y, panel['date']):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
