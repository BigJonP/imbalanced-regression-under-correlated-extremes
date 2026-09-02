"""S&P 500 volatility panel from the Stooq bulk daily archive.

Stooq's endpoints are behind a browser check, so the archive is downloaded
manually (https://stooq.com/db/h/ -> "Daily, US, ASCII" -> d_us_txt.zip) and
processed here. Stooq data is free for personal/research use but not
redistributable, so only the processing is committed. The universe is the
committed current-constituents snapshot (survivorship bias: acknowledged
limitation, standard for this kind of study).
"""

import io
import zipfile

import numpy as np
import pandas as pd

from dire.runs import REPO_ROOT

CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)  # ODC-PDDL (public domain)
UNIVERSE_FILE = REPO_ROOT / "configs" / "sp500_universe.csv"
_PARKINSON = 1.0 / (4.0 * np.log(2.0))
MAX_RANGE = 5.0  # high/low above this is a bad tick, not a session


def load_universe(path=UNIVERSE_FILE) -> list[str]:
    return pd.read_csv(path, comment="#")["Symbol"].str.strip().tolist()


def stooq_member_name(ticker: str) -> str:
    return ticker.lower().replace(".", "-") + ".us.txt"


def parse_stooq_daily(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip("<>").lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df[["date", "open", "high", "low", "close"]]


def parkinson_vol(high, low):
    return np.sqrt(_PARKINSON * np.log(np.asarray(high) / np.asarray(low)) ** 2)


def build_panel(zip_path, tickers, start="2000-01-01", min_obs=1000,
                max_range=MAX_RANGE) -> pd.DataFrame:
    """Long panel [unit, date, y] with y = daily Parkinson volatility.

    Sessions whose high/low clears `max_range` are dropped as bad ticks. An
    S&P 500 constituent does not trade a five-fold intraday range: the archive
    holds eight such rows, all at high/low near 10.2, which is a decimal error
    rather than a market event. The largest genuine range in the panel is 4.19
    (AIG, 2008-09-16), so the cut is an absolute physical bound with a clean gap
    on either side, not a quantile fitted to the data.
    """
    frames, missing = [], []
    with zipfile.ZipFile(zip_path) as zf:
        members = {name.rsplit("/", 1)[-1]: name for name in zf.namelist() if name.endswith(".txt")}
        for t in tickers:
            member = members.get(stooq_member_name(t))
            if member is None:
                missing.append(t)
                continue
            df = parse_stooq_daily(zf.read(member).decode())
            df = df[(df["date"] >= start) & (df["low"] > 0) & (df["high"] > df["low"])]
            df = df[df["high"] / df["low"] < max_range]
            if len(df) < min_obs:
                missing.append(t)
                continue
            frames.append(
                pd.DataFrame({"unit": t, "date": df["date"], "y": parkinson_vol(df["high"], df["low"])})
            )
    if not frames:
        raise ValueError("no tickers could be built from the archive")
    panel = pd.concat(frames, ignore_index=True).sort_values(["unit", "date"], ignore_index=True)
    panel.attrs["skipped_tickers"] = missing
    return panel
