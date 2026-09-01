"""Long panel [unit, date, y, ...] -> supervised frame: features at t, target at t+1."""

import numpy as np
import pandas as pd

UNIT, DATE, VALUE = "unit", "date", "y"
TARGET_RAW, TARGET_LOG, TARGET_DATE = "target_raw", "target_log", "target_date"
FEATURE_PREFIX = "f_"
HAR_FEATURES = ["f_log_y", "f_log_roll5", "f_log_roll22"]


def feature_columns(df) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]


def build_supervised(panel, covariates=(), extra_lags=(), add_target_dow=False):
    """Target is the next observed day per unit; every feature uses info at or before t.

    Features: log y_t, HAR-style rolling means (5, 22), the cross-unit mean of
    log y_t, optional covariates at t, optional extra lags of log y, and optionally
    the day-of-week of the target date (deterministic, so not leakage).
    """
    df = panel.sort_values([UNIT, DATE]).reset_index(drop=True).copy()
    if (df[VALUE] <= 0).any() or df[VALUE].isna().any():
        raise ValueError("y must be positive and non-missing")

    g = df.groupby(UNIT, sort=False)
    df["f_log_y"] = np.log(df[VALUE])
    logs = df.groupby(UNIT, sort=False)["f_log_y"]
    df["f_log_roll5"] = logs.transform(lambda s: s.rolling(5).mean())
    df["f_log_roll22"] = logs.transform(lambda s: s.rolling(22).mean())
    df["f_log_cross"] = df.groupby(DATE)["f_log_y"].transform("mean")
    for k in extra_lags:
        df[f"f_log_lag{k}"] = logs.shift(k)
    for c in covariates:
        df[f"f_{c}"] = df[c]

    df[TARGET_RAW] = g[VALUE].shift(-1)
    df[TARGET_DATE] = g[DATE].shift(-1)
    with np.errstate(invalid="ignore"):
        df[TARGET_LOG] = np.log(df[TARGET_RAW])
    if add_target_dow:
        df["f_target_dow"] = pd.to_datetime(df[TARGET_DATE]).dt.dayofweek

    keep = [UNIT, DATE, TARGET_DATE, TARGET_RAW, TARGET_LOG]
    keep += feature_columns(df)
    keep += [c for c in df.columns if c.startswith("latent_")]
    out = df[keep].dropna(subset=feature_columns(df) + [TARGET_RAW])
    return out.reset_index(drop=True)
