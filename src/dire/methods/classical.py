"""Classical baselines: HAR-RV, seasonal naive, temperature GBM.

All predict on the raw scale via exp of a log-scale prediction (no smearing
correction; every method is treated the same way in Phase 4).
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from dire.data.panel import HAR_FEATURES, TARGET_LOG, feature_columns


class HARBaseline:
    """Corsi (2009) HAR, pooled OLS across units on log volatility."""

    def fit(self, df):
        self._model = LinearRegression().fit(df[HAR_FEATURES], df[TARGET_LOG])
        return self

    def predict(self, df):
        return np.exp(self._model.predict(df[HAR_FEATURES]))


class SeasonalNaive:
    """Same weekday last week: y_hat_{t+1} = y_{t-6}. Needs extra_lags=(6,)."""

    def fit(self, df):
        if "f_log_lag6" not in df.columns:
            raise ValueError("SeasonalNaive needs build_supervised(..., extra_lags=(6,))")
        return self

    def predict(self, df):
        return np.exp(df["f_log_lag6"].to_numpy())


class TemperatureGBM:
    """Gradient boosting on all features (temperature and calendar included)."""

    def __init__(self, seed=0):
        self._seed = seed

    def fit(self, df):
        self._features = feature_columns(df)
        self._model = HistGradientBoostingRegressor(random_state=self._seed).fit(
            df[self._features], df[TARGET_LOG]
        )
        return self

    def predict(self, df):
        return np.exp(self._model.predict(df[self._features]))
