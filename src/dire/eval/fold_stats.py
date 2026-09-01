"""Statistics fitted on training folds only; tests/test_leakage.py proves it."""

import numpy as np

from dire.data.diagnostics import intraclass_correlation, standardize_within_unit
from dire.data.panel import DATE, TARGET_LOG, UNIT, feature_columns

N_BINS = 50


def target_icc(train_df):
    z = standardize_within_unit(train_df[TARGET_LOG], train_df[UNIT])
    return np.array([intraclass_correlation(z, train_df[DATE])])


def lds_bin_edges(train_df):
    t = train_df[TARGET_LOG]
    return np.linspace(t.min(), t.max(), N_BINS + 1)


def lds_quantile_edges(train_df):
    return np.quantile(train_df[TARGET_LOG], np.linspace(0, 1, N_BINS + 1))


def market_scaler(train_df):
    f = train_df[feature_columns(train_df)]
    return np.vstack([f.mean().to_numpy(), f.std().to_numpy()])


FOLD_STATISTICS = {
    "intraclass_correlation": target_icc,
    "lds_bin_edges": lds_bin_edges,
    "lds_quantile_edges": lds_quantile_edges,
    "market_scaler": market_scaler,
}
