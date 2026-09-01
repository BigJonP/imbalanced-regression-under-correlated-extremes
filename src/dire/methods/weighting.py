"""Sample weights, all fitted on the training slice passed in. Mean weight is 1."""

import numpy as np
import pandas as pd

from dire.data.diagnostics import intraclass_correlation, standardize_within_unit
from dire.data.panel import DATE, TARGET_LOG, UNIT
from dire.eval.fold_stats import N_BINS, lds_bin_edges

KERNEL_SIZE = 5
KERNEL_SIGMA = 2.0


def _bin_index(train_df):
    edges = lds_bin_edges(train_df)
    return np.clip(np.digitize(train_df[TARGET_LOG], edges) - 1, 0, N_BINS - 1)


def _normalize(w):
    w = np.asarray(w, dtype=float)
    return w / w.mean()


def uniform_weights(train_df):
    return np.ones(len(train_df))


def inverse_weights(train_df, power=1.0):
    bins = _bin_index(train_df)
    counts = np.bincount(bins, minlength=N_BINS).astype(float)
    return _normalize((1.0 / counts[bins]) ** power)


def sqinv_weights(train_df):
    return inverse_weights(train_df, power=0.5)


def _gaussian_kernel(size=KERNEL_SIZE, sigma=KERNEL_SIGMA):
    x = np.arange(size) - size // 2
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    return k / k.sum()


def lds_weights(train_df):
    """Yang et al. (2021): inverse of the kernel-smoothed target density."""
    bins = _bin_index(train_df)
    counts = np.bincount(bins, minlength=N_BINS).astype(float)
    smoothed = np.convolve(counts, _gaussian_kernel(), mode="same")
    return _normalize(1.0 / smoothed[bins])


def _event_lds_weights(event_means):
    """LDS machinery applied at the event level: rarity of day-mean targets."""
    edges = np.linspace(event_means.min(), event_means.max(), N_BINS + 1)
    bins = np.clip(np.digitize(event_means, edges) - 1, 0, N_BINS - 1)
    counts = np.bincount(bins, minlength=N_BINS).astype(float)
    smoothed = np.convolve(counts, _gaussian_kernel(), mode="same")
    return _normalize(1.0 / smoothed[bins])


def lds_deff_weights(train_df):
    """Design-effect-corrected LDS.

    An event (day) with m rows and intraclass correlation rho carries only
    1/deff of a row's worth of independent row-level information
    (deff = 1 + (m - 1) * rho, Kish). That share keeps its row-level LDS
    weight, discounted by deff; the remaining share is one shared story,
    weighted by the rarity of the DAY-MEAN target among days and split across
    the day's rows. At rho = 0 this reduces exactly to LDS; at high rho it
    counts each story once instead of each copy. Note: a pure per-event
    division by deff is a no-op on balanced panels (constant rescale), which
    is why the event-level channel exists.
    """
    row_w = lds_weights(train_df)
    z = standardize_within_unit(train_df[TARGET_LOG], train_df[UNIT])
    rho = max(float(intraclass_correlation(z, train_df[DATE])), 0.0)

    by_event = train_df.groupby(DATE)[TARGET_LOG]
    event_means = by_event.mean()
    event_w = pd.Series(_event_lds_weights(event_means.to_numpy()), index=event_means.index)
    m = by_event.size()
    deff = 1.0 + (m - 1) * rho

    per_row_event = _normalize((event_w / m).loc[train_df[DATE]].to_numpy())
    share = (1.0 / deff).loc[train_df[DATE]].to_numpy()
    return _normalize(share * row_w + (1.0 - share) * per_row_event)


WEIGHTINGS = {
    "none": uniform_weights,
    "inverse": inverse_weights,
    "sqinv": sqinv_weights,
    "lds": lds_weights,
    "lds_deff": lds_deff_weights,
}
