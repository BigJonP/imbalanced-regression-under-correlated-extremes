"""Sample weights, all fitted on the training slice passed in. Mean weight is 1."""

from functools import partial

import numpy as np
import pandas as pd

from dire.data.events import episode_labels

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


def lds_weights(train_df, kernel_size=KERNEL_SIZE, sigma=KERNEL_SIGMA):
    """Yang et al. (2021): inverse of the kernel-smoothed target density."""
    bins = _bin_index(train_df)
    counts = np.bincount(bins, minlength=N_BINS).astype(float)
    smoothed = np.convolve(counts, _gaussian_kernel(kernel_size, sigma), mode="same")
    return _normalize(1.0 / smoothed[bins])


def lds_capped_weights(train_df, cap=10.0):
    """Is plain clipping enough? LDS with weights capped at `cap` x the mean."""
    return _normalize(np.minimum(lds_weights(train_df), cap))


def _event_lds_weights(event_means):
    """LDS machinery applied at the event level: rarity of day-mean targets."""
    edges = np.linspace(event_means.min(), event_means.max(), N_BINS + 1)
    bins = np.clip(np.digitize(event_means, edges) - 1, 0, N_BINS - 1)
    counts = np.bincount(bins, minlength=N_BINS).astype(float)
    smoothed = np.convolve(counts, _gaussian_kernel(), mode="same")
    return _normalize(1.0 / smoothed[bins])


def day_events(train_df):
    """The frozen primary definition: one event per calendar day."""
    return pd.Series(train_df[DATE].to_numpy(), index=train_df.index)


def episode_events(train_df, q=0.95, gap=1):
    """The episode variant: runs of extreme days merged across gaps <= `gap`.

    Ordinary days keep their own identity; only the extreme stretches (a
    week-long heat wave, a crash week) collapse into one event. Thresholds come
    from the training frame passed in, which is all this function ever sees.
    """
    day_mean = train_df.groupby(DATE)[TARGET_LOG].mean()
    labels = episode_labels(day_mean >= day_mean.quantile(q), gap=gap)
    ids = labels.where(labels.notna(), pd.Series(day_mean.index.astype(str), index=labels.index))
    return pd.Series(ids.loc[train_df[DATE]].to_numpy(), index=train_df.index)


def lds_deff_weights(train_df, rho_scale=1.0, event_fn=day_events):
    """Design-effect-corrected LDS.

    An event (a day by default) with m rows and intraclass correlation rho
    carries only 1/deff of a row's worth of independent row-level information
    (deff = 1 + (m - 1) * rho, Kish). That share keeps its row-level LDS
    weight, discounted by deff; the remaining share is one shared story,
    weighted by the rarity of the EVENT-MEAN target among events and split
    across the event's rows. At rho = 0 this reduces exactly to LDS; at high
    rho it counts each story once instead of each copy.
    """
    row_w = lds_weights(train_df)
    ev = event_fn(train_df)
    z = standardize_within_unit(train_df[TARGET_LOG], train_df[UNIT])
    rho = min(max(float(intraclass_correlation(z, ev)) * rho_scale, 0.0), 0.999)

    by_event = train_df.groupby(ev.to_numpy())[TARGET_LOG]
    event_means = by_event.mean()
    event_w = pd.Series(_event_lds_weights(event_means.to_numpy()), index=event_means.index)
    m = by_event.size()
    deff = 1.0 + (m - 1) * rho

    per_row_event = _normalize((event_w / m).loc[ev].to_numpy())
    share = (1.0 / deff).loc[ev].to_numpy()
    return _normalize(share * row_w + (1.0 - share) * per_row_event)


WEIGHTINGS = {
    "none": uniform_weights,
    "inverse": inverse_weights,
    "sqinv": sqinv_weights,
    "lds": lds_weights,
    "lds_deff": lds_deff_weights,
    # ablation variants
    "lds_cap": lds_capped_weights,
    "lds_narrow": partial(lds_weights, kernel_size=1),
    "lds_wide": partial(lds_weights, kernel_size=9, sigma=3.0),
    "lds_deff_lo": partial(lds_deff_weights, rho_scale=0.5),
    "lds_deff_hi": partial(lds_deff_weights, rho_scale=1.5),
    "lds_deff_episode": partial(lds_deff_weights, event_fn=episode_events),
}
