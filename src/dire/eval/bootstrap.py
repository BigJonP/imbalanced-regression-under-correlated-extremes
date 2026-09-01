"""Cluster bootstrap over events (never over rows).

Rows of one event rise and fall together, so resampling rows would understate
uncertainty by roughly the design effect. Every replicate here resamples whole
events with replacement.
"""

import numpy as np
import pandas as pd


def _event_groups(events):
    return pd.Series(np.arange(len(events))).groupby(np.asarray(events)).indices


def cluster_bootstrap_ci(y, pred, events, metric_fn, n_boot=1000, seed=0, ci=0.95):
    """Percentile CI of metric_fn(y, pred) under whole-event resampling."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    groups = _event_groups(events)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(len(keys), size=len(keys), replace=True)
        rows = np.concatenate([groups[keys[i]] for i in sampled])
        stats[b] = metric_fn(y[rows], pred[rows])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return {"point": float(metric_fn(y, pred)), "lo": float(lo), "hi": float(hi)}


def paired_cluster_bootstrap(y, pred_a, pred_b, events, metric_fn, n_boot=1000, seed=0, ci=0.95):
    """CI and two-sided p-value for metric(a) - metric(b), both methods scored
    on the same resampled events per replicate."""
    y = np.asarray(y, float)
    pred_a, pred_b = np.asarray(pred_a, float), np.asarray(pred_b, float)
    groups = _event_groups(events)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(len(keys), size=len(keys), replace=True)
        rows = np.concatenate([groups[keys[i]] for i in sampled])
        diffs[b] = metric_fn(y[rows], pred_a[rows]) - metric_fn(y[rows], pred_b[rows])
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(diffs, [alpha, 1.0 - alpha])
    frac_pos = float((diffs > 0).mean())
    return {
        "diff": float(metric_fn(y, pred_a) - metric_fn(y, pred_b)),
        "lo": float(lo),
        "hi": float(hi),
        "p_value": float(2.0 * min(frac_pos, 1.0 - frac_pos)),
    }
