"""Cluster bootstrap over events (never over rows).

Rows of one event rise and fall together, so resampling rows would understate
uncertainty by roughly the design effect. Every replicate here resamples whole
events with replacement.
"""

import numpy as np
import pandas as pd


def _event_groups(events):
    return pd.Series(np.arange(len(events))).groupby(np.asarray(events)).indices


def event_sums_bootstrap_ci(event_rows, n_col="n", sse_col="sse", n_boot=1000, seed=0, ci=0.95):
    """Same cluster bootstrap, driven by per-day error sums instead of rows.

    MSE over a resample of whole days is sum(sse) / sum(n), so the day-level
    sums that `dire.experiments` records are enough to resample any grid result
    without keeping every prediction.
    """
    n = np.asarray(event_rows[n_col], float)
    sse = np.asarray(event_rows[sse_col], float)
    keep = n > 0
    n, sse = n[keep], sse[keep]
    if len(n) == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(n), size=(n_boot, len(n)))
    stats = sse[draws].sum(axis=1) / n[draws].sum(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return {"point": float(sse.sum() / n.sum()), "lo": float(lo), "hi": float(hi)}


def event_sums_ratio_ci(rows_a, rows_b, event_col="date", n_col="n", sse_col="sse",
                        n_boot=1000, seed=0, ci=0.95):
    """CI for method A's MSE divided by method B's, over the same resampled days.

    The two methods must be scored on the identical resample or the interval is
    meaningless: a draw that happens to miss the crisis days makes *every*
    method look good, and an unpaired interval reports that shared swing as
    uncertainty about the comparison, which can span orders of magnitude.
    Paired, it cancels. Both methods see the same days, so the row counts cancel
    too and the ratio is just sum(sse_a) / sum(sse_b).
    """
    merged = rows_a[[event_col, n_col, sse_col]].merge(
        rows_b[[event_col, n_col, sse_col]], on=event_col, suffixes=("_a", "_b")
    )
    merged = merged[(merged[f"{n_col}_a"] > 0) & (merged[f"{n_col}_b"] > 0)]
    sse_a = merged[f"{sse_col}_a"].to_numpy(float)
    sse_b = merged[f"{sse_col}_b"].to_numpy(float)
    if len(sse_a) == 0:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sse_a), size=(n_boot, len(sse_a)))
    stats = sse_a[draws].sum(axis=1) / sse_b[draws].sum(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return {"point": float(sse_a.sum() / sse_b.sum()), "lo": float(lo), "hi": float(hi)}


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
