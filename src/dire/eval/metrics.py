"""Scoring: overall, tail, SERA, per-event, and the memorization gap.

Thresholds and the relevance function are always derived from TRAINING targets,
never from the slice being scored.
"""

import numpy as np
import pandas as pd


def mse(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return float(np.mean((pred - y) ** 2))


def mae(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return float(np.mean(np.abs(pred - y)))


def tail_threshold(train_targets, q):
    return float(np.quantile(np.asarray(train_targets, float), q))


def tail_mse(y, pred, threshold):
    """MSE on rows whose TRUE value is at or above the (train-derived) threshold."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    mask = y >= threshold
    return float(np.mean((pred[mask] - y[mask]) ** 2)) if mask.any() else float("nan")


def relevance_fn(train_targets):
    """Piecewise-linear relevance: 0 up to the train median, 1 at the extreme
    anchor Q3 + 1.5 IQR (capped at the train max), linear in between. A
    simplification of Ribeiro & Moniz (2020)'s pchip through the same boxplot
    control points."""
    t = np.asarray(train_targets, float)
    q1, med, q3 = np.quantile(t, [0.25, 0.5, 0.75])
    anchor = min(q3 + 1.5 * (q3 - q1), float(t.max()))
    if anchor <= med:
        anchor = med + 1e-12

    def phi(y):
        return np.clip((np.asarray(y, float) - med) / (anchor - med), 0.0, 1.0)

    return phi


def sera(y, pred, relevance, n_steps=101):
    """Squared Error-Relevance Area (Ribeiro & Moniz 2020): integrate, over the
    relevance cutoff t in [0, 1], the summed squared error of all rows with
    relevance >= t. Emphasizes extremes without an arbitrary single cutoff."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    rel = relevance(y)
    err2 = (pred - y) ** 2
    ts = np.linspace(0.0, 1.0, n_steps)
    ser = np.array([err2[rel >= t].sum() for t in ts])
    return float(np.trapezoid(ser, ts))


def per_event_mse(y, pred, events):
    """Aggregate within an event before averaging across events. The plain
    row-level average is exactly the over-counting this paper is about."""
    err2 = pd.Series((np.asarray(pred, float) - np.asarray(y, float)) ** 2)
    return float(err2.groupby(np.asarray(events)).mean().mean())


def per_event_tail_mse(y, pred, events, threshold):
    y = np.asarray(y, float)
    mask = y >= threshold
    if not mask.any():
        return float("nan")
    return per_event_mse(y[mask], np.asarray(pred, float)[mask], np.asarray(events)[mask])
