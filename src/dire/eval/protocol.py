"""The scorecard: everything a fitted method is judged on, for one fold.

All reference quantities (tail thresholds, relevance) come from the training
slice; events are calendar days (the frozen definition). The extreme gap,
val-tail error minus train-tail error, is the memorization diagnostic: a
crammer aces the extremes it has seen and flunks the ones it has not.
"""

import numpy as np

from dire.data.panel import DATE, TARGET_RAW
from dire.eval import metrics as M

TAIL_QUANTILES = (0.95, 0.90)


def score_predictions(train_df, train_pred, val_df, val_pred):
    y_tr = train_df[TARGET_RAW].to_numpy(dtype=float)
    y_va = val_df[TARGET_RAW].to_numpy(dtype=float)
    events = val_df[DATE].to_numpy()
    relevance = M.relevance_fn(y_tr)

    scores = {
        "mse": M.mse(y_va, val_pred),
        "mae": M.mae(y_va, val_pred),
        "sera": M.sera(y_va, val_pred, relevance),
        "per_event_mse": M.per_event_mse(y_va, val_pred, events),
        "n_val_rows": int(len(val_df)),
        "n_val_events": int(val_df[DATE].nunique()),
    }
    for q in TAIL_QUANTILES:
        thr = M.tail_threshold(y_tr, q)
        key = f"tail{int(q * 100)}"
        scores[f"{key}_mse"] = M.tail_mse(y_va, val_pred, thr)
        scores[f"per_event_{key}_mse"] = M.per_event_tail_mse(y_va, val_pred, events, thr)
        scores[f"n_{key}_rows"] = int((y_va >= thr).sum())
        train_tail = M.tail_mse(y_tr, np.asarray(train_pred, float), thr)
        scores[f"train_{key}_mse"] = train_tail
        scores[f"extreme_gap_{key}"] = scores[f"{key}_mse"] - train_tail
    return scores


def score_method(method, train_df, val_df):
    """Fit-free scoring: the method must already be fitted."""
    return score_predictions(train_df, method.predict(train_df), val_df, method.predict(val_df))
