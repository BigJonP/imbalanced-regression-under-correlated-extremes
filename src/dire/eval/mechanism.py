"""Why reweighting fails under clustering: concentration, seen-vs-unseen, dependence.

Three views of the same claim. `weight_concentration` measures how much of the
training attention lands on a handful of events. `seen_vs_unseen` splits extreme
error by whether the model trained on that event. `leave_one_event_out` removes
one extreme event at a time and refits, so a method that leans on a single
historical crisis shows it.
"""

import numpy as np
import pandas as pd

from dire.data.panel import DATE, TARGET_LOG, TARGET_RAW
from dire.eval import metrics as M
from dire.eval.splits import inner_split
from dire.methods.registry import build_method
from dire.methods.weighting import WEIGHTINGS


def weight_concentration(train_df, weights, k=10):
    """Share of total training weight carried by the top-k events.

    `equal_share` is what those events would get if every event counted the
    same, so the ratio of the two is how many times over the method is paying
    attention to its favourite days.
    """
    w = pd.Series(np.asarray(weights, float), index=train_df.index)
    by_event = w.groupby(train_df[DATE].to_numpy()).sum()
    total = float(by_event.sum())
    ranked = by_event.sort_values(ascending=False)
    n_events = len(ranked)
    top_pct = max(int(round(0.01 * n_events)), 1)
    return {
        "n_events": n_events,
        f"top{k}_share": float(ranked.iloc[:k].sum() / total),
        "top1pct_share": float(ranked.iloc[:top_pct].sum() / total),
        "equal_share": float(k / n_events),
        "concentration": float((ranked.iloc[:k].sum() / total) / (k / n_events)),
    }


def extreme_events(df, q=0.95, threshold=None):
    """Days whose mean target clears `q` of day means. Pass `threshold` to reuse
    a training-derived cut on a slice that must not set its own."""
    day_mean = df.groupby(DATE)[TARGET_LOG].mean()
    thr = day_mean.quantile(q) if threshold is None else threshold
    return day_mean[day_mean >= thr].index, float(thr)


def seen_vs_unseen(scores, tail="tail95"):
    """Extreme error on the events the model trained on, against events it never
    saw. Both sides use the same training-derived cut, so the only thing that
    differs is whether the model has read that story before. A ratio above 1
    means the method does worse on extremes it has not met, which is what
    memorizing a handful of historical crises looks like from outside.
    """
    seen, unseen = f"train_{tail}_mse", f"{tail}_mse"
    g = scores.groupby("method")[[seen, unseen]].mean()
    return pd.DataFrame({
        "seen_extreme_error": g[seen],
        "unseen_extreme_error": g[unseen],
        "unseen_over_seen": g[unseen] / g[seen],
    }).sort_values("unseen_over_seen", ascending=False)


def unseen_event_spread(event_rows, fit_df, val_df, q=0.95, tail="tail95"):
    """Per-event error across the unseen extreme events, method by method.

    The aggregate above can hide its shape: a method can look average while
    failing badly on two or three particular days. This keeps the per-event
    numbers so that concentration is visible.
    """
    _, thr = extreme_events(fit_df, q=q)
    unseen_days, _ = extreme_events(val_df, q=q, threshold=thr)
    n_col, sse_col = f"n_{tail}", f"sse_{tail}"

    out = []
    for (method, seed, fold), g in event_rows.groupby(["method", "seed", "fold"]):
        g = g[g[n_col] > 0]
        per_event = (g[sse_col] / g[n_col]).to_numpy()
        is_unseen = pd.DatetimeIndex(g[DATE]).isin(unseen_days)
        if not is_unseen.any():
            continue
        vals = per_event[is_unseen]
        out.append({
            "method": method, "seed": seed, "fold": fold,
            "n_unseen_extreme_events": int(is_unseen.sum()),
            "mean_error": float(vals.mean()), "median_error": float(np.median(vals)),
            "worst_event_error": float(vals.max()),
            "worst_share": float(vals.max() / vals.sum()),
        })
    return pd.DataFrame(out)


def leave_one_event_out(supervised, methods, fold, n_events=8, seeds=(0,), q=0.95,
                        model_kwargs=None, log=None):
    """Drop one extreme training day at a time, refit, and watch the tail error.

    A method that learned what extremes look like barely notices. A method that
    memorized its handful of crises moves when one of them is taken away, so the
    spread across removals is the quantity of interest.
    """
    from dire.eval.splits import split_frame

    train, val = split_frame(supervised, fold)
    fit_df, es_df = inner_split(train)
    target_days, _ = extreme_events(fit_df, q=q)
    day_mean = fit_df.groupby(DATE)[TARGET_LOG].mean().loc[target_days]
    worst = list(day_mean.sort_values(ascending=False).index[:n_events])

    y_va = val[TARGET_RAW].to_numpy(dtype=float)
    rows = []
    for m in methods:
        for seed in seeds:
            for dropped in [None, *worst]:
                sub = fit_df if dropped is None else fit_df[fit_df[DATE] != dropped]
                fitted = build_method(m, seed=int(seed), **(model_kwargs or {})).fit(sub, es_df)
                pred = fitted.predict(val)
                thr = M.tail_threshold(sub[TARGET_RAW].to_numpy(dtype=float), q)
                rows.append({
                    "method": m, "seed": int(seed),
                    "dropped": "none" if dropped is None else str(pd.Timestamp(dropped).date()),
                    "tail_mse": M.tail_mse(y_va, pred, thr),
                    "mse": M.mse(y_va, pred),
                })
                if log:
                    log(f"loeo {m} seed={seed} dropped={rows[-1]['dropped']} "
                        f"tail={rows[-1]['tail_mse']:.3e}")
    return pd.DataFrame(rows)


def loeo_sensitivity(loeo_rows):
    """Per method: how far the validation tail error moves when one event goes,
    as a percentage of the error with every event present."""
    out = []
    for (method, seed), g in loeo_rows.groupby(["method", "seed"]):
        base = float(g.loc[g.dropped == "none", "tail_mse"].iloc[0])
        removed = g[g.dropped != "none"]["tail_mse"].to_numpy(dtype=float)
        swings = np.abs(removed - base) / base
        out.append({"method": method, "seed": seed, "base_tail_mse": base,
                    "mean_swing_pct": float(100 * swings.mean()),
                    "max_swing_pct": float(100 * swings.max())})
    return pd.DataFrame(out)


def weight_table(train_df, names=tuple(WEIGHTINGS), k=10):
    """Concentration for every registered weighting on one training frame."""
    return pd.DataFrame([
        {"weighting": n, **weight_concentration(train_df, WEIGHTINGS[n](train_df), k=k)}
        for n in names
    ])
