"""Turning grid results into the numbers the journal quotes.

Two rules run through everything here.

Ratios are taken **inside a cell** and then averaged, never the other way
round. A cell is one (grid, fold, seed): the log baseline is refitted in every
one of them, so dividing within the cell removes the fold's own difficulty
before anything is pooled. Ratio-of-means would instead let the single hardest
fold set every method's headline number.

Windows too thin to score are dropped, not averaged in. A validation block that
holds a handful of extreme rows reports an extreme-error that is mostly noise,
so `usable_windows` removes any (grid, fold) holding fewer than `min_rows` of
them.
"""

import numpy as np
import pandas as pd

BASELINE = "log_target"
MIN_TAIL_ROWS = 20
CELL = ["grid", "fold", "seed"]


def usable_windows(scores, tail="tail95", min_rows=MIN_TAIL_ROWS):
    """Drop (grid, fold) blocks holding fewer than `min_rows` extreme rows.

    The row count is a property of the window, not of the method, so it is the
    same for every method and seed inside a block.
    """
    n_col = f"n_{tail}_rows"
    keep = scores.groupby(["grid", "fold"])[n_col].transform("max") >= min_rows
    return scores[keep].reset_index(drop=True)


def cell_ratios(scores, metric, baseline=BASELINE, cell=CELL):
    """Per-cell ratio of `metric` to the baseline's score in the same cell."""
    cell = [c for c in cell if c in scores.columns]
    base = (scores[scores["method"] == baseline]
            .groupby(cell)[metric].mean().rename("_base"))
    out = scores.merge(base, left_on=cell, right_index=True, how="inner")
    out["ratio"] = out[metric] / out["_base"]
    return out.drop(columns="_base")


def ratio_table(scores, metrics, by=None, baseline=BASELINE, sort_by=None):
    """Mean per-cell ratio per method, optionally split by a column (e.g. rho)."""
    by = [by] if isinstance(by, str) else list(by or [])
    frames = []
    for metric in metrics:
        r = cell_ratios(scores, metric, baseline=baseline)
        frames.append(r.groupby(["method", *by])["ratio"].mean().rename(metric))
    table = pd.concat(frames, axis=1)
    if by:
        table = table.reset_index().pivot(index="method", columns=by[0], values=metrics[0])
    if sort_by is not None:
        table = table.sort_values(sort_by)
    return table


def spread_table(scores, metric, methods=None, baseline=BASELINE):
    """Per method: the mean, best and worst per-cell ratio, and the spread.

    A method can have a decent average and still be unusable, if the average is
    a good fit and a catastrophic one taking turns. The spread says which.
    """
    r = cell_ratios(scores, metric, baseline=baseline)
    if methods is not None:
        r = r[r["method"].isin(methods)]
    g = r.groupby("method")["ratio"]
    out = pd.DataFrame({"mean": g.mean(), "best": g.min(), "worst": g.max(), "n_fits": g.size()})
    out["spread"] = out["worst"] / out["best"]
    return out.sort_values("mean")


def paired_gain(scores, method, against, metric, n_boot=2000, seed=0, ci=0.95):
    """How much `method` improves on `against`, cell by cell.

    Both are scored on exactly the same fits, so the comparison is paired and
    the resampling unit is the cell.
    """
    cell = [c for c in CELL if c in scores.columns]
    wide = (scores[scores["method"].isin([method, against])]
            .pivot_table(index=cell, columns="method", values=metric))
    wide = wide.dropna()
    if wide.empty or method not in wide or against not in wide:
        return {"n_cells": 0, "win_rate": float("nan"), "mean_gain_pct": float("nan"),
                "lo": float("nan"), "hi": float("nan")}
    gain = 100.0 * (1.0 - wide[method] / wide[against])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(gain), size=(n_boot, len(gain)))
    stats = gain.to_numpy()[draws].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return {"n_cells": int(len(gain)), "win_rate": float((gain > 0).mean()),
            "mean_gain_pct": float(gain.mean()), "lo": float(lo), "hi": float(hi)}


def dose_response(scores, method, metric, dose="rho", baseline=BASELINE,
                  n_boot=2000, seed=0, ci=0.95):
    """Slope of the method's per-cell ratio against the dose, with a CI.

    The claim under test is that reweighting gets *worse* as units move together
    more, which is a positive slope. Cells are resampled within each dose level,
    so the CI carries the fold-and-seed noise the point estimate hides.
    """
    r = cell_ratios(scores, metric, baseline=baseline)
    r = r[r["method"] == method].dropna(subset=["ratio", dose])
    x, y = r[dose].to_numpy(float), r["ratio"].to_numpy(float)
    if len(np.unique(x)) < 2:
        return {"slope": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": len(x)}
    levels = [np.flatnonzero(x == v) for v in np.unique(x)]
    rng = np.random.default_rng(seed)
    slopes = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([lv[rng.integers(0, len(lv), len(lv))] for lv in levels])
        slopes[b] = np.polyfit(x[idx], y[idx], 1)[0]
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(slopes, [alpha, 1.0 - alpha])
    return {"slope": float(np.polyfit(x, y, 1)[0]), "lo": float(lo), "hi": float(hi),
            "n": int(len(x))}


def markdown(table, labels=None, float_fmt="{:.2f}", index_name="method", int_cols=()):
    """Small markdown renderer, so the journal tables come out of the code."""
    df = table.copy()
    for col in int_cols:
        df[col] = df[col].map(lambda v: f"{int(v):,}")
    if labels is not None:
        df.index = [labels.get(i, i) for i in df.index]
    header = f"| {index_name} | " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|---" * (len(df.columns) + 1) + "|"
    lines = [header, rule]
    for name, row in df.iterrows():
        lines.append(f"| {name} | " + " | ".join(_cell(v, float_fmt) for v in row) + " |")
    return "\n".join(lines)


def _cell(v, float_fmt):
    if not isinstance(v, (int, float, np.integer, np.floating)) or not np.isfinite(v):
        return str(v)
    return float_fmt.format(v)
