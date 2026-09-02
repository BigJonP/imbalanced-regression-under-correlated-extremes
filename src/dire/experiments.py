"""Grid runner: (method x seed x fold) on one supervised frame, tidy scores out.

Every grid is a tracked Run: results/<run_id>/ holds the config, git SHA,
per-fit metrics (metrics.jsonl), the final scores.csv, and per-day error sums
(event_errors.parquet) for the cluster bootstrap.

The fold's validation block is scored and nothing else. Fitting, early stopping
and every weight are confined to the inner split of the training slice, so no
part of the model is chosen on the rows it is graded against.
"""

import numpy as np
import pandas as pd

from dire.data.panel import DATE, TARGET_RAW
from dire.eval import metrics as M
from dire.eval.protocol import TAIL_QUANTILES, score_predictions
from dire.eval.splits import TemporalSplits, inner_split, split_frame
from dire.methods.registry import build_method
from dire.runs import Run

CLASSICAL = {"har", "seasonal_naive", "temp_gbm"}


def event_error_rows(fit_df, val_df, val_pred):
    """Per-day squared-error sums and counts, overall and per tail.

    MSE over any resample of whole days is sum(sse) / sum(n), so these sums
    reconstruct every squared-error metric exactly while staying small enough
    to keep for a 456k-row S&P fold.
    """
    y = val_df[TARGET_RAW].to_numpy(dtype=float)
    err2 = (np.asarray(val_pred, float) - y) ** 2
    y_tr = fit_df[TARGET_RAW].to_numpy(dtype=float)
    out = pd.DataFrame({DATE: val_df[DATE].to_numpy(), "err2": err2, "y": y})

    by_day = out.groupby(DATE)["err2"]
    rows = pd.DataFrame({"n": by_day.size(), "sse": by_day.sum()})
    for q in TAIL_QUANTILES:
        thr = M.tail_threshold(y_tr, q)
        tail = out[out["y"] >= thr].groupby(DATE)["err2"]
        key = f"tail{int(q * 100)}"
        rows[f"n_{key}"] = tail.size().reindex(rows.index, fill_value=0)
        rows[f"sse_{key}"] = tail.sum().reindex(rows.index, fill_value=0.0)
    return rows.reset_index()


def run_grid(name, supervised, methods, seeds, n_folds=3, last_fold_only=False,
             model_kwargs=None, results_dir=None, extra=None, log=None):
    config = {
        "seed": int(seeds[0]),
        "grid": name,
        "methods": list(methods),
        "seeds": [int(s) for s in seeds],
        "n_folds": n_folds,
        "last_fold_only": last_fold_only,
        "model_kwargs": {k: str(v) for k, v in (model_kwargs or {}).items()},
        "extra": {k: str(v) for k, v in (extra or {}).items()},
    }
    run = Run(config, name=name, results_dir=results_dir)
    splits = TemporalSplits(supervised[DATE], n_folds=n_folds)
    folds = splits.folds[-1:] if last_fold_only else splits.folds
    rows, event_rows = [], []
    for fi, fold in enumerate(folds):
        train, val = split_frame(supervised, fold)
        fit_df, es_df = inner_split(train)
        for m in methods:
            for seed in seeds:
                kwargs = {} if m in CLASSICAL else dict(model_kwargs or {})
                fitted = build_method(m, seed=int(seed), **kwargs).fit(fit_df, es_df)
                val_pred = fitted.predict(val)
                ident = {"grid": name, "method": m, "seed": int(seed), "fold": fi}
                scored = score_predictions(fit_df, fitted.predict(fit_df), val, val_pred)
                row = {**ident, **(extra or {}), **scored}
                rows.append(row)
                event_rows.append(event_error_rows(fit_df, val, val_pred).assign(**ident))
                run.log_metrics(row)
                if log:
                    log(f"{name} fold={fi} {m} seed={seed} tail95={row['tail95_mse']:.3e}")
    scores = pd.DataFrame(rows)
    scores.to_csv(run.dir / "scores.csv", index=False)
    pd.concat(event_rows, ignore_index=True).to_parquet(
        run.dir / "event_errors.parquet", index=False
    )
    run.finalize({"n_rows": len(scores)})
    return scores
