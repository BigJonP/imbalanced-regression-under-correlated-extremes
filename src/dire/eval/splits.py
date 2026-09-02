"""Walk-forward temporal splits with an embargo and a sealed holdout.

Frozen layout: the last `holdout_frac` of dates is the sealed holdout; an
embargo of `embargo` positions separates it from the working timeline. The
second half of the working timeline is cut into `n_folds` sequential validation
blocks; fold k trains on every date ending `embargo` positions before its
block. Embargo >= 1 guarantees a training row's next-day target never lands in
validation. "Test rows" for leakage purposes are validation + holdout dates;
embargo dates belong to neither side.

Inside a fold, `inner_split` cuts a further early-stopping slice off the end of
the training dates, so nothing about the fitted model is chosen on the block it
is scored against.
"""

import numpy as np
import pandas as pd

from dire.data.panel import DATE


class TemporalSplits:
    def __init__(self, dates, n_folds, embargo=5, holdout_frac=0.15):
        if n_folds < 1 or embargo < 1 or not 0 < holdout_frac < 0.5:
            raise ValueError("need n_folds >= 1, embargo >= 1, holdout_frac in (0, 0.5)")
        self.timeline = pd.DatetimeIndex(np.sort(pd.unique(pd.to_datetime(dates))))
        self.embargo = embargo
        n_holdout = int(round(len(self.timeline) * holdout_frac))
        self._holdout = self.timeline[len(self.timeline) - n_holdout:]
        working = self.timeline[: len(self.timeline) - n_holdout - embargo]

        val_start = len(working) // 2
        blocks = np.array_split(np.arange(val_start, len(working)), n_folds)
        self.folds = []
        for block in blocks:
            if len(block) == 0 or block[0] - embargo <= 0:
                raise ValueError("not enough dates for this fold layout")
            self.folds.append((working[: block[0] - embargo], working[block]))

    def holdout_dates(self, confirm=False):
        """Sealed: reading the holdout requires confirm=True, on purpose."""
        if confirm is not True:
            raise RuntimeError(
                "the holdout is sealed until final evaluation; pass confirm=True "
                "only when you mean it"
            )
        return self._holdout

    def test_dates(self, fold):
        """Validation + holdout dates: the rows leakage tests corrupt or delete."""
        train, val = fold
        return val.union(self._holdout)


def split_frame(supervised, fold):
    train_dates, val_dates = fold
    d = supervised[DATE]
    return (
        supervised[d.isin(train_dates)].reset_index(drop=True),
        supervised[d.isin(val_dates)].reset_index(drop=True),
    )


def inner_split(train_df, frac=0.2, embargo=5):
    """Cut an early-stopping set off the end of a training frame.

    The model must never see the fold's validation block while fitting, so the
    epoch it stops at is chosen here instead: the last `frac` of training dates
    become the early-stopping set, separated from the fitting dates by the same
    embargo used between train and validation. Returns (fit_df, es_df).
    """
    if not 0 < frac < 0.5 or embargo < 1:
        raise ValueError("need frac in (0, 0.5) and embargo >= 1")
    dates = pd.DatetimeIndex(np.sort(pd.unique(pd.to_datetime(train_df[DATE]))))
    cut = int(round(len(dates) * (1.0 - frac)))
    if cut - embargo <= 0 or cut >= len(dates):
        raise ValueError("not enough training dates for an early-stopping split")
    d = pd.to_datetime(train_df[DATE])
    return (
        train_df[d.isin(dates[: cut - embargo])].reset_index(drop=True),
        train_df[d.isin(dates[cut:])].reset_index(drop=True),
    )
