"""Phase 2 gate: nothing from validation, holdout, or the future can touch training.

Rebuilt from the suite this repo historically had. All tests run on synthetic
data only, so a fresh clone passes before any download.
"""

import numpy as np
import pandas as pd
import pytest

from dire.data.panel import DATE, TARGET_DATE, TARGET_LOG, TARGET_RAW, build_supervised, feature_columns
from dire.data.synthetic import generate_panel
from dire.eval.fold_stats import FOLD_STATISTICS
from dire.eval.splits import TemporalSplits, split_frame

FOLD_COUNTS = [1, 3, 5]


@pytest.fixture(scope="module")
def panel():
    return generate_panel(40, 400, rho=0.4, seed=11).drop(columns="latent_common")


@pytest.fixture(scope="module")
def supervised(panel):
    return build_supervised(panel)


def _splits(supervised, n_folds):
    return TemporalSplits(supervised[DATE], n_folds=n_folds)


def _shuffle_after(panel, cutoff, seed=0):
    rng = np.random.default_rng(seed)
    out = panel.copy()
    mask = out[DATE] > cutoff
    out.loc[mask, "y"] = rng.permutation(out.loc[mask, "y"].to_numpy())
    return out


# --- sealed holdout --------------------------------------------------------

def test_holdout_is_sealed_by_default(supervised):
    splits = _splits(supervised, 3)
    with pytest.raises(RuntimeError, match="sealed"):
        splits.holdout_dates()
    with pytest.raises(RuntimeError, match="sealed"):
        splits.holdout_dates(confirm="yes")
    assert len(splits.holdout_dates(confirm=True)) > 0


@pytest.mark.parametrize("n_folds", FOLD_COUNTS)
def test_holdout_never_appears_in_train_or_val(supervised, n_folds):
    splits = _splits(supervised, n_folds)
    holdout = splits.holdout_dates(confirm=True)
    for fold in splits.folds:
        train_df, val_df = split_frame(supervised, fold)
        for df in (train_df, val_df):
            assert not df[DATE].isin(holdout).any()
            assert not df[TARGET_DATE].isin(holdout).any(), "a target reached into the holdout"


# --- target construction ---------------------------------------------------

def test_target_is_next_day_not_same_day(supervised):
    assert (supervised[TARGET_DATE] > supervised[DATE]).all()


def test_y_log_matches_y_raw(supervised):
    assert np.allclose(np.exp(supervised[TARGET_LOG]), supervised[TARGET_RAW], rtol=1e-12)


# --- future shuffle --------------------------------------------------------

def test_features_invariant_to_future_shuffle(panel, supervised):
    cutoff = supervised[DATE].sort_values().unique()[250]
    shuffled = build_supervised(_shuffle_after(panel, cutoff))
    before, before_shuffled = (
        df[df[DATE] <= cutoff].reset_index(drop=True) for df in (supervised, shuffled)
    )
    assert len(before) == len(before_shuffled)
    for col in feature_columns(supervised):
        assert np.array_equal(before[col], before_shuffled[col]), f"{col} saw the future"


def test_future_shuffle_test_can_actually_fail(panel, supervised):
    # the canary: a deliberately leaky feature (the target itself) must trip
    # exactly the comparison the test above relies on
    cutoff = supervised[DATE].sort_values().unique()[250]
    leaky = supervised.assign(f_leak=supervised[TARGET_LOG])
    leaky_shuffled = build_supervised(_shuffle_after(panel, cutoff)).assign(
        f_leak=lambda df: df[TARGET_LOG]
    )
    before = leaky[leaky[DATE] <= cutoff].reset_index(drop=True)
    before_shuffled = leaky_shuffled[leaky_shuffled[DATE] <= cutoff].reset_index(drop=True)
    assert not np.array_equal(before["f_leak"], before_shuffled["f_leak"])


# --- fold statistics -------------------------------------------------------

@pytest.mark.parametrize("stat_name", list(FOLD_STATISTICS))
@pytest.mark.parametrize("n_folds", FOLD_COUNTS)
def test_fold_statistics_invariant_to_corrupting_test_rows(panel, supervised, n_folds, stat_name):
    stat = FOLD_STATISTICS[stat_name]
    splits = _splits(supervised, n_folds)
    for fold in splits.folds:
        reference = stat(split_frame(supervised, fold)[0])
        corrupted = panel.copy()
        mask = corrupted[DATE].isin(splits.test_dates(fold))
        corrupted.loc[mask, "y"] = corrupted.loc[mask, "y"] * 100 + 7
        train_df = split_frame(build_supervised(corrupted), fold)[0]
        assert np.array_equal(reference, stat(train_df)), (
            f"{stat_name} changed when test rows were corrupted"
        )


@pytest.mark.parametrize("stat_name", list(FOLD_STATISTICS))
@pytest.mark.parametrize("n_folds", FOLD_COUNTS)
def test_fold_statistics_invariant_to_deleting_test_rows(panel, supervised, n_folds, stat_name):
    stat = FOLD_STATISTICS[stat_name]
    splits = _splits(supervised, n_folds)
    for fold in splits.folds:
        reference = stat(split_frame(supervised, fold)[0])
        pruned = panel[~panel[DATE].isin(splits.test_dates(fold))]
        train_df = split_frame(build_supervised(pruned), fold)[0]
        assert np.array_equal(reference, stat(train_df)), (
            f"{stat_name} changed when test rows were deleted"
        )


# --- embargo ---------------------------------------------------------------

@pytest.mark.parametrize("n_folds", FOLD_COUNTS)
def test_embargo_between_train_and_validation(supervised, n_folds):
    splits = _splits(supervised, n_folds)
    for train_dates, val_dates in splits.folds:
        gap = splits.timeline.get_loc(val_dates.min()) - splits.timeline.get_loc(train_dates.max())
        assert gap > splits.embargo


def test_fold_statistics_do_depend_on_training_rows(panel, supervised):
    # guard against vacuous passes above: corrupting TRAIN rows must move every statistic
    splits = _splits(supervised, 3)
    fold = splits.folds[0]
    corrupted = panel.copy()
    mask = corrupted[DATE].isin(fold[0])
    corrupted.loc[mask, "y"] = corrupted.loc[mask, "y"] * 100 + 7
    train_ref = split_frame(supervised, fold)[0]
    train_bad = split_frame(build_supervised(corrupted), fold)[0]
    for name, stat in FOLD_STATISTICS.items():
        assert not np.array_equal(stat(train_ref), stat(train_bad)), f"{name} ignores its input"
