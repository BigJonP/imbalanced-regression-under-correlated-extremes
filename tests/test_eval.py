"""Phase 4 gate: metrics, SERA, per-event scoring, the gap, cluster bootstrap."""

import numpy as np
import pandas as pd
import pytest

from dire.eval import metrics as M
from dire.eval.bootstrap import cluster_bootstrap_ci, paired_cluster_bootstrap
from dire.eval.protocol import score_predictions


def test_mse_mae_hand_values():
    y, pred = np.array([1.0, 2.0, 3.0]), np.array([1.0, 4.0, 2.0])
    assert M.mse(y, pred) == pytest.approx(5.0 / 3.0)
    assert M.mae(y, pred) == pytest.approx(1.0)


def test_tail_mse_uses_only_the_tail():
    y = np.array([1.0, 1.0, 10.0, 12.0])
    pred = np.array([99.0, 99.0, 11.0, 13.0])
    assert M.tail_mse(y, pred, threshold=10.0) == pytest.approx(1.0)
    assert np.isnan(M.tail_mse(y, pred, threshold=100.0))


def test_relevance_shape():
    train = np.concatenate([np.ones(50), np.linspace(1, 30, 50)])
    phi = M.relevance_fn(train)
    med = np.median(train)
    assert phi(np.array([med - 1]))[0] == 0.0
    assert phi(np.array([train.max()]))[0] == 1.0
    grid = phi(np.linspace(train.min(), train.max(), 100))
    assert (np.diff(grid) >= -1e-12).all()


def test_sera_zero_for_perfect_and_extreme_sensitive():
    rng = np.random.default_rng(0)
    y = np.exp(rng.normal(size=500))
    phi = M.relevance_fn(y)
    assert M.sera(y, y, phi) == 0.0
    err = np.zeros_like(y)
    low_err, high_err = err.copy(), err.copy()
    low_err[np.argmin(y)] = 1.0
    high_err[np.argmax(y)] = 1.0
    assert M.sera(y, y + high_err, phi) > M.sera(y, y + low_err, phi)


def test_per_event_average_tames_one_giant_event():
    # one 10-row event with error 3, ten 1-row events with error 1
    y = np.zeros(20)
    pred = np.concatenate([np.full(10, 3.0), np.ones(10)])
    events = np.concatenate([np.zeros(10), np.arange(1, 11)])
    row_level = M.mse(y, pred)
    event_level = M.per_event_mse(y, pred, events)
    assert row_level == pytest.approx(5.0)
    assert event_level == pytest.approx((9.0 + 10.0) / 11.0)
    assert event_level < row_level


def test_extreme_gap_flags_a_crammer():
    train = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=50), "target_raw": np.linspace(1, 10, 50)}
    )
    val = pd.DataFrame(
        {"date": pd.date_range("2020-06-01", periods=50), "target_raw": np.linspace(1, 10, 50)}
    )
    crammer_train = train["target_raw"].to_numpy()          # perfect on seen data
    crammer_val = np.full(50, val["target_raw"].mean())     # useless on unseen
    scores = score_predictions(train, crammer_train, val, crammer_val)
    assert scores["extreme_gap_tail95"] > 0
    assert scores["train_tail95_mse"] == pytest.approx(0.0)
    for key in ("mse", "mae", "sera", "per_event_mse", "tail90_mse"):
        assert np.isfinite(scores[key])


def test_cluster_bootstrap_ci_sane_and_deterministic():
    rng = np.random.default_rng(1)
    events = np.repeat(np.arange(40), 25)
    shock = np.repeat(rng.normal(0, 2, 40), 25)  # errors shared within event
    y = rng.normal(size=1000)
    pred = y + shock + rng.normal(0, 0.1, 1000)
    a = cluster_bootstrap_ci(y, pred, events, M.mse, n_boot=200, seed=5)
    b = cluster_bootstrap_ci(y, pred, events, M.mse, n_boot=200, seed=5)
    assert a == b
    assert a["lo"] <= a["point"] <= a["hi"]


def test_cluster_bootstrap_wider_than_row_bootstrap():
    # errors are constant within events, so real uncertainty lives at the event
    # level; a row bootstrap would report far too narrow an interval
    rng = np.random.default_rng(2)
    events = np.repeat(np.arange(30), 40)
    pred_err = np.repeat(rng.normal(0, 1, 30), 40)
    y = np.zeros(1200)
    ci_cluster = cluster_bootstrap_ci(y, pred_err, events, M.mse, n_boot=300, seed=0)
    rows = np.arange(1200)
    row_stats = [
        M.mse(y[idx], pred_err[idx])
        for idx in (rng.choice(rows, 1200) for _ in range(300))
    ]
    row_width = np.quantile(row_stats, 0.975) - np.quantile(row_stats, 0.025)
    assert (ci_cluster["hi"] - ci_cluster["lo"]) > 2 * row_width


def test_paired_bootstrap_separates_clear_winner():
    rng = np.random.default_rng(3)
    events = np.repeat(np.arange(50), 10)
    y = rng.normal(size=500)
    good = y + rng.normal(0, 0.1, 500)
    bad = y + rng.normal(0, 1.0, 500)
    res = paired_cluster_bootstrap(y, good, bad, events, M.mse, n_boot=300, seed=0)
    assert res["diff"] < 0 and res["hi"] < 0 and res["p_value"] < 0.05


def test_paired_bootstrap_no_false_certainty_on_tie():
    rng = np.random.default_rng(4)
    events = np.repeat(np.arange(50), 10)
    y = rng.normal(size=500)
    a = y + rng.normal(0, 0.5, 500)
    b = y + rng.normal(0, 0.5, 500)
    res = paired_cluster_bootstrap(y, a, b, events, M.mse, n_boot=300, seed=0)
    assert res["lo"] < 0 < res["hi"]
