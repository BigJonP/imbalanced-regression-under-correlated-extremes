"""Phase 4 gate: metrics, SERA, per-event scoring, the gap, cluster bootstrap."""

import numpy as np
import pandas as pd
import pytest

from dire.eval import metrics as M
from dire.eval.bootstrap import cluster_bootstrap_ci, paired_cluster_bootstrap
from dire.eval.mechanism import loeo_sensitivity, weight_concentration
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


# --- mechanism -------------------------------------------------------------

def _clustered_frame(n_days=40, n_units=10, seed=0):
    rng = np.random.default_rng(seed)
    days = np.repeat(pd.date_range("2020-01-01", periods=n_days), n_units)
    level = np.repeat(rng.lognormal(0.0, 1.0, n_days), n_units)
    return pd.DataFrame({"date": days, "unit": np.tile(np.arange(n_units), n_days),
                         "target_raw": level, "target_log": np.log(level)})


def test_uniform_weights_are_not_concentrated():
    df = _clustered_frame()
    flat = weight_concentration(df, np.ones(len(df)), k=5)
    assert flat["concentration"] == pytest.approx(1.0)
    assert flat["top5_share"] == pytest.approx(flat["equal_share"])


def test_concentration_rises_when_weight_piles_onto_few_days():
    df = _clustered_frame()
    top_days = df.groupby("date")["target_log"].mean().nlargest(5).index
    piled = np.where(df["date"].isin(top_days), 50.0, 1.0)
    assert weight_concentration(df, piled, k=5)["concentration"] > 5.0


def test_loeo_sensitivity_is_zero_for_a_method_that_ignores_the_dropped_event():
    rows = pd.DataFrame({
        "method": ["steady"] * 3 + ["fragile"] * 3,
        "seed": [0] * 6,
        "dropped": ["none", "2020-01-01", "2020-01-02"] * 2,
        "tail_mse": [2.0, 2.0, 2.0, 2.0, 3.0, 1.0],
    })
    out = loeo_sensitivity(rows).set_index("method")
    assert out.loc["steady", "mean_swing_pct"] == pytest.approx(0.0)
    assert out.loc["fragile", "mean_swing_pct"] == pytest.approx(50.0)


# --- report helpers --------------------------------------------------------

def _grid_scores():
    """Two folds; fold 1 is ten times harder, and holds too few extreme rows."""
    rows = []
    for fold, (scale, n_tail) in enumerate([(1.0, 300), (10.0, 4)]):
        for seed in range(2):
            for method, factor in [("log_target", 1.0), ("lds", 2.0), ("lds_deff", 0.5)]:
                rows.append({"grid": "g", "method": method, "seed": seed, "fold": fold,
                             "rho": 0.4, "tail95_mse": scale * factor,
                             "n_tail95_rows": n_tail})
    return pd.DataFrame(rows)


def test_usable_windows_drops_the_thin_block():
    from dire.eval.report import usable_windows

    kept = usable_windows(_grid_scores())
    assert kept["fold"].unique().tolist() == [0]


def test_cell_ratios_divide_out_fold_difficulty():
    from dire.eval.report import cell_ratios

    r = cell_ratios(_grid_scores(), "tail95_mse")
    lds = r[r["method"] == "lds"]["ratio"]
    assert np.allclose(lds, 2.0)  # same ratio in the easy and the hard fold


def test_ratio_of_means_would_have_been_dominated_by_one_fold():
    """Why cell_ratios exists: pooling raw scores first lets the hard fold speak
    for every method, and the two answers disagree."""
    from dire.eval.report import cell_ratios

    scores = _grid_scores()
    scores.loc[(scores.fold == 1) & (scores.method == "lds"), "tail95_mse"] = 500.0
    pooled = scores.groupby("method")["tail95_mse"].mean()
    naive = pooled["lds"] / pooled["log_target"]
    honest = cell_ratios(scores, "tail95_mse").query("method == 'lds'")["ratio"].mean()
    assert naive == pytest.approx(45.6, abs=0.1)   # the hard fold speaks for everyone
    assert honest == pytest.approx(26.0, abs=0.1)  # each fold speaks for itself


def test_spread_table_separates_steady_from_erratic():
    from dire.eval.report import spread_table

    scores = _grid_scores()
    scores.loc[(scores.fold == 1) & (scores.method == "lds"), "tail95_mse"] = 500.0
    out = spread_table(scores, "tail95_mse")
    assert out.loc["lds", "spread"] > 20.0
    assert out.loc["lds_deff", "spread"] == pytest.approx(1.0)


def test_paired_gain_is_paired_and_signed():
    from dire.eval.report import paired_gain

    out = paired_gain(_grid_scores(), "lds_deff", "lds", "tail95_mse")
    assert out["n_cells"] == 4
    assert out["win_rate"] == 1.0
    assert out["mean_gain_pct"] == pytest.approx(75.0)


def test_dose_response_recovers_a_planted_slope():
    from dire.eval.report import dose_response

    rows = []
    for rho in [0.0, 0.2, 0.4, 0.6, 0.8]:
        for fold in range(3):
            rows.append({"grid": f"g{rho}", "method": "log_target", "seed": 0,
                         "fold": fold, "rho": rho, "tail95_mse": 1.0})
            rows.append({"grid": f"g{rho}", "method": "lds", "seed": 0,
                         "fold": fold, "rho": rho, "tail95_mse": 1.0 + 2.0 * rho})
    out = dose_response(pd.DataFrame(rows), "lds", "tail95_mse")
    assert out["slope"] == pytest.approx(2.0)
    assert out["lo"] == pytest.approx(2.0) and out["hi"] == pytest.approx(2.0)


def test_dose_response_interval_covers_the_truth_under_noise():
    from dire.eval.report import dose_response

    rng = np.random.default_rng(0)
    rows = []
    for rho in [0.0, 0.2, 0.4, 0.6, 0.8]:
        for fold in range(3):
            for seed in range(5):
                ident = {"grid": f"g{rho}", "seed": seed, "fold": fold, "rho": rho}
                rows.append({**ident, "method": "log_target", "tail95_mse": 1.0})
                rows.append({**ident, "method": "lds",
                             "tail95_mse": 1.0 + 2.0 * rho + rng.normal(scale=0.3)})
    out = dose_response(pd.DataFrame(rows), "lds", "tail95_mse")
    assert out["lo"] < 2.0 < out["hi"]
    assert out["hi"] - out["lo"] > 0.05  # a real interval, not a collapsed one


def test_markdown_renders_a_table():
    from dire.eval.report import markdown

    table = pd.DataFrame({"a": [1.0], "b": [2.0]}, index=["lds"])
    text = markdown(table, labels={"lds": "LDS"})
    assert text.splitlines()[0] == "| method | a | b |"
    assert text.splitlines()[-1] == "| LDS | 1.00 | 2.00 |"


def test_paired_ratio_bootstrap_is_narrower_than_the_unpaired_one():
    """Why event_sums_ratio_ci exists: crisis days move every method at once, so
    an unpaired interval on a ratio reports that shared swing as uncertainty
    about the comparison."""
    from dire.eval.bootstrap import event_sums_bootstrap_ci, event_sums_ratio_ci

    rng = np.random.default_rng(0)
    day_hardness = np.exp(rng.normal(0, 1.5, 200))  # a few days dominate
    a = pd.DataFrame({"date": np.arange(200), "n": 50, "sse": 50 * 2.0 * day_hardness})
    b = pd.DataFrame({"date": np.arange(200), "n": 50, "sse": 50 * 1.0 * day_hardness})

    paired = event_sums_ratio_ci(a, b, n_boot=500)
    assert paired["point"] == pytest.approx(2.0)
    assert paired["hi"] - paired["lo"] < 1e-9  # the shared hardness cancels exactly

    ca, cb = (event_sums_bootstrap_ci(f, n_boot=500) for f in (a, b))
    unpaired_width = (ca["hi"] - ca["lo"]) / cb["point"]
    # taken separately the shared hardness does not cancel: the interval spans
    # a large fraction of the ratio it is supposed to be pinning down
    assert unpaired_width > 0.4 * paired["point"]


def test_paired_ratio_bootstrap_still_reports_real_disagreement():
    from dire.eval.bootstrap import event_sums_ratio_ci

    rng = np.random.default_rng(1)
    a = pd.DataFrame({"date": np.arange(200), "n": 10,
                      "sse": rng.exponential(2.0, 200)})
    b = pd.DataFrame({"date": np.arange(200), "n": 10,
                      "sse": rng.exponential(1.0, 200)})
    out = event_sums_ratio_ci(a, b, n_boot=500)
    assert out["lo"] < out["point"] < out["hi"]
    assert out["lo"] > 1.0  # a genuine two-fold gap is still detected
