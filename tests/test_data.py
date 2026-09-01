"""Phase 1 gate: generators, diagnostics, builders, events, classical baselines."""

import numpy as np
import pandas as pd
import pytest

from dire.data.diagnostics import design_effect, intraclass_correlation
from dire.data.events import (
    EVENT, assign_day_events, assign_episode_events, episode_labels, heatwave_flags,
)
from dire.data.panel import HAR_FEATURES, build_supervised, feature_columns
from dire.data.sp500 import parkinson_vol, parse_stooq_daily, stooq_member_name
from dire.data.synthetic import generate_panel
from dire.methods.classical import HARBaseline, SeasonalNaive, TemperatureGBM

RHO_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8]


# --- diagnostics -----------------------------------------------------------

def _gaussian_clusters(rho, n_groups=3000, m=30, seed=0):
    rng = np.random.default_rng(seed)
    common = np.repeat(rng.standard_normal(n_groups), m)
    idio = rng.standard_normal(n_groups * m)
    values = np.sqrt(rho) * common + np.sqrt(1 - rho) * idio
    return values, np.repeat(np.arange(n_groups), m)


@pytest.mark.parametrize("rho", [0.0, 0.4])
def test_icc_recovers_known_correlation(rho):
    values, groups = _gaussian_clusters(rho)
    assert intraclass_correlation(values, groups) == pytest.approx(rho, abs=0.03)


def test_icc_needs_multiple_groups():
    with pytest.raises(ValueError):
        intraclass_correlation([1.0, 2.0], ["a", "a"])


# --- the null gate ---------------------------------------------------------

def test_null_gate_design_effect_is_one_at_rho_zero():
    panel = generate_panel(80, 4000, rho=0.0, seed=0, phi_common=0.3, phi_idio=0.3)
    deff = design_effect(np.log(panel["y"]), panel["date"])
    assert deff == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("rho", RHO_SWEEP)
def test_synthetic_icc_matches_rho(rho):
    panel = generate_panel(80, 4000, rho=rho, seed=1, phi_common=0.3, phi_idio=0.3)
    icc = intraclass_correlation(np.log(panel["y"]), panel["date"])
    assert icc == pytest.approx(rho, abs=0.05)


# --- synthetic panel -------------------------------------------------------

def test_synthetic_reproducible_and_fresh_events():
    a = generate_panel(10, 200, rho=0.4, seed=7)
    b = generate_panel(10, 200, rho=0.4, seed=7)
    c = generate_panel(10, 200, rho=0.4, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["y"], c["y"])


def test_extremes_cluster_by_day_only_under_correlation():
    def distinct_extreme_dates(rho):
        panel = generate_panel(100, 1500, rho=rho, seed=2)
        top = panel.nlargest(len(panel) // 100, "y")
        return top["date"].nunique()

    assert distinct_extreme_dates(0.8) < 0.5 * distinct_extreme_dates(0.0)


def test_synthetic_rejects_bad_rho():
    with pytest.raises(ValueError):
        generate_panel(5, 50, rho=1.0, seed=0)


# --- supervised builder ----------------------------------------------------

def _toy_panel(n_units=3, n_days=40):
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rows = [
        {"unit": f"u{i}", "date": d, "y": float(1 + i + 10 * t)}
        for i in range(n_units)
        for t, d in enumerate(dates)
    ]
    return pd.DataFrame(rows)


def test_target_is_next_observed_day():
    sup = build_supervised(_toy_panel())
    row = sup.iloc[0]
    assert row["target_date"] > row["date"]
    merged = sup.merge(
        _toy_panel().rename(columns={"date": "target_date", "y": "y_next"}),
        on=["unit", "target_date"],
    )
    assert np.allclose(merged["target_raw"], merged["y_next"])


def test_y_log_matches_y_raw():
    sup = build_supervised(_toy_panel())
    assert np.allclose(sup["target_log"], np.log(sup["target_raw"]))


def test_features_use_only_the_past():
    panel = _toy_panel()
    sup = build_supervised(panel)
    cutoff = sup["date"].iloc[len(sup) // 2]
    corrupted = panel.copy()
    future = corrupted["date"] > cutoff
    corrupted.loc[future, "y"] = corrupted.loc[future, "y"] * 100 + 5
    sup2 = build_supervised(corrupted)
    early, early2 = sup[sup["date"] < cutoff], sup2[sup2["date"] < cutoff]
    for col in feature_columns(sup):
        assert np.allclose(early[col], early2[col]), f"{col} looked into the future"


def test_min_history_rows_dropped():
    sup = build_supervised(_toy_panel(n_days=40))
    assert sup.groupby("unit").size().eq(40 - 22).all()


def test_extra_lags_and_covariates_and_dow():
    panel = _toy_panel()
    panel["temp_max"] = 20.0
    sup = build_supervised(panel, covariates=("temp_max",), extra_lags=(6,), add_target_dow=True)
    assert {"f_temp_max", "f_log_lag6", "f_target_dow"} <= set(sup.columns)


# --- events ----------------------------------------------------------------

def test_day_events_one_per_date():
    sup = assign_day_events(_toy_panel())
    assert sup.groupby(EVENT)["date"].nunique().eq(1).all()


def test_episode_merging():
    dates = pd.date_range("2020-07-01", periods=8)
    flags = pd.Series([True, True, False, True, False, False, True, True], index=dates)
    labels = episode_labels(flags, gap=1)
    assert labels.iloc[0] == labels.iloc[1] == labels.iloc[3]  # gap of 1 merged
    assert labels.iloc[6] != labels.iloc[3]                    # gap of 2 splits
    assert labels.iloc[2] is None


def test_episode_events_span_units():
    panel = _toy_panel(n_units=2, n_days=10)
    flag_date = panel["date"].unique()[5]
    flags = pd.Series(
        [d == flag_date for d in panel["date"].unique()], index=panel["date"].unique()
    )
    out = assign_episode_events(panel, flags)
    on_day = out[out["date"] == flag_date]
    assert on_day[EVENT].nunique() == 1
    assert on_day[EVENT].iloc[0].startswith("ep_")


def test_heatwave_flags_threshold():
    temps = pd.Series(np.arange(100.0), index=pd.date_range("2020-01-01", periods=100))
    assert heatwave_flags(temps, q=0.95).sum() == 5


# --- stooq parsing ---------------------------------------------------------

STOOQ_FIXTURE = """<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>
AAPL.US,D,20200102,000000,74.06,75.15,73.8,75.09,135480400,0
AAPL.US,D,20200103,000000,74.29,75.14,74.13,74.36,146322800,0
"""


def test_parse_stooq_daily():
    df = parse_stooq_daily(STOOQ_FIXTURE)
    assert list(df.columns) == ["date", "open", "high", "low", "close"]
    assert df["date"].iloc[0] == pd.Timestamp("2020-01-02")
    assert df["high"].iloc[0] == 75.15


def test_parkinson_matches_hand_computation():
    expected = np.sqrt(np.log(75.15 / 73.8) ** 2 / (4 * np.log(2)))
    assert parkinson_vol([75.15], [73.8])[0] == pytest.approx(expected)


def test_stooq_member_name_handles_share_classes():
    assert stooq_member_name("BRK.B") == "brk-b.us.txt"


# --- classical baselines ---------------------------------------------------

def test_seasonal_naive_exact_on_weekly_pattern():
    # 7-day calendar, matching the load panel this baseline is built for
    dates = pd.date_range("2020-01-06", periods=60)
    panel = pd.DataFrame(
        [{"unit": "z", "date": d, "y": float(10 + d.dayofweek)} for d in dates]
    )
    sup = build_supervised(panel, extra_lags=(6,))
    preds = SeasonalNaive().fit(sup).predict(sup)
    assert np.allclose(preds, sup["target_raw"])


def test_har_beats_unconditional_mean_on_persistent_series():
    panel = generate_panel(30, 800, rho=0.3, seed=3)
    sup = build_supervised(panel.drop(columns="latent_common"))
    half = len(sup) // 2
    train, test = sup.iloc[:half], sup.iloc[half:]
    har_mse = np.mean((HARBaseline().fit(train).predict(test) - test["target_raw"]) ** 2)
    mean_mse = np.mean((train["target_raw"].mean() - test["target_raw"]) ** 2)
    assert har_mse < mean_mse


def test_temperature_gbm_runs():
    panel = _toy_panel()
    panel["temp_max"] = np.linspace(0, 30, len(panel))
    sup = build_supervised(panel, covariates=("temp_max",), add_target_dow=True)
    preds = TemperatureGBM(seed=0).fit(sup).predict(sup)
    assert np.isfinite(preds).all() and (preds > 0).all()


# --- opsd parsing ----------------------------------------------------------

def test_build_load_panel_from_fixtures(tmp_path):
    from dire.data.opsd import build_load_panel

    hours = pd.date_range("2018-07-01", periods=48, freq="h", tz="UTC")
    load = 40000 + np.arange(48) % 24 * 500  # peak = 51500 each day
    csv = tmp_path / "opsd.csv"
    pd.DataFrame(
        {"utc_timestamp": hours, "DE_load_actual_entsoe_transparency": load}
    ).to_csv(csv, index=False)

    temp = tmp_path / "DE.json"
    temp.write_text(
        '{"daily": {"time": ["2018-07-01", "2018-07-02"], "temperature_2m_max": [30.5, 33.0]}}'
    )
    panel = build_load_panel(csv, {"DE": temp})
    assert list(panel.columns) == ["unit", "date", "y", "temp_max"]
    assert len(panel) == 2
    assert (panel["y"] == 51500).all()
    assert panel["temp_max"].tolist() == [30.5, 33.0]
