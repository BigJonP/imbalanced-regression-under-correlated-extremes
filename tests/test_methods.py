"""Phase 3 gate: weighting, sampling, the shared MLP, modern DIR options, registry."""

import numpy as np
import pandas as pd
import pytest

from dire.data.panel import DATE, TARGET_LOG, TARGET_RAW, build_supervised
from dire.data.synthetic import generate_panel
from dire.methods.mlp import MLPRegressor
from dire.methods.registry import METHODS, build_method
from dire.methods.sampling import RARE_QUANTILE, SAMPLERS, _rare_mask
from dire.methods.weighting import WEIGHTINGS

FAST = dict(hidden=(16, 16), max_epochs=12, patience=4)


@pytest.fixture(scope="module")
def sup():
    return build_supervised(generate_panel(30, 300, rho=0.4, seed=3).drop(columns="latent_common"))


@pytest.fixture(scope="module")
def sup_hi():
    return build_supervised(generate_panel(60, 400, rho=0.8, seed=4).drop(columns="latent_common"))


@pytest.fixture(scope="module")
def split(sup):
    dates = np.sort(sup[DATE].unique())
    cut = dates[len(dates) * 2 // 3]
    return sup[sup[DATE] <= cut], sup[sup[DATE] > cut]


# --- weighting -------------------------------------------------------------

@pytest.mark.parametrize("name", list(WEIGHTINGS))
def test_weights_positive_mean_one(sup, name):
    w = WEIGHTINGS[name](sup)
    assert (w > 0).all() and np.isclose(w.mean(), 1.0)


def test_inverse_upweights_rarest(sup):
    w = WEIGHTINGS["inverse"](sup)
    top = sup[TARGET_LOG] >= sup[TARGET_LOG].quantile(0.99)
    assert w[top.to_numpy()].mean() > 5 * w[~top.to_numpy()].mean()


def test_lds_is_smoother_than_inverse(sup):
    # smoothness = total variation of log weight across occupied neighboring bins
    from dire.methods.weighting import _bin_index

    bins = _bin_index(sup)
    def tv(weights):
        per_bin = pd.Series(weights).groupby(bins).mean()
        return np.abs(np.diff(np.log(per_bin.to_numpy()))).sum()

    assert tv(WEIGHTINGS["lds"](sup)) < tv(WEIGHTINGS["inverse"](sup))


def test_deff_discounts_big_correlated_days(sup_hi):
    lds = WEIGHTINGS["lds"](sup_hi)
    deff = WEIGHTINGS["lds_deff"](sup_hi)
    day_weight = pd.DataFrame({"d": sup_hi[DATE], "lds": lds, "deff": deff}).groupby("d").sum()
    top_days = day_weight["lds"].nlargest(10).index
    share_lds = day_weight.loc[top_days, "lds"].sum() / day_weight["lds"].sum()
    share_deff = day_weight.loc[top_days, "deff"].sum() / day_weight["deff"].sum()
    # multiplicity is corrected (strictly less concentration on the big days)
    # while rare-region emphasis is kept (far above the uniform 10/n_days share)
    assert share_deff < 0.95 * share_lds
    assert share_deff > 3 * (10 / day_weight.shape[0])


def test_deff_matches_lds_when_uncorrelated():
    sup0 = build_supervised(
        generate_panel(60, 1500, rho=0.0, seed=5, phi_common=0.3, phi_idio=0.3).drop(
            columns="latent_common"
        )
    )
    lds, deff = WEIGHTINGS["lds"](sup0), WEIGHTINGS["lds_deff"](sup0)
    assert np.corrcoef(lds, deff)[0, 1] > 0.99
    assert np.abs(deff / lds - 1).max() < 0.2


# --- sampling --------------------------------------------------------------

def test_under_keeps_all_rare(sup):
    rng = np.random.default_rng(0)
    out = SAMPLERS["random_under"](sup, rng)
    threshold = sup[TARGET_LOG].quantile(RARE_QUANTILE)
    assert int((out[TARGET_LOG] >= threshold).sum()) == int(_rare_mask(sup).sum())
    assert len(out) < len(sup)


def test_over_balances_and_keeps_originals(sup):
    out = SAMPLERS["random_over"](sup, np.random.default_rng(0))
    assert len(out) > len(sup)
    rare_n = (out[TARGET_LOG] >= sup[TARGET_LOG].quantile(RARE_QUANTILE)).sum()
    assert np.isclose(rare_n, len(out) - rare_n, rtol=0.02)


def test_smoter_synthesizes_within_rare_range(sup):
    out = SAMPLERS["smoter"](sup, np.random.default_rng(0))
    synth = out.iloc[len(sup):]
    rare = sup[_rare_mask(sup)]
    assert len(synth) > 0
    assert synth[TARGET_LOG].between(rare[TARGET_LOG].min(), rare[TARGET_LOG].max()).all()
    assert np.allclose(np.exp(synth[TARGET_LOG]), synth[TARGET_RAW])


def test_cluster_aware_appends_whole_days(sup):
    out = SAMPLERS["cluster_aware"](sup, np.random.default_rng(0))
    appended = out.iloc[len(sup):]
    assert len(appended) > 0
    day_sizes = sup.groupby(DATE).size()
    for d, size in appended.groupby(DATE).size().items():
        assert size % day_sizes[d] == 0, "a day was appended in fragments"


# --- the shared MLP --------------------------------------------------------

def test_mlp_beats_predicting_the_mean(split):
    train, test = split
    model = MLPRegressor(seed=0, target="log", **FAST).fit(train, test)
    mse = np.mean((model.predict(test) - test[TARGET_RAW]) ** 2)
    mse_mean = np.mean((train[TARGET_RAW].mean() - test[TARGET_RAW]) ** 2)
    assert mse < mse_mean


def test_mlp_deterministic_per_seed(split):
    train, test = split
    p1 = MLPRegressor(seed=7, **FAST).fit(train).predict(test)
    p2 = MLPRegressor(seed=7, **FAST).fit(train).predict(test)
    p3 = MLPRegressor(seed=8, **FAST).fit(train).predict(test)
    assert np.array_equal(p1, p2)
    assert not np.array_equal(p1, p3)


def test_sample_weight_changes_the_fit(split):
    train, test = split
    base = MLPRegressor(seed=0, **FAST).fit(train).predict(test)
    w = np.where(train[TARGET_LOG] > train[TARGET_LOG].median(), 10.0, 0.1)
    weighted = MLPRegressor(seed=0, **FAST).fit(train, sample_weight=w).predict(test)
    assert not np.array_equal(base, weighted)


def test_log_target_predicts_positive(split):
    train, test = split
    preds = MLPRegressor(seed=0, target="log", **FAST).fit(train).predict(test)
    assert (preds > 0).all()


def test_early_stopping_stops(split):
    train, test = split
    model = MLPRegressor(seed=0, hidden=(16, 16), max_epochs=200, patience=2).fit(train, test)
    assert model.n_epochs_ < 200


@pytest.mark.parametrize("option", [dict(loss="bmc"), dict(ranksim_lambda=1.0), dict(fds=True)])
def test_dir_options_are_wired_in(split, option):
    train, test = split
    base = MLPRegressor(seed=0, **FAST).fit(train, test).predict(test)
    alt = MLPRegressor(seed=0, **FAST, **option).fit(train, test).predict(test)
    assert np.isfinite(alt).all()
    assert not np.array_equal(base, alt)


# --- registry --------------------------------------------------------------

@pytest.mark.parametrize("name", list(METHODS))
def test_every_method_fits_and_predicts(name):
    panel = generate_panel(20, 220, rho=0.4, seed=6).drop(columns="latent_common")
    panel["temp_max"] = 15 + 10 * np.sin(np.arange(len(panel)) / 500.0)
    frame = build_supervised(panel, covariates=("temp_max",), extra_lags=(6,), add_target_dow=True)
    dates = np.sort(frame[DATE].unique())
    train = frame[frame[DATE] <= dates[150]]
    test = frame[frame[DATE] > dates[150]]
    method = build_method(name, seed=0, **({} if name in ("har", "seasonal_naive", "temp_gbm")
                                           else dict(hidden=(16,), max_epochs=4, patience=2)))
    preds = method.fit(train, test).predict(test)
    assert preds.shape == (len(test),)
    assert np.isfinite(preds).all()
