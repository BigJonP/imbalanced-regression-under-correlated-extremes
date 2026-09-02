"""Phase 5 gate: the grid runner produces complete, deterministic, tracked results."""

import json

import numpy as np
import pandas as pd
import pytest

from dire.data.panel import DATE, TARGET_RAW, build_supervised
from dire.data.synthetic import generate_panel
from dire.experiments import run_grid
from dire.methods.weighting import WEIGHTINGS


@pytest.fixture(scope="module")
def sup():
    return build_supervised(generate_panel(15, 220, rho=0.4, seed=1).drop(columns="latent_common"))


def test_run_grid_end_to_end(tmp_path, sup):
    scores = run_grid(
        "tiny", sup, ["log_target", "har"], seeds=[0, 1], n_folds=2, last_fold_only=True,
        model_kwargs=dict(hidden=(8,), max_epochs=3, patience=2),
        results_dir=tmp_path, extra={"rho": 0.4},
    )
    assert len(scores) == 4
    expected = {"grid", "method", "seed", "fold", "rho", "mse", "mae", "sera",
                "tail95_mse", "per_event_tail95_mse", "extreme_gap_tail95",
                "train_tail95_mse", "n_val_rows"}
    assert expected <= set(scores.columns)
    assert np.isfinite(scores["mse"]).all()

    (run_dir,) = tmp_path.iterdir()
    assert (run_dir / "scores.csv").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["git"]["sha"]
    assert len((run_dir / "metrics.jsonl").read_text().strip().splitlines()) == 4


def test_run_grid_deterministic(tmp_path, sup):
    kw = dict(model_kwargs=dict(hidden=(8,), max_epochs=3, patience=2), last_fold_only=True)
    a = run_grid("a", sup, ["lds"], seeds=[3], results_dir=tmp_path / "a", **kw)
    b = run_grid("b", sup, ["lds"], seeds=[3], results_dir=tmp_path / "b", **kw)
    # every metric, not just two, and NaN-safe: a tiny fixture window can hold
    # no extreme rows, and NaN == NaN would fail for the wrong reason
    pd.testing.assert_frame_equal(a.drop(columns="grid"), b.drop(columns="grid"))


def test_event_errors_reconstruct_the_scores(tmp_path, sup):
    scores = run_grid(
        "sums", sup, ["log_target", "har"], seeds=[0], n_folds=2, last_fold_only=True,
        model_kwargs=dict(hidden=(8,), max_epochs=3, patience=2), results_dir=tmp_path,
    )
    (run_dir,) = tmp_path.iterdir()
    events = pd.read_parquet(run_dir / "event_errors.parquet")
    for method in scores["method"]:
        g = events[events["method"] == method]
        assert np.isclose(g["sse"].sum() / g["n"].sum(),
                          scores.loc[scores.method == method, "mse"].iloc[0])
        tail = g[g["n_tail95"] > 0]
        assert np.isclose(tail["sse_tail95"].sum() / tail["n_tail95"].sum(),
                          scores.loc[scores.method == method, "tail95_mse"].iloc[0])


def test_empty_tail_window_reports_nan_not_a_number():
    # the rho = 0.8 trap: a validation block can hold no extreme rows at all, and
    # that must surface as NaN rather than quietly averaging into the scoreboard
    from dire.eval.protocol import score_predictions

    train = pd.DataFrame({DATE: pd.to_datetime(["2020-01-01"] * 4),
                          TARGET_RAW: [1.0, 2.0, 3.0, 100.0]})
    val = pd.DataFrame({DATE: pd.to_datetime(["2020-02-01"] * 3), TARGET_RAW: [1.0, 1.1, 1.2]})
    scored = score_predictions(train, np.zeros(4), val, np.zeros(3))
    assert scored["n_tail95_rows"] == 0
    assert np.isnan(scored["tail95_mse"]) and np.isnan(scored["per_event_tail95_mse"])


def test_ablation_weightings_behave(sup):
    lds = WEIGHTINGS["lds"](sup)
    capped = WEIGHTINGS["lds_cap"](sup)
    assert capped.max() < lds.max() or lds.max() <= 10.0
    assert not np.array_equal(WEIGHTINGS["lds_deff_lo"](sup), WEIGHTINGS["lds_deff_hi"](sup))
    assert not np.array_equal(WEIGHTINGS["lds_narrow"](sup), WEIGHTINGS["lds_wide"](sup))
