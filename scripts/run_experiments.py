#!/usr/bin/env python3
"""Phase 5 experiment presets.

Usage: run_experiments.py [sweep|load|ablations|sp500|all]
Each preset writes results/experiments/<name>.csv (plus a tracked run dir).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from dire.data.io import PROCESSED_DIR
from dire.data.panel import DATE, build_supervised
from dire.data.synthetic import generate_panel
from dire.eval.mechanism import leave_one_event_out, weight_table
from dire.eval.splits import TemporalSplits, inner_split, split_frame
from dire.experiments import run_grid
from dire.runs import RESULTS_DIR

OUT = RESULTS_DIR / "experiments"
RHOS = [0.0, 0.2, 0.4, 0.6, 0.8]
SEEDS5 = [0, 1, 2, 3, 4]
SWEEP_METHODS = ["vanilla", "log_target", "inverse", "sqinv", "lds", "lds_fds", "lds_deff",
                 "under", "over", "smoter", "cluster", "fds", "ranksim", "bmc", "har"]


def stamp(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _save(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    stamp(f"wrote {OUT / f'{name}.csv'} ({len(df)} rows)")


def _synthetic_sup(rho, data_seed):
    panel = generate_panel(60, 800, rho=rho, seed=data_seed).drop(columns="latent_common")
    return build_supervised(panel)


def sweep():
    frames = []
    for i, rho in enumerate(RHOS):
        sup = _synthetic_sup(rho, 100 + i)
        # full 3-fold CV: at high rho, extremes cluster so hard that a single
        # validation window can contain zero extreme days
        frames.append(run_grid(f"sweep_rho{int(rho * 10):02d}", sup, SWEEP_METHODS, SEEDS5,
                               n_folds=3, extra={"rho": rho}, log=stamp))
    _save(pd.concat(frames, ignore_index=True), "synthetic_sweep")


def load():
    panel = pd.read_parquet(PROCESSED_DIR / "load_panel.parquet")
    sup = build_supervised(panel, covariates=("temp_max",), extra_lags=(6,), add_target_dow=True)
    methods = SWEEP_METHODS + ["seasonal_naive", "temp_gbm"]
    _save(run_grid("load", sup, methods, SEEDS5, n_folds=3, log=stamp), "load")


def ablations():
    sup = _synthetic_sup(0.8, 104)  # same data as the rho=0.8 sweep point
    methods = ["lds", "lds_cap", "lds_narrow", "lds_wide",
               "lds_deff", "lds_deff_lo", "lds_deff_hi", "lds_deff_episode"]
    _save(run_grid("ablations", sup, methods, [0, 1, 2], n_folds=3,
                   extra={"rho": 0.8}, log=stamp), "ablations")


def sp500():
    panel = pd.read_parquet(PROCESSED_DIR / "sp500_vol_panel.parquet")
    sup = build_supervised(panel)
    methods = ["vanilla", "log_target", "inverse", "sqinv", "lds", "lds_deff",
               "over", "cluster", "bmc", "har"]
    _save(run_grid("sp500", sup, methods, [0, 1, 2], n_folds=3,
                   model_kwargs=dict(batch_size=4096, max_epochs=60), log=stamp), "sp500")


def mechanism():
    """The deep-dive: where the training weight lands, and what one event is worth."""
    frames = []
    for name, sup in [("synthetic_rho08", _synthetic_sup(0.8, 104)),
                      ("load", build_supervised(
                          pd.read_parquet(PROCESSED_DIR / "load_panel.parquet"),
                          covariates=("temp_max",), extra_lags=(6,), add_target_dow=True)),
                      ("sp500", build_supervised(
                          pd.read_parquet(PROCESSED_DIR / "sp500_vol_panel.parquet")))]:
        fold = TemporalSplits(sup[DATE], n_folds=3).folds[-1]
        fit_df, _ = inner_split(split_frame(sup, fold)[0])
        frames.append(weight_table(fit_df).assign(dataset=name))
        stamp(f"weight concentration done: {name}")
    _save(pd.concat(frames, ignore_index=True), "weight_concentration")

    sup = _synthetic_sup(0.8, 104)  # same panel as the ablations and the rho = 0.8 sweep point
    fold = TemporalSplits(sup[DATE], n_folds=3).folds[-1]
    _save(leave_one_event_out(sup, ["log_target", "lds", "lds_deff", "cluster"], fold,
                              n_events=8, seeds=(0, 1, 2), log=stamp), "loeo")


PRESETS = {"sweep": sweep, "load": load, "ablations": ablations, "sp500": sp500,
           "mechanism": mechanism}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    names = list(PRESETS) if which == "all" else [which]
    for n in names:
        stamp(f"=== {n} ===")
        PRESETS[n]()
    stamp("all done")
