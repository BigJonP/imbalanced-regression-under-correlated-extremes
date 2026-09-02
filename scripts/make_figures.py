#!/usr/bin/env python3
"""Rebuild every journal figure and every journal table from the tracked results.

Usage: make_figures.py [name ...]   (no arguments: everything)

Nothing in `journal/` is drawn by hand. Each figure below reads either a
processed panel or `results/experiments/*.csv`, so the journal can be
regenerated from the run directories alone.

Form notes that apply throughout. Ratios against a baseline of 1.0 span orders
of magnitude, so they are drawn as dots on a log axis and never as bars: a bar
has to start at zero to be honest, and zero is not on this scale. Correlation
is a magnitude, so it is encoded by one hue getting darker, never by cycling
categorical colors. Methods are identity, and there are up to seventeen of
them, so identity is carried by position on the y-axis with color reserved for
the two-way split the panel is actually about.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dire.data.diagnostics import design_effect, intraclass_correlation, standardize_within_unit
from dire.data.events import episode_labels, heatwave_flags
from dire.data.io import PROCESSED_DIR
from dire.data.panel import DATE, TARGET_LOG, build_supervised
from dire.data.synthetic import generate_panel
from dire.eval import report as R
from dire.eval.mechanism import loeo_sensitivity, seen_vs_unseen, weight_table
from dire.eval.splits import TemporalSplits, inner_split, split_frame
from dire.methods.weighting import WEIGHTINGS
from dire.runs import REPO_ROOT, RESULTS_DIR

JOURNAL = REPO_ROOT / "journal"
EXPERIMENTS = RESULTS_DIR / "experiments"

# Validated against the CVD checks on the #fcfcfb surface: worst adjacent pair
# dE 9.2 (deutan), 27.6 (normal). AQUA sits below 3:1 contrast, so every panel
# using it carries direct labels or a companion table in the journal.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
RHO_RAMP = ["#cfe0f5", "#9ec2eb", "#6ba3e0", "#2a78d6", "#17497f"]  # sequential: rho is a magnitude

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.labelcolor": "#52514e",
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": SURFACE,
})

RHOS = [0.0, 0.2, 0.4, 0.6, 0.8]
SWEEP_UNITS, SWEEP_DAYS = 60, 800
LABEL = {
    "log_target": "log baseline", "vanilla": "plain model", "inverse": "inverse frequency",
    "sqinv": "sqrt-inverse", "lds": "LDS", "lds_cap": "LDS, weights capped",
    "lds_narrow": "LDS, narrow kernel", "lds_wide": "LDS, wide kernel",
    "lds_deff": "LDS + our correction", "lds_deff_lo": "correction, rho halved",
    "lds_deff_hi": "correction, rho inflated 50%",
    "lds_deff_episode": "correction, episode events",
    "over": "oversampling", "under": "undersampling", "smoter": "SMOTER",
    "cluster": "whole-day resampling", "fds": "FDS", "lds_fds": "LDS + FDS",
    "ranksim": "RankSim", "bmc": "Balanced MSE", "har": "classical HAR",
    "seasonal_naive": "seasonal naive", "temp_gbm": "temperature GBM",
    "none": "no weighting",
}


def title(ax, text, y=1.10, x=0.0):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=11.5, color=INK, weight="bold")


def save(fig, name):
    JOURNAL.mkdir(exist_ok=True)
    fig.savefig(JOURNAL / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote journal/{name}")


def load(name):
    return pd.read_csv(EXPERIMENTS / f"{name}.csv")


def sweep_panel(rho, seed):
    return generate_panel(SWEEP_UNITS, SWEEP_DAYS, rho=rho, seed=seed)


def sp500_panel():
    return pd.read_parquet(PROCESSED_DIR / "sp500_vol_panel.parquet")


def load_panel():
    return pd.read_parquet(PROCESSED_DIR / "load_panel.parquet")


def dotplot(ax, frame, columns, colors, labels, order, logx=True):
    """Methods on the y-axis, value by position, one dot per series.

    Position carries identity for as many methods as the panel needs; color is
    left free for the two-way comparison the panel exists to make.
    """
    ypos = np.arange(len(order))
    a = frame.reindex(order)
    if len(columns) == 2:
        ax.hlines(ypos, a[columns[0]], a[columns[1]], color=AXIS, linewidth=1.4, zorder=1)
    for col, color, lab in zip(columns, colors, labels):
        ax.scatter(a[col], ypos, s=52, color=color, zorder=3, label=lab,
                   edgecolors=SURFACE, linewidths=1.2)
    ax.axvline(1.0, color="#52514e", linewidth=1.1, zorder=0)
    if logx:
        ax.set_xscale("log")
    ax.set_yticks(ypos)
    ax.set_yticklabels([LABEL.get(m, m) for m in order], fontsize=8.5)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.grid(axis="y", visible=False)
    return ypos


# --- Phase 1: the data -----------------------------------------------------

def fig_synthetic_extremes():
    """Two things at once: extremes stack into shared days as rho rises, and the
    day means carry a right tail, which is what makes those days crises."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for j, (rho, color) in enumerate([(0.0, BLUE), (0.8, ORANGE)]):
        p = sweep_panel(rho, seed=104)
        d = p.assign(ly=np.log(p.y))
        units = {u: i for i, u in enumerate(sorted(d.unit.unique()))}
        days = {v: i for i, v in enumerate(sorted(d.date.unique()))}

        ax = axes[0, j]
        e = d[d.ly >= d.ly.quantile(0.98)]
        ax.scatter(e.date.map(days), e.unit.map(units), s=3, color=color, alpha=0.55,
                   linewidths=0, rasterized=True)
        ax.set_title(f"correlation = {rho}", fontsize=10, color=INK, pad=8)
        ax.set_xlabel("trading day")
        ax.set_ylabel("stock" if j == 0 else "")
        ax.grid(False)

        ax = axes[1, j]
        dm = d.groupby("date").ly.mean()
        ax.hist(dm, bins=45, color=color, alpha=0.85, linewidth=0)
        ax.set_xlabel("day-mean log volatility")
        ax.set_ylabel("days" if j == 0 else "")
        ax.text(0.96, 0.92, f"skew {dm.skew():+.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#52514e")

    title(axes[0, 0], "The top 2% of observations: scattered, then stacked", x=-0.10)
    title(axes[1, 0], "Crisis days arrive only once the days are shared", x=-0.10, y=1.14)
    fig.tight_layout(h_pad=3.2)
    save(fig, "synthetic_extremes.png")


def fig_icc_knob():
    """The dial does what it claims, and the null gate reads 1.00."""
    requested = np.linspace(0.0, 0.8, 9)
    measured, deffs = [], []
    for rho in requested:
        p = generate_panel(80, 2000, rho=float(rho), seed=7)
        z = standardize_within_unit(np.log(p.y), p.unit)
        measured.append(intraclass_correlation(z, p.date))
        deffs.append(design_effect(z, p.date))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot([0, 0.85], [0, 0.85], color=AXIS, linewidth=1.4, zorder=1)
    ax.scatter(requested, measured, s=60, color=BLUE, zorder=3,
               edgecolors=SURFACE, linewidths=1.2)
    ax.set_xlabel("correlation we asked for")
    ax.set_ylabel("correlation we measured back")
    title(ax, "The dial is honest")

    ax = axes[1]
    ax.plot(requested, deffs, color=BLUE, linewidth=2, zorder=2)
    ax.scatter(requested, deffs, s=42, color=BLUE, zorder=3,
               edgecolors=SURFACE, linewidths=1.2)
    ax.axhline(1.0, color="#52514e", linewidth=1.1, zorder=1)
    ax.annotate(f"{deffs[0]:.2f} at zero correlation", xy=(0.0, deffs[0]),
                xytext=(0.30, max(deffs) * 0.10), fontsize=8.5, color="#52514e",
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=1))
    ax.set_xlabel("correlation we asked for")
    ax.set_ylabel("redundancy factor of a day")
    title(ax, "and the smoke detector works")
    fig.tight_layout(w_pad=3.0)
    save(fig, "icc_knob.png")


def fig_sp500_market_vol():
    p = sp500_panel()
    daily = p.groupby("date").y.mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily.index, daily.to_numpy(), color=BLUE, linewidth=0.8)
    ax.set_ylim(0, daily.max() * 1.30)
    for when, text, dx in [("2008-10-10", "2008 crisis", -60), ("2020-03-16", "COVID crash", 55)]:
        t = pd.Timestamp(when)
        ax.annotate(text, xy=(t, daily.loc[:t].iloc[-1]), xytext=(dx, 16),
                    textcoords="offset points", ha="center", fontsize=9, color="#52514e",
                    arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=1))
    ax.set_ylabel("average volatility across the day's stocks")
    ax.set_xlabel("")
    title(ax, "Crises are shared days, not scattered stocks", y=1.06)
    fig.tight_layout()
    save(fig, "sp500_market_vol.png")


def fig_heatwaves():
    p = load_panel()
    temp = p.groupby("date").temp_max.mean()
    flags = heatwave_flags(temp)
    eps = episode_labels(flags)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(temp.index, temp.to_numpy(), color=AXIS, linewidth=0.8, zorder=1)
    hot = temp[flags]
    ax.scatter(hot.index, hot.to_numpy(), s=14, color=ORANGE, zorder=3,
               linewidths=0, label="flagged heat-wave day")
    biggest = (temp[flags].groupby(eps[flags]).mean().nlargest(3).index)
    for ep in biggest:
        block = temp[eps == ep]
        ax.annotate(str(block.idxmax().date()), xy=(block.idxmax(), block.max()),
                    xytext=(0, 16), textcoords="offset points", ha="center",
                    fontsize=8.5, color="#52514e",
                    arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=1))
    ax.set_ylabel("mean daily max temperature, 10 capitals (C)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    title(ax, "The detector finds the heat waves nobody had to point out", y=1.06)
    fig.tight_layout()
    save(fig, "heatwaves.png")


# --- Phase 2: the splits ---------------------------------------------------

def fig_timeline_splits():
    sup = build_supervised(sweep_panel(0.4, seed=3))
    splits = TemporalSplits(sup[DATE], n_folds=3)
    timeline = splits.timeline
    pos = {d: i for i, d in enumerate(timeline)}
    holdout = splits.holdout_dates(confirm=True)

    fig, ax = plt.subplots(figsize=(11, 3.4))
    for k, (train, val) in enumerate(splits.folds):
        y = len(splits.folds) - 1 - k
        fit, es = inner_split(sup[sup[DATE].isin(train)])
        bars = [
            (fit[DATE].map(pos).min(), fit[DATE].map(pos).max(), BLUE, "fitting"),
            (es[DATE].map(pos).min(), es[DATE].map(pos).max(), AQUA, "early stopping"),
            (pos[val[0]], pos[val[-1]], ORANGE, "scored block"),
        ]
        for lo, hi, color, lab in bars:
            ax.barh(y, hi - lo, left=lo, height=0.5, color=color, linewidth=0,
                    label=lab if k == 0 else None)
        ax.text(-14, y, f"fold {k}", ha="right", va="center", fontsize=9, color="#52514e")
    ax.barh(-1, len(holdout), left=pos[holdout[0]], height=0.5, color="#52514e",
            linewidth=0, label="sealed final exam")
    ax.text(-14, -1, "sealed", ha="right", va="center", fontsize=9, color="#52514e")
    ax.set_yticks([])
    ax.set_xlabel("trading day")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.28))
    title(ax, "Every gap between two coloured blocks is an embargo", y=1.08, x=-0.03)
    fig.tight_layout()
    save(fig, "timeline_splits.png")


# --- Phase 3: where the training attention lands ---------------------------

def _rho08_fit_frame():
    """The fitting slice of the last fold of the rho = 0.8 panel, which is the
    same frame the ablations and the mechanism deep-dive use."""
    sup = build_supervised(sweep_panel(0.8, seed=104))
    fold = TemporalSplits(sup[DATE], n_folds=3).folds[-1]
    return inner_split(split_frame(sup, fold)[0])[0]


def fig_weight_share(fit_df=None):
    fit_df = _rho08_fit_frame() if fit_df is None else fit_df
    table = weight_table(fit_df, names=("none", "lds", "lds_deff", "inverse"), k=10)
    table = table.set_index("weighting")
    equal = float(table["equal_share"].iloc[0])
    order = ["none", "lds_deff", "lds", "inverse"]
    shares = table.loc[order, "top10_share"] * 100

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ypos = np.arange(len(order))
    ax.barh(ypos, shares, height=0.55, color=BLUE, linewidth=0)
    ax.axvline(equal * 100, color="#52514e", linewidth=1.2, zorder=3)
    ax.text(equal * 100 + 0.6, -0.72, f"an equal share is {equal * 100:.1f}%",
            fontsize=8.5, color="#52514e", va="center")
    for y, v in zip(ypos, shares):
        ax.text(v + 0.6, y, f"{v:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels([LABEL[m] for m in order], fontsize=9)
    ax.set_ylim(len(order) - 0.4, -1.0)
    ax.set_xlabel("share of all training attention (%)")
    ax.grid(axis="y", visible=False)
    title(ax, "Training attention on the 10 most extreme days", y=1.14, x=-0.30)
    fig.tight_layout()
    save(fig, "weight_share.png")
    return shares, equal


# --- Phase 4: the referee --------------------------------------------------

# --- Phase 5: the experiments ----------------------------------------------

SWEEP_SHOWN = ["log_target", "har", "cluster", "over", "vanilla", "lds_deff", "lds", "inverse"]


def fig_sweep_dose_response(sweep):
    """Three panels for one question: does reweighting get worse as units move
    together more, and does the correction buy anything when they do."""
    usable = R.usable_windows(sweep)
    ratios = R.cell_ratios(usable, "tail95_mse")
    dose = R.dose_response(usable, "lds", "tail95_mse")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    for m, color in [("lds", ORANGE), ("lds_deff", BLUE), ("vanilla", AQUA)]:
        g = ratios[ratios.method == m].groupby("rho")["ratio"]
        ax.plot(g.mean().index, g.mean().to_numpy(), color=color, linewidth=2, zorder=3)
        ax.scatter(g.mean().index, g.mean().to_numpy(), s=40, color=color, zorder=4,
                   edgecolors=SURFACE, linewidths=1.2)
        last = g.mean()
        ax.annotate(LABEL[m], xy=(last.index[-1], last.iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", fontsize=8.5, color=color, va="center")
    ax.axhline(1.0, color="#52514e", linewidth=1.1, zorder=1)
    ax.set_xlim(-0.05, 1.02)
    ax.set_xlabel("how much units move together")
    ax.set_ylabel("tail error vs log baseline")
    title(ax, "Wild-day error against the dial", y=1.08, x=-0.14)

    ax = axes[1]
    g = ratios[ratios.method == "lds"]
    ax.scatter(g.rho + np.random.default_rng(0).normal(0, 0.008, len(g)), g.ratio,
               s=14, color=ORANGE, alpha=0.4, linewidths=0, zorder=2)
    xs = np.linspace(0, 0.8, 50)
    fit = np.polyfit(g.rho, g.ratio, 1)
    ax.plot(xs, np.polyval(fit, xs), color=ORANGE, linewidth=2, zorder=3)
    ax.axhline(1.0, color="#52514e", linewidth=1.1, zorder=1)
    ax.text(0.03, 0.93, f"slope {dose['slope']:+.2f}\n95% range "
                        f"[{dose['lo']:+.2f}, {dose['hi']:+.2f}]",
            transform=ax.transAxes, va="top", fontsize=9, color="#52514e")
    ax.set_xlabel("how much units move together")
    ax.set_ylabel("LDS tail error, per fit")
    title(ax, "The dose response, fitted", y=1.08, x=-0.14)

    ax = axes[2]
    gains = []
    for rho in sorted(usable.rho.unique()):
        gains.append({"rho": rho, **R.paired_gain(usable[usable.rho == rho],
                                                  "lds_deff", "lds", "tail95_mse")})
    gains = pd.DataFrame(gains)
    ax.hlines(gains.rho, gains.lo, gains.hi, color=BLUE, linewidth=1.4, alpha=0.55, zorder=2)
    ax.scatter(gains.mean_gain_pct, gains.rho, s=52, color=BLUE, zorder=3,
               edgecolors=SURFACE, linewidths=1.2)
    ax.axvline(0.0, color="#52514e", linewidth=1.1, zorder=1)
    ax.set_yticks(sorted(usable.rho.unique()))
    ax.set_ylabel("how much units move together")
    ax.set_xlabel("what the correction buys over LDS (%)")
    ax.grid(axis="y", visible=False)
    title(ax, "and what the correction buys", y=1.08, x=-0.14)

    fig.tight_layout(w_pad=3.4)
    save(fig, "sweep_dose_response.png")
    return dose, gains


def fig_real_data_contest(loadscores, spscores):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    out = {}
    for ax, (name, scores, heading) in zip(axes, [
        ("load", loadscores, "EU electricity: a day holds 10 zones"),
        ("sp500", spscores, "S&P 500: a day holds around 400 stocks"),
    ]):
        usable = R.usable_windows(scores)
        table = R.ratio_table(usable, ["tail95_mse", "mse"]).sort_values("tail95_mse")
        order = list(table.index)
        dotplot(ax, table, ["mse", "tail95_mse"], [BLUE, ORANGE],
                ["overall", "tail"], order)
        ax.set_xlabel("error against the log baseline (1.0 = baseline)")
        title(ax, heading, y=1.06, x=-0.30)
        out[name] = table
    axes[1].legend(frameon=False, fontsize=9, ncol=2, loc="upper center",
                   bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout(w_pad=2.0)
    save(fig, "real_data_contest.png")
    return out


def fig_mechanism_weight_share(conc):
    """How many times their fair share the ten biggest days get, per weighting.

    Dots on a log axis: the values run from 1x to nearly 100x, and a bar would
    have to start at zero to be honest.
    """
    wide = conc.pivot(index="weighting", columns="dataset", values="concentration")
    cols = ["synthetic_rho08", "load", "sp500"]
    names = {"synthetic_rho08": "synthetic, correlation 0.8",
             "load": "EU electricity", "sp500": "S&P 500"}
    wide = wide[cols]
    order = list(wide["sp500"].sort_values().index)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ypos = np.arange(len(order))
    for col, color in zip(cols, [AQUA, BLUE, ORANGE]):
        ax.scatter(wide.loc[order, col], ypos, s=56, color=color, zorder=3,
                   label=names[col], edgecolors=SURFACE, linewidths=1.2)
    for y, m in zip(ypos, order):
        ax.text(wide.loc[m, "sp500"] * 1.14, y, f"{wide.loc[m, 'sp500']:.0f}x",
                va="center", fontsize=8.5, color="#52514e")
    ax.axvline(1.0, color="#52514e", linewidth=1.1, zorder=1)
    ax.set_xscale("log")
    ax.set_yticks(ypos)
    ax.set_yticklabels([LABEL.get(m, m) for m in order], fontsize=9)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.75, wide.to_numpy().max() * 2.6)
    ax.set_xlabel("times their equal share of training attention, top 10 days")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.15))
    title(ax, "Concentration follows how redundant the dataset is", y=1.08, x=-0.28)
    fig.tight_layout()
    save(fig, "mechanism_weight_share.png")
    return wide


def fig_loeo(loeo):
    sens = loeo_sensitivity(loeo)
    order = list(sens.groupby("method").mean_swing_pct.mean().sort_values().index)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ypos = np.arange(len(order))
    rng = np.random.default_rng(0)
    for y, m in zip(ypos, order):
        g = sens[sens.method == m]
        ax.scatter(g.mean_swing_pct, y + rng.normal(0, 0.06, len(g)), s=30,
                   color=AXIS, zorder=2, linewidths=0)
        ax.scatter([g.mean_swing_pct.mean()], [y], s=68, color=ORANGE, zorder=3,
                   edgecolors=SURFACE, linewidths=1.2)
        ax.text(g.mean_swing_pct.mean() + 1.2, y, f"{g.mean_swing_pct.mean():.0f}%",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels([LABEL[m] for m in order], fontsize=9)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("how far the tail error moves when one crisis day is removed (%)")
    ax.scatter([], [], s=30, color=AXIS, linewidths=0, label="one model seed")
    ax.scatter([], [], s=68, color=ORANGE, edgecolors=SURFACE, linewidths=1.2,
               label="average over seeds")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.20))
    title(ax, "Taking one crisis out of the room", y=1.12, x=-0.28)
    fig.tight_layout()
    save(fig, "loeo.png")
    return sens.groupby("method")[["mean_swing_pct", "max_swing_pct"]].mean()


def fig_ablations(abl):
    usable = R.usable_windows(abl)
    table = R.ratio_table(usable, ["tail95_mse", "mse"], baseline="lds")
    table = table.sort_values("tail95_mse")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    # everything here sits within a factor of three, so a linear axis reads
    # better than the log one the cross-method panels need
    dotplot(ax, table, ["mse", "tail95_mse"], [BLUE, ORANGE],
            ["overall", "tail"], list(table.index), logx=False)
    ax.set_xlim(0.35, 1.35)
    ax.set_xlabel("error against plain LDS (1.0 = plain LDS)")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    title(ax, "Ablations, on the most correlated synthetic panel", y=1.09, x=-0.30)
    fig.tight_layout()
    save(fig, "ablations.png")
    return table


# --- the tables the journal quotes -----------------------------------------

def section(name):
    print(f"\n\n{'=' * 72}\n{name}\n{'=' * 72}")


def table_datasets():
    section("Phase 1: the three datasets")
    rows = []
    for name, panel in [("synthetic, correlation 0.8", sweep_panel(0.8, seed=104)),
                        ("S&P 500 volatility", sp500_panel()),
                        ("EU electricity", load_panel())]:
        ly = np.log(panel.y)
        z = standardize_within_unit(ly, panel.unit)
        dm = panel.assign(ly=ly).groupby("date").ly.mean()
        rows.append({"dataset": name, "rows": len(panel), "units": panel.unit.nunique(),
                     "days": panel.date.nunique(),
                     "ICC": intraclass_correlation(z, panel.date),
                     "deff": design_effect(z, panel.date),
                     "row skew": ly.skew(), "day-mean skew": dm.skew(),
                     "day-mean kurt": dm.kurt()})
    out = pd.DataFrame(rows).set_index("dataset")
    print(R.markdown(out, index_name="dataset", int_cols=("rows", "units", "days")))
    return out


def table_synthetic_moments():
    section("Phase 1: what the corrected generator produces")
    rows = []
    for i, rho in enumerate(RHOS):
        p = sweep_panel(rho, seed=100 + i)
        ly = np.log(p.y)
        z = standardize_within_unit(ly, p.unit)
        dm = p.assign(ly=ly).groupby("date").ly.mean()
        rows.append({"rho": rho, "measured ICC": intraclass_correlation(z, p.date),
                     "redundancy factor": design_effect(z, p.date),
                     "row skew": ly.skew(), "day-mean skew": dm.skew(),
                     "day-mean kurt": dm.kurt()})
    out = pd.DataFrame(rows).set_index("rho")
    print(R.markdown(out, index_name="rho"))
    return out


def table_tail_rows(sweep):
    section("Phase 5: extreme rows per validation window")
    out = sweep.groupby(["rho", "fold"])["n_tail95_rows"].first().unstack()
    out.columns = [f"window {c}" for c in out.columns]
    print(R.markdown(out, index_name="rho", float_fmt="{:.0f}"))
    return out


def table_sweep(sweep):
    section("Phase 5: the sweep, tail error against the log baseline")
    usable = R.usable_windows(sweep)
    table = R.ratio_table(usable, ["tail95_mse"], by="rho")
    table = table.reindex([m for m in SWEEP_SHOWN if m in table.index])
    print(R.markdown(table, labels=LABEL, index_name="method"))

    print("\nnull gate, correction against LDS at rho = 0:")
    print(" ", R.paired_gain(usable[usable.rho == 0.0], "lds_deff", "lds", "tail95_mse"))
    print("\ncorrection against LDS wherever units do move together:")
    print(" ", R.paired_gain(usable[usable.rho > 0.0], "lds_deff", "lds", "tail95_mse"))
    return table


MEMORIZATION_SHOWN = ["inverse", "lds", "lds_deff", "over", "vanilla", "cluster", "log_target"]


def table_memorization_by_rho(sweep):
    """The designed diagnostic, split by the dose. Both halves are ratios to the
    log baseline, so their ratio already divides the baseline's own gap out."""
    section("Phase 5: seen against unseen extremes, by rho")
    usable = R.usable_windows(sweep)
    seen = R.cell_ratios(usable, "train_tail95_mse").groupby(["method", "rho"])["ratio"].mean()
    unseen = R.cell_ratios(usable, "tail95_mse").groupby(["method", "rho"])["ratio"].mean()
    for label, tab in [("LDS on the extremes it trained on", seen.unstack().loc[["lds"]]),
                       ("LDS on the extremes it never saw", unseen.unstack().loc[["lds"]])]:
        print(R.markdown(tab.rename(index={"lds": label}), index_name="", float_fmt="{:.2f}"))
    ratio = (unseen / seen).unstack().reindex(MEMORIZATION_SHOWN)
    print("\nunseen over seen, every method:")
    print(R.markdown(ratio, labels=LABEL, index_name="method"))
    return ratio


def table_seen_vs_unseen(scores, name):
    section(f"Phase 5: seen against unseen extremes, {name}")
    usable = R.usable_windows(scores)
    ratios = {}
    for metric in ("train_tail95_mse", "tail95_mse"):
        ratios[metric] = R.cell_ratios(usable, metric).groupby("method")["ratio"].mean()
    out = pd.DataFrame({"error on extremes it trained on": ratios["train_tail95_mse"],
                        "error on extremes it never saw": ratios["tail95_mse"]})
    out["unseen over seen"] = (out.iloc[:, 1] / out.iloc[:, 0])
    print(R.markdown(out.sort_values("unseen over seen", ascending=False), labels=LABEL))
    return out


def table_spread(scores, name, methods):
    section(f"Phase 5: how steady each method is, {name}")
    out = R.spread_table(R.usable_windows(scores), "tail95_mse", methods=methods)
    print(R.markdown(out[["mean", "best", "worst", "spread"]], labels=LABEL))
    return out


def main(which):
    which = set(which)
    if "tables" in which:  # the tables quote what the figures compute
        which |= {"phase3", "phase5"}
    print("figures\n" + "-" * 40)
    need_grids = {"phase5", "tables"} & which
    if need_grids:
        sweep, abl = load("synthetic_sweep"), load("ablations")
        spscores, loadscores = load("sp500"), load("load")
        loeo, conc = load("loeo"), load("weight_concentration")

    if "phase1" in which:
        fig_synthetic_extremes()
        fig_icc_knob()
        fig_sp500_market_vol()
        fig_heatwaves()
    if "phase2" in which:
        fig_timeline_splits()
    if "phase3" in which:
        shares, equal = fig_weight_share()
    if "phase5" in which:
        dose, gains = fig_sweep_dose_response(sweep)
        real = fig_real_data_contest(loadscores, spscores)
        wide = fig_mechanism_weight_share(conc)
        swings = fig_loeo(loeo)
        ablt = fig_ablations(abl)

    if "tables" in which:
        table_datasets()
        table_synthetic_moments()
        if "phase3" in which:
            section("Phase 3: training attention on the top 10 days")
            print((shares.rename("share of all attention (%)").to_frame()
                   .assign(**{"an equal share would be (%)": equal * 100})).round(1))
        table_tail_rows(sweep)
        table_sweep(sweep)
        table_memorization_by_rho(sweep)
        section("Phase 5: dose response and what the correction buys")
        print(" slope:", dose)
        print(R.markdown(gains.set_index("rho")[["win_rate", "mean_gain_pct", "lo", "hi"]],
                         index_name="rho"))
        for name, table in real.items():
            section(f"Phase 5: {name}, error against the log baseline")
            print(R.markdown(table.rename(columns={"tail95_mse": "tail error",
                                                   "mse": "overall error"}), labels=LABEL))
        section("Phase 5: concentration on the top 10 days")
        print(R.markdown(wide, labels=LABEL, index_name="weighting", float_fmt="{:.1f}"))
        section("Phase 5: leave one crisis out")
        print(R.markdown(swings, labels=LABEL, float_fmt="{:.1f}"))
        section("Phase 5: ablations, against plain LDS")
        print(R.markdown(ablt.rename(columns={"tail95_mse": "tail error",
                                              "mse": "overall error"}), labels=LABEL))
        for scores, name in [(sweep, "synthetic sweep"), (loadscores, "EU electricity"),
                             (spscores, "S&P 500")]:
            table_seen_vs_unseen(scores, name)
        table_spread(spscores, "S&P 500", None)
        table_spread(sweep[sweep.rho == 0.8], "synthetic, correlation 0.8", None)


ALL = ["phase1", "phase2", "phase3", "phase5", "tables"]

if __name__ == "__main__":
    main(sys.argv[1:] or ALL)
