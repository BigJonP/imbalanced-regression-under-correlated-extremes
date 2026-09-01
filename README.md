# Imbalanced Regression Under Correlated Extremes

---

## What this paper investigates

Deep Imbalanced Regression (DIR) methods address a real problem: when the values
you care about are rare, a model trained on ordinary loss will largely ignore
them. Methods such as Label Distribution Smoothing (LDS, Yang et al., ICML 2021)
correct for this by estimating the label density and assigning each sample a
weight inversely proportional to it, so rare regions of the target contribute
more to the loss.

However, one potential issue with DIR is that assigning substantially larger
weights to rare samples may cause the model to overfit to the specific rare events
observed in the training data. Rather than learning the underlying characteristics
of the phenomenon and generalizing them to unseen events, the model may instead
learn patterns that are specific to the individual historical events represented
in the training set. This could manifest as lower prediction error on
rare/extreme samples in the training set, while performance on rare/extreme
samples in the test set deteriorates.

This issue may be particularly pronounced when working with strongly correlated
data. A single rare event can generate a large number of highly correlated
observations, all of which may exhibit similar extreme behaviour. When DIR is
applied, each of these observations receives a large weight, potentially causing
the same underlying event to have a disproportionately large influence on the
model during training. Consequently, the model may effectively overfit to a small
number of historical events rather than learning characteristics that generalize
across different occurrences of the rare phenomenon.

In this paper, we investigate whether this effect occurs systematically across
several datasets with different degrees of temporal and cross-sectional
correlation. We then compare DIR with alternative data-sampling and loss-weighting
approaches to determine which methods are most effective for learning rare events
while maintaining generalization to unseen events in strongly correlated data.

---

## Datasets

Correlation is treated as a **design variable**, not an accident of whichever
dataset was available:

| tier | data | why it is here |
|---|---|---|
| synthetic | factor panel, correlation swept from 0 to 0.8 | correlation is *set*, so any effect can be traced against it |
| real, strongly correlated | S&P 500 equity volatility | measured standardized ICC = 0.435, deff ≈ 186 |
| real, differently correlated | electricity load across zones | heat waves as the structural analogue of market stress; measured standardized ICC ≈ 0.75 |

The synthetic tier doubles as a correctness check on the whole pipeline: at zero
correlation the design effect must come out at 1.00, and if it does not, the
diagnostic is wrong rather than the data.

## Methods compared

- **Loss weighting** — none, inverse frequency, LDS, and a design-effect
  correction that discounts the sample count for how much the samples repeat one
  another.
- **Data sampling** — none, random under/oversampling, SMOTER, and
  **cluster-aware sampling** that resamples whole events rather than individual
  rows. If the concern is that one event is counted hundreds of times, sampling at
  the event level addresses it structurally rather than by reweighting.
- **Modern DIR regularizers** — FDS (feature distribution smoothing), RankSim,
  and Balanced MSE, so the finding is about the reweighting principle as the
  field practices it today, not about one 2021 method.
- **The honest baseline** — a plain model on a log-transformed target, with no
  weighting or sampling at all. Log-transforming a strongly right-skewed target
  compresses the tail, so an ordinary squared-error loss stops ignoring the
  region that matters. It estimates no densities, assigns no weights, and cannot
  concentrate training mass on a handful of events — which makes it the bar
  every method above must clear before any stronger claim is made.

---

## Core claims

1. **Dose–response.** The benefit of DIR reweighting on extreme-value *test*
   error degrades as intra-event correlation rises, all else held fixed. The
   synthetic correlation sweep measures exactly this.
2. **Mechanism.** Under clustering, inverse-density weights concentrate the
   effective training mass on a few historical events, and the model memorizes
   them: train-set extreme error falls while unseen-event extreme error rises.
   The arithmetic is stark. With intraclass correlation ρ = 0.4, a crash day
   touching all 500 stocks has design effect 1 + 499 × 0.4 ≈ 200.6, so its 500
   rows carry 500 / 200.6 ≈ 2.5 rows' worth of independent information — and an
   equicorrelated event can never contribute more than 1/ρ = 2.5 effective
   observations, no matter how many stocks trade that day. LDS counts, and
   upweights, all 500.
3. **Correction.** Fixes that respect event structure — design-effect-discounted
   weights and event-level sampling — keep the rare-region benefit without the
   generalization loss, and the log-target baseline is harder to beat than the
   DIR literature suggests.

---

## Positioning

As of September 2026: no published work studies DIR
failure as a function of sample correlation, and none imports the survey-sampling
design effect into loss weighting. The standard DIR benchmarks (IMDB-WIKI-DIR,
AgeDB-DIR, STS-B-DIR) are essentially i.i.d. — which is precisely the point.
The neighbours this paper builds upon and differes from are: 

- **Yang et al., ICML 2021** (arXiv:2102.09554) — LDS/FDS; the method under study.
- **“Deconstructing deep imbalanced regression”**, Artificial Intelligence Review,
  2026 (doi:10.1007/s10462-026-11570-1) — the field's current map and benchmark;
  its evaluation protocols are i.i.d., which is this paper's opening.
- **Cui et al., CVPR 2019** (arXiv:1901.05555) — class-balanced loss via the
  “effective number of samples”: the classification analogue. Their redundancy
  comes from feature-space overlap; ours from clustered dependence among rows
  generated by one event.
- **Ribeiro & Moniz, Machine Learning 2020** — SERA, the imbalanced-regression
  community's evaluation metric; reported here for exactly that reason.
- **Modern DIR baselines** — Focal-R; RankSim (ICML 2022, arXiv:2205.15236);
  Balanced MSE (CVPR 2022); ConR (ICLR 2024, arXiv:2309.06651); Dist Loss
  (arXiv:2411.15216).
---

## Research roadmap

### Phase 1 — data (three tiers, scripted downloads; see DATASETS.md) (done 2026-09-01)
- [x] Synthetic factor panel (`dire.data.synthetic`): heavy-tailed common
      factor, ICC(log y | day) = ρ by construction; a new seed draws fresh
      unseen events.
- [x] The ρ = 0 null gate: measured design effect = 1.00, enforced in
      `tests/test_data.py`.
- [x] S&P 500 volatility: 492 tickers, 2.7M rows, 2000–2026, Parkinson target;
      measured standardized ICC = 0.435 (the ≈ 0.41 claim confirmed),
      deff ≈ 186 — an average market day carries ~2 independent observations.
- [x] Electricity load: OPSD zonal peak load (CC-BY-4.0) + ERA5 capital
      temperatures, fully scripted; measured standardized ICC = 0.745,
      deff ≈ 7.7 across 10 zones.
- [x] Event definitions frozen (DATASETS.md): day = primary cluster everywhere;
      heat-wave episodes at q = 0.95 with train-only thresholds; the 2018/2019/
      2020 European heat waves are exactly what gets flagged.
- [x] Classical baselines wired in (`dire.methods.classical`): HAR-RV,
      seasonal-naive, temperature GBM.

### Phase 2 — leakage tests (done 2026-09-01)
- [x] Sealed holdout (`dire.eval.splits.TemporalSplits`): access requires
      confirm=True; never intersects train/val, targets included.
- [x] Target is next-day, never same-day; log target consistent with raw target.
- [x] Features invariant to shuffling future rows — plus the canary test proving
      the check can fail, and a guard that fold statistics cannot pass vacuously.
- [x] Per-fold statistics (LDS bin and quantile edges, market scaler, ICC;
      `dire.eval.fold_stats`) invariant to corrupting or deleting test rows,
      for 1/3/5 walk-forward folds with a 5-day embargo.
- [x] Every suite run is appended to `TESTLOG.md` by
      `scripts/run_tests.py`; the paper's reproducibility statement points at
      that log.

### Phase 3 — methods (identical architecture, optimizer, and tuning budget across methods) (done 2026-09-01)
- [x] Weighting (`dire.methods.weighting`): none · inverse frequency ·
      sqrt-inverse · LDS · design-effect-corrected LDS. The correction keeps
      1/deff of the row-level LDS weight (deff = 1 + (m − 1)ρ̂, ρ̂ estimated on
      the training slice only) and judges the remaining share at the event
      level: day-mean rarity, split across the day's rows. A pure division by
      deff is a no-op on balanced panels — the tests exposed this before any
      experiment ran.
- [x] Sampling (`dire.methods.sampling`): none · random under/over · SMOTER ·
      cluster-aware oversampling of whole events.
- [x] Modern DIR (`dire.methods.mlp`): FDS · RankSim · Balanced MSE, plus the
      shared MLP with early stopping and per-seed determinism.
- [x] Honest baseline: the same MLP on the log target.
- [x] Grid restraint: 17 named methods behind one fit/predict interface
      (`dire.methods.registry`), sampling crossed with the vanilla loss only.

### Phase 4 — evaluation protocol
- [ ] Blocked temporal splits with an embargo gap; no event ever straddles a
      split.
- [ ] Metrics: overall MSE/MAE; tail MSE on the top 5% and 10%; SERA; and the
      train-vs-test extreme-error gap as the memorization diagnostic.
- [ ] Per-event errors: aggregate within an event before averaging across
      events — the row-level average is exactly the mistake this paper is about.
- [ ] At least 5 seeds; cluster bootstrap over events (never over rows) for
      confidence intervals and paired comparisons.

### Phase 5 — experiments
- [ ] The central figure: extreme-region test error (and the train–test gap)
      against ρ, per method, on the synthetic sweep.
- [ ] Both real datasets, all methods, blocked temporal CV.
- [ ] Mechanism deep-dive: per-event decomposition (seen vs unseen extreme
      events); share of total loss weight carried by the top-k events;
      leave-one-event-out on a subsample.
- [ ] Ablations: LDS kernel width; weight capping alone (is clipping enough?);
      event-definition sensitivity; robustness to error in ρ̂.

### Phase 6 — analysis
- [ ] `HYPOTHESES.md` 
- [ ] Effect sizes with cluster-bootstrap intervals, not significance stars.
- [ ] Datasets where the effect fails to appear are reported.

