# Imbalanced Regression Under Correlated Extremes

## What this paper investigates

Deep Imbalanced Regression (DIR) addresses a real problem: when the values you
care about are rare, a model trained on ordinary loss largely ignores them.
Methods such as Label Distribution Smoothing (LDS, Yang et al., ICML 2021)
estimate the label density and weight each sample inversely to it, so rare
regions of the target contribute more to the loss.

Our concern is what that does under correlation. A single rare event can
generate hundreds of highly correlated observations: every S&P 500 stock on the
day Lehman fell is one story told 500 times. DIR weights each of those rows
heavily, so one event can dominate training, and the model may overfit a handful
of historical events instead of learning what extremes look like in general.
The signature would be low error on training extremes and high error on unseen
ones.

This paper tests whether that happens systematically, across datasets with
different degrees of correlation, and compares DIR against sampling and
weighting alternatives that respect event structure.

## Datasets

Correlation is a **design variable**, not an accident of whichever dataset was
available. See DATASETS.md for provenance and DATASETS.md plus JOURNAL.md Phase
1 for the measured numbers.

| tier | data | why it is here |
|---|---|---|
| synthetic | factor panel, correlation swept 0 to 0.8 | correlation is *set*, so any effect traces back to it |
| real, strongly correlated | S&P 500 equity volatility | standardized ICC 0.435, deff ~ 186, day-mean skew +1.07 |
| real, differently correlated | electricity load across zones | heat waves as the analogue of market stress; ICC 0.746, deff ~ 7.7, day-mean skew -0.30 |

The synthetic tier doubles as a correctness check: at zero correlation the
design effect must come out at 1.00 (measured 0.99), and if it does not, the
diagnostic is broken rather than the data.

The panel has to match the real one on *two* properties. Clustering alone is not
enough: days that move together but whose averages are symmetric give correlated
observations and no crises, so there is nothing for a tail method to overfit.
The common factor is therefore built to stay right-skewed after filtering,
calibrated so the day-mean skew at the S&P's own ICC matches the S&P's.

## Methods compared

- **Loss weighting**: none, inverse frequency, sqrt-inverse, LDS, and a
  design-effect correction that discounts the sample count by how much the
  samples repeat one another.
- **Data sampling**: none, random under/oversampling, SMOTER, and
  **cluster-aware sampling** that resamples whole events rather than rows. If
  the concern is that one event is counted hundreds of times, sampling at the
  event level addresses it structurally rather than by reweighting.
- **Modern DIR regularizers**: FDS, RankSim, Balanced MSE, so the finding is
  about the reweighting principle as the field practices it today, not one 2021
  method.
- **The honest baseline**: a plain model on a log-transformed target, no
  weighting or sampling. Logs compress a right-skewed tail so ordinary squared
  error stops ignoring the region that matters. It estimates no densities,
  assigns no weights, and cannot concentrate training mass on a few events,
  which makes it the bar every method above must clear.

## Core claims

Stated before the experiments ran. Each carries the verdict they returned; the
numbers behind them are in JOURNAL.md Phase 5.

1. **Dose-response.** DIR reweighting's benefit on extreme-value *test* error
   degrades as intra-event correlation rises, all else fixed. The synthetic
   sweep measures exactly this.
   > **Supported.** LDS tail error against the log baseline climbs with rho at
   > slope **+2.22, 95% CI [+1.16, +3.37]**; inverse frequency **+2.41
   > [+1.33, +3.74]**. Reweighting goes from ~1.2x the baseline at rho = 0 to
   > 3.2-3.5x at rho = 0.8. The effect is specific to that family: sampling and
   > the plain model move only 0.85 to ~1.15 across the same dial and HAR does
   > not move, so it is not the panel getting harder. The correction halves the
   > slope to **+1.14 [+0.52, +1.73]** without removing it.
2. **Mechanism.** Under clustering, inverse-density weights concentrate
   effective training mass on a few historical events and the model memorizes
   them: train-set extreme error falls while unseen-event extreme error rises.
   The arithmetic is stark. An average S&P day holds ~400 stocks at ICC 0.435,
   giving design effect 1 + 426 x 0.435 ~ 186 on a Kish mean day of 427, so
   those rows carry about 2.3 independent observations. An equicorrelated event
   can never contribute more than 1/ICC = 2.3 no matter how many stocks trade.
   LDS counts, and upweights, all 400.
   > **Concentration confirmed everywhere; memorization confirmed on the
   > synthetic panel and contradicted on S&P 500.** LDS gives the top 10 days
   > **83.8x** their equal share on S&P 500 against **2.5x** on electricity,
   > tracking deff 186 against 7.7. On the sweep the diagnostic fires and scales
   > with the dose: LDS's unseen-over-seen ratio goes **0.98, 0.91, 1.00, 1.26,
   > 3.15** across rho = 0 to 0.8, while sampling and the plain model reach only
   > ~1.5. On S&P 500 that ratio is **0.75** (LDS is worse on its own training
   > extremes) and the failure mode is instability instead: 1.00x to 13.71x
   > across 9 fits, a 14-fold spread, where the correction spans 1.2-fold. One
   > mechanism does not cover both datasets.
3. **Correction.** Fixes that respect event structure, design-effect-discounted
   weights and event-level sampling, keep the rare-region benefit without the
   generalization loss, and the log-target baseline is harder to beat than the
   DIR literature suggests.
   > **A real improvement, not a fix.** It reduces exactly to LDS at rho = 0 as
   > designed (gain +0.0%, CI [0.0%, 0.0%], all 15 fits), beats LDS by
   > **12.2% (95% CI [+2.0%, +22.0%])** wherever rho > 0, and turns S&P 500's
   > 4.78x tail error into **0.88x**; event-level sampling lands at 0.88x and
   > plain oversampling at 0.79x. On electricity it is 11% *worse* than plain
   > LDS, which the concentration numbers predict. Three caveats the ablations
   > force: at rho = 0.8 it still scores 2.23x the log baseline; plain weight
   > capping matches it on tail error (0.70 against 0.70) and beats it on
   > overall error (0.72 against 0.85); and merging days into multi-day episodes
   > beats both (0.51), so the frozen day-level event definition is wrong for
   > this data. **The log-baseline claim holds and is the strongest result
   > here**: taking logs beats every reweighting method on both real datasets
   > and across the whole sweep.

## Positioning

As of September 2026 no published work studies DIR failure as a function of
sample correlation, and none imports the survey-sampling design effect into loss
weighting. The standard DIR benchmarks (IMDB-WIKI-DIR, AgeDB-DIR, STS-B-DIR) are
essentially i.i.d., which is precisely the point. The neighbours this paper
builds on and differs from:

- **Yang et al., ICML 2021** (arXiv:2102.09554): LDS/FDS, the method under study.
- **"Deconstructing deep imbalanced regression"**, Artificial Intelligence
  Review 2026 (doi:10.1007/s10462-026-11570-1): the field's current map and
  benchmark; its evaluation protocols are i.i.d., which is this paper's opening.
- **Cui et al., CVPR 2019** (arXiv:1901.05555): class-balanced loss via the
  "effective number of samples", the classification analogue. Their redundancy
  comes from feature-space overlap, ours from clustered dependence among rows
  generated by one event.
- **Ribeiro & Moniz, Machine Learning 2020**: SERA, the imbalanced-regression
  community's evaluation metric. Implemented in `dire.eval.metrics.sera` and
  recorded per fit, but not reported in the current results.
- **Modern DIR baselines**: Focal-R; RankSim (ICML 2022, arXiv:2205.15236);
  Balanced MSE (CVPR 2022); ConR (ICLR 2024, arXiv:2309.06651); Dist Loss
  (arXiv:2411.15216).

## Research roadmap

Phases 1 to 5 are done and narrated in JOURNAL.md.

### Phase 1, data (done 2026-09-01, generator corrected 2026-09-02)

- [x] Synthetic factor panel (`dire.data.synthetic`): right-skewed common factor
      via `crisis_factor`, ICC(log y | day) = rho by construction, `shot_weight`
      calibrated so day-mean skew at rho = 0.435 matches the S&P's +1.07.
- [x] Gates in `tests/test_data.py`: design effect 0.99 at rho = 0, ICC
      recovered across the sweep, and right skew asserted on the factor, the day
      means and the top percentile of days, since ICC alone cannot see whether
      the panel has crises in it.
- [x] S&P 500 volatility: 492 tickers, 2,716,698 rows, 2000-2026, Parkinson
      target, standardized ICC 0.435, deff ~ 186.
- [x] Electricity load: OPSD zonal peak plus ERA5 capital temperatures, 20,978
      rows, standardized ICC 0.746, deff ~ 7.7 across 10 zones.
- [x] Data-quality filters on both real panels at construction time, so they are
      not fold statistics: H/L >= 5 dropped from the S&P panel (8 rows), peaks
      above 2x a zone's median dropped from the load panel (1 row).
- [x] Event definitions frozen (DATASETS.md); classical baselines wired in
      (`dire.methods.classical`).

### Phase 2, leakage tests (done 2026-09-01)

- [x] Sealed holdout (`dire.eval.splits.TemporalSplits`): access requires
      `confirm=True` and it has not been opened; never intersects train or val.
- [x] Target is next-day, never same-day; features invariant to shuffling future
      rows, with a canary test proving the check can fail.
- [x] Per-fold statistics (`dire.eval.fold_stats`) invariant to corrupting or
      deleting test rows, for 1/3/5 folds with a 5-day embargo.
- [x] Early stopping never sees the scored block: `inner_split` carves the last
      20% of training dates off behind the same embargo, and the fitted model is
      proven invariant to corrupting validation targets.
- [x] Every suite run appended to `TESTLOG.md` by `scripts/run_tests.py`.

### Phase 3, methods (done 2026-09-01)

Identical architecture, optimizer and tuning budget across methods.

- [x] Weighting (`dire.methods.weighting`): none, inverse frequency,
      sqrt-inverse, LDS, design-effect-corrected LDS. The correction keeps
      1/deff of the row-level LDS weight (deff = 1 + (m - 1) rho-hat, estimated
      on the training slice only) and judges the rest at the event level:
      day-mean rarity, split across the day's rows.
- [x] Sampling (`dire.methods.sampling`) and modern DIR (`dire.methods.mlp`),
      plus the shared MLP with early stopping and per-seed determinism.
- [x] 17 named methods behind one fit/predict interface
      (`dire.methods.registry`), plus 6 ablation variants used only in Phase 5.

### Phase 4, evaluation protocol (done 2026-09-01)

- [x] Blocked temporal splits with an embargo; no event straddles a split.
- [x] Metrics (`dire.eval.metrics`): overall MSE, tail MSE on the top 5% with
      train-derived thresholds, and the train-vs-val tail-error gap as the
      memorization diagnostic (`dire.eval.protocol.score_predictions`).
- [x] Cluster bootstrap over events, never rows (`dire.eval.bootstrap`):
      percentile CIs and paired comparisons on identical event resamples; tests
      confirm honest intervals are more than 2x wider than row bootstrapping
      would claim.

### Phase 5, experiments (run 2026-09-01, rerun on corrected data 2026-09-02)

- [x] The central result: tail test error and the train-vs-unseen gap against
      rho, per method (5 rho x 15 methods x 5 seeds x 3 folds). Dose-response
      slope +2.22, 95% CI [+1.16, +3.37]; the gap widens 0.98 to 3.15 across the
      dial. All fifteen windows hold enough tail rows to score, thinnest 46.
- [x] Both real datasets under blocked temporal CV: electricity with all 17
      methods x 5 seeds x 3 folds, S&P 500 with the 10-method core x 3 seeds x
      3 folds.
- [x] Mechanism deep-dive (`dire.eval.mechanism`): top-k weight share, seen vs
      unseen decomposition, leave-one-event-out refits.
- [x] Ablations: LDS kernel width, weight capping, event-definition sensitivity,
      robustness to error in rho-hat.
- [x] Reproducible reporting: `scripts/make_figures.py` rebuilds every
      `journal/*.png` and prints every table the journal quotes, from
      `results/experiments/*.csv`. The aggregation lives in `dire.eval.report`
      and is unit-tested: per-cell ratios against the refitted log baseline
      (never ratio-of-means, which lets the hardest fold speak for every
      method), windows below 20 tail rows dropped rather than averaged in, and
      cluster-bootstrap intervals over fits.

### Phase 6, analysis

- [ ] `HYPOTHESES.md`
- [ ] Effect sizes with cluster-bootstrap intervals, not significance stars.
- [ ] Datasets where the effect fails to appear are reported.
