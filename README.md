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
| real, strongly correlated | S&P 500 equity volatility | measured intra-class correlation ≈ 0.41 |
| real, differently correlated | electricity load across zones | heat waves as the structural analogue of market stress |

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

The target venue is TMLR, where the bar is claims–evidence alignment, so the
claims are stated up front where the experiments can falsify them:

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

Checked against the literature in September 2026: no published work studies DIR
failure as a function of sample correlation, and none imports the survey-sampling
design effect into loss weighting. The standard DIR benchmarks (IMDB-WIKI-DIR,
AgeDB-DIR, STS-B-DIR) are essentially i.i.d. — which is precisely the point.
The neighbours this paper must cite and differ from:

- **Yang et al., ICML 2021** (arXiv:2102.09554) — LDS/FDS; the method under study.
- **“Deconstructing deep imbalanced regression”**, Artificial Intelligence Review,
  2026 (doi:10.1007/s10462-026-11570-1) — the field's current map and benchmark;
  its evaluation protocols are i.i.d., which is this paper's opening.
- **Moniz, Branco & Torgo, 2017** (Int. J. Data Science and Analytics) —
  resampling for imbalanced *time-series* forecasting; the closest data-level
  relative. No event clustering, no deep nets, no design effect. Likewise their
  2021 spatio-temporal extension.
- **Cui et al., CVPR 2019** (arXiv:1901.05555) — class-balanced loss via the
  “effective number of samples”: the classification analogue. Their redundancy
  comes from feature-space overlap; ours from clustered dependence among rows
  generated by one event.
- **Ribeiro & Moniz, Machine Learning 2020** — SERA, the imbalanced-regression
  community's evaluation metric; reported here for exactly that reason.
- **Modern DIR baselines** — Focal-R; RankSim (ICML 2022, arXiv:2205.15236);
  Balanced MSE (CVPR 2022); ConR (ICLR 2024, arXiv:2309.06651); Dist Loss
  (arXiv:2411.15216).
- **Evaluation under dependence** — Bergmeir & Benítez 2012 on time-series CV;
  Roberts et al. 2017 (Ecography) on blocked CV; hv-blocked CV; Lones, “How to
  avoid machine learning pitfalls” (arXiv:2108.02497).
- **Extremes in time series** — Ding et al., KDD 2019 (extreme value loss with
  memory networks); MBB-RW (Scientific Reports, 2025), block-bootstrap
  resampling with relevance weighting.
- **Kish, 1965** — the design effect itself: deff = 1 + (m − 1)ρ.

---

## Research roadmap

Venue: TMLR (rolling submissions, claims–evidence bar, code encouraged).
Compute envelope: one GPU; MLP-scale models throughout.

### Phase 0 — infrastructure (done 2026-09-01)
- [x] `git init` and first commit. No CI by decision: the suite is run manually
      with `uv run scripts/run_tests.py`, and every run is appended to
      `TESTLOG.md` together with the tree state that was tested.
- [x] Layout: `src/dire/` (data, methods, eval), `configs/` (one YAML per run),
      `scripts/`, `tests/`, `paper/`.
- [x] Every run writes `results/<run_id>/` carrying its config, git SHA, seed,
      and a metrics manifest (`dire.runs.Run`, gated by
      `tests/test_infrastructure.py`).
- [x] `pyproject.toml` with `uv.lock`; all seeds set in one place
      (`dire.seeding.set_all_seeds`).

### Phase 1 — data (three tiers, scripted downloads, documented licenses)
- [ ] Synthetic factor panel: N assets × T days, heavy-tailed common factor,
      correlation knob ρ ∈ {0, 0.2, 0.4, 0.6, 0.8}; fresh unseen events can be
      generated at test time.
- [ ] The ρ = 0 null gate: the measured design effect must come out 1.00 before
      anything else runs.
- [ ] S&P 500 volatility: free EOD OHLCV from a source with a documented license
      (e.g. a Stooq dump — not ToS-gray scraping); target = next-day Parkinson or
      Garman–Klass volatility; re-measure and report the ICC (the ≈ 0.41 above).
- [ ] Electricity load: GEFCom2014 or Open Power System Data zonal load with
      temperature; heat waves as the cross-zone correlated extreme.
- [ ] Event definitions frozen per dataset before any experiment (finance:
      calendar day, optionally multi-day episodes; load: heat-wave episodes
      spanning zones).
- [ ] Classical baselines wired in: HAR-RV for volatility; seasonal-naive and a
      temperature GBM for load.

### Phase 2 — leakage tests (rebuilding the suite this repo once had)
- [ ] Sealed holdout: untouched by default and never intersecting train/val.
- [ ] Target is next-day, never same-day; log target consistent with raw target.
- [ ] Features invariant to shuffling future rows — plus a test proving that
      test can fail.
- [ ] Per-fold statistics (LDS bin and quantile edges, scalers, ICC) invariant
      to corrupting or deleting test rows.
- [ ] Every suite run is appended to `TESTLOG.md` by
      `scripts/run_tests.py`; the paper's reproducibility statement points at
      that log.

### Phase 3 — methods (one shared interface; identical architecture, optimizer, and tuning budget across methods)
- [ ] Weighting: none · inverse frequency · LDS · design-effect-corrected LDS
      (each weight divided by its event's 1 + (m − 1)ρ̂, with ρ̂ estimated on
      training folds only).
- [ ] Sampling: none · random under/over · SMOTER · cluster-aware sampling of
      whole events.
- [ ] Modern DIR: FDS · RankSim · Balanced MSE (ConR optional).
- [ ] Honest baseline: plain MLP on the log target.
- [ ] Grid restraint: sampling crossed with the vanilla loss only; roughly 15–20
      configs × 5 seeds — sized for one GPU.

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
- [ ] `HYPOTHESES.md` committed — and thereby timestamped — before the first
      real-data run.
- [ ] Effect sizes with cluster-bootstrap intervals, not significance stars.
- [ ] Datasets where the effect fails to appear are reported, not dropped.

### Phase 7 — paper
- [ ] TMLR format, sources under `paper/`; the intro leads with the worked
      design-effect example.
- [ ] One small proposition: under cluster sampling, the variance of the
      weighted gradient estimator inflates by the design effect.
- [ ] Pre-empted objections, in text: (a) “isn't this just blocked CV /
      clustered standard errors?” — evaluation correction is necessary but not
      sufficient; the training loss itself is distorted, and that is what gets
      quantified and fixed; (b) “history contains only one 2008” — the synthetic
      tier generates genuinely unseen events, and the real tiers are evaluated
      leave-event-out.
- [ ] Limitations and reproducibility statement; anonymized code mirror for
      review.

### Phase 8 — release and submission
- [ ] One command per figure and table; download scripts with checksums; a
      license file.
- [ ] TMLR submission on OpenReview; optionally a workshop version afterwards
      for visibility.
