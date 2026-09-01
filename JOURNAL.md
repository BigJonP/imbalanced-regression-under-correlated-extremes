# Research Journal

## What this project is about

Machine-learning models are bad at predicting rare, extreme outcomes market
crashes, record heat waves because they see so few of them. The standard fix
("Deep Imbalanced Regression") is to make rare examples count extra during
training. Our suspicion: when one crisis produces hundreds of near-identical
examples (every S&P stock on the day Lehman fell is the *same story* told 500
times), counting them all extra makes the model **memorize the few crises it
has seen** instead of learning what extremes look like in general. We're testing
whether that's true, and whether counting each *event* once rather than each
row fixes it.

---


## 2026-09-01 — Phase 1: the three datasets

### 1. A fake market where we control how much stocks move together

We built a simulator of daily volatility (how wildly a stock's price swings)
for ~100 fake stocks, with one dial: **how much all stocks move together**
(called ρ, "rho", from 0 to 0.8). Because we control the dial, any effect we
find later can be traced directly to it. The picture below shows the whole
thesis in one glance — at ρ = 0 the extreme observations are scattered
everywhere; at ρ = 0.8 they stack into vertical stripes: a few terrible *days*,
each hitting every stock at once.

![extremes scatter vs stripe](journal/synthetic_extremes.png)

### 2. Checking the dial actually works

We dial in a correlation, generate data, and measure the correlation back out.
The dots land on the diagonal the dial does what it claims. At ρ = 0 the
"redundancy alarm" (design effect, more below) correctly reads exactly 1.00.
That's our built-in smoke detector: if it ever reads anything else at zero
correlation, the measuring tools are broken, not the data. This check now runs
automatically in the test suite.

![measured vs requested correlation](journal/icc_knob.png)

### 3. The real stock market, 2000–2026

From a free daily-price archive we computed daily volatility for 
**492 S&P 500 stocks over 6,705 trading days 2.7 million observations**. 
The spikes below are the 2008 financial crisis and the 2020 COVID crash: 
crises are *shared* days, exactly the correlated-extremes situation we 
want to study. Measured togetherness: **0.435** The punchline number: an
average day contains ~405 stock-observations, but because they move together 
so much, they carry only about **2 days' worth of independent information**
(redundancy factor ≈ 186). A reweighting method that counts all 405 is
massively over-counting one story.

![S&P 500 market-wide volatility](journal/sp500_market_vol.png)

### 4. Electricity demand during heat waves

Second real dataset, different domain: daily peak electricity demand for **10
European countries over ~5.7 years** (plus each capital's temperature). Here the
"crisis" is a heat wave hitting the whole continent at once. Our automatic
heat-wave detector (no hand-tuning — just "hotter than 95% of days") finds
exactly the right events: the record July 2019 wave, August 2018, August 2020.
Zones move together even more than stocks: togetherness **0.745**, so a 10-zone
day carries only ~1.3 zones' worth of independent information.

![European temperatures with heat waves highlighted](journal/heatwaves.png)

One lesson learned here: measured naively, the togetherness came out *negative*
(−0.09), because Germany's demand is ten times Portugal's and that gap drowns
out the co-movement. Each zone has to be measured against its own normal first.
That trap is now handled in our measuring code.

### 5. What counts as "one event", and the simple opponents

We froze the definitions before running any experiments, so we can't
accidentally tune them later: an **event = one calendar day** (all stocks/zones
on that day belong to it), with multi-day episodes (like a week-long heat wave)
as a variant. We also wired up the simple opponents every fancy method must
beat: "tomorrow looks like the recent past" (HAR), "same day last week"
(seasonal naive), and a standard model fed the temperature (gradient boosting).

## The three datasets at a glance

| dataset | rows | units | days | togetherness (ICC) | redundancy factor | one event is |
|---|---|---|---|---|---|---|
| synthetic market | any | any | any | dialed: 0 → 0.8 | 1 → ~64 | a trading day |
| S&P 500 volatility | 2,716,706 | 492 stocks | 6,705 | 0.435 | ≈ 186 | a trading day |
| EU electricity | 20,979 | 10 zones | 2,100 | 0.745 | ≈ 7.7 | a day / heat-wave episode |

*Togetherness (ICC): 0 = units move independently, 1 = in lockstep. Redundancy
factor (design effect): how over-counted a day is when every row is treated as
independent — 405 stock-rows ÷ 186 ≈ 2 truly independent observations.*


## 2026-09-01 Phase 2: the paranoia suite

To avoid data leakage and ensure the integrity of out testing we do the following:

- **A sealed final exam.** The last stretch of time is locked away. The code
  refuses to hand it over unless you explicitly confirm, and tests prove no
  training or validation slice ever touches it.
- **Quarantine gaps.** Between every study period and its test period we skip
  a few days, so the last study day's "tomorrow" cannot land in the test.
- **Nothing is computed from the future.** Every number the pipeline learns
  from data (histogram bins, scalers, the togetherness measure) is fitted on
  the study slice only. The proof is blunt: we multiply all test data by 100,
  or delete it entirely, and verify nothing on the study side moves by a
  single bit. 24 variations of this run automatically.
- **We test the tests.** A smoke detector you never test is decoration. We
  plant a deliberately cheating feature (the answer itself) and confirm the
  leak detector goes off. It does. We also corrupt the training slice and
  confirm every fitted number *does* change, so no check can pass vacuously.

![how the timeline is split](journal/timeline_splits.png)

