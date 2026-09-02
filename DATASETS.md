# Datasets

Raw and processed data live under `data/` (gitignored: regenerable, or not
redistributable). Checksums land in `data/processed/*_manifest.json` at build
time.

## Synthetic factor panel (no download)

`dire.data.synthetic.generate_panel(n_units, n_days, rho, seed)`:

    log y_it = mu + sigma * (sqrt(rho) * h_t + sqrt(1 - rho) * z_it)

`h` is a common factor, `z` is Gaussian AR(1) noise, both standardized, so
ICC(log y | day) = rho by construction and a new seed draws fresh unseen events.

`h` comes from `crisis_factor`: a persistent Gaussian AR(1) base plus a strictly
positive shot-noise term (Poisson arrivals, exponential jumps, geometric decay)
added *after* the filter, then standardized. The shot noise is what makes the
panel right-skewed, so it holds crisis days and not merely correlated ones.
`shot_weight` is calibrated so the day-mean skew at the S&P's own ICC of 0.435
matches the S&P's +1.07.

Gates in `tests/test_data.py`: design effect 0.99 at rho = 0, ICC recovered
across the sweep, and right skew asserted on the common factor, the day means
and the top percentile of days.

## S&P 500 volatility (one manual step)

Stooq bot-checks scripted access, so fetch the archive once in a browser
(https://stooq.com/db/h/, "Daily, US, ASCII"), save it as
`data/raw/stooq/d_us_txt.zip`, then run `uv run scripts/download_sp500.py`.

- **Universe** `configs/sp500_universe.csv`, committed snapshot (2026-09-01) of
  github.com/datasets/s-and-p-500-companies, ODC-PDDL. Current constituents
  only: survivorship bias is an acknowledged limitation.
- **Licence** Stooq is free for personal and research use but not
  redistributable, hence gitignored and processing-only in git.
- **Target** daily Parkinson volatility sqrt(ln^2(H/L) / (4 ln 2)), from
  2000-01-01, tickers with at least 1000 usable days.
- **Filters** H > L > 0, and H/L >= 5 dropped as bad ticks: 8 rows, all near
  10x, against a largest genuine session range of 4.19.
- **Built** 2026-09-02 from the 2026-08-31 archive. 2,716,698 rows, 492 tickers,
  6,705 days (2000-01-03 to 2026-08-31); 11 tickers skipped for short history.
  ICC(log vol | day) raw 0.363, standardized 0.435, deff(std) ~ 186 at Kish mean
  cluster size 427, so a market day carries about 2.3 independent observations.
  The ceiling is 1/ICC = 2.3 however many stocks trade.

## Electricity load (fully scripted)

`uv run scripts/download_load_data.py` fetches and builds everything.

- **Load** Open Power System Data `time_series_60min_singleindex.csv`, version
  2020-10-06, CC-BY-4.0, from ENTSO-E Transparency. 10 zones
  (AT BE DE ES FR GB IT NL PL PT), 2015-01-01 to 2020-09-30.
- **Temperature** ERA5 daily max at each capital via the Open-Meteo archive API,
  CC-BY-4.0 (Open-Meteo); ERA5 (c) Copernicus/ECMWF.
- **Target** daily peak of hourly actual load per zone (UTC days, at least 20
  valid hours).
- **Filter** peaks above 2x the zone's own median dropped as bad readings:
  1 row, FR 2020-07-07 at 158 GW.
- **Built** 20,978 rows, 2,100 days. ICC(log peak | day) raw -0.088,
  within-unit standardized 0.746, deff(std) ~ 7.7. The raw number is dominated
  by static between-zone scale (DE ~ 80 GW against PT ~ 8 GW); the standardized
  one is the co-movement that matters.

## Event definitions (frozen)

- **Primary, all datasets** one event per calendar day, the cross-sectional
  cluster (`dire.data.events.assign_day_events`).
- **Episode variant (ablations)** runs of flagged dates merged across gaps of at
  most 1 day.
- **Heat waves** cross-zone mean daily max temperature at or above its q = 0.95
  quantile, computed on training dates only inside experiments. Full-sample
  check: 27 episodes, hottest 2019-07-25, 2020-08-08, 2018-08-03, which are the
  record 2019 European wave and the 2018 and 2020 waves.
