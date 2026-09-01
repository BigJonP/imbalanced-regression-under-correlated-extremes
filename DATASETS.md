# Datasets

All raw and processed data live under `data/` (gitignored — regenerable or not
redistributable). Checksums of everything are written to
`data/processed/*_manifest.json` at build time.

## Synthetic factor panel — no download

`dire.data.synthetic.generate_panel(n_units, n_days, rho, seed)`.
log y_it = μ + σ(√ρ·h_t + √(1−ρ)·z_it) with h a heavy-tailed (Student-t, ν=5)
AR(1) common factor and z Gaussian AR(1), both unit variance, so
ICC(log y | day) = ρ by construction. A new seed draws fresh, unseen events from
the same process. Null gate: at ρ = 0 the measured design effect is 1.00
(`tests/test_data.py::test_null_gate_design_effect_is_one_at_rho_zero`).

## S&P 500 volatility — one manual step

- Universe: `configs/sp500_universe.csv`, committed snapshot (fetched
  2026-09-01) of github.com/datasets/s-and-p-500-companies — ODC-PDDL (public
  domain). Current constituents only: survivorship bias is an acknowledged
  limitation.
- Prices: Stooq bulk daily US archive. Stooq bot-checks scripted access, so
  fetch once in a browser: https://stooq.com/db/h/ → "Daily, US, ASCII" →
  save as `data/raw/stooq/d_us_txt.zip`, then run
  `uv run scripts/download_sp500.py`. Stooq data is free for personal/research
  use, not redistributable — hence gitignored, processing-only in git.
- Target: daily Parkinson volatility √(ln²(H/L)/(4 ln 2)), days with H > L > 0,
  from 2000-01-01, tickers with ≥ 1000 usable days.
- Built 2026-09-01 from the archive dated 2026-08-31: 2,716,706 rows, 492
  tickers, 6,705 trading days (2000-01-03..2026-08-31); 11 tickers skipped for
  short history (recent IPOs/spin-offs: FDXF, GEHC, GEV, HONA, KVUE, Q, RDDT,
  SNDK, ...). Measured ICC(log vol | day): raw = 0.363, standardized = 0.435,
  deff(std) ≈ 186 at mean cluster size ~405 — an average market day carries
  ~2.2 independent observations. Highest market-vol days: 2008-10-10,
  2020-03-18/19, 2008-09-19, 2008-10-09.

## Electricity load — fully scripted

`uv run scripts/download_load_data.py` fetches and builds everything:

- Load: Open Power System Data, `time_series_60min_singleindex.csv`, version
  2020-10-06 — CC-BY-4.0, primary source ENTSO-E Transparency. 10 zones
  (AT BE DE ES FR GB IT NL PL PT), 2015-01-01..2020-09-30.
- y = daily peak of hourly actual load per zone (UTC days, ≥ 20 valid hours).
- Temperature: ERA5 daily max at each capital via the Open-Meteo archive API —
  CC-BY-4.0 (Open-Meteo), ERA5 © Copernicus/ECMWF.
- Measured on the full panel (20,979 rows, 2,100 days): ICC(log peak | day)
  raw = −0.088, within-unit standardized = 0.745, deff(std) ≈ 7.7. Raw pooled
  ICC is dominated by static between-zone scale (DE ≈ 80 GW vs PT ≈ 8 GW);
  the standardized number is the co-movement that matters. It still includes
  the shared weekly cycle — residual ICC after seasonal structure is the Phase 4
  refinement.

## Event definitions (frozen)

- Primary, all datasets: one event per calendar day — the cross-sectional
  cluster (`dire.data.events.assign_day_events`).
- Episode variant (ablations): runs of flagged dates merged across gaps ≤ 1 day.
- Heat waves: flag = cross-zone mean of daily max temperature ≥ its q = 0.95
  quantile; the quantile is computed on training dates only inside experiments.
  Full-sample sanity check: 27 episodes, hottest flagged dates 2019-07-25,
  2020-08-08, 2018-08-03 — the record 2019 European heat wave and the 2018/2020
  waves, as expected.
