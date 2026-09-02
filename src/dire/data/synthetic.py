"""Synthetic factor panel: within-day correlation of the log-target is set exactly by `rho`."""

import numpy as np
import pandas as pd

MU_LOG = -4.6
SIGMA_LOG = 0.75
SHOT_LAM = 0.10
SHOT_DECAY = 0.80
SHOT_SCALE = 0.7
SHOT_WEIGHT = 2.2


def _ar1(innovations, phi):
    out = np.empty_like(innovations)
    out[0] = innovations[0]
    c = np.sqrt(1.0 - phi**2)
    for t in range(1, len(innovations)):
        out[t] = phi * out[t - 1] + c * innovations[t]
    return out


def crisis_factor(n_days, rng, phi=0.96, lam=SHOT_LAM, decay=SHOT_DECAY,
                  scale=SHOT_SCALE, weight=SHOT_WEIGHT):
    """Unit-variance common factor that keeps a right-skewed tail: persistent
    Gaussian base plus a strictly positive shot-noise crisis term.

    The obvious construction, an AR(1) of heavy-tailed innovations, does not
    work and this generator used to get it wrong. At phi = 0.96 each filtered
    value averages roughly 1 / (1 - phi) = 25 innovations, so the central limit
    theorem eats whatever tail was fed in and the factor comes out Gaussian:
    correlated days, but no crisis days. Persistence and heavy tails cannot both
    come from one filtered series.

    Shot noise survives the filter because it is added after it. Poisson
    arrivals at rate `lam` with exponential jump sizes decaying geometrically,
    so the term is strictly positive and right-skewed by construction. `weight`
    is calibrated so that at the S&P 500's own ICC (0.435) the panel's day-mean
    log-target skew matches the S&P's 1.07.

    A symmetric on/off crisis regime was tried first and does not work at these
    panel lengths: only a couple of episodes get drawn, so standardizing turns
    the crisis block into a second mode rather than a tail and the skew comes
    out negative.
    """
    c = np.zeros(n_days)
    for t in range(1, n_days):
        c[t] = decay * c[t - 1] + (rng.random() < lam) * rng.exponential(scale)
    h = _ar1(rng.standard_normal(n_days), phi) + weight * c
    return (h - h.mean()) / h.std()


def generate_panel(
    n_units,
    n_days,
    rho,
    seed,
    phi_common=0.96,
    phi_idio=0.90,
    mu_log=MU_LOG,
    sigma_log=SIGMA_LOG,
    shot_lam=SHOT_LAM,
    shot_decay=SHOT_DECAY,
    shot_scale=SHOT_SCALE,
    shot_weight=SHOT_WEIGHT,
    start="2000-01-03",
):
    """Long panel [unit, date, y, latent_common].

    log y_it = mu + sigma * (sqrt(rho) * h_t + sqrt(1 - rho) * z_it), where h is
    the right-skewed common factor above and z is Gaussian AR(1) idiosyncratic
    noise, both unit variance — so ICC(log y | day) = rho by construction. A new
    seed with the same parameters draws fresh, unseen events from the same
    process.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    rng = np.random.default_rng(seed)
    h = crisis_factor(n_days, rng, phi=phi_common, lam=shot_lam, decay=shot_decay,
                      scale=shot_scale, weight=shot_weight)
    z = _ar1(rng.standard_normal((n_days, n_units)), phi_idio)
    z = (z - z.mean()) / z.std()
    log_y = mu_log + sigma_log * (np.sqrt(rho) * h[:, None] + np.sqrt(1.0 - rho) * z)

    dates = pd.bdate_range(start, periods=n_days)
    units = [f"u{i:04d}" for i in range(n_units)]
    return pd.DataFrame(
        {
            "unit": np.tile(units, n_days),
            "date": np.repeat(dates, n_units),
            "y": np.exp(log_y).ravel(),
            "latent_common": np.repeat(h, n_units),
        }
    )
