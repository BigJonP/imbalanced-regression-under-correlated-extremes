"""Synthetic factor panel: within-day correlation of the log-target is set exactly by `rho`."""

import numpy as np
import pandas as pd

MU_LOG = -4.6
SIGMA_LOG = 0.75


def _standardized_t(rng, size, nu):
    if nu <= 2:
        raise ValueError("nu must exceed 2 for finite variance")
    return rng.standard_t(nu, size=size) / np.sqrt(nu / (nu - 2))


def _ar1(innovations, phi):
    out = np.empty_like(innovations)
    out[0] = innovations[0]
    c = np.sqrt(1.0 - phi**2)
    for t in range(1, len(innovations)):
        out[t] = phi * out[t - 1] + c * innovations[t]
    return out


def generate_panel(
    n_units,
    n_days,
    rho,
    seed,
    nu=5,
    phi_common=0.96,
    phi_idio=0.90,
    mu_log=MU_LOG,
    sigma_log=SIGMA_LOG,
    start="2000-01-03",
):
    """Long panel [unit, date, y, latent_common].

    log y_it = mu + sigma * (sqrt(rho) * h_t + sqrt(1 - rho) * z_it), where h is a
    heavy-tailed AR(1) common factor and z is Gaussian AR(1) idiosyncratic noise,
    both unit variance — so ICC(log y | day) = rho by construction. A new seed with
    the same parameters draws fresh, unseen events from the same process.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    rng = np.random.default_rng(seed)
    h = _ar1(_standardized_t(rng, n_days, nu), phi_common)
    z = _ar1(rng.standard_normal((n_days, n_units)), phi_idio)
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
