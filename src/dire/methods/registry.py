"""The frozen method grid: sampler + weights + model options behind one interface."""

import numpy as np

from dire.methods import classical
from dire.methods.mlp import MLPRegressor
from dire.methods.sampling import SAMPLERS
from dire.methods.weighting import WEIGHTINGS

METHODS = {
    "vanilla": {},
    "log_target": dict(target="log"),
    "inverse": dict(weighting="inverse"),
    "sqinv": dict(weighting="sqinv"),
    "lds": dict(weighting="lds"),
    "lds_deff": dict(weighting="lds_deff"),
    "under": dict(sampling="random_under"),
    "over": dict(sampling="random_over"),
    "smoter": dict(sampling="smoter"),
    "cluster": dict(sampling="cluster_aware"),
    "fds": dict(fds=True),
    "lds_fds": dict(weighting="lds", fds=True),
    "ranksim": dict(ranksim_lambda=1.0),
    "bmc": dict(loss="bmc"),
    "lds_cap": dict(weighting="lds_cap"),
    "lds_narrow": dict(weighting="lds_narrow"),
    "lds_wide": dict(weighting="lds_wide"),
    "lds_deff_lo": dict(weighting="lds_deff_lo"),
    "lds_deff_hi": dict(weighting="lds_deff_hi"),
    "lds_deff_episode": dict(weighting="lds_deff_episode"),
    "har": dict(classical="har"),
    "seasonal_naive": dict(classical="seasonal_naive"),
    "temp_gbm": dict(classical="temp_gbm"),
}
_CLASSICAL = {
    "har": lambda seed: classical.HARBaseline(),
    "seasonal_naive": lambda seed: classical.SeasonalNaive(),
    "temp_gbm": lambda seed: classical.TemperatureGBM(seed=seed),
}


class Method:
    """fit(train_df, val_df) / predict(df) -> raw-scale predictions, any method."""

    def __init__(self, name, seed=0, **overrides):
        spec = dict(METHODS[name])
        spec.update(overrides)
        self.name, self.seed = name, seed
        self._classical = spec.pop("classical", None)
        self._sampling = spec.pop("sampling", "none")
        self._weighting = spec.pop("weighting", "none")
        self._model_kwargs = spec

    def fit(self, train_df, val_df=None):
        if self._classical is not None:
            self._model = _CLASSICAL[self._classical](self.seed).fit(train_df)
            return self
        rng = np.random.default_rng(self.seed)
        train_df = SAMPLERS[self._sampling](train_df, rng)
        weights = WEIGHTINGS[self._weighting](train_df)
        self._model = MLPRegressor(seed=self.seed, **self._model_kwargs).fit(
            train_df, val_df, sample_weight=weights
        )
        return self

    def predict(self, df):
        return np.asarray(self._model.predict(df))


def build_method(name, seed=0, **overrides):
    return Method(name, seed, **overrides)
