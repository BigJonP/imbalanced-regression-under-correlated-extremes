"""Resampling of the training slice. Rare = top 20% of the train target."""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from dire.data.panel import DATE, TARGET_LOG, TARGET_RAW, feature_columns

RARE_QUANTILE = 0.8
SMOTER_NEIGHBORS = 5


def _rare_mask(df):
    return df[TARGET_LOG] >= df[TARGET_LOG].quantile(RARE_QUANTILE)


def no_sampling(df, rng):
    return df


def random_under(df, rng):
    """Keep every rare row, subsample normal rows to a 1:1 ratio."""
    mask = _rare_mask(df)
    rare, normal = df[mask], df[~mask]
    keep = rng.choice(len(normal), size=min(len(rare), len(normal)), replace=False)
    return pd.concat([rare, normal.iloc[np.sort(keep)]], ignore_index=True)


def random_over(df, rng):
    """Duplicate rare rows with replacement up to a 1:1 ratio."""
    mask = _rare_mask(df)
    rare = df[mask]
    extra = int((~mask).sum()) - len(rare)
    if extra <= 0:
        return df
    picks = rng.choice(len(rare), size=extra, replace=True)
    return pd.concat([df, rare.iloc[picks]], ignore_index=True)


def smoter(df, rng):
    """Torgo/Branco SMOTER: interpolate rare rows in feature space, targets by
    inverse distance to the two parents. Synthetic rows keep the seed row's
    unit/date metadata."""
    mask = _rare_mask(df)
    rare = df[mask].reset_index(drop=True)
    n_synth = int((~mask).sum()) - len(rare)
    if n_synth <= 0 or len(rare) <= SMOTER_NEIGHBORS:
        return df
    feats = feature_columns(df)
    F = rare[feats].to_numpy(dtype=float)
    scale = F.std(axis=0)
    scale[scale == 0] = 1.0
    _, idx = (
        NearestNeighbors(n_neighbors=SMOTER_NEIGHBORS + 1).fit(F / scale).kneighbors(F / scale)
    )
    seeds = rng.integers(0, len(rare), n_synth)
    nbrs = idx[seeds, rng.integers(1, SMOTER_NEIGHBORS + 1, n_synth)]
    u = rng.random((n_synth, 1))
    new_F = F[seeds] + u * (F[nbrs] - F[seeds])
    d1 = np.linalg.norm((new_F - F[seeds]) / scale, axis=1)
    d2 = np.linalg.norm((new_F - F[nbrs]) / scale, axis=1)
    t1, t2 = rare[TARGET_LOG].to_numpy()[seeds], rare[TARGET_LOG].to_numpy()[nbrs]
    total = d1 + d2
    t = np.where(total > 0, (t1 * d2 + t2 * d1) / np.where(total > 0, total, 1.0), (t1 + t2) / 2.0)
    synth = rare.iloc[seeds].reset_index(drop=True).copy()
    synth[feats] = new_F
    synth[TARGET_LOG] = t
    synth[TARGET_RAW] = np.exp(t)
    return pd.concat([df, synth], ignore_index=True)


def cluster_aware(df, rng):
    """Oversample whole rare events (days). One crisis day counts once, however
    many rows it has; balance happens at the event level, not the row level."""
    event_stat = df.groupby(DATE)[TARGET_LOG].mean()
    rare_events = event_stat.index[event_stat >= event_stat.quantile(RARE_QUANTILE)]
    extra = (len(event_stat) - len(rare_events)) - len(rare_events)
    if extra <= 0:
        return df
    picks = rng.choice(len(rare_events), size=extra, replace=True)
    blocks = [df[df[DATE] == rare_events[i]] for i in picks]
    return pd.concat([df, *blocks], ignore_index=True)


SAMPLERS = {
    "none": no_sampling,
    "random_under": random_under,
    "random_over": random_over,
    "smoter": smoter,
    "cluster_aware": cluster_aware,
}
