"""Event definitions, frozen before any experiment is run.

Primary definition for all three datasets: one event per calendar day (the
cross-sectional cluster). Episode variants exist for ablations and for
heat waves: runs of flagged dates, merging runs separated by at most `gap`
unflagged dates. Heat-wave flag: cross-zone mean of daily max temperature at or
above its q = 0.95 quantile — the quantile must be computed on training dates
only when used inside an experiment.
"""

import pandas as pd

from dire.data.panel import DATE

EVENT = "event_id"


def assign_day_events(df):
    out = df.copy()
    out[EVENT] = "d_" + pd.to_datetime(out[DATE]).dt.strftime("%Y%m%d")
    return out


def episode_labels(flags: pd.Series, gap=1) -> pd.Series:
    """flags: bool Series indexed by sorted unique dates -> episode label or None."""
    flags = flags.sort_index()
    labels, current, since_true = [], 0, None
    for is_hot in flags.astype(bool):
        if is_hot:
            if since_true is None or since_true > gap:
                current += 1
            labels.append(f"ep_{current:04d}")
            since_true = 0
        else:
            labels.append(None)
            if since_true is not None:
                since_true += 1
    return pd.Series(labels, index=flags.index)


def assign_episode_events(df, flags: pd.Series, gap=1):
    """Rows on flagged dates share their episode's event id; other dates stay day events."""
    out = assign_day_events(df)
    eps = episode_labels(flags, gap=gap)
    mapped = pd.to_datetime(out[DATE]).map(eps)
    out.loc[mapped.notna(), EVENT] = mapped[mapped.notna()]
    return out


def heatwave_flags(temp_by_date: pd.Series, q=0.95) -> pd.Series:
    return temp_by_date >= temp_by_date.quantile(q)
