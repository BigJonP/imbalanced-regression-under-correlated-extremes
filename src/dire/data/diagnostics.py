"""Intraclass correlation and design effect for clustered panels."""

import numpy as np
import pandas as pd


def _anova(values, groups) -> tuple[float, pd.Series]:
    df = pd.DataFrame({"v": np.asarray(values, dtype=float), "g": np.asarray(groups)}).dropna()
    sizes = df.groupby("g")["v"].size()
    k, n = len(sizes), int(sizes.sum())
    if k < 2 or n <= k:
        raise ValueError("need at least 2 groups and more observations than groups")
    means = df.groupby("g")["v"].mean()
    grand = df["v"].mean()
    msb = float((sizes * (means - grand) ** 2).sum()) / (k - 1)
    msw = float(((df["v"] - means.loc[df["g"]].to_numpy()) ** 2).sum()) / (n - k)
    n0 = (n - float((sizes**2).sum()) / n) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    return float(icc), sizes


def intraclass_correlation(values, groups) -> float:
    """One-way ANOVA ICC(1), unbalanced groups. Can be slightly negative by chance."""
    return _anova(values, groups)[0]


def design_effect(values, groups) -> float:
    """Kish deff = 1 + (m_bar - 1) * icc, with size-weighted mean cluster size."""
    icc, sizes = _anova(values, groups)
    m_bar = float((sizes**2).sum()) / float(sizes.sum())
    return 1.0 + (m_bar - 1.0) * icc
