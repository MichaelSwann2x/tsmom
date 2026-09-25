"""Performance metrics for TSMOM equity curves."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.std() == 0 or len(r) < 2:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / r.std())


def max_drawdown(cumulative: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    c = cumulative.dropna()
    if len(c) == 0:
        return 0.0
    peak = c.cummax()
    dd = c / peak - 1.0
    return float(dd.min())


def performance_summary(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> dict:
    """Compact dict of total return, ann. return, vol, Sharpe, max DD."""
    r = returns.dropna()
    if len(r) == 0:
        return {}
    cum = (1 + r).cumprod()
    n = len(r)
    total = float(cum.iloc[-1] - 1)
    ann_ret = float(cum.iloc[-1] ** (periods_per_year / n) - 1) if n > 0 else 0.0
    ann_vol = float(r.std() * np.sqrt(periods_per_year))
    return {
        "total_return": total,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe_ratio(r, periods_per_year),
        "max_drawdown": max_drawdown(cum),
        "n_obs": int(n),
    }
