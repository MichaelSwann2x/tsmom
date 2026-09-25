"""Multi-lookback ensemble TSMOM (new feature).

Averages signals across several lookbacks so the strategy is less sensitive
to a single arbitrary horizon.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .core import lookback_return, signal_from_returns


def multi_lookback_ensemble(
    prices: pd.Series,
    lookbacks: Sequence[int] = (21, 63, 126, 252),
    weights: Sequence[float] | None = None,
    target_vol: float | None = 0.15,
    vol_lookback: int = 60,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """Equal- or custom-weight average of TSMOM signals across lookbacks."""
    prices = prices.astype(float)
    lookbacks = list(lookbacks)
    if weights is None:
        weights = [1.0 / len(lookbacks)] * len(lookbacks)
    if len(weights) != len(lookbacks):
        raise ValueError("weights must match lookbacks")
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()

    daily = prices.pct_change()
    combined = pd.Series(0.0, index=prices.index)
    for lb, wi in zip(lookbacks, w):
        sig = signal_from_returns(lookback_return(prices, lb))
        combined = combined + wi * sig.fillna(0)

    df = pd.DataFrame(index=prices.index)
    df["price"] = prices
    df["daily_return"] = daily
    df["ensemble_signal"] = combined

    if target_vol is not None:
        vol = daily.rolling(vol_lookback).std() * np.sqrt(252)
        pos = (target_vol / vol).clip(upper=max_leverage)
        df["position_size"] = pos
        df["scaled_signal"] = combined * pos
        df["tsmom_return"] = df["scaled_signal"].shift(1) * daily
    else:
        df["tsmom_return"] = combined.shift(1) * daily

    df["cumulative_tsmom"] = (1 + df["tsmom_return"].fillna(0)).cumprod()
    df["cumulative_buyhold"] = (1 + daily.fillna(0)).cumprod()
    return df
