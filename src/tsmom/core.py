"""Core TSMOM signal, backtest, volatility scaling, and transaction costs."""

from __future__ import annotations

import numpy as np
import pandas as pd


def lookback_return(prices: pd.Series, lookback: int) -> pd.Series:
    """Simple return over the past `lookback` periods: (P_t - P_{t-k}) / P_{t-k}."""
    prices = prices.astype(float)
    return (prices - prices.shift(lookback)) / prices.shift(lookback)


def signal_from_returns(lookback_returns: pd.Series) -> pd.Series:
    """TSMOM signal = sign of lookback return (+1 / -1 / 0)."""
    return np.sign(lookback_returns)


def tsmom_backtest(prices: pd.Series, lookback: int = 252) -> pd.DataFrame:
    """Full single-asset TSMOM backtest (unscaled).

    signal_t = sign(r_{t-lookback:t})
    strategy return uses lagged signal: signal_{t-1} * r_t
    """
    prices = prices.astype(float)
    df = pd.DataFrame(index=prices.index)
    df["price"] = prices
    df["daily_return"] = prices.pct_change()
    df["lookback_return"] = lookback_return(prices, lookback)
    df["signal"] = signal_from_returns(df["lookback_return"])
    df["tsmom_return"] = df["signal"].shift(1) * df["daily_return"]
    df["cumulative_tsmom"] = (1 + df["tsmom_return"].fillna(0)).cumprod()
    df["cumulative_buyhold"] = (1 + df["daily_return"].fillna(0)).cumprod()
    return df


def tsmom_vol_scaled(
    prices: pd.Series,
    lookback: int = 252,
    target_vol: float = 0.15,
    vol_lookback: int = 60,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """TSMOM with volatility targeting (position size = target_vol / realized_vol)."""
    df = tsmom_backtest(prices, lookback)
    df["volatility"] = df["daily_return"].rolling(vol_lookback).std() * np.sqrt(252)
    pos = (target_vol / df["volatility"]).clip(upper=max_leverage)
    df["position_size"] = pos
    df["scaled_signal"] = df["signal"] * pos
    df["tsmom_return"] = df["scaled_signal"].shift(1) * df["daily_return"]
    df["cumulative_tsmom"] = (1 + df["tsmom_return"].fillna(0)).cumprod()
    return df


def tsmom_with_costs(
    prices: pd.Series,
    lookback: int = 252,
    cost_per_trade: float = 0.001,
    vol_scaled: bool = False,
    **vol_kwargs,
) -> pd.DataFrame:
    """TSMOM with proportional transaction costs charged on signal changes."""
    if vol_scaled:
        df = tsmom_vol_scaled(prices, lookback=lookback, **vol_kwargs)
        signal_col = "scaled_signal"
    else:
        df = tsmom_backtest(prices, lookback)
        signal_col = "signal"

    change = df[signal_col].diff().abs().fillna(0)
    df["turnover"] = change
    df["tsmom_return_net"] = df["tsmom_return"] - df["turnover"] * cost_per_trade
    df["cumulative_tsmom_net"] = (1 + df["tsmom_return_net"].fillna(0)).cumprod()
    return df
