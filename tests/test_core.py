"""Tests for tsmom package."""

import numpy as np
import pandas as pd
import pytest

from tsmom import (
    lookback_return,
    signal_from_returns,
    tsmom_backtest,
    tsmom_vol_scaled,
    tsmom_with_costs,
    performance_summary,
    multi_lookback_ensemble,
)


def _synthetic_prices(n=500, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, n)
    prices = 100 * np.cumprod(1 + rets)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx, name="px")


def test_lookback_and_signal():
    p = _synthetic_prices(100)
    lr = lookback_return(p, 20)
    sig = signal_from_returns(lr)
    assert set(sig.dropna().unique()).issubset({-1.0, 0.0, 1.0})


def test_backtest_runs():
    p = _synthetic_prices()
    df = tsmom_backtest(p, lookback=60)
    assert "tsmom_return" in df.columns
    assert df["cumulative_tsmom"].iloc[-1] > 0


def test_vol_scaled():
    p = _synthetic_prices()
    df = tsmom_vol_scaled(p, lookback=40, target_vol=0.12)
    assert "position_size" in df.columns
    assert df["position_size"].dropna().max() <= 2.0 + 1e-9


def test_costs_reduce_pnl():
    p = _synthetic_prices()
    plain = tsmom_backtest(p, lookback=40)
    costly = tsmom_with_costs(p, lookback=40, cost_per_trade=0.01)
    assert costly["cumulative_tsmom_net"].iloc[-1] <= plain["cumulative_tsmom"].iloc[-1] + 1e-9


def test_metrics():
    p = _synthetic_prices()
    df = tsmom_backtest(p, lookback=30)
    s = performance_summary(df["tsmom_return"])
    assert "sharpe" in s and "max_drawdown" in s
    assert s["max_drawdown"] <= 0


def test_ensemble():
    p = _synthetic_prices()
    df = multi_lookback_ensemble(p, lookbacks=(20, 40, 60), target_vol=0.1)
    assert "ensemble_signal" in df.columns
    assert df["cumulative_tsmom"].iloc[-1] > 0
