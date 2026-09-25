"""Time Series Momentum — improved package.

Faithful to Moskowitz, Ooi & Pedersen (2012) and the mirkovicdev educational
notebook, with volatility scaling, transaction costs, multi-lookback ensemble,
and a clean importable API.
"""

from .core import (
    lookback_return,
    signal_from_returns,
    tsmom_backtest,
    tsmom_vol_scaled,
    tsmom_with_costs,
)
from .metrics import performance_summary, max_drawdown, sharpe_ratio
from .ensemble import multi_lookback_ensemble

__version__ = "0.2.0"
__all__ = [
    "lookback_return",
    "signal_from_returns",
    "tsmom_backtest",
    "tsmom_vol_scaled",
    "tsmom_with_costs",
    "performance_summary",
    "max_drawdown",
    "sharpe_ratio",
    "multi_lookback_ensemble",
]
