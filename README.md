# TSMOM (improved)

Clean, importable implementation of **Time Series Momentum** as described by
Moskowitz, Ooi & Pedersen (2012) and the educational notebook at
[mirkovicdev/TSMOM](https://github.com/mirkovicdev/TSMOM).

## What changed

| Area | Original | This package |
|------|----------|--------------|
| Structure | Single notebook | `src/` package + tests |
| API | Copy cells | `from tsmom import tsmom_backtest` |
| Tooling | — | pyproject, ruff, pytest, uv-ready |
| Metrics | Ad-hoc plots | `performance_summary`, Sharpe, max DD |
| New | — | Multi-lookback ensemble, cleaner cost model |

Signal definition is unchanged: `sign` of lookback return, lagged one period, optional vol targeting.

## Install

```bash
uv pip install -e ".[dev]"
# or
pip install -e ".[dev]"
```

## Quick start

```python
import pandas as pd
from tsmom import (
    tsmom_backtest,
    tsmom_vol_scaled,
    tsmom_with_costs,
    multi_lookback_ensemble,
    performance_summary,
)

prices = ...  # pd.Series of prices

plain = tsmom_backtest(prices, lookback=126)
scaled = tsmom_vol_scaled(prices, lookback=126, target_vol=0.15)
costly = tsmom_with_costs(prices, lookback=126, cost_per_trade=0.001)
ens = multi_lookback_ensemble(prices, lookbacks=(21, 63, 126, 252))

print(performance_summary(scaled["tsmom_return"]))
```

## New feature: multi-lookback ensemble

Averages the ±1 signals of several lookbacks (equal weight by default), then
optionally volatility-scales the combined signal. Reduces dependence on any
single arbitrary horizon while keeping the same economic idea.

## Citation

```bibtex
@article{moskowitz2012time,
  title={Time series momentum},
  author={Moskowitz, Tobias J and Ooi, Yao Hua and Pedersen, Lasse Heje},
  journal={Journal of Financial Economics},
  year={2012}
}
```

## License

MIT. Methodology belongs to the original authors; this is an independent clean packaging of the public educational implementation.
