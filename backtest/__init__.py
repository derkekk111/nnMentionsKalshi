"""Backtesting module for Kalshi mention market trading simulation."""

from .simulator import BacktestSimulator, BacktestConfig
from .analysis import (
    compute_performance_metrics,
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_win_rate
)

__all__ = [
    'BacktestSimulator',
    'BacktestConfig',
    'compute_performance_metrics',
    'compute_sharpe_ratio',
    'compute_max_drawdown',
    'compute_win_rate'
]
