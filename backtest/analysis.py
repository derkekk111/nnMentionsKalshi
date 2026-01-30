"""
Performance analysis utilities for backtesting.

Computes:
- PnL metrics
- Sharpe ratio
- Maximum drawdown
- Win rate and other statistics
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252
) -> float:
    """
    Compute Sharpe ratio.
    
    Sharpe = (Mean return - Risk-free rate) / Std of returns
    
    Higher Sharpe = more consistent, stable strategy.
    Lower Sharpe = more volatile, unreliable strategy.
    
    Args:
        returns: Array of returns (can be per-trade or periodic)
        risk_free_rate: Risk-free rate (daily if returns are daily)
        annualization_factor: Factor to annualize (252 for daily)
        
    Returns:
        Sharpe ratio (annualized if factor provided)
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)
    
    if std_return == 0:
        return 0.0 if mean_return <= 0 else float('inf')
    
    sharpe = mean_return / std_return
    
    # Annualize
    if annualization_factor > 1:
        sharpe *= np.sqrt(annualization_factor)
    
    return float(sharpe)


def compute_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
    """
    Compute maximum drawdown.
    
    MDD = largest loss from peak to trough before recovery.
    
    Args:
        equity_curve: Array of equity values over time
        
    Returns:
        Tuple of (max_drawdown_pct, peak_idx, trough_idx)
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0
    
    equity_curve = np.array(equity_curve)
    
    # Running maximum
    running_max = np.maximum.accumulate(equity_curve)
    
    # Drawdown at each point
    drawdowns = (running_max - equity_curve) / running_max
    
    # Maximum drawdown
    max_dd_idx = np.argmax(drawdowns)
    max_dd = float(drawdowns[max_dd_idx])
    
    # Find peak index (most recent peak before trough)
    peak_idx = np.argmax(equity_curve[:max_dd_idx + 1])
    
    return max_dd, int(peak_idx), int(max_dd_idx)


def compute_win_rate(pnls: np.ndarray) -> float:
    """
    Compute win rate.
    
    Win rate = % of trades that are profitable.
    
    Args:
        pnls: Array of PnL values per trade
        
    Returns:
        Win rate (0-1)
    """
    if len(pnls) == 0:
        return 0.0
    
    pnls = np.array(pnls)
    wins = np.sum(pnls > 0)
    
    return float(wins / len(pnls))


def compute_profit_factor(pnls: np.ndarray) -> float:
    """
    Compute profit factor.
    
    Profit factor = Gross profit / Gross loss
    
    Args:
        pnls: Array of PnL values
        
    Returns:
        Profit factor (>1 is profitable)
    """
    pnls = np.array(pnls)
    
    gross_profit = np.sum(pnls[pnls > 0])
    gross_loss = abs(np.sum(pnls[pnls < 0]))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return float(gross_profit / gross_loss)


def compute_calmar_ratio(
    total_return: float,
    max_drawdown: float,
    periods: int = 1
) -> float:
    """
    Compute Calmar ratio.
    
    Calmar = Annualized return / Max drawdown
    
    Args:
        total_return: Total return (e.g., 0.20 for 20%)
        max_drawdown: Maximum drawdown (e.g., 0.10 for 10%)
        periods: Number of periods (for annualization)
        
    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return float('inf') if total_return > 0 else 0.0
    
    return float(total_return / max_drawdown)


def compute_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252
) -> float:
    """
    Compute Sortino ratio.
    
    Like Sharpe but only considers downside volatility.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        annualization_factor: Annualization factor
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    mean_return = np.mean(excess_returns)
    
    # Downside deviation
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        return float('inf') if mean_return > 0 else 0.0
    
    downside_std = np.sqrt(np.mean(negative_returns ** 2))
    
    if downside_std == 0:
        return float('inf') if mean_return > 0 else 0.0
    
    sortino = mean_return / downside_std
    
    if annualization_factor > 1:
        sortino *= np.sqrt(annualization_factor)
    
    return float(sortino)


def compute_performance_metrics(
    pnls: List[float],
    initial_bankroll: float,
    equity_curve: Optional[List[float]] = None
) -> Dict:
    """
    Compute comprehensive performance metrics.
    
    Args:
        pnls: List of PnL values per trade
        initial_bankroll: Starting capital
        equity_curve: Optional equity values over time
        
    Returns:
        Dict with all performance metrics
    """
    pnls = np.array(pnls)
    
    if len(pnls) == 0:
        return {
            'total_pnl': 0.0,
            'total_return': 0.0,
            'n_trades': 0
        }
    
    # Basic PnL metrics
    total_pnl = float(np.sum(pnls))
    total_return = total_pnl / initial_bankroll
    
    avg_pnl = float(np.mean(pnls))
    pnl_std = float(np.std(pnls))
    
    # Win/loss analysis
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    
    n_wins = len(wins)
    n_losses = len(losses)
    
    win_rate = compute_win_rate(pnls)
    profit_factor = compute_profit_factor(pnls)
    
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    
    max_win = float(np.max(wins)) if len(wins) > 0 else 0.0
    max_loss = float(np.min(losses)) if len(losses) > 0 else 0.0
    
    # Risk-adjusted metrics
    if equity_curve is not None and len(equity_curve) > 1:
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        
        sharpe = compute_sharpe_ratio(returns)
        sortino = compute_sortino_ratio(returns)
        max_dd, peak_idx, trough_idx = compute_max_drawdown(equity)
        calmar = compute_calmar_ratio(total_return, max_dd)
    else:
        # Use per-trade returns
        returns = pnls / initial_bankroll
        sharpe = compute_sharpe_ratio(returns, annualization_factor=1)
        sortino = compute_sortino_ratio(returns, annualization_factor=1)
        
        # Cumulative equity for drawdown
        cumulative_equity = initial_bankroll + np.cumsum(pnls)
        max_dd, peak_idx, trough_idx = compute_max_drawdown(cumulative_equity)
        calmar = compute_calmar_ratio(total_return, max_dd)
    
    # Expectancy
    expectancy = avg_pnl
    expectancy_r = (win_rate * avg_win + (1 - win_rate) * avg_loss) if avg_loss != 0 else avg_win
    
    return {
        # PnL metrics
        'total_pnl': total_pnl,
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'avg_pnl': avg_pnl,
        'pnl_std': pnl_std,
        
        # Win/loss
        'n_wins': n_wins,
        'n_losses': n_losses,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_win': max_win,
        'max_loss': max_loss,
        
        # Risk-adjusted
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd * 100,
        
        # Expectancy
        'expectancy': expectancy,
        'expectancy_r': expectancy_r
    }


def analyze_by_ticker(trades_df, metric: str = 'pnl') -> Dict:
    """
    Analyze performance by ticker.
    
    Args:
        trades_df: DataFrame with trade data
        metric: Metric to analyze ('pnl', 'win_rate', etc.)
        
    Returns:
        Dict with per-ticker metrics
    """
    if trades_df.empty:
        return {}
    
    results = {}
    
    for ticker in trades_df['ticker'].unique():
        ticker_trades = trades_df[trades_df['ticker'] == ticker]
        pnls = ticker_trades['pnl'].values
        
        results[ticker] = {
            'n_trades': len(ticker_trades),
            'total_pnl': float(np.sum(pnls)),
            'avg_pnl': float(np.mean(pnls)),
            'win_rate': compute_win_rate(pnls),
            'profit_factor': compute_profit_factor(pnls)
        }
    
    return results


def analyze_by_edge_bucket(trades_df, buckets: int = 5) -> Dict:
    """
    Analyze performance by edge magnitude.
    
    Args:
        trades_df: DataFrame with trade data
        buckets: Number of edge buckets
        
    Returns:
        Dict with per-bucket metrics
    """
    if trades_df.empty:
        return {}
    
    trades_df = trades_df.copy()
    trades_df['abs_edge'] = np.abs(trades_df['edge'])
    trades_df['edge_bucket'] = pd.cut(
        trades_df['abs_edge'],
        bins=buckets,
        labels=[f'Q{i+1}' for i in range(buckets)]
    )
    
    results = {}
    
    for bucket in trades_df['edge_bucket'].unique():
        if pd.isna(bucket):
            continue
            
        bucket_trades = trades_df[trades_df['edge_bucket'] == bucket]
        pnls = bucket_trades['pnl'].values
        
        results[str(bucket)] = {
            'n_trades': len(bucket_trades),
            'avg_edge': float(bucket_trades['abs_edge'].mean()),
            'total_pnl': float(np.sum(pnls)),
            'avg_pnl': float(np.mean(pnls)),
            'win_rate': compute_win_rate(pnls)
        }
    
    return results


# Need pandas for analyze functions
try:
    import pandas as pd
except ImportError:
    pd = None


def format_backtest_report(metrics: Dict) -> str:
    """
    Format backtest results as a human-readable report.
    
    Args:
        metrics: Dict from get_results()
        
    Returns:
        Formatted string report
    """
    if metrics.get('n_trades', 0) == 0:
        return "No trades executed."
    
    lines = [
        "=" * 60,
        "BACKTEST RESULTS",
        "=" * 60,
        "",
        "PERFORMANCE SUMMARY:",
        f"  Total PnL:        ${metrics.get('total_pnl', 0):.2f}",
        f"  Total Return:     {metrics.get('total_return', 0) * 100:.2f}%",
        f"  Final Bankroll:   ${metrics.get('final_bankroll', 0):.2f}",
        "",
        "TRADE STATISTICS:",
        f"  Total Trades:     {metrics.get('n_trades', 0)}",
        f"  Wins:             {metrics.get('n_wins', 0)}",
        f"  Losses:           {metrics.get('n_losses', 0)}",
        f"  Win Rate:         {metrics.get('win_rate', 0) * 100:.1f}%",
        f"  Profit Factor:    {metrics.get('profit_factor', 0):.2f}",
        "",
        "DIRECTION BREAKDOWN:",
        f"  Yes Trades:       {metrics.get('n_yes_trades', 0)}",
        f"  Yes Win Rate:     {metrics.get('yes_win_rate', 0) * 100:.1f}%",
        f"  Yes Avg PnL:      ${metrics.get('yes_avg_pnl', 0):.2f}",
        f"  No Trades:        {metrics.get('n_no_trades', 0)}",
        f"  No Win Rate:      {metrics.get('no_win_rate', 0) * 100:.1f}%",
        f"  No Avg PnL:       ${metrics.get('no_avg_pnl', 0):.2f}",
        "",
        "RISK METRICS:",
        f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.2f}",
        f"  Sortino Ratio:    {metrics.get('sortino_ratio', 0):.2f}",
        f"  Max Drawdown:     {metrics.get('max_drawdown_pct', 0):.1f}%",
        f"  Calmar Ratio:     {metrics.get('calmar_ratio', 0):.2f}",
        "",
        "TRADE DETAILS:",
        f"  Avg PnL:          ${metrics.get('avg_pnl', 0):.2f}",
        f"  Avg Win:          ${metrics.get('avg_win', 0):.2f}",
        f"  Avg Loss:         ${metrics.get('avg_loss', 0):.2f}",
        f"  Max Win:          ${metrics.get('max_win', 0):.2f}",
        f"  Max Loss:         ${metrics.get('max_loss', 0):.2f}",
        f"  Avg Edge:         {metrics.get('avg_edge', 0) * 100:.1f}%",
        f"  Avg Position:     ${metrics.get('avg_position_size', 0):.2f}",
        "=" * 60
    ]
    
    return "\n".join(lines)
