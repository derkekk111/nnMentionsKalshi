"""
BacktestSimulator: Trading simulation for Kalshi mention markets.

Simulates trading based on model predictions and tracks:
- PnL (Profit and Loss)
- MDD (Maximum Drawdown)
- Sharpe Ratio
- Win Rate
- Individual trade tracking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    # Capital management
    initial_bankroll: float = 10000.0
    max_risk_per_trade: float = 0.05  # 5% of bankroll
    max_position_size: float = 100.0  # Max dollars per trade
    
    # Trading rules
    edge_threshold: float = 0.05  # Minimum edge to trade
    confidence_threshold: float = 0.0  # Minimum model confidence
    
    # Position sizing
    use_kelly_criterion: bool = False
    kelly_fraction: float = 0.25  # Fraction of Kelly to use
    use_model_position: bool = True  # Use model's position sizing
    
    # Fees and slippage
    fee_per_trade: float = 0.0  # Fee in dollars
    slippage: float = 0.01  # Price slippage (1 cent)
    
    # Risk management
    daily_loss_limit: float = 0.10  # 10% daily loss limit
    max_concurrent_positions: int = 50


@dataclass
class Trade:
    """Represents a single trade."""
    
    # Identifiers
    trade_id: int
    ticker: str
    keyword: str
    date_str: str
    earnings_date: datetime
    
    # Trade details
    direction: str  # 'yes' or 'no'
    position_size: float  # Dollars risked
    entry_price: float  # Price paid (0-1)
    
    # Model predictions
    model_prob: float
    model_position: float
    
    # Market data
    market_price: float
    edge: float
    
    # Outcome (filled after resolution)
    actual_outcome: Optional[int] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    
    def resolve(self, actual_outcome: int):
        """Resolve trade with actual outcome."""
        self.actual_outcome = actual_outcome
        
        if self.direction == 'yes':
            # Bought Yes at entry_price
            # If outcome is Yes (1): get $1, profit = 1 - entry_price
            # If outcome is No (0): get $0, loss = entry_price
            self.exit_price = float(actual_outcome)
            pnl_per_contract = actual_outcome - self.entry_price
        else:
            # Bought No at (1 - entry_price) equivalent
            # If outcome is No (0): get $1, profit = entry_price
            # If outcome is Yes (1): get $0, loss = 1 - entry_price
            self.exit_price = 1.0 - float(actual_outcome)
            pnl_per_contract = self.entry_price - actual_outcome
        
        # Scale by position size (in terms of contracts)
        # If entry_price is 0.70, each contract costs $0.70
        # Position size in dollars / entry price = number of contracts
        if self.direction == 'yes':
            n_contracts = self.position_size / self.entry_price
        else:
            n_contracts = self.position_size / (1 - self.entry_price)
        
        self.pnl = pnl_per_contract * n_contracts
        self.pnl_percent = self.pnl / self.position_size if self.position_size > 0 else 0


class BacktestSimulator:
    """
    Simulates trading on historical Kalshi mention markets.
    
    Takes model predictions and simulates executing trades based on
    predicted probabilities vs market prices.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize simulator.
        
        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        
        # State
        self.bankroll = self.config.initial_bankroll
        self.initial_bankroll = self.config.initial_bankroll
        
        # Tracking
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_pnl: Dict[str, float] = {}
        
        # Counters
        self.trade_counter = 0
    
    def reset(self):
        """Reset simulator to initial state."""
        self.bankroll = self.config.initial_bankroll
        self.trades = []
        self.equity_curve = []
        self.daily_pnl = {}
        self.trade_counter = 0
    
    def _calculate_position_size(
        self,
        edge: float,
        model_position: float,
        entry_price: float,
        direction: str
    ) -> float:
        """
        Calculate position size for a trade.
        
        Args:
            edge: Model probability - market price
            model_position: Model's suggested position (0-1)
            entry_price: Entry price (market price)
            direction: 'yes' or 'no'
            
        Returns:
            Position size in dollars
        """
        # Max risk based on bankroll
        max_risk = self.bankroll * self.config.max_risk_per_trade
        max_risk = min(max_risk, self.config.max_position_size)
        
        if max_risk <= 0:
            return 0.0
        
        # Kelly criterion (optional)
        if self.config.use_kelly_criterion:
            # f* = (p*b - q) / b
            # p = probability of win
            # b = odds (payout / stake)
            # q = 1 - p
            
            if direction == 'yes':
                p = entry_price + edge  # Our estimated true probability
                b = (1 - entry_price) / entry_price  # Odds
            else:
                p = 1 - (entry_price + edge)  # Probability of No
                b = entry_price / (1 - entry_price)  # Odds
            
            q = 1 - p
            
            if b > 0:
                kelly = (p * b - q) / b
                kelly = max(0, kelly) * self.config.kelly_fraction
                kelly_size = kelly * self.bankroll
            else:
                kelly_size = 0
            
            position = min(kelly_size, max_risk)
        else:
            # Use model's position sizing
            if self.config.use_model_position:
                position = model_position * max_risk
            else:
                # Scale by edge
                edge_factor = min(abs(edge) / 0.2, 1.0)  # Max at 20% edge
                position = edge_factor * max_risk
        
        return position
    
    def should_trade(
        self,
        model_prob: float,
        market_price: float,
        model_position: float = 1.0
    ) -> Tuple[bool, str, float]:
        """
        Determine if we should trade and in what direction.
        
        Args:
            model_prob: Model's predicted probability
            market_price: Current market price
            model_position: Model's position sizing suggestion
            
        Returns:
            Tuple of (should_trade, direction, edge)
        """
        # Calculate edge
        edge = model_prob - market_price
        
        # Check edge threshold
        if abs(edge) < self.config.edge_threshold:
            return False, '', edge
        
        # Check confidence threshold
        if model_position < self.config.confidence_threshold:
            return False, '', edge
        
        # Determine direction
        direction = 'yes' if edge > 0 else 'no'
        
        return True, direction, edge
    
    def execute_trade(
        self,
        ticker: str,
        keyword: str,
        date_str: str,
        earnings_date: datetime,
        model_prob: float,
        market_price: float,
        model_position: float = 1.0,
        actual_outcome: Optional[int] = None
    ) -> Optional[Trade]:
        """
        Execute a trade if conditions are met.
        
        Args:
            ticker: Stock ticker
            keyword: Keyword being traded
            date_str: Date string (e.g., '25JUL31')
            earnings_date: Earnings call datetime
            model_prob: Model's predicted probability
            market_price: Current market price (0-1)
            model_position: Model's position sizing (0-1)
            actual_outcome: Actual outcome (1=mentioned, 0=not)
            
        Returns:
            Trade object if executed, None otherwise
        """
        # Check if we should trade
        should, direction, edge = self.should_trade(
            model_prob, market_price, model_position
        )
        
        if not should:
            return None
        
        # Calculate entry price with slippage
        if direction == 'yes':
            entry_price = market_price + self.config.slippage
        else:
            entry_price = market_price - self.config.slippage
        
        # Clip to valid range
        entry_price = np.clip(entry_price, 0.01, 0.99)
        
        # Calculate position size
        position_size = self._calculate_position_size(
            edge, model_position, entry_price, direction
        )
        
        if position_size <= 0:
            return None
        
        # Apply fee
        position_size -= self.config.fee_per_trade
        
        if position_size <= 0:
            return None
        
        # Create trade
        self.trade_counter += 1
        trade = Trade(
            trade_id=self.trade_counter,
            ticker=ticker,
            keyword=keyword,
            date_str=date_str,
            earnings_date=earnings_date,
            direction=direction,
            position_size=position_size,
            entry_price=entry_price,
            model_prob=model_prob,
            model_position=model_position,
            market_price=market_price,
            edge=edge
        )
        
        # Resolve trade if outcome is known
        if actual_outcome is not None:
            trade.resolve(actual_outcome)
            self.bankroll += trade.pnl
            
            # Track equity
            self.equity_curve.append((earnings_date, self.bankroll))
            
            # Track daily PnL
            date_key = earnings_date.strftime('%Y-%m-%d')
            if date_key not in self.daily_pnl:
                self.daily_pnl[date_key] = 0.0
            self.daily_pnl[date_key] += trade.pnl
        
        self.trades.append(trade)
        return trade
    
    def run_backtest(
        self,
        predictions: np.ndarray,
        positions: Optional[np.ndarray],
        labels: np.ndarray,
        market_prices: np.ndarray,
        metadata: List[Dict]
    ) -> Dict:
        """
        Run full backtest on predictions.
        
        Args:
            predictions: Model probability predictions
            positions: Model position predictions (optional)
            labels: Actual outcomes (0 or 1)
            market_prices: Market prices at time of prediction
            metadata: List of dicts with ticker, keyword, date info
            
        Returns:
            Dict with backtest results
        """
        self.reset()
        
        if positions is None:
            positions = np.ones_like(predictions)
        
        # Sort by date to process chronologically
        dates = [m['earnings_date'] for m in metadata]
        sort_idx = np.argsort(dates)
        
        for i in sort_idx:
            meta = metadata[i]
            
            self.execute_trade(
                ticker=meta['ticker'],
                keyword=meta['keyword'],
                date_str=meta['date_str'],
                earnings_date=meta['earnings_date'],
                model_prob=float(predictions[i]),
                market_price=float(market_prices[i]),
                model_position=float(positions[i]),
                actual_outcome=int(labels[i])
            )
        
        return self.get_results()
    
    def get_results(self) -> Dict:
        """
        Get comprehensive backtest results.
        
        Returns:
            Dict with all metrics and trade details
        """
        from .analysis import compute_performance_metrics
        
        if not self.trades:
            return {
                'n_trades': 0,
                'final_bankroll': self.bankroll,
                'total_return': 0.0,
                'message': 'No trades executed'
            }
        
        # Extract trade data
        resolved_trades = [t for t in self.trades if t.pnl is not None]
        
        if not resolved_trades:
            return {
                'n_trades': len(self.trades),
                'n_resolved': 0,
                'final_bankroll': self.bankroll,
                'message': 'No trades resolved yet'
            }
        
        pnls = [t.pnl for t in resolved_trades]
        
        # Compute metrics
        metrics = compute_performance_metrics(
            pnls=pnls,
            initial_bankroll=self.initial_bankroll,
            equity_curve=[e[1] for e in self.equity_curve]
        )
        
        # Add trade details
        metrics['n_trades'] = len(resolved_trades)
        metrics['final_bankroll'] = self.bankroll
        
        # Win/loss breakdown
        wins = [t for t in resolved_trades if t.pnl > 0]
        losses = [t for t in resolved_trades if t.pnl < 0]
        
        metrics['n_wins'] = len(wins)
        metrics['n_losses'] = len(losses)
        metrics['win_rate'] = len(wins) / len(resolved_trades)
        
        # Direction breakdown
        yes_trades = [t for t in resolved_trades if t.direction == 'yes']
        no_trades = [t for t in resolved_trades if t.direction == 'no']
        
        metrics['n_yes_trades'] = len(yes_trades)
        metrics['n_no_trades'] = len(no_trades)
        
        if yes_trades:
            metrics['yes_win_rate'] = sum(1 for t in yes_trades if t.pnl > 0) / len(yes_trades)
            metrics['yes_avg_pnl'] = np.mean([t.pnl for t in yes_trades])
        else:
            metrics['yes_win_rate'] = 0.0
            metrics['yes_avg_pnl'] = 0.0
        
        if no_trades:
            metrics['no_win_rate'] = sum(1 for t in no_trades if t.pnl > 0) / len(no_trades)
            metrics['no_avg_pnl'] = np.mean([t.pnl for t in no_trades])
        else:
            metrics['no_win_rate'] = 0.0
            metrics['no_avg_pnl'] = 0.0
        
        # Edge analysis
        metrics['avg_edge'] = np.mean([abs(t.edge) for t in resolved_trades])
        metrics['avg_position_size'] = np.mean([t.position_size for t in resolved_trades])
        
        return metrics
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as a DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        records = []
        for t in self.trades:
            records.append({
                'trade_id': t.trade_id,
                'ticker': t.ticker,
                'keyword': t.keyword,
                'date_str': t.date_str,
                'earnings_date': t.earnings_date,
                'direction': t.direction,
                'position_size': t.position_size,
                'entry_price': t.entry_price,
                'model_prob': t.model_prob,
                'model_position': t.model_position,
                'market_price': t.market_price,
                'edge': t.edge,
                'actual_outcome': t.actual_outcome,
                'pnl': t.pnl,
                'pnl_percent': t.pnl_percent
            })
        
        return pd.DataFrame(records)
    
    def get_equity_df(self) -> pd.DataFrame:
        """Get equity curve as a DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()
        
        return pd.DataFrame(
            self.equity_curve,
            columns=['date', 'equity']
        )
