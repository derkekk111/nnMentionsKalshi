"""
Visualization utilities for Kalshi mention prediction evaluation.

Creates:
1. Equity curve
2. Calibration plot
3. Model vs Market probability scatter
4. Prediction error histogram
5. Trade timeline with PnL
6. Feature importance (SHAP)
7. Training history plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#27AE60',
    'danger': '#E74C3C',
    'warning': '#F39C12',
    'neutral': '#7F8C8D'
}


def plot_equity_curve(
    equity_values: np.ndarray,
    dates: Optional[List[datetime]] = None,
    title: str = 'Equity Curve',
    initial_value: float = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot equity curve over time.
    
    Shows how the trading account balance changes as the model executes trades.
    
    Args:
        equity_values: Array of equity values
        dates: Optional list of dates corresponding to equity values
        title: Plot title
        initial_value: Initial equity (for drawdown highlighting)
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if dates is not None:
        x = dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
    else:
        x = range(len(equity_values))
    
    # Plot equity curve
    ax.plot(x, equity_values, color=COLORS['primary'], linewidth=2, label='Equity')
    
    # Add initial value line
    if initial_value is not None:
        ax.axhline(y=initial_value, color=COLORS['neutral'], linestyle='--', 
                   alpha=0.7, label=f'Initial: ${initial_value:,.0f}')
    
    # Highlight drawdown periods
    running_max = np.maximum.accumulate(equity_values)
    drawdown = (running_max - equity_values) / running_max
    
    # Fill areas where equity is below running max
    ax.fill_between(x, equity_values, running_max, 
                    where=equity_values < running_max,
                    alpha=0.3, color=COLORS['danger'], label='Drawdown')
    
    # Formatting
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Equity ($)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add summary stats
    final_value = equity_values[-1]
    max_dd = np.max(drawdown) * 100
    total_return = (final_value - equity_values[0]) / equity_values[0] * 100
    
    stats_text = f'Final: ${final_value:,.0f}\nReturn: {total_return:.1f}%\nMax DD: {max_dd:.1f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    title: str = 'Calibration Plot',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot calibration diagram.
    
    Shows how well predicted probabilities match actual frequencies.
    A perfectly calibrated model lies on the diagonal.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        n_bins: Number of calibration bins
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Compute calibration curve
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_counts = []
    bin_accuracies = []
    bin_confidences = []
    
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            mask = (y_pred >= lower) & (y_pred <= upper)
        else:
            mask = (y_pred >= lower) & (y_pred < upper)
        
        count = np.sum(mask)
        bin_counts.append(count)
        
        if count > 0:
            bin_accuracies.append(np.mean(y_true[mask]))
            bin_confidences.append(np.mean(y_pred[mask]))
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)
    
    bin_counts = np.array(bin_counts)
    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    
    # Main calibration plot
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], linestyle='--', color=COLORS['neutral'], 
             label='Perfect Calibration', linewidth=2)
    
    # Actual calibration
    valid_mask = ~np.isnan(bin_confidences)
    ax1.plot(bin_confidences[valid_mask], bin_accuracies[valid_mask], 
             marker='o', color=COLORS['primary'], linewidth=2, markersize=8,
             label='Model Calibration')
    
    # Error bars based on bin counts
    for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
        if not np.isnan(conf) and count > 5:
            # Approximate standard error
            se = np.sqrt(acc * (1 - acc) / count) if acc > 0 and acc < 1 else 0.1
            ax1.errorbar(conf, acc, yerr=se, fmt='none', color=COLORS['primary'], alpha=0.5)
    
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Compute ECE
    total_samples = np.sum(bin_counts)
    ece = np.nansum(bin_counts / total_samples * np.abs(bin_confidences - bin_accuracies))
    ax1.text(0.98, 0.02, f'ECE: {ece:.3f}', transform=ax1.transAxes,
             horizontalalignment='right', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Histogram of predictions
    ax2.bar(bin_centers, bin_counts, width=1/n_bins * 0.8, 
            color=COLORS['primary'], alpha=0.7, edgecolor='white')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_xlim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_probability_scatter(
    model_probs: np.ndarray,
    market_prices: np.ndarray,
    outcomes: np.ndarray,
    title: str = 'Model vs Market Probability',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Scatter plot comparing model probability to market price.
    
    Points above the diagonal indicate model thinks probability is higher
    than market (buy Yes opportunity). Points below indicate buy No opportunity.
    
    Args:
        model_probs: Model probability predictions
        market_prices: Market prices (0-1)
        outcomes: Actual outcomes (0 or 1)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate by outcome
    mentioned = outcomes == 1
    not_mentioned = outcomes == 0
    
    # Plot
    ax.scatter(market_prices[mentioned], model_probs[mentioned], 
               c=COLORS['success'], alpha=0.6, s=50, label='Mentioned (Yes)', edgecolors='white')
    ax.scatter(market_prices[not_mentioned], model_probs[not_mentioned], 
               c=COLORS['danger'], alpha=0.6, s=50, label='Not Mentioned (No)', edgecolors='white')
    
    # Diagonal line (no edge)
    ax.plot([0, 1], [0, 1], linestyle='--', color=COLORS['neutral'], 
            linewidth=2, label='No Edge')
    
    # Edge threshold lines
    edge_threshold = 0.05
    ax.fill_between([0, 1], [edge_threshold, 1 + edge_threshold], 
                    [0, 1], alpha=0.1, color=COLORS['success'], label=f'+{edge_threshold*100:.0f}% edge')
    ax.fill_between([0, 1], [0, 1],
                    [-edge_threshold, 1 - edge_threshold], alpha=0.1, color=COLORS['danger'])
    
    ax.set_xlabel('Market Price', fontsize=12)
    ax.set_ylabel('Model Probability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Add correlation
    corr = np.corrcoef(market_prices, model_probs)[0, 1]
    ax.text(0.98, 0.02, f'Correlation: {corr:.3f}', transform=ax.transAxes,
            horizontalalignment='right', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_prediction_histogram(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    title: str = 'Prediction Distribution',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Histogram of prediction errors and distributions.
    
    Args:
        y_pred: Predicted probabilities
        y_true: True binary labels
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left: Prediction distribution by class
    ax1 = axes[0]
    
    mentioned = y_true == 1
    not_mentioned = y_true == 0
    
    bins = np.linspace(0, 1, 21)
    
    ax1.hist(y_pred[mentioned], bins=bins, alpha=0.7, color=COLORS['success'],
             label='Mentioned', density=True, edgecolor='white')
    ax1.hist(y_pred[not_mentioned], bins=bins, alpha=0.7, color=COLORS['danger'],
             label='Not Mentioned', density=True, edgecolor='white')
    
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Predictions by Outcome', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Prediction error distribution
    ax2 = axes[1]
    
    errors = y_pred - y_true
    
    ax2.hist(errors, bins=30, alpha=0.7, color=COLORS['primary'],
             edgecolor='white', density=True)
    ax2.axvline(x=0, color=COLORS['neutral'], linestyle='--', linewidth=2)
    ax2.axvline(x=np.mean(errors), color=COLORS['danger'], linestyle='-', 
                linewidth=2, label=f'Mean: {np.mean(errors):.3f}')
    
    ax2.set_xlabel('Prediction Error (Pred - Actual)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Error Distribution', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add stats
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    ax2.text(0.98, 0.98, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}', 
             transform=ax2.transAxes, verticalalignment='top',
             horizontalalignment='right', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_trade_timeline(
    trades_df,
    title: str = 'Trade Timeline',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Timeline visualization of individual trades with PnL.
    
    Args:
        trades_df: DataFrame with trade data (from BacktestSimulator)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    import pandas as pd
    
    if trades_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No trades to display', ha='center', va='center', fontsize=14)
        return fig
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, 
                              gridspec_kw={'height_ratios': [2, 1]})
    
    # Top: Individual trade PnLs
    ax1 = axes[0]
    
    trades_df = trades_df.sort_values('earnings_date')
    x = range(len(trades_df))
    
    colors = [COLORS['success'] if pnl > 0 else COLORS['danger'] 
              for pnl in trades_df['pnl']]
    
    ax1.bar(x, trades_df['pnl'], color=colors, alpha=0.7, edgecolor='white')
    ax1.axhline(y=0, color=COLORS['neutral'], linestyle='-', linewidth=1)
    
    ax1.set_xlabel('Trade Number', fontsize=12)
    ax1.set_ylabel('PnL ($)', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Bottom: Cumulative PnL
    ax2 = axes[1]
    
    cumulative_pnl = trades_df['pnl'].cumsum()
    
    ax2.fill_between(x, 0, cumulative_pnl, 
                     where=cumulative_pnl >= 0, alpha=0.3, color=COLORS['success'])
    ax2.fill_between(x, 0, cumulative_pnl, 
                     where=cumulative_pnl < 0, alpha=0.3, color=COLORS['danger'])
    ax2.plot(x, cumulative_pnl, color=COLORS['primary'], linewidth=2)
    ax2.axhline(y=0, color=COLORS['neutral'], linestyle='-', linewidth=1)
    
    ax2.set_xlabel('Trade Number', fontsize=12)
    ax2.set_ylabel('Cumulative PnL ($)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    from sklearn.metrics import confusion_matrix
    
    y_pred_binary = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(cm, cmap='Blues')
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Add labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Not Mentioned', 'Mentioned'])
    ax.set_yticklabels(['Not Mentioned', 'Mentioned'])
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, f'{cm[i, j]}',
                          ha='center', va='center', fontsize=16,
                          color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    # Add metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}'
    ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes,
            verticalalignment='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict,
    title: str = 'Training History',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Plot training history (loss, metrics over epochs).
    
    Args:
        history: Dict with 'train_loss', 'val_loss', etc.
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss
    ax1 = axes[0, 0]
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], label='Train', color=COLORS['primary'])
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], label='Validation', color=COLORS['secondary'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Learning rate
    ax2 = axes[0, 1]
    if 'lr' in history:
        ax2.plot(epochs, history['lr'], color=COLORS['warning'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    
    # AUC-ROC
    ax3 = axes[1, 0]
    if 'val_metrics' in history and len(history['val_metrics']) > 0:
        aucs = [m.get('auc_roc', 0.5) for m in history['val_metrics']]
        ax3.plot(epochs[:len(aucs)], aucs, color=COLORS['success'])
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('AUC-ROC')
        ax3.set_title('Validation AUC-ROC')
        ax3.axhline(y=0.5, color=COLORS['neutral'], linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)
    
    # Log-loss
    ax4 = axes[1, 1]
    if 'val_metrics' in history and len(history['val_metrics']) > 0:
        log_losses = [m.get('log_loss', 0) for m in history['val_metrics']]
        ax4.plot(epochs[:len(log_losses)], log_losses, color=COLORS['danger'])
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Log-Loss')
        ax4.set_title('Validation Log-Loss')
        ax4.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_evaluation_dashboard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    market_prices: np.ndarray,
    trades_df=None,
    equity_curve: Optional[np.ndarray] = None,
    history: Optional[Dict] = None,
    save_dir: Optional[str] = None
) -> Dict[str, plt.Figure]:
    """
    Create comprehensive evaluation dashboard with all plots.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        market_prices: Market prices
        trades_df: DataFrame with trade data
        equity_curve: Equity values over time
        history: Training history dict
        save_dir: Directory to save all plots
        
    Returns:
        Dict mapping plot names to Figure objects
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    figures = {}
    
    # 1. Calibration plot
    fig = plot_calibration(y_true, y_pred)
    figures['calibration'] = fig
    if save_dir:
        fig.savefig(os.path.join(save_dir, 'calibration.png'), dpi=150, bbox_inches='tight')
    
    # 2. Probability scatter
    fig = plot_probability_scatter(y_pred, market_prices, y_true)
    figures['probability_scatter'] = fig
    if save_dir:
        fig.savefig(os.path.join(save_dir, 'probability_scatter.png'), dpi=150, bbox_inches='tight')
    
    # 3. Prediction histogram
    fig = plot_prediction_histogram(y_pred, y_true)
    figures['prediction_histogram'] = fig
    if save_dir:
        fig.savefig(os.path.join(save_dir, 'prediction_histogram.png'), dpi=150, bbox_inches='tight')
    
    # 4. Confusion matrix
    fig = plot_confusion_matrix(y_true, y_pred)
    figures['confusion_matrix'] = fig
    if save_dir:
        fig.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    
    # 5. Equity curve (if available)
    if equity_curve is not None and len(equity_curve) > 0:
        fig = plot_equity_curve(equity_curve)
        figures['equity_curve'] = fig
        if save_dir:
            fig.savefig(os.path.join(save_dir, 'equity_curve.png'), dpi=150, bbox_inches='tight')
    
    # 6. Trade timeline (if available)
    if trades_df is not None and len(trades_df) > 0:
        fig = plot_trade_timeline(trades_df)
        figures['trade_timeline'] = fig
        if save_dir:
            fig.savefig(os.path.join(save_dir, 'trade_timeline.png'), dpi=150, bbox_inches='tight')
    
    # 7. Training history (if available)
    if history is not None and len(history.get('train_loss', [])) > 0:
        fig = plot_training_history(history)
        figures['training_history'] = fig
        if save_dir:
            fig.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    
    plt.close('all')
    
    return figures
