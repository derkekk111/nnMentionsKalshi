"""
Evaluation metrics for Kalshi mention prediction.

Implements:
- Log-loss: Penalizes confident wrong predictions
- Brier score: MSE for probabilities
- ECE: Expected Calibration Error
- AUC-ROC: Ranking quality
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    log_loss as sklearn_log_loss,
    brier_score_loss,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def compute_log_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-7
) -> float:
    """
    Compute log loss (binary cross-entropy).
    
    Log-loss penalizes confident wrong predictions heavily.
    -log(0.01) = 4.6 vs -log(0.99) = 0.01
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        eps: Small value for numerical stability
        
    Returns:
        Log loss value (lower is better)
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return sklearn_log_loss(y_true, y_pred)


def compute_brier_score(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute Brier score.
    
    Brier = mean((pred - actual)^2)
    Equivalent to MSE for probability predictions.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        
    Returns:
        Brier score (lower is better, 0-1 range)
    """
    return brier_score_loss(y_true, y_pred)


def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, Dict]:
    """
    Compute Expected Calibration Error.
    
    ECE measures how well predicted probabilities match actual frequencies.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        Tuple of (ECE value, bin_details dict)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_details = []
    
    total_samples = len(y_true)
    ece = 0.0
    
    for i in range(n_bins):
        # Find samples in this bin
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            # Include upper boundary for last bin
            mask = (y_pred >= lower) & (y_pred <= upper)
        else:
            mask = (y_pred >= lower) & (y_pred < upper)
        
        bin_samples = np.sum(mask)
        
        if bin_samples > 0:
            # Average predicted probability in bin
            avg_confidence = np.mean(y_pred[mask])
            # Actual accuracy in bin
            avg_accuracy = np.mean(y_true[mask])
            # Calibration error for this bin
            bin_error = np.abs(avg_confidence - avg_accuracy)
            
            # Weight by proportion of samples
            ece += (bin_samples / total_samples) * bin_error
            
            bin_details.append({
                'bin': i,
                'lower': lower,
                'upper': upper,
                'samples': int(bin_samples),
                'avg_confidence': float(avg_confidence),
                'avg_accuracy': float(avg_accuracy),
                'calibration_error': float(bin_error)
            })
        else:
            bin_details.append({
                'bin': i,
                'lower': lower,
                'upper': upper,
                'samples': 0,
                'avg_confidence': None,
                'avg_accuracy': None,
                'calibration_error': None
            })
    
    return float(ece), {'bins': bin_details}


def compute_auc_roc(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute AUC-ROC score.
    
    Measures ranking quality (probability of ranking a random positive
    higher than a random negative).
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        
    Returns:
        AUC-ROC score (0.5 = random, 1.0 = perfect)
    """
    # Need both classes present
    if len(np.unique(y_true)) < 2:
        return 0.5
    
    return roc_auc_score(y_true, y_pred)


def compute_accuracy_at_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute classification metrics at a specific threshold.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dict with accuracy, precision, recall, f1
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Handle edge cases
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred_binary)) < 2:
        return {
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    return {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0)
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, int]:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dict with TP, TN, FP, FN
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    return {
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    n_bins: int = 10
) -> Dict:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        n_bins: Number of bins for ECE
        
    Returns:
        Dict with all metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Probability metrics
    log_loss_val = compute_log_loss(y_true, y_pred)
    brier_val = compute_brier_score(y_true, y_pred)
    ece_val, ece_details = compute_ece(y_true, y_pred, n_bins)
    auc_val = compute_auc_roc(y_true, y_pred)
    
    # Classification metrics
    class_metrics = compute_accuracy_at_threshold(y_true, y_pred, threshold)
    conf_matrix = compute_confusion_matrix(y_true, y_pred, threshold)
    
    # Summary statistics
    n_samples = len(y_true)
    n_positive = int(np.sum(y_true))
    positive_rate = n_positive / n_samples if n_samples > 0 else 0
    
    return {
        # Probability metrics (primary)
        'log_loss': log_loss_val,
        'brier_score': brier_val,
        'ece': ece_val,
        'auc_roc': auc_val,
        
        # Classification metrics
        'accuracy': class_metrics['accuracy'],
        'precision': class_metrics['precision'],
        'recall': class_metrics['recall'],
        'f1': class_metrics['f1'],
        
        # Confusion matrix
        'confusion_matrix': conf_matrix,
        
        # Calibration details
        'calibration': ece_details,
        
        # Summary
        'n_samples': n_samples,
        'n_positive': n_positive,
        'positive_rate': positive_rate,
        
        # Prediction statistics
        'pred_mean': float(np.mean(y_pred)),
        'pred_std': float(np.std(y_pred)),
        'pred_min': float(np.min(y_pred)),
        'pred_max': float(np.max(y_pred))
    }


def compute_trading_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    market_prices: np.ndarray,
    edge_threshold: float = 0.05
) -> Dict:
    """
    Compute trading-specific metrics.
    
    Args:
        y_true: True binary outcomes
        y_pred: Predicted probabilities
        market_prices: Market prices (0-1)
        edge_threshold: Minimum edge to trade
        
    Returns:
        Dict with trading metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    market_prices = np.asarray(market_prices).flatten()
    
    # Calculate edge
    edge = y_pred - market_prices
    
    # Trades with sufficient edge
    trade_mask = np.abs(edge) >= edge_threshold
    n_trades = np.sum(trade_mask)
    
    if n_trades == 0:
        return {
            'n_trades': 0,
            'win_rate': 0.0,
            'avg_edge': 0.0,
            'total_pnl': 0.0,
            'avg_pnl_per_trade': 0.0
        }
    
    # Trade direction (positive edge = buy Yes, negative = buy No)
    trade_direction = np.sign(edge[trade_mask])
    
    # PnL calculation
    # Buy Yes: PnL = outcome - price
    # Buy No: PnL = (1 - outcome) - (1 - price) = price - outcome
    pnl_yes = y_true[trade_mask] - market_prices[trade_mask]
    pnl_no = market_prices[trade_mask] - y_true[trade_mask]
    
    pnl = np.where(trade_direction >= 0, pnl_yes, pnl_no)
    
    # Win rate
    wins = np.sum(pnl > 0)
    win_rate = wins / n_trades
    
    return {
        'n_trades': int(n_trades),
        'n_trades_yes': int(np.sum(trade_direction >= 0)),
        'n_trades_no': int(np.sum(trade_direction < 0)),
        'win_rate': float(win_rate),
        'avg_edge': float(np.mean(np.abs(edge[trade_mask]))),
        'total_pnl': float(np.sum(pnl)),
        'avg_pnl_per_trade': float(np.mean(pnl)),
        'pnl_std': float(np.std(pnl)),
        'max_win': float(np.max(pnl)),
        'max_loss': float(np.min(pnl))
    }


def format_metrics_report(metrics: Dict) -> str:
    """
    Format metrics as a human-readable report.
    
    Args:
        metrics: Dict from compute_all_metrics
        
    Returns:
        Formatted string report
    """
    lines = [
        "=" * 60,
        "EVALUATION METRICS REPORT",
        "=" * 60,
        "",
        "PROBABILITY METRICS (lower is better for loss metrics):",
        f"  Log-Loss:     {metrics['log_loss']:.4f}",
        f"  Brier Score:  {metrics['brier_score']:.4f}",
        f"  ECE:          {metrics['ece']:.4f}",
        f"  AUC-ROC:      {metrics['auc_roc']:.4f}",
        "",
        "CLASSIFICATION METRICS:",
        f"  Accuracy:     {metrics['accuracy']:.4f}",
        f"  Precision:    {metrics['precision']:.4f}",
        f"  Recall:       {metrics['recall']:.4f}",
        f"  F1 Score:     {metrics['f1']:.4f}",
        "",
        "CONFUSION MATRIX:",
        f"  True Positives:  {metrics['confusion_matrix']['true_positives']}",
        f"  True Negatives:  {metrics['confusion_matrix']['true_negatives']}",
        f"  False Positives: {metrics['confusion_matrix']['false_positives']}",
        f"  False Negatives: {metrics['confusion_matrix']['false_negatives']}",
        "",
        "DATASET SUMMARY:",
        f"  Total Samples:   {metrics['n_samples']}",
        f"  Positive Rate:   {metrics['positive_rate']:.2%}",
        f"  Pred Mean:       {metrics['pred_mean']:.4f}",
        f"  Pred Std:        {metrics['pred_std']:.4f}",
        "=" * 60
    ]
    
    return "\n".join(lines)
