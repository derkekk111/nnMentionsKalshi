"""Evaluation and visualization module for Kalshi mention market prediction."""

from .visualize import (
    plot_equity_curve,
    plot_calibration,
    plot_probability_scatter,
    plot_prediction_histogram,
    plot_trade_timeline,
    plot_confusion_matrix,
    plot_training_history,
    create_evaluation_dashboard
)

__all__ = [
    'plot_equity_curve',
    'plot_calibration',
    'plot_probability_scatter',
    'plot_prediction_histogram',
    'plot_trade_timeline',
    'plot_confusion_matrix',
    'plot_training_history',
    'create_evaluation_dashboard'
]
