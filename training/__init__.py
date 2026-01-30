"""Training module for Kalshi mention market prediction."""

from .trainer import Trainer, TrainingConfig
from .metrics import (
    compute_log_loss,
    compute_brier_score,
    compute_ece,
    compute_auc_roc,
    compute_all_metrics
)

__all__ = [
    'Trainer',
    'TrainingConfig',
    'compute_log_loss',
    'compute_brier_score',
    'compute_ece',
    'compute_auc_roc',
    'compute_all_metrics'
]
