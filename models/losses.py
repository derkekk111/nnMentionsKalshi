"""
Custom loss functions for Kalshi mention prediction.

Includes:
- CombinedLoss: Multi-task loss for probability + position sizing
- PositionLoss: PnL-aware loss for position sizing head
- CalibrationLoss: Regularization for calibrated probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reduces loss for well-classified examples, focusing on hard cases.
    FL(p) = -alpha * (1-p)^gamma * log(p)
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            pred: Predicted probabilities (batch_size, 1)
            target: Binary targets (batch_size, 1)
            
        Returns:
            Loss value
        """
        # Clamp predictions for numerical stability
        pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
        
        # Binary cross entropy
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        # Focal weighting
        pt = target * pred + (1 - target) * (1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class PositionLoss(nn.Module):
    """
    PnL-aware loss for position sizing.
    
    Penalizes:
    - Large positions on incorrect predictions
    - Small positions on correct predictions with good edge
    """
    
    def __init__(
        self,
        edge_threshold: float = 0.1,
        max_position: float = 1.0
    ):
        """
        Initialize position loss.
        
        Args:
            edge_threshold: Minimum edge to justify a position
            max_position: Maximum allowed position size
        """
        super().__init__()
        self.edge_threshold = edge_threshold
        self.max_position = max_position
    
    def forward(
        self,
        position: torch.Tensor,
        pred_prob: torch.Tensor,
        market_price: torch.Tensor,
        actual_outcome: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute position sizing loss.
        
        Args:
            position: Predicted position size (0-1)
            pred_prob: Predicted probability
            market_price: Market price (0-1)
            actual_outcome: Actual outcome (0 or 1)
            
        Returns:
            Loss value
        """
        # Calculate edge (model prob - market price)
        edge = pred_prob - market_price
        
        # Determine optimal trade direction
        # Positive edge = buy Yes, negative edge = buy No
        trade_direction = torch.sign(edge)
        
        # Calculate PnL per trade (simplified)
        # If buy Yes: PnL = outcome - price
        # If buy No: PnL = (1 - outcome) - (1 - price) = price - outcome
        pnl_yes = actual_outcome - market_price
        pnl_no = market_price - actual_outcome
        
        # Select PnL based on trade direction
        pnl = torch.where(trade_direction >= 0, pnl_yes, pnl_no)
        
        # Realized PnL = position * pnl
        realized_pnl = position * pnl
        
        # Loss = negative PnL (we want to maximize PnL)
        # Add regularization for position sizing
        position_reg = 0.01 * (position ** 2)  # Penalize large positions
        
        # Only trade when edge exceeds threshold
        edge_mask = (torch.abs(edge) >= self.edge_threshold).float()
        
        # Final loss: negative PnL + regularization
        loss = -realized_pnl + position_reg
        loss = loss * edge_mask + (1 - edge_mask) * (position ** 2)  # Penalize positions without edge
        
        return loss.mean()


class CalibrationLoss(nn.Module):
    """
    Expected Calibration Error (ECE) as a differentiable loss.
    
    Encourages predicted probabilities to match empirical frequencies.
    """
    
    def __init__(
        self,
        n_bins: int = 10,
        temperature: float = 1.0
    ):
        """
        Initialize calibration loss.
        
        Args:
            n_bins: Number of calibration bins
            temperature: Softmax temperature for soft binning
        """
        super().__init__()
        self.n_bins = n_bins
        self.temperature = temperature
        
        # Bin boundaries
        self.register_buffer(
            'bin_boundaries',
            torch.linspace(0, 1, n_bins + 1)
        )
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute calibration loss (differentiable ECE approximation).
        
        Args:
            pred: Predicted probabilities
            target: Binary targets
            
        Returns:
            Calibration loss
        """
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Move bin_boundaries to same device as pred
        bin_boundaries = self.bin_boundaries.to(pred.device)
        
        # Soft bin assignments using temperature-scaled sigmoid
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        bin_width = 1.0 / self.n_bins
        
        # Distance to each bin center
        distances = (pred.unsqueeze(1) - bin_centers.unsqueeze(0)).abs()
        
        # Soft assignment (closer = higher weight)
        weights = F.softmax(-distances / (self.temperature * bin_width), dim=1)
        
        # Weighted average prediction per bin
        weighted_preds = (weights * pred.unsqueeze(1)).sum(dim=0)
        bin_counts = weights.sum(dim=0) + 1e-7
        avg_pred_per_bin = weighted_preds / bin_counts
        
        # Weighted average accuracy per bin
        weighted_targets = (weights * target.unsqueeze(1)).sum(dim=0)
        avg_target_per_bin = weighted_targets / bin_counts
        
        # ECE: weighted average of |confidence - accuracy|
        calibration_error = torch.abs(avg_pred_per_bin - avg_target_per_bin)
        ece = (bin_counts / bin_counts.sum() * calibration_error).sum()
        
        return ece


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task learning.
    
    Combines:
    - Probability prediction loss (BCE or Focal)
    - Position sizing loss (PnL-aware)
    - Calibration regularization
    """
    
    def __init__(
        self,
        prob_weight: float = 1.0,
        position_weight: float = 0.5,
        calibration_weight: float = 0.1,
        use_focal_loss: bool = False,
        focal_gamma: float = 2.0
    ):
        """
        Initialize combined loss.
        
        Args:
            prob_weight: Weight for probability loss
            position_weight: Weight for position loss
            calibration_weight: Weight for calibration loss
            use_focal_loss: Whether to use focal loss for probability
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        
        self.prob_weight = prob_weight
        self.position_weight = position_weight
        self.calibration_weight = calibration_weight
        
        # Probability loss
        if use_focal_loss:
            self.prob_loss = FocalLoss(gamma=focal_gamma)
        else:
            self.prob_loss = nn.BCELoss()
        
        # Position loss
        self.position_loss = PositionLoss()
        
        # Calibration loss
        self.calibration_loss = CalibrationLoss()
    
    def forward(
        self,
        pred_prob: torch.Tensor,
        pred_position: Optional[torch.Tensor],
        target: torch.Tensor,
        market_price: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            pred_prob: Predicted probabilities
            pred_position: Predicted position sizes (optional)
            target: Binary targets
            market_price: Market prices for position loss (optional)
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Probability loss
        loss_prob = self.prob_loss(pred_prob, target)
        
        # Calibration loss
        loss_calib = self.calibration_loss(pred_prob, target)
        
        # Position loss (if applicable)
        loss_position = torch.tensor(0.0, device=pred_prob.device)
        if pred_position is not None and market_price is not None:
            loss_position = self.position_loss(
                pred_position, pred_prob, market_price, target
            )
        
        # Combine losses
        total_loss = (
            self.prob_weight * loss_prob +
            self.position_weight * loss_position +
            self.calibration_weight * loss_calib
        )
        
        components = {
            'total': total_loss.item(),
            'probability': loss_prob.item(),
            'position': loss_position.item(),
            'calibration': loss_calib.item()
        }
        
        return total_loss, components


class BrierLoss(nn.Module):
    """
    Brier Score as a loss function.
    
    Brier = mean((pred - actual)^2)
    Equivalent to MSE for probability predictions.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute Brier score loss."""
        loss = (pred - target) ** 2
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LogLoss(nn.Module):
    """
    Log Loss (Binary Cross-Entropy) with numerical stability.
    
    This is the standard metric for probability predictions.
    """
    
    def __init__(self, eps: float = 1e-7, reduction: str = 'mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute log loss."""
        pred = torch.clamp(pred, min=self.eps, max=1 - self.eps)
        
        loss = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
