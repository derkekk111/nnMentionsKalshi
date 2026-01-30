"""
Training pipeline for Kalshi mention prediction.

Features:
- Time-based train/val/test split (NO data leakage)
- Combined loss function (probability + position + calibration)
- Early stopping on validation log-loss
- Learning rate scheduling
- Comprehensive logging and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
import json

from .metrics import compute_all_metrics, compute_trading_metrics, format_metrics_report


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Optimizer settings
    optimizer: str = 'adamw'  # 'adam', 'adamw', 'sgd'
    scheduler: str = 'plateau'  # 'plateau', 'cosine', 'step', 'none'
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    
    # Loss weights
    prob_weight: float = 1.0
    position_weight: float = 0.5
    calibration_weight: float = 0.1
    use_focal_loss: bool = False
    
    # Checkpointing
    save_best: bool = True
    save_path: str = 'checkpoints'
    
    # Logging
    log_interval: int = 10
    verbose: bool = True


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """Check if training should stop."""
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """
    Trainer for MentionPredictor model.
    
    Handles the full training loop with:
    - Multi-task loss optimization
    - Validation monitoring
    - Early stopping
    - Checkpointing
    - Metrics computation
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        device: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: MentionPredictor model
            config: Training configuration
            device: Device to train on ('cpu', 'cuda', 'mps')
        """
        self.model = model
        self.config = config or TrainingConfig()
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = self._create_loss()
        
        # Initialize early stopping
        if self.config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                mode='min'
            )
        else:
            self.early_stopping = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'lr': []
        }
        
        # Best model state
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.best_epoch = 0
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.save_path)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        params = self.model.parameters()
        
        if self.config.optimizer == 'adam':
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adamw':
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on config."""
        if self.config.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience
            )
        elif self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.scheduler_patience,
                gamma=self.config.scheduler_factor
            )
        else:
            return None
    
    def _create_loss(self) -> nn.Module:
        """Create loss function."""
        from models.losses import CombinedLoss
        
        return CombinedLoss(
            prob_weight=self.config.prob_weight,
            position_weight=self.config.position_weight,
            calibration_weight=self.config.calibration_weight,
            use_focal_loss=self.config.use_focal_loss
        )
    
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.config.verbose:
            print(message)
    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Tuple[float, Dict]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, loss_components)
        """
        self.model.train()
        
        total_loss = 0.0
        total_components = {'probability': 0.0, 'position': 0.0, 'calibration': 0.0}
        n_batches = 0
        
        all_preds = []
        all_labels = []
        
        for batch_idx, (features, labels, metadata) in enumerate(train_loader):
            # Move to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Get market prices for position loss
            market_prices = torch.tensor(
                [m['market_price'] for m in metadata],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            prob_pred, position_pred = self.model(features, return_position=True)
            
            # Compute loss
            loss, components = self.criterion(
                prob_pred, position_pred, labels, market_prices
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            for key in total_components:
                total_components[key] += components.get(key, 0.0)
            n_batches += 1
            
            # Store predictions for metrics
            all_preds.extend(prob_pred.detach().cpu().numpy().flatten())
            all_labels.extend(labels.detach().cpu().numpy().flatten())
        
        # Average losses
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in total_components.items()}
        
        # Compute metrics
        metrics = compute_all_metrics(
            np.array(all_labels),
            np.array(all_preds)
        )
        
        return avg_loss, avg_components, metrics
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, Dict, Dict]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, loss_components, metrics)
        """
        self.model.eval()
        
        total_loss = 0.0
        total_components = {'probability': 0.0, 'position': 0.0, 'calibration': 0.0}
        n_batches = 0
        
        all_preds = []
        all_labels = []
        all_market_prices = []
        
        with torch.no_grad():
            for features, labels, metadata in val_loader:
                # Move to device
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Get market prices
                market_prices = torch.tensor(
                    [m['market_price'] for m in metadata],
                    dtype=torch.float32,
                    device=self.device
                ).unsqueeze(1)
                
                # Forward pass
                prob_pred, position_pred = self.model(features, return_position=True)
                
                # Compute loss
                loss, components = self.criterion(
                    prob_pred, position_pred, labels, market_prices
                )
                
                # Track metrics
                total_loss += loss.item()
                for key in total_components:
                    total_components[key] += components.get(key, 0.0)
                n_batches += 1
                
                # Store predictions
                all_preds.extend(prob_pred.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                all_market_prices.extend(market_prices.cpu().numpy().flatten())
        
        # Average losses
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in total_components.items()}
        
        # Compute metrics
        metrics = compute_all_metrics(
            np.array(all_labels),
            np.array(all_preds)
        )
        
        # Add trading metrics
        trading_metrics = compute_trading_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_market_prices)
        )
        metrics['trading'] = trading_metrics
        
        return avg_loss, avg_components, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
            
        Returns:
            Training history dict
        """
        self._log(f"\nStarting training for {self.config.epochs} epochs")
        self._log(f"Device: {self.device}")
        self._log(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self._log("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_components, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_components, val_metrics = self.validate(val_loader)
            
            # Update scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['lr'].append(current_lr)
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.best_epoch = epoch
                
                if self.config.save_best:
                    self._save_checkpoint(epoch, val_loss, is_best=True)
            
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_loss):
                    self._log(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
            
            # Logging
            if (epoch + 1) % self.config.log_interval == 0 or epoch == 0:
                epoch_time = time.time() - epoch_start
                self._log(
                    f"Epoch {epoch + 1:3d}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val AUC: {val_metrics['auc_roc']:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:.1f}s"
                )
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self._log(f"\nRestored best model from epoch {self.best_epoch + 1}")
        
        # Final evaluation
        total_time = time.time() - start_time
        self._log(f"\nTraining completed in {total_time / 60:.1f} minutes")
        
        # Test evaluation if provided
        if test_loader is not None:
            self._log("\nEvaluating on test set...")
            test_loss, test_components, test_metrics = self.validate(test_loader)
            self.history['test_metrics'] = test_metrics
            self._log(format_metrics_report(test_metrics))
        
        return self.history
    
    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ):
        """Save model checkpoint."""
        from models.mention_predictor import save_model
        
        filename = 'best_model.pt' if is_best else f'checkpoint_epoch_{epoch}.pt'
        path = self.checkpoint_dir / filename
        
        save_model(
            self.model,
            str(path),
            optimizer=self.optimizer,
            epoch=epoch,
            metrics={'val_loss': val_loss}
        )
    
    def predict(
        self,
        data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Generate predictions for a dataset.
        
        Args:
            data_loader: Data loader
            
        Returns:
            Tuple of (probabilities, positions, metadata_list)
        """
        self.model.eval()
        
        all_probs = []
        all_positions = []
        all_metadata = []
        
        with torch.no_grad():
            for features, labels, metadata in data_loader:
                features = features.to(self.device)
                
                prob, position = self.model(features, return_position=True)
                
                all_probs.extend(prob.cpu().numpy().flatten())
                if position is not None:
                    all_positions.extend(position.cpu().numpy().flatten())
                all_metadata.extend(metadata)
        
        return (
            np.array(all_probs),
            np.array(all_positions) if all_positions else None,
            all_metadata
        )
    
    def save_history(self, path: str):
        """Save training history to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {}
        for key, value in self.history.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    history_serializable[key] = value
                else:
                    history_serializable[key] = [float(v) for v in value]
            else:
                history_serializable[key] = value
        
        with open(path, 'w') as f:
            json.dump(history_serializable, f, indent=2, default=str)
