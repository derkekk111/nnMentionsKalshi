"""
MentionPredictor: Neural network for Kalshi mention market prediction.

Features:
- Configurable MLP architecture (2-4 hidden layers)
- Dual output heads:
  - Probability head: P(keyword mentioned)
  - Position head: Recommended trade position size [0, 1]
- BatchNorm and Dropout for regularization
- Optional residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for MentionPredictor model."""
    
    # Input dimensions
    n_features: int = 15
    
    # Architecture
    hidden_dims: List[int] = None  # Default: [128, 64, 32]
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    use_residual: bool = False
    activation: str = 'relu'  # 'relu', 'gelu', 'leaky_relu'
    
    # Output heads
    include_position_head: bool = True
    position_head_hidden: int = 16
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]


class MLPBlock(nn.Module):
    """
    MLP block with optional BatchNorm, Dropout, and residual connection.
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        use_residual: bool = False,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.use_residual = use_residual and (in_dim == out_dim)
        
        layers = [nn.Linear(in_dim, out_dim)]
        
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        
        # Activation
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
        
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class MentionPredictor(nn.Module):
    """
    Neural network for predicting Kalshi mention market outcomes.
    
    Architecture:
    - Input layer with feature normalization
    - MLP backbone with configurable depth
    - Dual output heads:
      - Probability head: Sigmoid -> P(mention)
      - Position head: Sigmoid -> position size [0, 1]
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the model.
        
        Args:
            config: Model configuration (uses defaults if None)
        """
        super().__init__()
        
        self.config = config or ModelConfig()
        
        # Input normalization layer
        self.input_norm = nn.BatchNorm1d(self.config.n_features)
        
        # Build MLP backbone
        dims = [self.config.n_features] + self.config.hidden_dims
        
        backbone_layers = []
        for i in range(len(dims) - 1):
            backbone_layers.append(
                MLPBlock(
                    dims[i], 
                    dims[i + 1],
                    dropout_rate=self.config.dropout_rate,
                    use_batch_norm=self.config.use_batch_norm,
                    use_residual=self.config.use_residual,
                    activation=self.config.activation
                )
            )
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # Probability head
        backbone_out_dim = self.config.hidden_dims[-1]
        self.prob_head = nn.Sequential(
            nn.Linear(backbone_out_dim, 1),
            nn.Sigmoid()
        )
        
        # Position sizing head (optional)
        self.include_position_head = self.config.include_position_head
        if self.include_position_head:
            self.position_head = nn.Sequential(
                nn.Linear(backbone_out_dim, self.config.position_head_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.position_head_hidden, 1),
                nn.Sigmoid()
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_position: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input features of shape (batch_size, n_features)
            return_position: Whether to return position size prediction
            
        Returns:
            prob: Probability predictions of shape (batch_size, 1)
            position: Position size predictions of shape (batch_size, 1) or None
        """
        # Normalize input
        x = self.input_norm(x)
        
        # Backbone
        features = self.backbone(x)
        
        # Probability prediction
        prob = self.prob_head(features)
        
        # Position sizing prediction
        position = None
        if return_position and self.include_position_head:
            position = self.position_head(features)
        
        return prob, position
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions only.
        
        Args:
            x: Input features
            
        Returns:
            Probability predictions
        """
        self.eval()
        with torch.no_grad():
            prob, _ = self.forward(x, return_position=False)
        return prob
    
    def predict_with_position(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get both probability and position predictions.
        
        Args:
            x: Input features
            
        Returns:
            Tuple of (probabilities, positions)
        """
        self.eval()
        with torch.no_grad():
            prob, position = self.forward(x, return_position=True)
        return prob, position
    
    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get intermediate backbone features for analysis.
        
        Args:
            x: Input features
            
        Returns:
            Backbone output features
        """
        x = self.input_norm(x)
        return self.backbone(x)
    
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        backbone = sum(
            p.numel() for p in self.backbone.parameters()
        )
        prob_head = sum(
            p.numel() for p in self.prob_head.parameters()
        )
        position_head = sum(
            p.numel() for p in self.position_head.parameters()
        ) if self.include_position_head else 0
        
        return {
            'total': total,
            'trainable': trainable,
            'backbone': backbone,
            'prob_head': prob_head,
            'position_head': position_head
        }


class EnsembleMentionPredictor(nn.Module):
    """
    Ensemble of MentionPredictor models for improved predictions.
    """
    
    def __init__(
        self,
        n_models: int = 5,
        config: Optional[ModelConfig] = None
    ):
        """
        Initialize ensemble.
        
        Args:
            n_models: Number of models in ensemble
            config: Shared model configuration
        """
        super().__init__()
        
        self.n_models = n_models
        self.models = nn.ModuleList([
            MentionPredictor(config) for _ in range(n_models)
        ])
    
    def forward(
        self, 
        x: torch.Tensor,
        return_position: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with ensemble averaging.
        
        Returns mean predictions across all models.
        """
        probs = []
        positions = []
        
        for model in self.models:
            prob, position = model(x, return_position=return_position)
            probs.append(prob)
            if position is not None:
                positions.append(position)
        
        # Average predictions
        avg_prob = torch.stack(probs).mean(dim=0)
        avg_position = None
        if positions:
            avg_position = torch.stack(positions).mean(dim=0)
        
        return avg_prob, avg_position
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get predictions with uncertainty estimates.
        
        Args:
            x: Input features
            
        Returns:
            Tuple of (mean_prob, std_prob, positions)
        """
        self.eval()
        with torch.no_grad():
            probs = []
            positions = []
            
            for model in self.models:
                prob, position = model(x, return_position=True)
                probs.append(prob)
                if position is not None:
                    positions.append(position)
            
            probs = torch.stack(probs)
            mean_prob = probs.mean(dim=0)
            std_prob = probs.std(dim=0)
            
            avg_position = None
            if positions:
                avg_position = torch.stack(positions).mean(dim=0)
            
        return mean_prob, std_prob, avg_position


def create_model(
    n_features: int,
    hidden_dims: Optional[List[int]] = None,
    dropout_rate: float = 0.3,
    include_position_head: bool = True
) -> MentionPredictor:
    """
    Convenience function to create a MentionPredictor model.
    
    Args:
        n_features: Number of input features
        hidden_dims: Hidden layer dimensions
        dropout_rate: Dropout rate
        include_position_head: Whether to include position sizing head
        
    Returns:
        Configured MentionPredictor model
    """
    config = ModelConfig(
        n_features=n_features,
        hidden_dims=hidden_dims or [128, 64, 32],
        dropout_rate=dropout_rate,
        include_position_head=include_position_head
    )
    
    return MentionPredictor(config)


def load_model(path: str, device: str = 'cpu') -> MentionPredictor:
    """
    Load a saved model.
    
    Args:
        path: Path to saved model checkpoint
        device: Device to load model to
        
    Returns:
        Loaded MentionPredictor model
    """
    # weights_only=False needed for loading custom config objects
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Reconstruct config
    config = checkpoint.get('config', ModelConfig())
    if isinstance(config, dict):
        config = ModelConfig(**config)
    
    # Create and load model
    model = MentionPredictor(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def save_model(
    model: MentionPredictor, 
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[Dict] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        path: Path to save to
        optimizer: Optional optimizer state
        epoch: Current epoch
        metrics: Optional metrics dict
    """
    checkpoint = {
        'config': model.config,
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'metrics': metrics or {}
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, path)
