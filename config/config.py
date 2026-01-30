"""
Configuration management for Kalshi mention prediction.

Centralizes all hyperparameters, paths, and settings.
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data-related configuration."""
    
    # Data paths
    data_dir: str = 'nn_directory'
    cache_dir: str = 'cache'
    
    # Tickers to include
    tickers: List[str] = field(default_factory=lambda: ['AAPL'])
    
    # Feature settings
    use_llm_probs: bool = False  # Requires API access to Jetstream
    use_news_features: bool = True
    lookback_quarters: int = 8
    news_lookback_days: int = 21
    
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Architecture
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    use_residual: bool = False
    activation: str = 'relu'
    
    # Output heads
    include_position_head: bool = True
    position_head_hidden: int = 16


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Optimizer
    optimizer: str = 'adamw'
    scheduler: str = 'plateau'
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
    log_interval: int = 10


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    
    # Capital
    initial_bankroll: float = 10000.0
    max_risk_per_trade: float = 0.05
    max_position_size: float = 100.0
    
    # Trading rules
    edge_threshold: float = 0.05
    confidence_threshold: float = 0.0
    
    # Position sizing
    use_kelly_criterion: bool = False
    kelly_fraction: float = 0.25
    use_model_position: bool = True
    
    # Costs
    fee_per_trade: float = 0.0
    slippage: float = 0.01


@dataclass 
class Config:
    """Master configuration containing all sub-configs."""
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    # Global settings
    seed: int = 42
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'
    verbose: bool = True
    output_dir: str = 'outputs'
    
    def __post_init__(self):
        """Ensure output directories exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.training.save_path, exist_ok=True)
        os.makedirs(self.data.cache_dir, exist_ok=True)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'backtest': asdict(self.backtest),
            'seed': self.seed,
            'device': self.device,
            'verbose': self.verbose,
            'output_dir': self.output_dir
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Config':
        """Create config from dictionary."""
        return cls(
            data=DataConfig(**d.get('data', {})),
            model=ModelConfig(**d.get('model', {})),
            training=TrainingConfig(**d.get('training', {})),
            backtest=BacktestConfig(**d.get('backtest', {})),
            seed=d.get('seed', 42),
            device=d.get('device', 'auto'),
            verbose=d.get('verbose', True),
            output_dir=d.get('output_dir', 'outputs')
        )


def load_config(path: str) -> Config:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        d = json.load(f)
    return Config.from_dict(d)


def save_config(config: Config, path: str):
    """Save configuration to JSON file."""
    with open(path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


# Preset configurations
PRESETS = {
    'debug': Config(
        data=DataConfig(tickers=['AAPL']),
        model=ModelConfig(hidden_dims=[32, 16]),
        training=TrainingConfig(epochs=10, batch_size=16, log_interval=1)
    ),
    'standard': Config(
        data=DataConfig(tickers=['AAPL']),
        model=ModelConfig(hidden_dims=[128, 64, 32]),
        training=TrainingConfig(epochs=100, batch_size=32)
    ),
    'production': Config(
        data=DataConfig(
            tickers=['AAPL', 'NVDA', 'TSLA', 'GOOGL', 'MSFT'],
            use_llm_probs=True
        ),
        model=ModelConfig(
            hidden_dims=[256, 128, 64, 32],
            dropout_rate=0.4
        ),
        training=TrainingConfig(
            epochs=200,
            batch_size=64,
            early_stopping_patience=30
        )
    )
}


def get_preset_config(name: str) -> Config:
    """Get a preset configuration by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]
