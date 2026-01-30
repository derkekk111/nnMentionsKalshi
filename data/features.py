"""
Feature engineering utilities for Kalshi mention prediction.

Handles feature extraction, normalization, and transformation.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature processing."""
    
    # Feature groups
    mention_features: List[str] = None
    probability_features: List[str] = None
    market_features: List[str] = None
    news_features: List[str] = None
    keyword_features: List[str] = None
    
    # Normalization settings
    normalize_method: str = 'standard'  # 'standard', 'minmax', or 'none'
    clip_outliers: bool = True
    outlier_std: float = 3.0
    
    def __post_init__(self):
        if self.mention_features is None:
            self.mention_features = [
                'mention_rate_2y', 'mention_count_last_4q', 
                'mention_recency', 'mention_trend'
            ]
        if self.probability_features is None:
            self.probability_features = ['llm_prob']
        if self.market_features is None:
            self.market_features = [
                'market_price', 'open_interest', 'volume', 'days_to_expiry'
            ]
        if self.news_features is None:
            self.news_features = [
                'news_keyword_mentions', 'news_volume', 'news_sentiment'
            ]
        if self.keyword_features is None:
            self.keyword_features = ['keyword_length', 'keyword_word_count']
    
    @property
    def all_features(self) -> List[str]:
        """Get all feature names in order."""
        return (
            self.mention_features + 
            self.probability_features + 
            self.market_features + 
            self.news_features + 
            self.keyword_features
        )


class FeatureExtractor:
    """
    Handles feature extraction and normalization for the mention prediction model.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature extractor.
        
        Args:
            config: Feature configuration (uses defaults if None)
        """
        self.config = config or FeatureConfig()
        self.scaler = None
        self.feature_stats = {}
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'FeatureExtractor':
        """
        Fit the feature extractor on training data.
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            self for chaining
        """
        feature_names = self.config.all_features
        
        # Extract features that exist in the DataFrame
        available_features = [f for f in feature_names if f in df.columns]
        X = df[available_features].values.astype(np.float32)
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip outliers if configured
        if self.config.clip_outliers:
            for i in range(X.shape[1]):
                mean = np.mean(X[:, i])
                std = np.std(X[:, i])
                if std > 0:
                    lower = mean - self.config.outlier_std * std
                    upper = mean + self.config.outlier_std * std
                    X[:, i] = np.clip(X[:, i], lower, upper)
        
        # Fit scaler
        if self.config.normalize_method == 'standard':
            self.scaler = StandardScaler()
        elif self.config.normalize_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
        
        if self.scaler is not None:
            self.scaler.fit(X)
        
        # Store feature statistics
        for i, name in enumerate(available_features):
            self.feature_stats[name] = {
                'mean': float(np.mean(X[:, i])),
                'std': float(np.std(X[:, i])),
                'min': float(np.min(X[:, i])),
                'max': float(np.max(X[:, i]))
            }
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            Normalized feature array
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
        
        feature_names = self.config.all_features
        available_features = [f for f in feature_names if f in df.columns]
        
        X = df[available_features].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip outliers
        if self.config.clip_outliers:
            for i, name in enumerate(available_features):
                if name in self.feature_stats:
                    stats = self.feature_stats[name]
                    lower = stats['mean'] - self.config.outlier_std * stats['std']
                    upper = stats['mean'] + self.config.outlier_std * stats['std']
                    X[:, i] = np.clip(X[:, i], lower, upper)
        
        # Apply scaler
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def to_tensor(self, X: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor."""
        return torch.from_numpy(X.astype(np.float32))
    
    def get_feature_importance_template(self) -> Dict[str, float]:
        """Get a template dict for feature importance tracking."""
        return {name: 0.0 for name in self.config.all_features}
    
    def save(self, path: str):
        """Save fitted extractor to disk."""
        import pickle
        
        state = {
            'config': self.config,
            'scaler': self.scaler,
            'feature_stats': self.feature_stats,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str) -> 'FeatureExtractor':
        """Load fitted extractor from disk."""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        extractor = cls(config=state['config'])
        extractor.scaler = state['scaler']
        extractor.feature_stats = state['feature_stats']
        extractor.is_fitted = state['is_fitted']
        
        return extractor


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional derived features from base features.
    
    Args:
        df: DataFrame with base features
        
    Returns:
        DataFrame with additional derived features
    """
    df = df.copy()
    
    # Probability-based features
    # Mention-based features
    if 'mention_rate_2y' in df.columns and 'mention_trend' in df.columns:
        # Adjusted rate based on trend
        df['adjusted_mention_rate'] = (
            df['mention_rate_2y'] * (1 + 0.1 * df['mention_trend'])
        )
    
    # Market-based features
    if 'open_interest' in df.columns and 'volume' in df.columns:
        # Liquidity indicator
        df['liquidity'] = np.log1p(df['open_interest'] + df['volume'])
    
    if 'market_price' in df.columns:
        # Price uncertainty (highest near 0.5)
        df['price_uncertainty'] = 1 - np.abs(2 * df['market_price'] - 1)
    
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between different feature groups.
    
    Args:
        df: DataFrame with base features
        
    Returns:
        DataFrame with interaction features added
    """
    df = df.copy()
    
    # News volume * keyword mentions
    if 'news_volume' in df.columns and 'news_keyword_mentions' in df.columns:
        # Proportion of news mentioning keyword
        df['news_mention_rate'] = np.where(
            df['news_volume'] > 0,
            df['news_keyword_mentions'] / df['news_volume'],
            0.0
        )
    
    # Market price * mention recency (recent mentions at low price = opportunity)
    if 'market_price' in df.columns and 'mention_recency' in df.columns:
        # Normalize recency to 0-1 (lower is more recent)
        max_recency = df['mention_recency'].max()
        if max_recency > 0:
            normalized_recency = df['mention_recency'] / max_recency
            df['recency_value'] = (1 - df['market_price']) * (1 - normalized_recency)
    
    return df
