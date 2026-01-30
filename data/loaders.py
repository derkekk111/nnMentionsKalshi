"""
DataLoader utilities for Kalshi mention prediction.

Handles train/validation splits with proper time-based separation
to prevent data leakage.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Dict, List, Optional, Tuple
from datetime import datetime


def time_based_split(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[Subset, Subset, Subset]:
    """
    Split dataset by time to prevent data leakage.
    
    Older earnings calls go to training, newer to validation/test.
    
    Args:
        dataset: MentionDataset instance
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        
    Returns:
        Tuple of (train_subset, val_subset, test_subset)
    """
    # Get earnings dates for all samples
    dates_and_indices = []
    
    for i in range(len(dataset)):
        _, _, metadata = dataset[i]
        date = metadata.get('earnings_date')
        if isinstance(date, datetime):
            dates_and_indices.append((date, i))
        else:
            # Try to parse date string
            try:
                date = pd.to_datetime(date)
                dates_and_indices.append((date, i))
            except:
                # Use epoch as fallback
                dates_and_indices.append((datetime(1970, 1, 1), i))
    
    # Sort by date
    dates_and_indices.sort(key=lambda x: x[0])
    sorted_indices = [idx for _, idx in dates_and_indices]
    
    # Calculate split points
    n = len(sorted_indices)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    # Create subsets
    train_indices = sorted_indices[:train_end]
    val_indices = sorted_indices[train_end:val_end]
    test_indices = sorted_indices[val_end:]
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    return train_subset, val_subset, test_subset


def time_based_split_by_date(
    dataset: Dataset,
    train_end_date: datetime,
    val_end_date: datetime
) -> Tuple[Subset, Subset, Subset]:
    """
    Split dataset using specific date cutoffs.
    
    Args:
        dataset: MentionDataset instance
        train_end_date: Last date for training data
        val_end_date: Last date for validation data
        
    Returns:
        Tuple of (train_subset, val_subset, test_subset)
    """
    train_indices = []
    val_indices = []
    test_indices = []
    
    for i in range(len(dataset)):
        _, _, metadata = dataset[i]
        date = metadata.get('earnings_date')
        
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        if date <= train_end_date:
            train_indices.append(i)
        elif date <= val_end_date:
            val_indices.append(i)
        else:
            test_indices.append(i)
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    return train_subset, val_subset, test_subset


def collate_with_metadata(
    batch: List[Tuple[torch.Tensor, torch.Tensor, Dict]]
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Custom collate function that preserves metadata.
    
    Args:
        batch: List of (features, label, metadata) tuples
        
    Returns:
        Tuple of (batched_features, batched_labels, list_of_metadata)
    """
    features = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    metadata = [item[2] for item in batch]
    
    return features, labels, metadata


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders with time-based splitting.
    
    Args:
        dataset: MentionDataset instance
        batch_size: Batch size for all loaders
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split by time
    train_subset, val_subset, test_subset = time_based_split(
        dataset, train_ratio, val_ratio, test_ratio
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_with_metadata
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_metadata
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_metadata
    )
    
    return train_loader, val_loader, test_loader


class CallLevelDataset(Dataset):
    """
    Dataset that groups keywords by earnings call.
    
    Each sample represents an entire earnings call with all its keywords.
    """
    
    def __init__(self, base_dataset: Dataset):
        """
        Initialize from a keyword-level dataset.
        
        Args:
            base_dataset: MentionDataset instance
        """
        self.base_dataset = base_dataset
        self.calls = []  # List of (ticker, date_str, indices)
        
        self._group_by_call()
    
    def _group_by_call(self):
        """Group samples by earnings call."""
        call_map = {}  # (ticker, date_str) -> list of indices
        
        for i in range(len(self.base_dataset)):
            _, _, metadata = self.base_dataset[i]
            key = (metadata['ticker'], metadata['date_str'])
            
            if key not in call_map:
                call_map[key] = []
            call_map[key].append(i)
        
        # Sort by date
        sorted_keys = sorted(
            call_map.keys(),
            key=lambda x: self.base_dataset[call_map[x][0]][2]['earnings_date']
        )
        
        self.calls = [(k[0], k[1], call_map[k]) for k in sorted_keys]
    
    def __len__(self) -> int:
        return len(self.calls)
    
    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Get all keywords for a single earnings call.
        
        Returns:
            features: Tensor of shape (n_keywords, n_features)
            labels: Tensor of shape (n_keywords, 1)
            metadata: List of metadata dicts for each keyword
        """
        ticker, date_str, indices = self.calls[idx]
        
        features_list = []
        labels_list = []
        metadata_list = []
        
        for i in indices:
            features, label, metadata = self.base_dataset[i]
            features_list.append(features)
            labels_list.append(label)
            metadata_list.append(metadata)
        
        features = torch.stack(features_list)
        labels = torch.stack(labels_list)
        
        return features, labels, metadata_list


def create_call_level_loader(
    dataset: Dataset,
    batch_size: int = 1,  # Usually 1 since calls have variable keywords
    shuffle: bool = False,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for call-level predictions.
    
    Args:
        dataset: CallLevelDataset instance
        batch_size: Batch size (usually 1)
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    def call_collate(batch):
        """Collate function for call-level data."""
        if len(batch) == 1:
            return batch[0]
        # For batch_size > 1, return list
        return batch
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=call_collate
    )


def get_dataset_statistics(dataset: Dataset) -> Dict:
    """
    Compute statistics about the dataset.
    
    Args:
        dataset: MentionDataset instance
        
    Returns:
        Dict with various statistics
    """
    n_samples = len(dataset)
    
    if n_samples == 0:
        return {'n_samples': 0}
    
    # Collect all samples
    all_labels = []
    all_features = []
    tickers = set()
    dates = set()
    
    for i in range(n_samples):
        features, label, metadata = dataset[i]
        all_features.append(features.numpy())
        all_labels.append(label.item())
        tickers.add(metadata['ticker'])
        dates.add(metadata['date_str'])
    
    all_features = np.stack(all_features)
    all_labels = np.array(all_labels)
    
    # Compute statistics
    stats = {
        'n_samples': n_samples,
        'n_tickers': len(tickers),
        'n_earnings_calls': len(dates),
        'n_features': all_features.shape[1],
        'positive_rate': float(np.mean(all_labels)),
        'feature_means': all_features.mean(axis=0).tolist(),
        'feature_stds': all_features.std(axis=0).tolist(),
        'tickers': list(tickers),
        'dates': sorted(list(dates))
    }
    
    return stats
