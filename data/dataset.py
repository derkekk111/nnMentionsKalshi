"""
MentionDataset: Unified dataset for Kalshi mention market prediction.

Combines data from:
- Defeat-Beta API (transcripts, news)
- Kalshi market data (prices, outcomes)
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
input_data_dir = project_root / 'input_data'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(input_data_dir))

from defeatbeta_api.data.ticker import Ticker


class MentionDataset(Dataset):
    """
    PyTorch Dataset for Kalshi earnings mention prediction.
    
    Each sample represents a single keyword for a specific earnings call,
    with features from historical transcripts, news, and market data.
    """
    
    def __init__(
        self,
        tickers: List[str],
        data_dir: str = None,
        use_cache: bool = True,
        fetch_llm_probs: bool = False,
        verbose: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            tickers: List of ticker symbols to include
            data_dir: Directory containing cached Kalshi data (defaults to nn_directory)
            use_cache: Whether to use cached data or fetch fresh
            fetch_llm_probs: Whether to fetch LLM probabilities (requires API access)
            verbose: Whether to print progress
        """
        self.tickers = [t.upper() for t in tickers]
        self.data_dir = Path(data_dir) if data_dir else project_root / 'nn_directory'
        self.use_cache = use_cache
        self.fetch_llm_probs = fetch_llm_probs
        self.verbose = verbose
        
        # Storage for processed samples
        self.samples: List[Dict[str, Any]] = []
        self.feature_names: List[str] = []
        
        # Load and process all data
        self._load_all_data()
        
    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def _parse_kalshi_date(self, date_str: str) -> datetime:
        """
        Parse Kalshi date format (e.g., '25JUL31') to datetime.
        
        Args:
            date_str: Date in format YYMMMDD (e.g., '25JUL31')
            
        Returns:
            datetime object
        """
        month_map = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
            'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
            'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        
        year = 2000 + int(date_str[:2])
        month = month_map[date_str[2:5]]
        day = int(date_str[5:7]) if len(date_str) > 5 else 1
        
        return datetime(year, month, day)
    
    def _load_kalshi_markets(self, ticker: str) -> Dict[str, Dict]:
        """
        Load Kalshi market data for a ticker.
        
        Returns dict mapping date_str -> {keyword -> {price, result, volume, etc}}
        """
        ticker_dir = self.data_dir / ticker
        if not ticker_dir.exists():
            self._log(f"  No data directory for {ticker}")
            return {}
        
        markets_by_date = {}
        
        for file_path in ticker_dir.glob(f'kalshi_response_{ticker}_*.json'):
            try:
                # Extract date from filename
                date_str = file_path.stem.split('_')[-1]
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                markets = data.get('markets', [])
                markets_by_date[date_str] = {}
                
                for market in markets:
                    keyword = market.get('custom_strike', {}).get('Word', '')
                    if not keyword:
                        continue
                    
                    # Extract market features
                    result_str = market.get('result', '').lower()
                    markets_by_date[date_str][keyword] = {
                        'result': 1 if result_str == 'yes' else 0,
                        'last_price': market.get('last_price', 50),
                        'open_interest': market.get('open_interest', 0),
                        'volume': market.get('volume', 0),
                        'yes_bid': market.get('yes_bid', 0),
                        'yes_ask': market.get('yes_ask', 100),
                        'close_time': market.get('close_time', ''),
                        'open_time': market.get('open_time', ''),
                    }
                    
            except Exception as e:
                self._log(f"  Error loading {file_path}: {e}")
                continue
        
        return markets_by_date
    
    def _load_transcripts(self, ticker: str) -> pd.DataFrame:
        """Load earnings call transcripts for a ticker."""
        try:
            ticker_obj = Ticker(ticker)
            transcripts = ticker_obj.earning_call_transcripts()
            df = transcripts.get_transcripts_list()
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # Standardize columns
            df['date'] = pd.to_datetime(df['report_date'])
            df['transcript'] = df['transcripts']
            df['year_quarter'] = (
                df['fiscal_year'].astype(str) + '-Q' + 
                df['fiscal_quarter'].astype(str)
            )
            
            return df.sort_values('date')
            
        except Exception as e:
            self._log(f"  Error loading transcripts for {ticker}: {e}")
            return pd.DataFrame()
    
    def _load_news(self, ticker: str, before_date: datetime, days: int = 21) -> pd.DataFrame:
        """Load news articles for a ticker in the period before an earnings call."""
        try:
            ticker_obj = Ticker(ticker)
            news_obj = ticker_obj.news()
            news_df = news_obj.get_news_list()
            
            if news_df is None or news_df.empty:
                return pd.DataFrame()
            
            # Filter by date range
            news_df['report_date'] = pd.to_datetime(news_df['report_date'])
            start_date = before_date - timedelta(days=days + 2)
            end_date = before_date - timedelta(days=2)
            
            filtered = news_df[
                (news_df['report_date'] >= start_date) & 
                (news_df['report_date'] <= end_date)
            ]
            
            return filtered.sort_values('report_date', ascending=False)
            
        except Exception as e:
            self._log(f"  Error loading news for {ticker}: {e}")
            return pd.DataFrame()
    
    def _analyze_mentions(
        self, 
        transcripts_df: pd.DataFrame, 
        keywords: List[str],
        target_date: datetime
    ) -> Dict[str, Dict]:
        """
        Analyze historical keyword mentions in transcripts.
        
        Returns dict mapping keyword -> mention statistics
        """
        from input_data.kalshi_mentions_prob import analyze_keyword_mentions
        
        if transcripts_df.empty:
            return {kw: self._empty_mention_stats() for kw in keywords}
        
        # Filter transcripts before target date
        two_years_before = target_date - timedelta(days=730)
        filtered_df = transcripts_df[
            (transcripts_df['date'] >= two_years_before) &
            (transcripts_df['date'] < target_date)
        ]
        
        if filtered_df.empty:
            return {kw: self._empty_mention_stats() for kw in keywords}
        
        try:
            mention_df, prob_df = analyze_keyword_mentions(
                filtered_df, keywords, verbose=False
            )
            
            stats = {}
            for keyword in keywords:
                kw_data = mention_df[mention_df['keyword'] == keyword]
                
                if kw_data.empty:
                    stats[keyword] = self._empty_mention_stats()
                    continue
                
                # Calculate mention statistics
                total_calls = len(kw_data)
                mentions = kw_data['mentioned'].sum()
                
                # Last 4 quarters
                last_4q = kw_data.tail(4)
                mentions_last_4q = last_4q['mentioned'].sum() if len(last_4q) > 0 else 0
                
                # Recency: days since last mention
                mentioned_dates = kw_data[kw_data['mentioned']]['date']
                if len(mentioned_dates) > 0:
                    last_mention = pd.to_datetime(mentioned_dates.max())
                    recency = (target_date - last_mention).days
                else:
                    recency = 730  # 2 years (never mentioned)
                
                # Trend: slope of mention frequency over time
                if len(kw_data) >= 4:
                    kw_data = kw_data.copy()
                    kw_data['time_idx'] = range(len(kw_data))
                    mentioned_vals = kw_data['mentioned'].astype(float).values
                    time_vals = kw_data['time_idx'].values
                    if np.std(mentioned_vals) > 0:
                        trend = np.corrcoef(time_vals, mentioned_vals)[0, 1]
                    else:
                        trend = 0.0
                else:
                    trend = 0.0
                
                stats[keyword] = {
                    'mention_rate_2y': mentions / total_calls if total_calls > 0 else 0.0,
                    'mention_count_last_4q': int(mentions_last_4q),
                    'mention_recency': float(recency),
                    'mention_trend': float(trend) if not np.isnan(trend) else 0.0,
                    'total_mentions': int(mentions),
                    'total_calls': int(total_calls)
                }
            
            return stats
            
        except Exception as e:
            self._log(f"  Error analyzing mentions: {e}")
            return {kw: self._empty_mention_stats() for kw in keywords}
    
    def _empty_mention_stats(self) -> Dict:
        """Return empty mention statistics."""
        return {
            'mention_rate_2y': 0.0,
            'mention_count_last_4q': 0,
            'mention_recency': 730.0,
            'mention_trend': 0.0,
            'total_mentions': 0,
            'total_calls': 0
        }
    
    def _extract_news_features(
        self, 
        news_df: pd.DataFrame, 
        keyword: str
    ) -> Dict[str, float]:
        """Extract news-related features for a keyword."""
        if news_df.empty:
            return {
                'news_keyword_mentions': 0,
                'news_volume': 0,
                'news_sentiment': 0.0
            }
        
        # Count keyword mentions in news
        keyword_lower = keyword.lower()
        keyword_mentions = 0
        
        for _, article in news_df.iterrows():
            content = str(article.get('news', '')) + ' ' + str(article.get('title', ''))
            if keyword_lower in content.lower():
                keyword_mentions += 1
        
        return {
            'news_keyword_mentions': keyword_mentions,
            'news_volume': len(news_df),
            'news_sentiment': 0.0  # Placeholder - could add sentiment model
        }
    
    def _get_llm_probabilities(
        self,
        ticker: str,
        keywords: List[str],
        transcripts_df: pd.DataFrame,
        target_date: datetime
    ) -> Dict[str, float]:
        """Get LLM probability estimates for keywords."""
        if not self.fetch_llm_probs:
            # Return default values if not fetching
            return {kw: 50.0 for kw in keywords}
        
        try:
            from input_data.LLM_Likelihood_DefeatBeta import get_keywords_probabilities_batch
            
            probs = get_keywords_probabilities_batch(
                keywords=keywords,
                ticker=ticker,
                transcripts_df=transcripts_df,
                context_date=target_date.strftime('%Y-%m-%d')
            )
            
            if probs is None:
                return {kw: 50.0 for kw in keywords}
            
            # Ensure all keywords have a probability
            result = {}
            for kw in keywords:
                result[kw] = float(probs.get(kw, 50.0))
            
            return result
            
        except Exception as e:
            self._log(f"  Error getting LLM probabilities: {e}")
            return {kw: 50.0 for kw in keywords}
    
    def _load_all_data(self):
        """Load and process data for all tickers."""
        self._log("Loading data for Kalshi mention prediction...")
        
        for ticker in self.tickers:
            self._log(f"\nProcessing {ticker}...")
            
            # Load Kalshi market data
            markets_by_date = self._load_kalshi_markets(ticker)
            if not markets_by_date:
                self._log(f"  No Kalshi data found for {ticker}")
                continue
            
            # Load transcripts
            transcripts_df = self._load_transcripts(ticker)
            
            # Process each earnings date
            for date_str, keywords_data in markets_by_date.items():
                try:
                    target_date = self._parse_kalshi_date(date_str)
                except Exception as e:
                    self._log(f"  Error parsing date {date_str}: {e}")
                    continue
                
                keywords = list(keywords_data.keys())
                self._log(f"  Processing {date_str} with {len(keywords)} keywords")
                
                # Get historical mention statistics
                mention_stats = self._analyze_mentions(
                    transcripts_df, keywords, target_date
                )
                
                # Load news for this period
                news_df = self._load_news(ticker, target_date)
                
                # Get LLM probabilities (batch)
                llm_probs = self._get_llm_probabilities(
                    ticker, keywords, transcripts_df, target_date
                )
                
                # Create samples for each keyword
                for keyword, market_data in keywords_data.items():
                    # Get mention stats
                    kw_mention_stats = mention_stats.get(
                        keyword, self._empty_mention_stats()
                    )
                    
                    # Get news features
                    news_features = self._extract_news_features(news_df, keyword)
                    
                    # Get LLM probability
                    llm_prob = llm_probs.get(keyword, 50.0)
                    
                    # Calculate days to expiry (from market open)
                    days_to_expiry = 4.0  # Default assumption
                    
                    # Build sample
                    sample = {
                        # Identifiers
                        'keyword': keyword,
                        'ticker': ticker,
                        'earnings_date': target_date,
                        'date_str': date_str,
                        
                        # Historical mention features
                        'mention_rate_2y': kw_mention_stats['mention_rate_2y'],
                        'mention_count_last_4q': kw_mention_stats['mention_count_last_4q'],
                        'mention_recency': kw_mention_stats['mention_recency'],
                        'mention_trend': kw_mention_stats['mention_trend'],
                        
                        # LLM probability
                        'llm_prob': llm_prob / 100.0,  # Normalize to 0-1
                        
                        # Market features
                        'market_price': market_data['last_price'] / 100.0,  # Normalize
                        'open_interest': market_data['open_interest'],
                        'volume': market_data['volume'],
                        'days_to_expiry': days_to_expiry,
                        
                        # News features
                        'news_keyword_mentions': news_features['news_keyword_mentions'],
                        'news_volume': news_features['news_volume'],
                        'news_sentiment': news_features['news_sentiment'],
                        
                        # Keyword characteristics
                        'keyword_length': len(keyword),
                        'keyword_word_count': len(keyword.split()),
                        
                        # Label
                        'mentioned': market_data['result']
                    }
                    
                    self.samples.append(sample)
        
        # Define feature names (order matters for tensor creation)
        self.feature_names = [
            'mention_rate_2y', 'mention_count_last_4q', 'mention_recency', 
            'mention_trend', 'llm_prob',
            'market_price', 'open_interest', 'volume', 'days_to_expiry',
            'news_keyword_mentions', 'news_volume', 'news_sentiment',
            'keyword_length', 'keyword_word_count'
        ]
        
        self._log(f"\nLoaded {len(self.samples)} samples total")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a single sample.
        
        Returns:
            features: Tensor of shape (n_features,)
            label: Tensor of shape (1,) with binary label
            metadata: Dict with identifiers and raw values
        """
        sample = self.samples[idx]
        
        # Extract features in order
        features = []
        for name in self.feature_names:
            value = sample.get(name, 0.0)
            features.append(float(value))
        
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor([sample['mentioned']], dtype=torch.float32)
        
        metadata = {
            'keyword': sample['keyword'],
            'ticker': sample['ticker'],
            'earnings_date': sample['earnings_date'],
            'date_str': sample['date_str'],
            'market_price': sample['market_price']
        }
        
        return features_tensor, label_tensor, metadata
    
    def get_feature_names(self) -> List[str]:
        """Get the names of features in order."""
        return self.feature_names.copy()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame for analysis."""
        return pd.DataFrame(self.samples)
    
    def get_call_level_data(self) -> pd.DataFrame:
        """
        Aggregate samples to earnings call level.
        
        Returns DataFrame with one row per (ticker, earnings_date) combination.
        """
        df = self.to_dataframe()
        
        if df.empty:
            return df
        
        # Group by ticker and earnings date
        grouped = df.groupby(['ticker', 'date_str']).agg({
            'mentioned': ['sum', 'count', 'mean'],
            'llm_prob': 'mean',
            'market_price': 'mean',
            'mention_rate_2y': 'mean',
            'keyword': list
        }).reset_index()
        
        # Flatten column names
        grouped.columns = [
            'ticker', 'date_str', 
            'total_mentions', 'keyword_count', 'mention_rate',
            'avg_llm_prob', 'avg_market_price', 'avg_mention_rate_2y', 'keywords'
        ]
        
        return grouped


def load_cached_dataset(
    cache_path: str,
    tickers: Optional[List[str]] = None
) -> MentionDataset:
    """
    Load a cached dataset from disk.
    
    Args:
        cache_path: Path to cached dataset file
        tickers: Optional list of tickers to filter
        
    Returns:
        MentionDataset instance
    """
    import pickle
    
    with open(cache_path, 'rb') as f:
        dataset = pickle.load(f)
    
    if tickers:
        tickers_upper = [t.upper() for t in tickers]
        dataset.samples = [
            s for s in dataset.samples 
            if s['ticker'] in tickers_upper
        ]
    
    return dataset


def save_dataset_cache(dataset: MentionDataset, cache_path: str):
    """Save dataset to disk for faster loading."""
    import pickle
    
    with open(cache_path, 'wb') as f:
        pickle.dump(dataset, f)
