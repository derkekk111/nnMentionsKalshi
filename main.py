#!/usr/bin/env python3
"""
Kalshi Mention Market Neural Network - Main Entry Point

Usage:
    python main.py train --ticker AAPL --epochs 100
    python main.py predict --ticker AAPL --date 25JUL31
    python main.py backtest --ticker AAPL
    python main.py evaluate --model checkpoints/best_model.pt
"""

import argparse
import sys
import os
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import Config, load_config, save_config, get_preset_config
from data.dataset import MentionDataset
from data.features import FeatureExtractor, FeatureConfig
from data.loaders import create_dataloaders, time_based_split
from models.mention_predictor import MentionPredictor, ModelConfig, create_model, save_model, load_model
from training.trainer import Trainer, TrainingConfig
from training.metrics import compute_all_metrics, format_metrics_report
from backtest.simulator import BacktestSimulator, BacktestConfig
from backtest.analysis import format_backtest_report
from evaluation.visualize import create_evaluation_dashboard


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> str:
    """Get the appropriate device."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_str


def train(args):
    """Train the model."""
    print("=" * 60)
    print("KALSHI MENTION PREDICTOR - TRAINING")
    print("=" * 60)
    
    # Load or create config
    if args.config:
        config = load_config(args.config)
    elif args.preset:
        config = get_preset_config(args.preset)
    else:
        config = Config()
    
    # Override with command line args
    if args.ticker:
        config.data.tickers = [t.strip().upper() for t in args.ticker.split(',')]
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    
    # Set seed and device
    set_seed(config.seed)
    device = get_device(config.device)
    
    print(f"\nConfiguration:")
    print(f"  Tickers: {config.data.tickers}")
    print(f"  Device: {device}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = MentionDataset(
        tickers=config.data.tickers,
        fetch_llm_probs=config.data.use_llm_probs,
        verbose=config.verbose
    )
    
    if len(dataset) == 0:
        print("ERROR: No data loaded. Check that data files exist.")
        return
    
    print(f"  Loaded {len(dataset)} samples")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=config.training.batch_size,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio
    )
    
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    
    # Create model
    n_features = len(dataset.get_feature_names())
    
    model_config = ModelConfig(
        n_features=n_features,
        hidden_dims=config.model.hidden_dims,
        dropout_rate=config.model.dropout_rate,
        use_batch_norm=config.model.use_batch_norm,
        include_position_head=config.model.include_position_head
    )
    
    model = MentionPredictor(model_config)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create trainer
    training_config = TrainingConfig(
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        optimizer=config.training.optimizer,
        scheduler=config.training.scheduler,
        early_stopping=config.training.early_stopping,
        early_stopping_patience=config.training.early_stopping_patience,
        save_best=config.training.save_best,
        save_path=config.training.save_path,
        log_interval=config.training.log_interval,
        verbose=config.verbose
    )
    
    trainer = Trainer(model, training_config, device=device)
    
    # Train
    history = trainer.train(train_loader, val_loader, test_loader)
    
    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    save_config(config, str(output_dir / 'config.json'))
    
    # Save history
    trainer.save_history(str(output_dir / 'training_history.json'))
    
    # Save final model
    save_model(model, str(output_dir / 'final_model.pt'))
    
    print(f"\nResults saved to {output_dir}")
    
    # Generate evaluation plots
    if args.plot:
        print("\nGenerating evaluation plots...")
        
        # Get test predictions
        probs, positions, metadata = trainer.predict(test_loader)
        labels = np.array([m['market_price'] for m in metadata])  # Placeholder
        market_prices = np.array([m['market_price'] for m in metadata])
        
        # Get actual labels from test set
        test_labels = []
        for i in range(len(test_loader.dataset)):
            if hasattr(test_loader.dataset, 'dataset'):
                # Subset
                idx = test_loader.dataset.indices[i]
                _, label, _ = test_loader.dataset.dataset[idx]
            else:
                _, label, _ = test_loader.dataset[i]
            test_labels.append(label.item())
        labels = np.array(test_labels)
        
        # Create plots
        create_evaluation_dashboard(
            y_true=labels,
            y_pred=probs,
            market_prices=market_prices,
            history=history,
            save_dir=str(output_dir / 'plots')
        )
        
        print(f"  Plots saved to {output_dir / 'plots'}")
    
    print("\nTraining complete!")


def predict(args):
    """Generate predictions for a specific date."""
    print("=" * 60)
    print("KALSHI MENTION PREDICTOR - PREDICTION")
    print("=" * 60)
    
    # Load model
    model_path = args.model or 'checkpoints/best_model.pt'
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
    
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)
    model.eval()
    
    device = get_device(args.device or 'auto')
    model = model.to(device)
    
    # Load dataset for the specified ticker and date
    ticker = args.ticker.upper()
    date_str = args.date
    
    print(f"Loading data for {ticker} - {date_str}...")
    
    dataset = MentionDataset(
        tickers=[ticker],
        verbose=True
    )
    
    if len(dataset) == 0:
        print("ERROR: No data found")
        return
    
    # Filter to target date if specified
    if date_str:
        target_samples = [
            i for i in range(len(dataset))
            if dataset.samples[i]['date_str'] == date_str
        ]
        
        if not target_samples:
            print(f"ERROR: No samples found for date {date_str}")
            print(f"Available dates: {set(s['date_str'] for s in dataset.samples)}")
            return
    else:
        target_samples = list(range(len(dataset)))
    
    # Generate predictions
    print(f"\nGenerating predictions for {len(target_samples)} keywords...")
    
    results = []
    
    for i in target_samples:
        features, label, metadata = dataset[i]
        features = features.unsqueeze(0).to(device)
        
        with torch.no_grad():
            prob, position = model(features, return_position=True)
        
        results.append({
            'keyword': metadata['keyword'],
            'model_prob': float(prob.item()),
            'position_size': float(position.item()) if position is not None else 1.0,
            'market_price': metadata['market_price'],
            'edge': float(prob.item()) - metadata['market_price'],
            'actual': label.item() if label.item() in [0, 1] else 'unknown'
        })
    
    # Sort by edge
    results.sort(key=lambda x: abs(x['edge']), reverse=True)
    
    # Print results
    print("\n" + "=" * 80)
    print(f"PREDICTIONS FOR {ticker} - {date_str or 'ALL DATES'}")
    print("=" * 80)
    print(f"{'Keyword':<30} {'Model':<8} {'Market':<8} {'Edge':<8} {'Position':<8} {'Actual'}")
    print("-" * 80)
    
    for r in results:
        direction = 'YES' if r['edge'] > 0.05 else ('NO' if r['edge'] < -0.05 else '-')
        print(f"{r['keyword']:<30} {r['model_prob']:.2%}  {r['market_price']:.2%}  "
              f"{r['edge']:+.2%}  {r['position_size']:.2f}     {r['actual']}")
    
    # Save to file
    if args.output:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


def backtest(args):
    """Run backtest simulation."""
    print("=" * 60)
    print("KALSHI MENTION PREDICTOR - BACKTEST")
    print("=" * 60)
    
    # Load model
    model_path = args.model or 'checkpoints/best_model.pt'
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
    
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)
    model.eval()
    
    device = get_device(args.device or 'auto')
    model = model.to(device)
    
    # Load dataset
    tickers = [t.strip().upper() for t in args.ticker.split(',')] if args.ticker else ['AAPL']
    
    print(f"Loading data for {tickers}...")
    dataset = MentionDataset(
        tickers=tickers,
        verbose=True
    )
    
    if len(dataset) == 0:
        print("ERROR: No data found")
        return
    
    # Generate predictions
    print(f"\nGenerating predictions for {len(dataset)} samples...")
    
    all_probs = []
    all_positions = []
    all_labels = []
    all_market_prices = []
    all_metadata = []
    
    for i in range(len(dataset)):
        features, label, metadata = dataset[i]
        features = features.unsqueeze(0).to(device)
        
        with torch.no_grad():
            prob, position = model(features, return_position=True)
        
        all_probs.append(prob.item())
        all_positions.append(position.item() if position is not None else 1.0)
        all_labels.append(label.item())
        all_market_prices.append(metadata['market_price'])
        all_metadata.append(metadata)
    
    predictions = np.array(all_probs)
    positions = np.array(all_positions)
    labels = np.array(all_labels)
    market_prices = np.array(all_market_prices)
    
    # Create backtest config
    bt_config = BacktestConfig(
        initial_bankroll=args.bankroll or 10000.0,
        max_risk_per_trade=args.max_risk or 0.05,
        edge_threshold=args.edge_threshold or 0.05,
        use_model_position=True
    )
    
    # Run backtest
    print("\nRunning backtest simulation...")
    simulator = BacktestSimulator(bt_config)
    
    results = simulator.run_backtest(
        predictions=predictions,
        positions=positions,
        labels=labels,
        market_prices=market_prices,
        metadata=all_metadata
    )
    
    # Print results
    print(format_backtest_report(results))
    
    # Generate plots
    if args.plot:
        print("\nGenerating backtest plots...")
        
        output_dir = Path(args.output_dir or 'outputs/backtest')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        trades_df = simulator.get_trades_df()
        equity_df = simulator.get_equity_df()
        
        create_evaluation_dashboard(
            y_true=labels,
            y_pred=predictions,
            market_prices=market_prices,
            trades_df=trades_df,
            equity_curve=equity_df['equity'].values if not equity_df.empty else None,
            save_dir=str(output_dir)
        )
        
        # Save trades
        trades_df.to_csv(output_dir / 'trades.csv', index=False)
        
        print(f"Results saved to {output_dir}")


def evaluate(args):
    """Evaluate model on test data."""
    print("=" * 60)
    print("KALSHI MENTION PREDICTOR - EVALUATION")
    print("=" * 60)
    
    # Load model
    model_path = args.model or 'checkpoints/best_model.pt'
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return
    
    print(f"\nLoading model from {model_path}...")
    model = load_model(model_path)
    model.eval()
    
    device = get_device(args.device or 'auto')
    model = model.to(device)
    
    # Load dataset
    tickers = [t.strip().upper() for t in args.ticker.split(',')] if args.ticker else ['AAPL']
    
    print(f"Loading data for {tickers}...")
    dataset = MentionDataset(
        tickers=tickers,
        verbose=True
    )
    
    if len(dataset) == 0:
        print("ERROR: No data found")
        return
    
    # Generate predictions
    all_probs = []
    all_labels = []
    all_market_prices = []
    
    for i in range(len(dataset)):
        features, label, metadata = dataset[i]
        features = features.unsqueeze(0).to(device)
        
        with torch.no_grad():
            prob, _ = model(features, return_position=False)
        
        all_probs.append(prob.item())
        all_labels.append(label.item())
        all_market_prices.append(metadata['market_price'])
    
    predictions = np.array(all_probs)
    labels = np.array(all_labels)
    market_prices = np.array(all_market_prices)
    
    # Compute metrics
    metrics = compute_all_metrics(labels, predictions)
    
    print(format_metrics_report(metrics))
    
    # Feature importance
    if args.importance:
        from evaluation.shap_analysis import analyze_feature_importance
        
        print("\nComputing feature importance...")
        
        features_list = []
        for i in range(len(dataset)):
            features, _, _ = dataset[i]
            features_list.append(features.numpy())
        
        X = np.stack(features_list)
        feature_names = dataset.get_feature_names()
        
        importances = analyze_feature_importance(
            model, X, labels, feature_names,
            method='permutation',
            save_dir=args.output_dir or 'outputs/evaluation'
        )
        
        print("\nFeature Importance:")
        for name, imp in sorted(importances.items(), key=lambda x: -abs(x[1])):
            print(f"  {name}: {imp:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Kalshi Mention Market Neural Network',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--ticker', '-t', help='Ticker(s) to train on (comma-separated)')
    train_parser.add_argument('--epochs', '-e', type=int, help='Number of epochs')
    train_parser.add_argument('--batch-size', '-b', type=int, help='Batch size')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    train_parser.add_argument('--config', '-c', help='Path to config file')
    train_parser.add_argument('--preset', choices=['debug', 'standard', 'production'], help='Use preset config')
    train_parser.add_argument('--plot', action='store_true', help='Generate evaluation plots')
    train_parser.add_argument('--output-dir', '-o', help='Output directory')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('--ticker', '-t', required=True, help='Ticker to predict')
    predict_parser.add_argument('--date', '-d', help='Earnings date (e.g., 25JUL31)')
    predict_parser.add_argument('--model', '-m', help='Path to model checkpoint')
    predict_parser.add_argument('--device', help='Device to use')
    predict_parser.add_argument('--output', '-o', help='Output CSV path')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest simulation')
    backtest_parser.add_argument('--ticker', '-t', help='Ticker(s) to backtest')
    backtest_parser.add_argument('--model', '-m', help='Path to model checkpoint')
    backtest_parser.add_argument('--bankroll', type=float, help='Initial bankroll')
    backtest_parser.add_argument('--max-risk', type=float, help='Max risk per trade')
    backtest_parser.add_argument('--edge-threshold', type=float, help='Minimum edge to trade')
    backtest_parser.add_argument('--plot', action='store_true', help='Generate plots')
    backtest_parser.add_argument('--output-dir', '-o', help='Output directory')
    backtest_parser.add_argument('--device', help='Device to use')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--ticker', '-t', help='Ticker(s) to evaluate')
    eval_parser.add_argument('--model', '-m', help='Path to model checkpoint')
    eval_parser.add_argument('--importance', action='store_true', help='Compute feature importance')
    eval_parser.add_argument('--output-dir', '-o', help='Output directory')
    eval_parser.add_argument('--device', help='Device to use')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict(args)
    elif args.command == 'backtest':
        backtest(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
