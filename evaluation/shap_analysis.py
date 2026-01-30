"""
Feature importance analysis using SHAP and gradient-based attribution.

Provides insights into which features matter most for predictions.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import warnings


def compute_gradient_attribution(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int = 1
) -> np.ndarray:
    """
    Compute gradient-based feature attribution.
    
    Uses the gradient of the output with respect to inputs to determine
    feature importance.
    
    Args:
        model: PyTorch model
        inputs: Input tensor (batch_size, n_features)
        target_class: Class to compute gradients for
        
    Returns:
        Attribution scores (batch_size, n_features)
    """
    model.eval()
    inputs = inputs.clone().requires_grad_(True)
    
    # Forward pass
    outputs, _ = model(inputs, return_position=False)
    
    # Backward pass
    model.zero_grad()
    outputs.sum().backward()
    
    # Get gradients - move to CPU for numpy conversion
    gradients = inputs.grad.detach().cpu().numpy()
    
    # Attribution = gradient * input (Gradient x Input method)
    attributions = gradients * inputs.detach().cpu().numpy()
    
    return attributions


def compute_integrated_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    n_steps: int = 50
) -> np.ndarray:
    """
    Compute Integrated Gradients attribution.
    
    More robust than simple gradients by integrating along a path
    from baseline to input.
    
    Args:
        model: PyTorch model
        inputs: Input tensor
        baseline: Baseline input (defaults to zeros)
        n_steps: Number of integration steps
        
    Returns:
        Attribution scores
    """
    model.eval()
    
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    
    # Scale inputs
    scaled_inputs = [
        baseline + (float(i) / n_steps) * (inputs - baseline)
        for i in range(n_steps + 1)
    ]
    
    gradients = []
    
    for scaled_input in scaled_inputs:
        scaled_input = scaled_input.clone().requires_grad_(True)
        
        outputs, _ = model(scaled_input, return_position=False)
        
        model.zero_grad()
        outputs.sum().backward()
        
        gradients.append(scaled_input.grad.detach().cpu().numpy())
    
    # Average gradients
    avg_gradients = np.mean(gradients, axis=0)
    
    # Integrated gradients - move to CPU for numpy conversion
    integrated_grads = (inputs.detach().cpu().numpy() - baseline.cpu().numpy()) * avg_gradients
    
    return integrated_grads


def compute_permutation_importance(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    metric: str = 'auc'
) -> Dict[str, float]:
    """
    Compute permutation importance.
    
    Measures feature importance by shuffling each feature and measuring
    the decrease in model performance.
    
    Args:
        model: PyTorch model
        X: Feature matrix
        y: True labels
        feature_names: Names of features
        n_repeats: Number of permutation repeats
        metric: Metric to use ('auc', 'accuracy', 'log_loss')
        
    Returns:
        Dict mapping feature name to importance score
    """
    from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
    
    model.eval()
    
    # Get model device and move tensors there
    device = next(model.parameters()).device
    X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
    
    with torch.no_grad():
        baseline_preds, _ = model(X_tensor, return_position=False)
        baseline_preds = baseline_preds.cpu().numpy().flatten()
    
    # Baseline score
    if metric == 'auc':
        baseline_score = roc_auc_score(y, baseline_preds)
    elif metric == 'accuracy':
        baseline_score = accuracy_score(y, (baseline_preds >= 0.5).astype(int))
    else:  # log_loss
        baseline_score = -log_loss(y, baseline_preds)  # Negative so higher is better
    
    importances = {}
    
    for i, feature_name in enumerate(feature_names):
        scores = []
        
        for _ in range(n_repeats):
            # Shuffle feature
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Get predictions - move tensor to same device as model
            X_tensor = torch.from_numpy(X_permuted.astype(np.float32)).to(device)
            with torch.no_grad():
                preds, _ = model(X_tensor, return_position=False)
                preds = preds.cpu().numpy().flatten()
            
            # Compute score
            if metric == 'auc':
                score = roc_auc_score(y, preds)
            elif metric == 'accuracy':
                score = accuracy_score(y, (preds >= 0.5).astype(int))
            else:
                score = -log_loss(y, preds)
            
            scores.append(baseline_score - score)
        
        importances[feature_name] = np.mean(scores)
    
    return importances


def try_compute_shap_values(
    model: torch.nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    n_background: int = 100
) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Try to compute SHAP values if shap is installed.
    
    Args:
        model: PyTorch model
        X_train: Training data for background
        X_test: Test data to explain
        feature_names: Feature names
        n_background: Number of background samples
        
    Returns:
        Tuple of (shap_values, feature_names) or None if shap not available
    """
    try:
        import shap
    except ImportError:
        warnings.warn("SHAP not installed. Install with: pip install shap")
        return None
    
    # Create a wrapper for the model
    def model_predict(x):
        model.eval()
        x_tensor = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            preds, _ = model(x_tensor, return_position=False)
        return preds.numpy()
    
    # Sample background
    if len(X_train) > n_background:
        idx = np.random.choice(len(X_train), n_background, replace=False)
        background = X_train[idx]
    else:
        background = X_train
    
    # Create explainer
    explainer = shap.KernelExplainer(model_predict, background)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_test)
    
    return shap_values, feature_names


def plot_feature_importance(
    importances: Dict[str, float],
    title: str = 'Feature Importance',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance bar chart.
    
    Args:
        importances: Dict mapping feature name to importance
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    # Sort by importance
    sorted_features = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['#27AE60' if v >= 0 else '#E74C3C' for v in values]
    
    ax.barh(range(len(names)), values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Importance')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def analyze_feature_importance(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    method: str = 'permutation',
    save_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Analyze and visualize feature importance.
    
    Args:
        model: PyTorch model
        X: Feature matrix
        y: True labels
        feature_names: Feature names
        method: 'permutation', 'gradient', or 'integrated_gradients'
        save_dir: Directory to save plots
        
    Returns:
        Dict mapping feature name to importance
    """
    import os
    
    if method == 'permutation':
        importances = compute_permutation_importance(
            model, X, y, feature_names
        )
    elif method == 'gradient':
        X_tensor = torch.from_numpy(X.astype(np.float32))
        attributions = compute_gradient_attribution(model, X_tensor)
        # Average absolute attribution per feature
        importances = {
            name: float(np.mean(np.abs(attributions[:, i])))
            for i, name in enumerate(feature_names)
        }
    elif method == 'integrated_gradients':
        X_tensor = torch.from_numpy(X.astype(np.float32))
        attributions = compute_integrated_gradients(model, X_tensor)
        importances = {
            name: float(np.mean(np.abs(attributions[:, i])))
            for i, name in enumerate(feature_names)
        }
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Plot
    fig = plot_feature_importance(
        importances,
        title=f'Feature Importance ({method.replace("_", " ").title()})'
    )
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            os.path.join(save_dir, f'feature_importance_{method}.png'),
            dpi=150, bbox_inches='tight'
        )
    
    plt.close(fig)
    
    return importances
