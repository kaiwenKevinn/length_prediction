"""
Evaluation and Visualization for TRAIL Length Prediction
Based on: DON'T STOP ME NOW: EMBEDDING BASED SCHEDULING FOR LLMS (ICLR 2025)

Generates:
1. Figure 2: MAE for length prediction using embeddings vs. layer
2. Figure 3: MAE comparison (BERT input vs. embedding with/without refinement)
3. Figure 4: Heatmap comparing ground truth vs predicted lengths
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import json

from config import get_default_config, PredictorConfig
from model import LengthPredictor, compute_metrics, load_model
from bayesian_smoothing import BayesianSmoother, OnlineLengthPredictor
from data_utils import (
    AlpacaDataLoader,
    LLaMAEmbeddingExtractor,
    EmbeddingDataset,
    create_dataloaders
)


def plot_layer_mae(
    results: Dict[int, Dict],
    save_path: Optional[str] = None,
    title: str = "Prediction w/ per-layer embeddings"
):
    """
    Plot MAE for each layer (Figure 2 in paper).
    
    Args:
        results: Dict mapping layer_idx to results dict with 'test_mae'
        save_path: Path to save figure
        title: Plot title
    """
    layers = sorted(results.keys())
    maes = [results[layer]['test_mae'] for layer in layers]
    
    plt.figure(figsize=(12, 6))
    plt.plot(layers, maes, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Layer index', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Highlight best layers (10-15)
    for i, layer in enumerate(layers):
        if 10 <= layer <= 15:
            plt.scatter([layer], [maes[i]], c='red', s=100, zorder=5)
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_mae_comparison(
    bert_mae: float,
    input_maes: Dict[str, float],
    refined_maes: Dict[str, float],
    layers: List[str],
    save_path: Optional[str] = None
):
    """
    Plot MAE comparison between BERT, input embedding, and refined predictions (Figure 3).
    
    Args:
        bert_mae: MAE for BERT predictions
        input_maes: Dict mapping layer name to input embedding MAE
        refined_maes: Dict mapping layer name to refined MAE
        layers: List of layer names (e.g., ['L10', 'L11', ...])
        save_path: Path to save figure
    """
    x = np.arange(len(layers))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    input_values = [input_maes[l] for l in layers]
    refined_values = [refined_maes[l] for l in layers]
    
    bars1 = ax.bar(x - width/2, input_values, width, label='Input', color='steelblue')
    bars2 = ax.bar(x + width/2, refined_values, width, label='Refined', color='darkorange')
    
    # Add BERT baseline
    ax.axhline(y=bert_mae, color='red', linestyle='--', linewidth=2, label='BERT')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax.set_title('MAE Comparison: BERT vs. Embedding-based Predictions', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_prediction_heatmap(
    ground_truth_bins: List[int],
    predicted_bins: List[int],
    num_bins: int = 10,
    save_path: Optional[str] = None,
    title: str = "Prediction Heatmap"
):
    """
    Plot heatmap comparing ground truth vs predicted length bins (Figure 4).
    
    Args:
        ground_truth_bins: List of ground truth bin indices
        predicted_bins: List of predicted bin indices
        num_bins: Number of bins (default: 10)
        save_path: Path to save figure
        title: Plot title
    """
    # Create confusion matrix
    confusion = np.zeros((num_bins, num_bins))
    for gt, pred in zip(ground_truth_bins, predicted_bins):
        confusion[pred, gt] += 1
    
    # Apply log scale (add 1 to avoid log(0))
    log_confusion = np.log1p(confusion)
    
    # Create bin labels
    bin_labels = [f'b{i+1}' for i in range(num_bins)]
    
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    ax = sns.heatmap(
        log_confusion,
        xticklabels=bin_labels,
        yticklabels=bin_labels,
        cmap='YlOrRd',
        annot=False,
        cbar_kws={'label': 'Log Count'}
    )
    
    plt.xlabel('Groundtruth Remaining Length', fontsize=12)
    plt.ylabel('Predicted Remaining Length', fontsize=12)
    plt.title(title, fontsize=14)
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def evaluate_with_smoothing(
    model: LengthPredictor,
    dataloader: torch.utils.data.DataLoader,
    smoother: BayesianSmoother,
    device: str = 'cpu'
) -> Tuple[float, float]:
    """
    Evaluate predictions with Bayesian smoothing.
    
    Args:
        model: Trained LengthPredictor
        dataloader: Test dataloader
        smoother: BayesianSmoother instance
        device: Device to use
        
    Returns:
        Tuple of (raw MAE, smoothed MAE)
    """
    model.eval()
    
    # Group samples by sequence
    sequences = {}
    for batch in dataloader:
        embeddings = batch['embedding']
        positions = batch['position']
        remaining = batch['remaining_length']
        
        for i in range(len(embeddings)):
            # Assuming sample_id is used to group sequences
            pos = positions[i].item()
            seq_id = remaining[i].item() + pos  # Approximate sequence grouping
            
            if seq_id not in sequences:
                sequences[seq_id] = []
            
            sequences[seq_id].append({
                'embedding': embeddings[i],
                'position': pos,
                'remaining': remaining[i].item()
            })
    
    raw_errors = []
    smoothed_errors = []
    
    with torch.no_grad():
        for seq_id, samples in sequences.items():
            # Sort by position
            samples = sorted(samples, key=lambda x: x['position'])
            
            state = None
            for sample in samples:
                embedding = sample['embedding'].unsqueeze(0).to(device)
                remaining = sample['remaining']
                
                # Raw prediction
                output = model(embedding)
                raw_pred = output['expected_length'].item()
                raw_errors.append(abs(raw_pred - remaining))
                
                # Smoothed prediction
                probs = output['probs'].squeeze(0).cpu()
                
                if state is None:
                    state = smoother.initialize(probs)
                else:
                    state, _ = smoother.update(state, probs)
                
                smoothed_pred = smoother.get_expected_length(state).item()
                smoothed_errors.append(abs(smoothed_pred - remaining))
    
    raw_mae = np.mean(raw_errors)
    smoothed_mae = np.mean(smoothed_errors)
    
    return raw_mae, smoothed_mae


def collect_predictions(
    model: LengthPredictor,
    dataloader: torch.utils.data.DataLoader,
    predictor_config: PredictorConfig,
    device: str = 'cpu'
) -> Tuple[List[int], List[int], List[float]]:
    """
    Collect predictions for visualization.
    
    Returns:
        Tuple of (ground_truth_bins, predicted_bins, remaining_lengths)
    """
    model.eval()
    
    ground_truth_bins = []
    predicted_bins = []
    remaining_lengths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting predictions"):
            embeddings = batch['embedding'].to(device)
            labels = batch['bin_label']
            remaining = batch['remaining_length']
            
            output = model(embeddings)
            pred_bins = output['logits'].argmax(dim=-1).cpu()
            
            ground_truth_bins.extend(labels.tolist())
            predicted_bins.extend(pred_bins.tolist())
            remaining_lengths.extend(remaining.tolist())
    
    return ground_truth_bins, predicted_bins, remaining_lengths


def full_evaluation(
    model_path: str,
    data: List[Dict],
    data_loader: AlpacaDataLoader,
    model_config,
    predictor_config: PredictorConfig,
    data_config,
    target_layer: int = 11,
    results_dir: str = "./results"
):
    """
    Run full evaluation and generate all figures.
    
    Args:
        model_path: Path to trained model
        data: Dataset samples
        data_loader: AlpacaDataLoader instance
        model_config: Model configuration
        predictor_config: Predictor configuration
        data_config: Data configuration
        target_layer: Target layer (default: 11)
        results_dir: Directory to save results
    """
    device = model_config.device
    
    # Load model
    model = load_model(model_path, predictor_config, device)
    
    # Extract embeddings
    extractor = LLaMAEmbeddingExtractor(model_config, predictor_config)
    embeddings_path = data_config.embeddings_cache_dir
    
    # Try to load cached embeddings
    if os.path.exists(os.path.join(embeddings_path, f"layer_{target_layer}_embeddings.pt")):
        embeddings = extractor.load_embeddings(embeddings_path, [target_layer])
    else:
        extractor.load_model()
        embeddings = extractor.extract_all_embeddings(
            data=data,
            layer_indices=[target_layer],
            data_loader=data_loader,
            save_dir=embeddings_path
        )
    
    # Create test dataloader
    _, _, test_loader = create_dataloaders(
        embeddings[target_layer],
        predictor_config,
        data_config
    )
    
    # Basic metrics
    metrics = compute_metrics(model, test_loader, device)
    print(f"\n=== Evaluation Results ===")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Loss: {metrics['loss']:.4f}")
    
    # Evaluate with smoothing
    smoother = BayesianSmoother(predictor_config)
    raw_mae, smoothed_mae = evaluate_with_smoothing(model, test_loader, smoother, device)
    
    print(f"\n=== With Bayesian Smoothing ===")
    print(f"Raw MAE: {raw_mae:.2f}")
    print(f"Smoothed MAE: {smoothed_mae:.2f}")
    print(f"Improvement: {(1 - smoothed_mae/raw_mae)*100:.1f}%")
    
    # Collect predictions for heatmap
    gt_bins, pred_bins, remaining = collect_predictions(
        model, test_loader, predictor_config, device
    )
    
    # Create results directory
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate heatmap
    plot_prediction_heatmap(
        gt_bins, pred_bins,
        num_bins=predictor_config.num_bins,
        save_path=os.path.join(results_dir, "prediction_heatmap.png"),
        title=f"Embedding-based Predictions (Layer {target_layer})"
    )
    
    # Save results
    results = {
        'mae': metrics['mae'],
        'accuracy': metrics['accuracy'],
        'loss': metrics['loss'],
        'raw_mae': raw_mae,
        'smoothed_mae': smoothed_mae,
        'improvement_percent': (1 - smoothed_mae/raw_mae)*100
    }
    
    with open(os.path.join(results_dir, "evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def plot_training_curves(history: Dict, save_path: Optional[str] = None):
    """
    Plot training curves (loss and MAE).
    
    Args:
        history: Training history dict
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history['train_mae'], label='Train')
    axes[1].plot(history['val_mae'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[2].plot(history['learning_rate'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule (Cosine Annealing)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def compute_output_length_distribution(data: List[Dict], data_loader: AlpacaDataLoader) -> Dict:
    """
    Compute output length distribution for dataset analysis (Figure 9 in paper).
    
    Args:
        data: Dataset samples
        data_loader: AlpacaDataLoader instance
        
    Returns:
        Dict with distribution statistics
    """
    from transformers import AutoTokenizer
    
    configs = get_default_config()
    tokenizer = AutoTokenizer.from_pretrained(configs["model"].model_name)
    
    output_lengths = []
    for sample in tqdm(data, desc="Computing output lengths"):
        output_text = sample["output"]
        if output_text:
            tokens = tokenizer.encode(output_text, add_special_tokens=False)
            output_lengths.append(len(tokens))
    
    return {
        'lengths': output_lengths,
        'mean': np.mean(output_lengths),
        'median': np.median(output_lengths),
        'std': np.std(output_lengths),
        'min': np.min(output_lengths),
        'max': np.max(output_lengths)
    }


def plot_output_length_cdf(lengths: List[int], save_path: Optional[str] = None):
    """
    Plot CDF of output lengths (Figure 9 in paper).
    
    Args:
        lengths: List of output lengths
        save_path: Path to save figure
    """
    sorted_lengths = np.sort(lengths)
    cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_lengths, cdf, linewidth=2)
    plt.xlabel('Output length', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    plt.title('Cumulative Distribution Function (CDF) of output lengths', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate TRAIL Length Predictor")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to evaluate on")
    parser.add_argument("--target_layer", type=int, default=11,
                       help="Target layer for evaluation")
    parser.add_argument("--results_dir", type=str, default="./results",
                       help="Directory to save results")
    parser.add_argument("--plot_history", type=str, default=None,
                       help="Path to training history JSON for plotting")
    
    args = parser.parse_args()
    
    # Load configs
    configs = get_default_config()
    model_config = configs["model"]
    predictor_config = configs["predictor"]
    data_config = configs["data"]
    
    # Plot training history if provided
    if args.plot_history:
        with open(args.plot_history, 'r') as f:
            history = json.load(f)
        plot_training_curves(
            history,
            save_path=os.path.join(args.results_dir, "training_curves.png")
        )
        return
    
    # Load data
    data_loader = AlpacaDataLoader(data_config)
    data = data_loader.load_dataset(num_samples=args.num_samples)
    
    # Run evaluation
    results = full_evaluation(
        model_path=args.model_path,
        data=data,
        data_loader=data_loader,
        model_config=model_config,
        predictor_config=predictor_config,
        data_config=data_config,
        target_layer=args.target_layer,
        results_dir=args.results_dir
    )
    
    print("\n=== Final Results ===")
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()


# ============================================================================
# Enhanced Evaluation with Entropy Features
# ============================================================================

def plot_ablation_results(
    results: Dict[str, Dict],
    save_path: Optional[str] = None
):
    """
    Plot ablation study results comparing different feature combinations.
    
    Args:
        results: Dict from run_ablation_study
        save_path: Path to save figure
    """
    models = list(results.keys())
    mae_values = [results[m]['mae'] for m in models]
    rmse_values = [results[m]['rmse'] for m in models]
    accuracy_values = [results[m]['accuracy'] for m in models]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MAE comparison
    bars1 = axes[0].bar(models, mae_values, color=['steelblue', 'darkorange', 'green'])
    axes[0].set_ylabel('MAE', fontsize=12)
    axes[0].set_title('Mean Absolute Error', fontsize=14)
    axes[0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, mae_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # RMSE comparison
    bars2 = axes[1].bar(models, rmse_values, color=['steelblue', 'darkorange', 'green'])
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('Root Mean Square Error', fontsize=14)
    axes[1].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, rmse_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Accuracy comparison
    bars3 = axes[2].bar(models, accuracy_values, color=['steelblue', 'darkorange', 'green'])
    axes[2].set_ylabel('Accuracy', fontsize=12)
    axes[2].set_title('Top-1 Accuracy', fontsize=14)
    axes[2].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars3, accuracy_values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_k_analysis_results(
    results: Dict[int, Dict],
    save_path: Optional[str] = None
):
    """
    Plot K-value analysis results.
    
    Args:
        results: Dict from analyze_k_values
        save_path: Path to save figure
    """
    k_values = sorted([k for k in results.keys() if isinstance(k, int)])
    mae_values = [results[k]['mae'] for k in k_values]
    rmse_values = [results[k]['rmse'] for k in k_values]
    accuracy_values = [results[k]['accuracy'] for k in k_values]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # MAE vs K
    axes[0].plot(k_values, mae_values, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('K (Number of Initial Tokens)', fontsize=12)
    axes[0].set_ylabel('MAE', fontsize=12)
    axes[0].set_title('MAE vs K', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Find optimal K
    best_k_idx = np.argmin(mae_values)
    axes[0].scatter([k_values[best_k_idx]], [mae_values[best_k_idx]], 
                   c='red', s=150, zorder=5, label=f'Best K={k_values[best_k_idx]}')
    axes[0].legend()
    
    # RMSE vs K
    axes[1].plot(k_values, rmse_values, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('K (Number of Initial Tokens)', fontsize=12)
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('RMSE vs K', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Accuracy vs K
    axes[2].plot(k_values, accuracy_values, 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('K (Number of Initial Tokens)', fontsize=12)
    axes[2].set_ylabel('Accuracy', fontsize=12)
    axes[2].set_title('Accuracy vs K', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_entropy_feature_importance(
    model,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot feature importance based on model weights.
    
    Args:
        model: Trained EnhancedLengthPredictor
        feature_names: Names of entropy features
        save_path: Path to save figure
    """
    if feature_names is None:
        feature_names = [
            'attn_mean', 'attn_std', 'attn_max', 'attn_min', 'attn_var',
            'attn_skew', 'attn_kurt', 'attn_trend', 'attn_vol', 'attn_peaks',
            'logits_mean', 'logits_std', 'logits_max', 'logits_min', 'logits_var',
            'logits_skew', 'logits_kurt', 'logits_trend', 'logits_vol', 'logits_peaks',
            'corr', 'ratio', 'diff', 'product'
        ]
    
    # Get first layer weights of entropy encoder
    if hasattr(model, 'entropy_encoder') and model.entropy_encoder is not None:
        weights = model.entropy_encoder.fc1.weight.data.abs().mean(dim=0).cpu().numpy()
        
        # Normalize
        weights = weights / weights.sum()
        
        # Sort by importance
        sorted_idx = np.argsort(weights)[::-1]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_weights = weights[sorted_idx]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(sorted_weights)), sorted_weights, color='steelblue')
        plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Importance (Normalized Weight)', fontsize=12)
        plt.title('Entropy Feature Importance', fontsize=14)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Highlight top 5
        for i in range(min(5, len(bars))):
            bars[i].set_color('darkorange')
        
        plt.tight_layout()
        
        if save_path:
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    else:
        print("Model does not have entropy encoder")


def plot_prediction_error_distribution(
    predictions: List[float],
    targets: List[float],
    save_path: Optional[str] = None
):
    """
    Plot distribution of prediction errors.
    
    Args:
        predictions: List of predicted lengths
        targets: List of ground truth lengths
        save_path: Path to save figure
    """
    errors = np.array(predictions) - np.array(targets)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Error histogram
    axes[0].hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].axvline(x=np.mean(errors), color='green', linestyle='-', linewidth=2, 
                   label=f'Mean Error: {np.mean(errors):.2f}')
    axes[0].set_xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Prediction Error Distribution', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot: Predicted vs Actual
    axes[1].scatter(targets, predictions, alpha=0.5, s=20)
    max_val = max(max(targets), max(predictions))
    axes[1].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Length', fontsize=12)
    axes[1].set_ylabel('Predicted Length', fontsize=12)
    axes[1].set_title('Predicted vs Actual Length', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\nError Statistics:")
    print(f"  Mean Error: {np.mean(errors):.2f}")
    print(f"  Std Error: {np.std(errors):.2f}")
    print(f"  MAE: {np.mean(np.abs(errors)):.2f}")
    print(f"  RMSE: {np.sqrt(np.mean(errors**2)):.2f}")
    print(f"  Median Error: {np.median(errors):.2f}")
    print(f"  95th Percentile Error: {np.percentile(np.abs(errors), 95):.2f}")


def compute_statistical_significance(
    baseline_errors: List[float],
    enhanced_errors: List[float],
    alpha: float = 0.05
) -> Dict[str, any]:
    """
    Compute statistical significance of improvement using paired t-test.
    
    Args:
        baseline_errors: Absolute errors from baseline model
        enhanced_errors: Absolute errors from enhanced model
        alpha: Significance level
        
    Returns:
        Dict with t-statistic, p-value, and whether improvement is significant
    """
    from scipy import stats
    
    baseline_errors = np.array(baseline_errors)
    enhanced_errors = np.array(enhanced_errors)
    
    # Paired t-test (one-sided: enhanced < baseline)
    t_stat, p_value = stats.ttest_rel(baseline_errors, enhanced_errors)
    p_value_onesided = p_value / 2 if t_stat > 0 else 1 - p_value / 2
    
    # Effect size (Cohen's d)
    diff = baseline_errors - enhanced_errors
    cohens_d = np.mean(diff) / np.std(diff)
    
    # Bootstrap confidence interval
    n_bootstrap = 1000
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(diff), size=len(diff), replace=True)
        bootstrap_diffs.append(np.mean(diff[idx]))
    
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    is_significant = p_value_onesided < alpha
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'p_value_onesided': float(p_value_onesided),
        'cohens_d': float(cohens_d),
        'mean_improvement': float(np.mean(diff)),
        'ci_95': (float(ci_lower), float(ci_upper)),
        'is_significant': bool(is_significant),
        'alpha': alpha
    }


def plot_comparison_with_baseline(
    baseline_results: Dict,
    enhanced_results: Dict,
    save_path: Optional[str] = None
):
    """
    Plot comparison between baseline (embedding only) and enhanced model.
    
    Args:
        baseline_results: Results dict from baseline model
        enhanced_results: Results dict from enhanced model
        save_path: Path to save figure
    """
    metrics = ['mae', 'rmse', 'accuracy', 'top3_accuracy']
    metric_labels = ['MAE', 'RMSE', 'Accuracy', 'Top-3 Accuracy']
    
    baseline_vals = [baseline_results.get(m, 0) for m in metrics]
    enhanced_vals = [enhanced_results.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (Embedding Only)', 
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, enhanced_vals, width, label='Enhanced (+ Entropy)', 
                   color='darkorange', alpha=0.8)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Baseline vs Enhanced Model Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    # Print improvement
    print("\nImprovement Summary:")
    for metric, label in zip(metrics, metric_labels):
        baseline_val = baseline_results.get(metric, 0)
        enhanced_val = enhanced_results.get(metric, 0)
        if metric in ['mae', 'rmse']:
            # Lower is better
            improvement = (baseline_val - enhanced_val) / baseline_val * 100
            print(f"  {label}: {baseline_val:.2f} → {enhanced_val:.2f} ({improvement:+.1f}%)")
        else:
            # Higher is better
            improvement = (enhanced_val - baseline_val) / baseline_val * 100
            print(f"  {label}: {baseline_val:.4f} → {enhanced_val:.4f} ({improvement:+.1f}%)")


def measure_inference_overhead(
    model,
    sample_embeddings: torch.Tensor,
    sample_entropy_features: torch.Tensor,
    num_iterations: int = 100,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Measure computational overhead of entropy feature computation.
    
    Args:
        model: EnhancedLengthPredictor
        sample_embeddings: Sample embedding batch
        sample_entropy_features: Sample entropy feature batch
        num_iterations: Number of iterations for timing
        device: Device to run on
        
    Returns:
        Dict with timing statistics
    """
    import time
    
    model = model.to(device)
    model.eval()
    embeddings = sample_embeddings.to(device)
    entropy_features = sample_entropy_features.to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(embeddings, entropy_features)
    
    # Measure with entropy
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(embeddings, entropy_features)
    torch.cuda.synchronize() if device == 'cuda' else None
    time_with_entropy = (time.perf_counter() - start) / num_iterations * 1000  # ms
    
    # Measure without entropy (embedding only)
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(embeddings, None)
    torch.cuda.synchronize() if device == 'cuda' else None
    time_embedding_only = (time.perf_counter() - start) / num_iterations * 1000  # ms
    
    overhead = time_with_entropy - time_embedding_only
    overhead_percent = (overhead / time_embedding_only) * 100
    
    return {
        'time_with_entropy_ms': time_with_entropy,
        'time_embedding_only_ms': time_embedding_only,
        'overhead_ms': overhead,
        'overhead_percent': overhead_percent,
        'batch_size': embeddings.shape[0],
        'num_iterations': num_iterations
    }
