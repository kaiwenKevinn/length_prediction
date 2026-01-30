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
