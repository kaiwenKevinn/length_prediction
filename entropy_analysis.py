"""
Entropy Analysis and Visualization for TRAIL Length Prediction
Provides tools for analyzing entropy patterns and their relationship to output length.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import json
from scipy import stats

from config import get_default_config
from entropy_utils import (
    EntropyCalculator, 
    EntropyFeatureExtractor, 
    SequenceEntropyAnalyzer,
    EntropyFeatures
)


def plot_entropy_time_series(
    attention_entropy_seq: List[float],
    logits_entropy_seq: List[float],
    output_length: int,
    save_path: Optional[str] = None,
    title: str = "Entropy Time Series"
):
    """
    Plot entropy values over generation time.
    
    Args:
        attention_entropy_seq: Sequence of attention entropy values
        logits_entropy_seq: Sequence of logits entropy values
        output_length: Total output length for reference
        save_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    tokens = list(range(len(attention_entropy_seq)))
    
    # Attention entropy
    axes[0].plot(tokens, attention_entropy_seq, 'b-', linewidth=2, label='Attention Entropy')
    axes[0].fill_between(tokens, attention_entropy_seq, alpha=0.3)
    axes[0].set_ylabel('Attention Entropy', fontsize=12)
    axes[0].set_title(f'{title} (Output Length: {output_length})', fontsize=14)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Logits entropy
    axes[1].plot(tokens, logits_entropy_seq, 'r-', linewidth=2, label='Logits Entropy')
    axes[1].fill_between(tokens, logits_entropy_seq, alpha=0.3, color='red')
    axes[1].set_xlabel('Token Position', fontsize=12)
    axes[1].set_ylabel('Logits Entropy', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_entropy_correlation(
    samples: List[Dict],  # Each dict has attention_seq, logits_seq, output_length
    save_path: Optional[str] = None
):
    """
    Plot correlation between entropy features and output length.
    
    Args:
        samples: List of dicts with entropy sequences and output lengths
        save_path: Path to save figure
    """
    # Extract features
    extractor = EntropyFeatureExtractor()
    
    output_lengths = []
    attn_means = []
    logits_means = []
    attn_stds = []
    logits_trends = []
    correlations = []
    
    for sample in samples:
        features = extractor.extract_features(
            sample['attention_seq'],
            sample['logits_seq']
        )
        
        output_lengths.append(sample['output_length'])
        attn_means.append(features.attention_entropy_mean)
        logits_means.append(features.logits_entropy_mean)
        attn_stds.append(features.attention_entropy_std)
        logits_trends.append(features.logits_entropy_trend)
        correlations.append(features.entropy_correlation)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Attention mean vs output length
    axes[0, 0].scatter(attn_means, output_lengths, alpha=0.5, s=20)
    axes[0, 0].set_xlabel('Attention Entropy Mean')
    axes[0, 0].set_ylabel('Output Length')
    axes[0, 0].set_title('Attention Entropy Mean vs Output Length')
    # Add correlation
    corr = stats.pearsonr(attn_means, output_lengths)[0]
    axes[0, 0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0, 0].transAxes,
                   fontsize=10, verticalalignment='top')
    
    # Logits mean vs output length
    axes[0, 1].scatter(logits_means, output_lengths, alpha=0.5, s=20, c='red')
    axes[0, 1].set_xlabel('Logits Entropy Mean')
    axes[0, 1].set_ylabel('Output Length')
    axes[0, 1].set_title('Logits Entropy Mean vs Output Length')
    corr = stats.pearsonr(logits_means, output_lengths)[0]
    axes[0, 1].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0, 1].transAxes,
                   fontsize=10, verticalalignment='top')
    
    # Attention std vs output length
    axes[0, 2].scatter(attn_stds, output_lengths, alpha=0.5, s=20, c='green')
    axes[0, 2].set_xlabel('Attention Entropy Std')
    axes[0, 2].set_ylabel('Output Length')
    axes[0, 2].set_title('Attention Entropy Volatility vs Output Length')
    corr = stats.pearsonr(attn_stds, output_lengths)[0]
    axes[0, 2].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0, 2].transAxes,
                   fontsize=10, verticalalignment='top')
    
    # Logits trend vs output length
    axes[1, 0].scatter(logits_trends, output_lengths, alpha=0.5, s=20, c='purple')
    axes[1, 0].set_xlabel('Logits Entropy Trend')
    axes[1, 0].set_ylabel('Output Length')
    axes[1, 0].set_title('Logits Entropy Trend vs Output Length')
    corr = stats.pearsonr(logits_trends, output_lengths)[0]
    axes[1, 0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1, 0].transAxes,
                   fontsize=10, verticalalignment='top')
    
    # Entropy correlation vs output length
    axes[1, 1].scatter(correlations, output_lengths, alpha=0.5, s=20, c='orange')
    axes[1, 1].set_xlabel('Attention-Logits Entropy Correlation')
    axes[1, 1].set_ylabel('Output Length')
    axes[1, 1].set_title('Entropy Correlation vs Output Length')
    corr = stats.pearsonr(correlations, output_lengths)[0]
    axes[1, 1].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top')
    
    # Output length distribution
    axes[1, 2].hist(output_lengths, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1, 2].set_xlabel('Output Length')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Output Length Distribution')
    
    plt.tight_layout()
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_entropy_by_length_bin(
    samples: List[Dict],
    num_bins: int = 5,
    save_path: Optional[str] = None
):
    """
    Plot entropy statistics grouped by output length bins.
    
    Args:
        samples: List of dicts with entropy sequences and output lengths
        num_bins: Number of length bins
        save_path: Path to save figure
    """
    extractor = EntropyFeatureExtractor()
    
    # Group samples by output length
    output_lengths = [s['output_length'] for s in samples]
    min_len, max_len = min(output_lengths), max(output_lengths)
    bin_edges = np.linspace(min_len, max_len + 1, num_bins + 1)
    
    bin_data = {i: {'attn_mean': [], 'logits_mean': [], 'attn_std': [], 'logits_std': []} 
                for i in range(num_bins)}
    
    for sample in samples:
        features = extractor.extract_features(sample['attention_seq'], sample['logits_seq'])
        bin_idx = min(np.digitize(sample['output_length'], bin_edges) - 1, num_bins - 1)
        
        bin_data[bin_idx]['attn_mean'].append(features.attention_entropy_mean)
        bin_data[bin_idx]['logits_mean'].append(features.logits_entropy_mean)
        bin_data[bin_idx]['attn_std'].append(features.attention_entropy_std)
        bin_data[bin_idx]['logits_std'].append(features.logits_entropy_std)
    
    # Create box plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    bin_labels = [f'{int(bin_edges[i])}-{int(bin_edges[i+1])}' for i in range(num_bins)]
    
    # Attention mean by bin
    data = [bin_data[i]['attn_mean'] for i in range(num_bins)]
    bp1 = axes[0, 0].boxplot(data, labels=bin_labels, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    axes[0, 0].set_xlabel('Output Length Bin')
    axes[0, 0].set_ylabel('Attention Entropy Mean')
    axes[0, 0].set_title('Attention Entropy Mean by Output Length')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Logits mean by bin
    data = [bin_data[i]['logits_mean'] for i in range(num_bins)]
    bp2 = axes[0, 1].boxplot(data, labels=bin_labels, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('darkorange')
        patch.set_alpha(0.7)
    axes[0, 1].set_xlabel('Output Length Bin')
    axes[0, 1].set_ylabel('Logits Entropy Mean')
    axes[0, 1].set_title('Logits Entropy Mean by Output Length')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Attention std by bin
    data = [bin_data[i]['attn_std'] for i in range(num_bins)]
    bp3 = axes[1, 0].boxplot(data, labels=bin_labels, patch_artist=True)
    for patch in bp3['boxes']:
        patch.set_facecolor('green')
        patch.set_alpha(0.7)
    axes[1, 0].set_xlabel('Output Length Bin')
    axes[1, 0].set_ylabel('Attention Entropy Std')
    axes[1, 0].set_title('Attention Entropy Volatility by Output Length')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Logits std by bin
    data = [bin_data[i]['logits_std'] for i in range(num_bins)]
    bp4 = axes[1, 1].boxplot(data, labels=bin_labels, patch_artist=True)
    for patch in bp4['boxes']:
        patch.set_facecolor('purple')
        patch.set_alpha(0.7)
    axes[1, 1].set_xlabel('Output Length Bin')
    axes[1, 1].set_ylabel('Logits Entropy Std')
    axes[1, 1].set_title('Logits Entropy Volatility by Output Length')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def plot_layer_entropy_analysis(
    per_layer_entropy: Dict[int, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot entropy analysis across different layers.
    
    Args:
        per_layer_entropy: Dict mapping layer_idx to list of entropy values
        save_path: Path to save figure
    """
    layers = sorted(per_layer_entropy.keys())
    means = [np.mean(per_layer_entropy[l]) for l in layers]
    stds = [np.std(per_layer_entropy[l]) for l in layers]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean entropy per layer
    axes[0].bar(layers, means, yerr=stds, capsize=3, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Layer Index', fontsize=12)
    axes[0].set_ylabel('Mean Attention Entropy', fontsize=12)
    axes[0].set_title('Attention Entropy by Layer', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Highlight best layers (typically 10-15)
    for i, layer in enumerate(layers):
        if 10 <= layer <= 15:
            axes[0].patches[i].set_facecolor('darkorange')
    
    # Entropy distribution per layer (heatmap)
    max_len = max(len(per_layer_entropy[l]) for l in layers)
    entropy_matrix = np.zeros((len(layers), max_len))
    entropy_matrix[:] = np.nan
    
    for i, layer in enumerate(layers):
        entropy_matrix[i, :len(per_layer_entropy[layer])] = per_layer_entropy[layer]
    
    im = axes[1].imshow(entropy_matrix, aspect='auto', cmap='viridis')
    axes[1].set_xlabel('Token Position', fontsize=12)
    axes[1].set_ylabel('Layer Index', fontsize=12)
    axes[1].set_title('Attention Entropy Heatmap', fontsize=14)
    axes[1].set_yticks(range(len(layers)))
    axes[1].set_yticklabels(layers)
    plt.colorbar(im, ax=axes[1], label='Entropy')
    
    plt.tight_layout()
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def analyze_entropy_patterns(
    samples: List[Dict],
    save_dir: str = "./results/entropy_analysis"
) -> Dict[str, any]:
    """
    Comprehensive entropy pattern analysis.
    
    Args:
        samples: List of dicts with entropy sequences and output lengths
        save_dir: Directory to save results
        
    Returns:
        Dict with analysis results
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    analyzer = SequenceEntropyAnalyzer()
    extractor = EntropyFeatureExtractor()
    
    results = {
        'pattern_counts': {},
        'length_predictions': {'short': 0, 'medium': 0, 'long': 0, 'unknown': 0},
        'feature_correlations': {},
        'statistics': {}
    }
    
    all_features = []
    output_lengths = []
    
    for sample in tqdm(samples, desc="Analyzing entropy patterns"):
        # Extract features
        features = extractor.extract_features(
            sample['attention_seq'],
            sample['logits_seq']
        )
        all_features.append(features)
        output_lengths.append(sample['output_length'])
        
        # Analyze pattern
        pattern = analyzer.analyze_entropy_pattern(
            sample['attention_seq'],
            sample['logits_seq']
        )
        
        pattern_type = pattern['pattern_type']
        results['pattern_counts'][pattern_type] = \
            results['pattern_counts'].get(pattern_type, 0) + 1
        
        # Length prediction
        prediction = analyzer.predict_length_indicator(
            sample['attention_seq'],
            sample['logits_seq']
        )
        results['length_predictions'][prediction] += 1
    
    # Compute feature correlations with output length
    feature_names = [
        'attention_entropy_mean', 'attention_entropy_std', 'attention_entropy_trend',
        'logits_entropy_mean', 'logits_entropy_std', 'logits_entropy_trend',
        'entropy_correlation', 'entropy_ratio_mean'
    ]
    
    for fname in feature_names:
        values = [getattr(f, fname) for f in all_features]
        corr, p_value = stats.pearsonr(values, output_lengths)
        results['feature_correlations'][fname] = {
            'correlation': float(corr),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }
    
    # Overall statistics
    results['statistics'] = {
        'num_samples': len(samples),
        'mean_output_length': float(np.mean(output_lengths)),
        'std_output_length': float(np.std(output_lengths)),
        'mean_attention_entropy': float(np.mean([f.attention_entropy_mean for f in all_features])),
        'mean_logits_entropy': float(np.mean([f.logits_entropy_mean for f in all_features]))
    }
    
    # Save results
    results_path = os.path.join(save_dir, "entropy_analysis_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")
    
    # Generate visualizations
    plot_entropy_correlation(samples, save_path=os.path.join(save_dir, "entropy_correlation.png"))
    plot_entropy_by_length_bin(samples, save_path=os.path.join(save_dir, "entropy_by_length.png"))
    
    return results


def plot_feature_correlation_heatmap(
    features: List[EntropyFeatures],
    save_path: Optional[str] = None
):
    """
    Plot correlation heatmap between all entropy features.
    
    Args:
        features: List of EntropyFeatures instances
        save_path: Path to save figure
    """
    feature_names = [
        'attn_mean', 'attn_std', 'attn_max', 'attn_min', 'attn_trend',
        'logits_mean', 'logits_std', 'logits_max', 'logits_min', 'logits_trend',
        'correlation', 'ratio', 'diff', 'product'
    ]
    
    # Build feature matrix
    feature_matrix = []
    for f in features:
        row = [
            f.attention_entropy_mean, f.attention_entropy_std, 
            f.attention_entropy_max, f.attention_entropy_min, f.attention_entropy_trend,
            f.logits_entropy_mean, f.logits_entropy_std,
            f.logits_entropy_max, f.logits_entropy_min, f.logits_entropy_trend,
            f.entropy_correlation, f.entropy_ratio_mean,
            f.entropy_diff_mean, f.entropy_product_mean
        ]
        feature_matrix.append(row)
    
    feature_matrix = np.array(feature_matrix)
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(feature_matrix.T)
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        xticklabels=feature_names,
        yticklabels=feature_names,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True
    )
    
    plt.title('Entropy Feature Correlation Matrix', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def demo_entropy_analysis():
    """Demonstrate entropy analysis with synthetic data"""
    print("=" * 60)
    print("Entropy Analysis Demo")
    print("=" * 60)
    
    # Generate synthetic samples
    np.random.seed(42)
    samples = []
    
    for _ in range(100):
        # Simulate that longer outputs have higher/more variable entropy
        output_length = np.random.randint(50, 400)
        base_entropy = 2.0 + output_length / 200  # Higher for longer outputs
        
        seq_len = min(15, output_length)
        attention_seq = [
            base_entropy + np.random.normal(0, 0.5) + i * 0.01
            for i in range(seq_len)
        ]
        logits_seq = [
            base_entropy + np.random.normal(0, 0.3) + np.random.normal(0, 0.1) * i
            for i in range(seq_len)
        ]
        
        samples.append({
            'attention_seq': attention_seq,
            'logits_seq': logits_seq,
            'output_length': output_length
        })
    
    print(f"Generated {len(samples)} synthetic samples")
    
    # Run analysis
    results = analyze_entropy_patterns(samples, save_dir="./demo_results")
    
    print("\n--- Pattern Counts ---")
    for pattern, count in results['pattern_counts'].items():
        print(f"  {pattern}: {count}")
    
    print("\n--- Feature Correlations with Output Length ---")
    for fname, info in results['feature_correlations'].items():
        sig = "*" if info['significant'] else ""
        print(f"  {fname}: r={info['correlation']:.3f}{sig}")
    
    print("\n--- Length Predictions ---")
    for pred, count in results['length_predictions'].items():
        print(f"  {pred}: {count}")


def main():
    """Main function for entropy analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Entropy Analysis for TRAIL")
    parser.add_argument("--mode", type=str, default="demo",
                       choices=["demo", "analyze"],
                       help="Mode: 'demo' for synthetic data, 'analyze' for real data")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to saved entropy data (for analyze mode)")
    parser.add_argument("--save_dir", type=str, default="./results/entropy_analysis",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        demo_entropy_analysis()
    else:
        if args.data_path is None:
            print("Please provide --data_path for analyze mode")
            return
            
        # Load data and analyze
        data = torch.load(args.data_path)
        samples = []
        for i in range(len(data.get('attention_seqs', []))):
            samples.append({
                'attention_seq': data['attention_seqs'][i],
                'logits_seq': data['logits_seqs'][i],
                'output_length': data['output_lengths'][i]
            })
        
        analyze_entropy_patterns(samples, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
