"""
Entropy Utilities for TRAIL Length Prediction
Implements Attention Entropy and Logits Entropy computation and feature extraction.

Theoretical Basis:
- High attention entropy → dispersed attention → complex reasoning → longer output
- High logits entropy → model uncertainty → longer generation trajectory
- Entropy stability correlates with output length: stable low entropy → short output
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from scipy import stats
import warnings


@dataclass
class EntropyFeatures:
    """Container for extracted entropy features"""
    # Attention entropy features
    attention_entropy_mean: float = 0.0
    attention_entropy_std: float = 0.0
    attention_entropy_max: float = 0.0
    attention_entropy_min: float = 0.0
    attention_entropy_var: float = 0.0
    attention_entropy_skew: float = 0.0
    attention_entropy_kurtosis: float = 0.0
    attention_entropy_trend: float = 0.0  # Slope of linear fit
    attention_entropy_volatility: float = 0.0  # Rolling std
    attention_entropy_peak_count: int = 0  # Number of local maxima
    
    # Logits entropy features
    logits_entropy_mean: float = 0.0
    logits_entropy_std: float = 0.0
    logits_entropy_max: float = 0.0
    logits_entropy_min: float = 0.0
    logits_entropy_var: float = 0.0
    logits_entropy_skew: float = 0.0
    logits_entropy_kurtosis: float = 0.0
    logits_entropy_trend: float = 0.0
    logits_entropy_volatility: float = 0.0
    logits_entropy_peak_count: int = 0
    
    # Cross-entropy features (interaction between attention and logits entropy)
    entropy_correlation: float = 0.0  # Pearson correlation
    entropy_ratio_mean: float = 0.0   # Mean ratio of attention/logits entropy
    entropy_diff_mean: float = 0.0    # Mean difference
    entropy_product_mean: float = 0.0  # Mean product (interaction term)
    
    # Raw sequences for time-series analysis
    attention_entropy_sequence: List[float] = field(default_factory=list)
    logits_entropy_sequence: List[float] = field(default_factory=list)
    
    # Per-layer attention entropy (for layer analysis)
    per_layer_attention_entropy: Dict[int, List[float]] = field(default_factory=dict)
    
    def to_tensor(self, include_sequences: bool = False) -> torch.Tensor:
        """Convert features to tensor for model input"""
        features = [
            # Attention entropy features (10 features)
            self.attention_entropy_mean,
            self.attention_entropy_std,
            self.attention_entropy_max,
            self.attention_entropy_min,
            self.attention_entropy_var,
            self.attention_entropy_skew,
            self.attention_entropy_kurtosis,
            self.attention_entropy_trend,
            self.attention_entropy_volatility,
            float(self.attention_entropy_peak_count),
            
            # Logits entropy features (10 features)
            self.logits_entropy_mean,
            self.logits_entropy_std,
            self.logits_entropy_max,
            self.logits_entropy_min,
            self.logits_entropy_var,
            self.logits_entropy_skew,
            self.logits_entropy_kurtosis,
            self.logits_entropy_trend,
            self.logits_entropy_volatility,
            float(self.logits_entropy_peak_count),
            
            # Cross-entropy features (4 features)
            self.entropy_correlation,
            self.entropy_ratio_mean,
            self.entropy_diff_mean,
            self.entropy_product_mean,
        ]
        
        return torch.tensor(features, dtype=torch.float32)
    
    @staticmethod
    def feature_dim(include_sequences: bool = False) -> int:
        """Return the dimension of feature tensor"""
        return 24  # 10 + 10 + 4 aggregated features


class EntropyCalculator:
    """
    Calculate attention entropy and logits entropy from LLM internals.
    
    Attention Entropy: H(A) = -Σ(a_i * log(a_i))
    - Measures how dispersed/focused the attention is
    - High entropy = attention spread across many tokens
    - Low entropy = attention focused on few tokens
    
    Logits Entropy: H(P) = -Σ(p_i * log(p_i))
    - Measures model's uncertainty in next token prediction
    - High entropy = model uncertain, many plausible next tokens
    - Low entropy = model confident, few likely next tokens
    """
    
    def __init__(self, config=None):
        """
        Args:
            config: EntropyConfig instance
        """
        self.config = config
        self.eps = 1e-10  # Small constant to avoid log(0)
        
    def compute_attention_entropy(
        self,
        attention_weights: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute entropy of attention weight distribution.
        
        Args:
            attention_weights: Attention weights of shape 
                [batch, num_heads, seq_len, seq_len] or [num_heads, seq_len, seq_len]
            layer_idx: Optional layer index for logging
            
        Returns:
            Entropy values of shape [batch, num_heads] or [num_heads]
        """
        # Ensure attention weights sum to 1 along last dimension
        # attention_weights shape: [..., seq_len, seq_len]
        # We compute entropy over the last dimension (attention distribution)
        
        # Clamp to avoid log(0)
        attn = torch.clamp(attention_weights, min=self.eps)
        
        # Compute entropy: H = -sum(p * log(p))
        entropy = -torch.sum(attn * torch.log(attn), dim=-1)
        
        # Average over sequence positions (second to last dim)
        # This gives entropy per head
        entropy = entropy.mean(dim=-1)
        
        return entropy
    
    def compute_logits_entropy(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute entropy of logits probability distribution.
        
        Args:
            logits: Model output logits of shape [batch, vocab_size] or [vocab_size]
            temperature: Temperature for softmax scaling
            
        Returns:
            Entropy value(s)
        """
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Convert to probabilities
        probs = F.softmax(scaled_logits, dim=-1)
        
        # Clamp to avoid log(0)
        probs = torch.clamp(probs, min=self.eps)
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        
        return entropy
    
    def compute_attention_entropy_all_layers(
        self,
        attention_outputs: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """
        Compute attention entropy for all layers.
        
        Args:
            attention_outputs: Dict mapping layer_idx to attention weights
            
        Returns:
            Dict mapping layer_idx to entropy values
        """
        entropies = {}
        for layer_idx, attn_weights in attention_outputs.items():
            entropies[layer_idx] = self.compute_attention_entropy(attn_weights, layer_idx)
        return entropies
    
    def aggregate_attention_entropy(
        self,
        layer_entropies: Dict[int, torch.Tensor],
        aggregation: str = "mean"
    ) -> torch.Tensor:
        """
        Aggregate attention entropy across layers.
        
        Args:
            layer_entropies: Dict mapping layer_idx to entropy values
            aggregation: Aggregation method ("mean", "max", "weighted")
            
        Returns:
            Aggregated entropy value
        """
        if not layer_entropies:
            return torch.tensor(0.0)
            
        # Stack all layer entropies
        all_entropies = torch.stack(list(layer_entropies.values()))
        
        if aggregation == "mean":
            return all_entropies.mean()
        elif aggregation == "max":
            return all_entropies.max()
        elif aggregation == "weighted":
            # Weight later layers more (they capture higher-level features)
            weights = torch.arange(1, len(layer_entropies) + 1, dtype=torch.float32)
            weights = weights / weights.sum()
            return (all_entropies.mean(dim=-1) * weights).sum()
        else:
            return all_entropies.mean()


class EntropyFeatureExtractor:
    """
    Extract statistical features from entropy sequences.
    
    Features include:
    - Basic statistics: mean, std, max, min, var
    - Higher-order statistics: skewness, kurtosis
    - Time-series features: trend, volatility, peak count
    - Cross-feature interactions: correlation, ratio, difference
    """
    
    def __init__(self, config=None):
        self.config = config
        self.calculator = EntropyCalculator(config)
        
    def compute_basic_stats(self, values: np.ndarray) -> Dict[str, float]:
        """Compute basic statistical features"""
        if len(values) == 0:
            return {
                "mean": 0.0, "std": 0.0, "max": 0.0, 
                "min": 0.0, "var": 0.0
            }
            
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "max": float(np.max(values)),
            "min": float(np.min(values)),
            "var": float(np.var(values))
        }
    
    def compute_higher_order_stats(self, values: np.ndarray) -> Dict[str, float]:
        """Compute skewness and kurtosis"""
        if len(values) < 3:
            return {"skew": 0.0, "kurtosis": 0.0}
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skew = stats.skew(values) if len(values) >= 3 else 0.0
            kurtosis = stats.kurtosis(values) if len(values) >= 4 else 0.0
            
        return {
            "skew": float(skew) if not np.isnan(skew) else 0.0,
            "kurtosis": float(kurtosis) if not np.isnan(kurtosis) else 0.0
        }
    
    def compute_trend(self, values: np.ndarray) -> float:
        """Compute linear trend (slope) of values"""
        if len(values) < 2:
            return 0.0
            
        x = np.arange(len(values))
        try:
            slope, _, _, _, _ = stats.linregress(x, values)
            return float(slope) if not np.isnan(slope) else 0.0
        except:
            return 0.0
    
    def compute_volatility(self, values: np.ndarray, window: int = 3) -> float:
        """Compute rolling standard deviation (volatility)"""
        if len(values) < window:
            return float(np.std(values)) if len(values) > 0 else 0.0
            
        # Compute rolling std
        rolling_std = []
        for i in range(len(values) - window + 1):
            rolling_std.append(np.std(values[i:i+window]))
            
        return float(np.mean(rolling_std))
    
    def count_peaks(self, values: np.ndarray, threshold_percentile: float = 75) -> int:
        """Count number of local maxima above threshold"""
        if len(values) < 3:
            return 0
            
        threshold = np.percentile(values, threshold_percentile)
        peaks = 0
        
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                if values[i] > threshold:
                    peaks += 1
                    
        return peaks
    
    def compute_cross_features(
        self,
        attention_entropy: np.ndarray,
        logits_entropy: np.ndarray
    ) -> Dict[str, float]:
        """Compute interaction features between attention and logits entropy"""
        if len(attention_entropy) == 0 or len(logits_entropy) == 0:
            return {
                "correlation": 0.0,
                "ratio_mean": 0.0,
                "diff_mean": 0.0,
                "product_mean": 0.0
            }
        
        # Ensure same length
        min_len = min(len(attention_entropy), len(logits_entropy))
        attn = attention_entropy[:min_len]
        logits = logits_entropy[:min_len]
        
        # Correlation
        if len(attn) >= 2:
            try:
                corr, _ = stats.pearsonr(attn, logits)
                corr = float(corr) if not np.isnan(corr) else 0.0
            except:
                corr = 0.0
        else:
            corr = 0.0
        
        # Ratio (attention / logits, with safety for division)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(logits > 1e-10, attn / logits, 0.0)
            ratio_mean = float(np.mean(ratios))
        
        # Difference
        diff_mean = float(np.mean(attn - logits))
        
        # Product (interaction term)
        product_mean = float(np.mean(attn * logits))
        
        return {
            "correlation": corr,
            "ratio_mean": ratio_mean if not np.isnan(ratio_mean) else 0.0,
            "diff_mean": diff_mean,
            "product_mean": product_mean
        }
    
    def extract_features(
        self,
        attention_entropy_seq: List[float],
        logits_entropy_seq: List[float],
        per_layer_attention: Optional[Dict[int, List[float]]] = None
    ) -> EntropyFeatures:
        """
        Extract all features from entropy sequences.
        
        Args:
            attention_entropy_seq: Sequence of attention entropy values
            logits_entropy_seq: Sequence of logits entropy values
            per_layer_attention: Optional per-layer attention entropy
            
        Returns:
            EntropyFeatures instance
        """
        attn_arr = np.array(attention_entropy_seq)
        logits_arr = np.array(logits_entropy_seq)
        
        # Attention entropy features
        attn_basic = self.compute_basic_stats(attn_arr)
        attn_higher = self.compute_higher_order_stats(attn_arr)
        attn_trend = self.compute_trend(attn_arr)
        attn_volatility = self.compute_volatility(attn_arr)
        attn_peaks = self.count_peaks(attn_arr)
        
        # Logits entropy features
        logits_basic = self.compute_basic_stats(logits_arr)
        logits_higher = self.compute_higher_order_stats(logits_arr)
        logits_trend = self.compute_trend(logits_arr)
        logits_volatility = self.compute_volatility(logits_arr)
        logits_peaks = self.count_peaks(logits_arr)
        
        # Cross features
        cross = self.compute_cross_features(attn_arr, logits_arr)
        
        return EntropyFeatures(
            # Attention entropy
            attention_entropy_mean=attn_basic["mean"],
            attention_entropy_std=attn_basic["std"],
            attention_entropy_max=attn_basic["max"],
            attention_entropy_min=attn_basic["min"],
            attention_entropy_var=attn_basic["var"],
            attention_entropy_skew=attn_higher["skew"],
            attention_entropy_kurtosis=attn_higher["kurtosis"],
            attention_entropy_trend=attn_trend,
            attention_entropy_volatility=attn_volatility,
            attention_entropy_peak_count=attn_peaks,
            
            # Logits entropy
            logits_entropy_mean=logits_basic["mean"],
            logits_entropy_std=logits_basic["std"],
            logits_entropy_max=logits_basic["max"],
            logits_entropy_min=logits_basic["min"],
            logits_entropy_var=logits_basic["var"],
            logits_entropy_skew=logits_higher["skew"],
            logits_entropy_kurtosis=logits_higher["kurtosis"],
            logits_entropy_trend=logits_trend,
            logits_entropy_volatility=logits_volatility,
            logits_entropy_peak_count=logits_peaks,
            
            # Cross features
            entropy_correlation=cross["correlation"],
            entropy_ratio_mean=cross["ratio_mean"],
            entropy_diff_mean=cross["diff_mean"],
            entropy_product_mean=cross["product_mean"],
            
            # Raw sequences
            attention_entropy_sequence=attention_entropy_seq,
            logits_entropy_sequence=logits_entropy_seq,
            per_layer_attention_entropy=per_layer_attention or {}
        )


class EntropyNormalizer:
    """
    Normalize entropy values across different layers and sequences.
    Uses running statistics for online normalization.
    """
    
    def __init__(self, momentum: float = 0.1):
        self.momentum = momentum
        self.running_mean = {}
        self.running_var = {}
        self.count = {}
        
    def update_stats(self, key: str, values: np.ndarray):
        """Update running statistics for a given key"""
        if len(values) == 0:
            return
            
        batch_mean = np.mean(values)
        batch_var = np.var(values)
        
        if key not in self.running_mean:
            self.running_mean[key] = batch_mean
            self.running_var[key] = batch_var
            self.count[key] = len(values)
        else:
            # Exponential moving average
            self.running_mean[key] = (1 - self.momentum) * self.running_mean[key] + \
                                     self.momentum * batch_mean
            self.running_var[key] = (1 - self.momentum) * self.running_var[key] + \
                                    self.momentum * batch_var
            self.count[key] += len(values)
    
    def normalize(self, key: str, values: np.ndarray) -> np.ndarray:
        """Normalize values using running statistics"""
        if key not in self.running_mean or len(values) == 0:
            return values
            
        mean = self.running_mean[key]
        std = np.sqrt(self.running_var[key] + 1e-8)
        
        return (values - mean) / std
    
    def save_stats(self, path: str):
        """Save normalization statistics"""
        import json
        stats = {
            "running_mean": self.running_mean,
            "running_var": self.running_var,
            "count": self.count
        }
        with open(path, 'w') as f:
            json.dump(stats, f)
    
    def load_stats(self, path: str):
        """Load normalization statistics"""
        import json
        with open(path, 'r') as f:
            stats = json.load(f)
        self.running_mean = stats["running_mean"]
        self.running_var = stats["running_var"]
        self.count = stats["count"]


class SequenceEntropyAnalyzer:
    """
    Analyze entropy patterns across a sequence generation.
    Provides insights into the relationship between entropy patterns and output length.
    """
    
    def __init__(self, config=None):
        self.config = config
        self.calculator = EntropyCalculator(config)
        self.extractor = EntropyFeatureExtractor(config)
        
    def analyze_entropy_pattern(
        self,
        attention_seq: List[float],
        logits_seq: List[float]
    ) -> Dict[str, any]:
        """
        Analyze entropy patterns to identify generation characteristics.
        
        Returns indicators for:
        - Complex reasoning (high sustained entropy)
        - Self-correction (entropy spikes)
        - Context introduction (attention entropy jumps)
        - Confident generation (low stable entropy)
        """
        attn_arr = np.array(attention_seq)
        logits_arr = np.array(logits_seq)
        
        if len(attn_arr) < 3 or len(logits_arr) < 3:
            return {
                "pattern_type": "insufficient_data",
                "complexity_score": 0.0,
                "confidence_score": 0.0,
                "stability_score": 0.0
            }
        
        # Complexity score: high mean entropy indicates complex reasoning
        complexity_score = (np.mean(logits_arr) + np.mean(attn_arr)) / 2
        
        # Confidence score: inverse of logits entropy
        confidence_score = 1.0 / (1.0 + np.mean(logits_arr))
        
        # Stability score: inverse of entropy variance
        stability_score = 1.0 / (1.0 + np.var(logits_arr) + np.var(attn_arr))
        
        # Detect patterns
        attn_trend = self.extractor.compute_trend(attn_arr)
        logits_trend = self.extractor.compute_trend(logits_arr)
        
        # Determine pattern type
        if complexity_score > 2.0 and stability_score < 0.5:
            pattern_type = "complex_reasoning"
        elif np.std(logits_arr) > np.mean(logits_arr) * 0.5:
            pattern_type = "self_correction"
        elif attn_trend > 0.1:
            pattern_type = "context_expansion"
        elif confidence_score > 0.7 and stability_score > 0.7:
            pattern_type = "confident_generation"
        else:
            pattern_type = "mixed"
        
        return {
            "pattern_type": pattern_type,
            "complexity_score": float(complexity_score),
            "confidence_score": float(confidence_score),
            "stability_score": float(stability_score),
            "attention_trend": float(attn_trend),
            "logits_trend": float(logits_trend)
        }
    
    def predict_length_indicator(
        self,
        attention_seq: List[float],
        logits_seq: List[float]
    ) -> str:
        """
        Predict whether output will be short, medium, or long based on entropy patterns.
        
        Based on theoretical assumptions:
        - Stable low entropy → short output
        - Unstable high entropy → long output
        """
        analysis = self.analyze_entropy_pattern(attention_seq, logits_seq)
        
        if analysis["pattern_type"] == "insufficient_data":
            return "unknown"
        
        complexity = analysis["complexity_score"]
        stability = analysis["stability_score"]
        
        # Decision logic based on theoretical assumptions
        if complexity < 1.5 and stability > 0.6:
            return "short"
        elif complexity > 2.5 or stability < 0.3:
            return "long"
        else:
            return "medium"


def demo_entropy_calculation():
    """Demonstrate entropy calculation with synthetic data"""
    print("=" * 60)
    print("Entropy Calculation Demo")
    print("=" * 60)
    
    calculator = EntropyCalculator()
    extractor = EntropyFeatureExtractor()
    
    # Simulate attention weights (batch=1, heads=8, seq_len=10)
    attention_weights = F.softmax(torch.randn(1, 8, 10, 10), dim=-1)
    
    # Compute attention entropy
    attn_entropy = calculator.compute_attention_entropy(attention_weights)
    print(f"\nAttention entropy per head: {attn_entropy.numpy()}")
    print(f"Mean attention entropy: {attn_entropy.mean().item():.4f}")
    
    # Simulate logits (batch=1, vocab_size=32000)
    logits = torch.randn(1, 32000)
    
    # Compute logits entropy
    logits_entropy = calculator.compute_logits_entropy(logits)
    print(f"\nLogits entropy: {logits_entropy.item():.4f}")
    
    # Simulate a sequence of entropy values (K=15 tokens)
    K = 15
    attn_seq = [float(calculator.compute_attention_entropy(
        F.softmax(torch.randn(1, 8, 10+i, 10+i), dim=-1)
    ).mean()) for i in range(K)]
    
    logits_seq = [float(calculator.compute_logits_entropy(
        torch.randn(1, 32000)
    )) for _ in range(K)]
    
    # Extract features
    features = extractor.extract_features(attn_seq, logits_seq)
    
    print(f"\n--- Extracted Features ---")
    print(f"Attention entropy mean: {features.attention_entropy_mean:.4f}")
    print(f"Attention entropy trend: {features.attention_entropy_trend:.4f}")
    print(f"Logits entropy mean: {features.logits_entropy_mean:.4f}")
    print(f"Logits entropy volatility: {features.logits_entropy_volatility:.4f}")
    print(f"Entropy correlation: {features.entropy_correlation:.4f}")
    
    # Analyze pattern
    analyzer = SequenceEntropyAnalyzer()
    analysis = analyzer.analyze_entropy_pattern(attn_seq, logits_seq)
    
    print(f"\n--- Pattern Analysis ---")
    print(f"Pattern type: {analysis['pattern_type']}")
    print(f"Complexity score: {analysis['complexity_score']:.4f}")
    print(f"Confidence score: {analysis['confidence_score']:.4f}")
    print(f"Stability score: {analysis['stability_score']:.4f}")
    
    # Feature tensor
    tensor = features.to_tensor()
    print(f"\n--- Feature Tensor ---")
    print(f"Shape: {tensor.shape}")
    print(f"Feature dimension: {EntropyFeatures.feature_dim()}")


if __name__ == "__main__":
    demo_entropy_calculation()
