"""
Configuration for TRAIL Length Prediction
Based on: DON'T STOP ME NOW: EMBEDDING BASED SCHEDULING FOR LLMS (ICLR 2025)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class ModelConfig:
    """LLaMA model configuration"""
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    num_layers: int = 32
    hidden_dim: int = 4096
    max_output_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    

@dataclass
class PredictorConfig:
    """MLP Predictor configuration"""
    input_dim: int = 4096  # LLaMA hidden dimension
    hidden_dim: int = 512  # First FC layer output
    num_bins: int = 10     # Number of length bins (k=10)
    max_length: int = 512  # Maximum output length
    
    # Training config
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 0.01
    weight_decay: float = 0.01  # AdamW weight decay
    
    # Layer selection
    target_layers: List[int] = field(default_factory=lambda: list(range(10, 16)))  # Layers 10-15
    best_layer: int = 11  # Best performing layer based on paper
    
    def get_bin_boundaries(self) -> List[float]:
        """Get bin boundaries for length classification"""
        bin_size = self.max_length / self.num_bins
        boundaries = [i * bin_size for i in range(self.num_bins + 1)]
        return boundaries
    
    def get_bin_centers(self) -> List[float]:
        """Get center values for each bin"""
        boundaries = self.get_bin_boundaries()
        centers = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(self.num_bins)]
        return centers
    
    def length_to_bin(self, length: int) -> int:
        """Convert length to bin index"""
        bin_size = self.max_length / self.num_bins
        bin_idx = int(length / bin_size)
        return min(bin_idx, self.num_bins - 1)  # Clamp to last bin


@dataclass
class DataConfig:
    """Data configuration"""
    dataset_name: str = "tatsu-lab/alpaca"
    num_profile_samples: int = 1000  # Samples for layer profiling
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    max_input_length: int = 512
    max_output_length: int = 512
    seed: int = 42
    
    # Cache paths
    embeddings_cache_dir: str = "./cache/embeddings"
    model_save_dir: str = "./checkpoints"
    results_dir: str = "./results"


@dataclass
class BayesianConfig:
    """Bayesian smoothing configuration"""
    num_bins: int = 10
    max_length: int = 512
    
    @property
    def bin_size(self) -> float:
        return self.max_length / self.num_bins
    
    @property
    def transition_prob(self) -> float:
        """Probability of transitioning to adjacent bin"""
        return 1.0 / self.bin_size


@dataclass
class EntropyConfig:
    """Entropy feature extraction configuration"""
    
    # Number of initial tokens to collect entropy from (K in paper)
    num_initial_tokens: int = 15  # Recommended K=10-20
    
    # Which layers to extract attention entropy from
    attention_layers: List[int] = field(default_factory=lambda: [10, 11, 12, 13, 14, 15])
    
    # Aggregation method for multi-layer attention entropy
    attention_aggregation: str = "mean"  # "mean", "max", "weighted"
    
    # Temperature for logits entropy computation
    logits_temperature: float = 1.0
    
    # Normalization settings
    normalize_entropy: bool = True
    normalization_momentum: float = 0.1
    
    # Feature extraction settings
    compute_attention_entropy: bool = True
    compute_logits_entropy: bool = True
    compute_cross_features: bool = True
    
    # Time-series analysis settings
    volatility_window: int = 3
    peak_threshold_percentile: float = 75.0
    
    # Cache paths
    entropy_cache_dir: str = "./cache/entropy"
    normalizer_path: str = "./cache/entropy/normalizer_stats.json"
    
    @property
    def feature_dim(self) -> int:
        """Total dimension of entropy features"""
        dim = 0
        if self.compute_attention_entropy:
            dim += 10  # 10 attention entropy features
        if self.compute_logits_entropy:
            dim += 10  # 10 logits entropy features
        if self.compute_cross_features:
            dim += 4   # 4 cross-entropy features
        return dim


@dataclass
class EnhancedPredictorConfig:
    """
    Enhanced MLP Predictor configuration with entropy features.
    Supports multiple fusion strategies.
    """
    
    # Original embedding config
    embedding_dim: int = 4096  # LLaMA hidden dimension
    embedding_hidden_dim: int = 512  # Embedding branch hidden dim
    
    # Entropy feature config
    entropy_feature_dim: int = 24  # From EntropyFeatures.feature_dim()
    entropy_hidden_dim: int = 64   # Entropy branch hidden dim
    
    # Fusion config
    fusion_method: str = "concat"  # "concat", "attention", "gating", "bilinear"
    fusion_hidden_dim: int = 256   # Hidden dim after fusion
    
    # Output config
    num_bins: int = 10
    max_length: int = 512
    
    # Feature flags
    use_embedding_features: bool = True
    use_entropy_features: bool = True
    
    # Training config
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 0.01
    weight_decay: float = 0.01
    
    # Dropout for regularization
    dropout_rate: float = 0.1
    
    # Time-series encoder config (for entropy sequences)
    use_sequence_encoder: bool = False  # Use CNN/RNN for entropy sequences
    sequence_encoder_type: str = "cnn"  # "cnn", "lstm", "gru"
    sequence_hidden_dim: int = 32
    
    def get_bin_boundaries(self) -> List[float]:
        """Get bin boundaries for length classification"""
        bin_size = self.max_length / self.num_bins
        boundaries = [i * bin_size for i in range(self.num_bins + 1)]
        return boundaries
    
    def get_bin_centers(self) -> List[float]:
        """Get center values for each bin"""
        boundaries = self.get_bin_boundaries()
        centers = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(self.num_bins)]
        return centers
    
    def length_to_bin(self, length: int) -> int:
        """Convert length to bin index"""
        bin_size = self.max_length / self.num_bins
        bin_idx = int(length / bin_size)
        return min(bin_idx, self.num_bins - 1)
    
    @property
    def total_input_dim(self) -> int:
        """Total input dimension based on enabled features"""
        dim = 0
        if self.use_embedding_features:
            dim += self.embedding_dim
        if self.use_entropy_features:
            dim += self.entropy_feature_dim
        return dim


@dataclass
class ExperimentConfig:
    """Configuration for ablation studies and experiments"""
    
    # Ablation flags
    ablation_embedding_only: bool = False
    ablation_attention_entropy_only: bool = False
    ablation_logits_entropy_only: bool = False
    ablation_entropy_only: bool = False  # Both entropy types, no embedding
    
    # K-value experiment
    k_values_to_test: List[int] = field(default_factory=lambda: [5, 10, 15, 20, 25])
    
    # Layer analysis
    layers_to_analyze: List[int] = field(default_factory=lambda: list(range(32)))
    
    # Statistical testing
    significance_level: float = 0.05
    num_bootstrap_samples: int = 1000
    
    # Logging
    log_interval: int = 100
    save_attention_maps: bool = False  # Saves memory if False
    
    # Results paths
    ablation_results_dir: str = "./results/ablation"
    layer_analysis_dir: str = "./results/layer_analysis"
    k_analysis_dir: str = "./results/k_analysis"


def get_default_config():
    """Get default configuration"""
    return {
        "model": ModelConfig(),
        "predictor": PredictorConfig(),
        "enhanced_predictor": EnhancedPredictorConfig(),
        "data": DataConfig(),
        "bayesian": BayesianConfig(),
        "entropy": EntropyConfig(),
        "experiment": ExperimentConfig()
    }
