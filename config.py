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


def get_default_config():
    """Get default configuration"""
    return {
        "model": ModelConfig(),
        "predictor": PredictorConfig(),
        "data": DataConfig(),
        "bayesian": BayesianConfig()
    }
