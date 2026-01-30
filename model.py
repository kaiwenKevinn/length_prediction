"""
MLP Predictor Model for TRAIL Length Prediction
Based on: DON'T STOP ME NOW: EMBEDDING BASED SCHEDULING FOR LLMS (ICLR 2025)

Architecture:
    Input Embedding (4096D)
         ↓
    FC Layer (4096 → 512)
         ↓
    ReLU Activation
         ↓
    FC Layer (512 → k=10)
         ↓
    Softmax → Probability Distribution over bins
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class LengthPredictor(nn.Module):
    """
    MLP-based Length Predictor for predicting remaining output length.
    
    This predictor takes LLaMA layer embeddings as input and outputs
    a probability distribution over k length bins.
    """
    
    def __init__(self, config):
        """
        Args:
            config: PredictorConfig instance
        """
        super().__init__()
        self.config = config
        
        # Two-layer MLP as described in the paper
        self.fc1 = nn.Linear(config.input_dim, config.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.hidden_dim, config.num_bins)
        
        # Store bin centers for expected length calculation
        self.register_buffer(
            'bin_centers',
            torch.tensor(config.get_bin_centers(), dtype=torch.float32)
        )
        
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the predictor.
        
        Args:
            embeddings: Input embeddings of shape [batch_size, hidden_dim]
            
        Returns:
            Dict containing:
                - 'logits': Raw output logits [batch_size, num_bins]
                - 'probs': Probability distribution [batch_size, num_bins]
                - 'expected_length': Expected length based on probabilities [batch_size]
        """
        # MLP forward pass
        x = self.fc1(embeddings)
        x = self.relu(x)
        logits = self.fc2(x)
        
        # Softmax for probability distribution
        probs = F.softmax(logits, dim=-1)
        
        # Calculate expected length: sum of (prob * bin_center)
        expected_length = (probs * self.bin_centers).sum(dim=-1)
        
        return {
            'logits': logits,
            'probs': probs,
            'expected_length': expected_length
        }
    
    def predict_bin(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict the most likely bin for each embedding.
        
        Args:
            embeddings: Input embeddings of shape [batch_size, hidden_dim]
            
        Returns:
            Predicted bin indices of shape [batch_size]
        """
        output = self.forward(embeddings)
        return output['probs'].argmax(dim=-1)
    
    def predict_length(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict expected length for each embedding.
        
        Args:
            embeddings: Input embeddings of shape [batch_size, hidden_dim]
            
        Returns:
            Expected lengths of shape [batch_size]
        """
        output = self.forward(embeddings)
        return output['expected_length']


class LengthPredictorEnsemble(nn.Module):
    """
    Ensemble of predictors for multiple layers.
    Used during layer selection phase to compare different layers.
    """
    
    def __init__(self, config, layer_indices):
        """
        Args:
            config: PredictorConfig instance
            layer_indices: List of layer indices to create predictors for
        """
        super().__init__()
        self.config = config
        self.layer_indices = layer_indices
        
        # Create a predictor for each layer
        self.predictors = nn.ModuleDict({
            str(layer_idx): LengthPredictor(config)
            for layer_idx in layer_indices
        })
        
    def forward(self, layer_embeddings: Dict[int, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Forward pass for all layer predictors.
        
        Args:
            layer_embeddings: Dict mapping layer_idx to embeddings [batch_size, hidden_dim]
            
        Returns:
            Dict mapping layer_idx to prediction outputs
        """
        outputs = {}
        for layer_idx, embeddings in layer_embeddings.items():
            if str(layer_idx) in self.predictors:
                outputs[layer_idx] = self.predictors[str(layer_idx)](embeddings)
        return outputs
    
    def get_predictor(self, layer_idx: int) -> LengthPredictor:
        """Get predictor for specific layer"""
        return self.predictors[str(layer_idx)]


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute CrossEntropy loss for length prediction.
    
    Args:
        logits: Predicted logits [batch_size, num_bins]
        labels: Ground truth bin indices [batch_size]
        reduction: Loss reduction method ('mean', 'sum', 'none')
        
    Returns:
        Loss value
    """
    return F.cross_entropy(logits, labels, reduction=reduction)


def compute_mae(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    """
    Compute Mean Absolute Error for length prediction.
    
    Args:
        predictions: Predicted lengths [batch_size]
        targets: Ground truth lengths [batch_size]
        
    Returns:
        MAE value
    """
    return torch.abs(predictions - targets.float()).mean()


def compute_metrics(
    model: LengthPredictor,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Compute evaluation metrics on a dataset.
    
    Args:
        model: LengthPredictor model
        dataloader: DataLoader for evaluation
        device: Device to use
        
    Returns:
        Dict containing:
            - 'loss': Average cross-entropy loss
            - 'mae': Mean absolute error
            - 'accuracy': Top-1 accuracy
    """
    model.eval()
    
    total_loss = 0.0
    total_mae = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch['embedding'].to(device)
            labels = batch['bin_label'].to(device)
            remaining_lengths = batch['remaining_length'].float().to(device)
            
            # Forward pass
            output = model(embeddings)
            
            # Compute loss
            loss = compute_loss(output['logits'], labels)
            total_loss += loss.item() * len(labels)
            
            # Compute MAE
            mae = compute_mae(output['expected_length'], remaining_lengths)
            total_mae += mae.item() * len(labels)
            
            # Compute accuracy
            predicted_bins = output['logits'].argmax(dim=-1)
            total_correct += (predicted_bins == labels).sum().item()
            
            total_samples += len(labels)
    
    return {
        'loss': total_loss / total_samples,
        'mae': total_mae / total_samples,
        'accuracy': total_correct / total_samples
    }


def save_model(model: LengthPredictor, path: str, config=None):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(path: str, config=None, device: str = 'cpu') -> LengthPredictor:
    """Load model from checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    
    if config is None:
        config = checkpoint.get('config')
        
    if config is None:
        raise ValueError("Config not found in checkpoint and not provided")
        
    model = LengthPredictor(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Model loaded from {path}")
    return model
