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


# ============================================================================
# Enhanced Models with Entropy Features
# ============================================================================

class EntropyEncoder(nn.Module):
    """
    Encoder for entropy features.
    Can optionally use time-series encoder (CNN/RNN) for entropy sequences.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Simple MLP encoder for aggregated entropy features
        self.fc1 = nn.Linear(config.entropy_feature_dim, config.entropy_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc2 = nn.Linear(config.entropy_hidden_dim, config.entropy_hidden_dim)
        
    def forward(self, entropy_features: torch.Tensor) -> torch.Tensor:
        """
        Encode entropy features.
        
        Args:
            entropy_features: [batch_size, entropy_feature_dim]
            
        Returns:
            Encoded features [batch_size, entropy_hidden_dim]
        """
        x = self.fc1(entropy_features)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EmbeddingEncoder(nn.Module):
    """
    Encoder for LLaMA embeddings (same as original LengthPredictor hidden layer).
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.fc1 = nn.Linear(config.embedding_dim, config.embedding_hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode embeddings.
        
        Args:
            embeddings: [batch_size, embedding_dim]
            
        Returns:
            Encoded features [batch_size, embedding_hidden_dim]
        """
        x = self.fc1(embeddings)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ConcatFusion(nn.Module):
    """Simple concatenation fusion"""
    
    def __init__(self, embedding_dim: int, entropy_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim + entropy_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, embedding_features: torch.Tensor, entropy_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([embedding_features, entropy_features], dim=-1)
        return self.relu(self.fc(combined))


class AttentionFusion(nn.Module):
    """Attention-based fusion between embedding and entropy features"""
    
    def __init__(self, embedding_dim: int, entropy_dim: int, output_dim: int):
        super().__init__()
        self.query = nn.Linear(embedding_dim, output_dim)
        self.key = nn.Linear(entropy_dim, output_dim)
        self.value = nn.Linear(entropy_dim, output_dim)
        self.output = nn.Linear(output_dim * 2, output_dim)
        self.scale = output_dim ** 0.5
        
    def forward(self, embedding_features: torch.Tensor, entropy_features: torch.Tensor) -> torch.Tensor:
        # embedding_features: [batch, embedding_dim]
        # entropy_features: [batch, entropy_dim]
        
        q = self.query(embedding_features)  # [batch, output_dim]
        k = self.key(entropy_features)      # [batch, output_dim]
        v = self.value(entropy_features)    # [batch, output_dim]
        
        # Compute attention score
        attn = torch.sum(q * k, dim=-1, keepdim=True) / self.scale  # [batch, 1]
        attn = torch.sigmoid(attn)  # Use sigmoid for gating
        
        # Apply attention to value
        attended = attn * v  # [batch, output_dim]
        
        # Combine with original embedding
        combined = torch.cat([q, attended], dim=-1)  # [batch, output_dim * 2]
        return self.output(combined)


class GatingFusion(nn.Module):
    """Gating mechanism for feature fusion"""
    
    def __init__(self, embedding_dim: int, entropy_dim: int, output_dim: int):
        super().__init__()
        total_dim = embedding_dim + entropy_dim
        self.gate = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.Sigmoid()
        )
        self.transform = nn.Linear(total_dim, output_dim)
        
    def forward(self, embedding_features: torch.Tensor, entropy_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([embedding_features, entropy_features], dim=-1)
        gate = self.gate(combined)
        transformed = self.transform(combined)
        return gate * transformed


class BilinearFusion(nn.Module):
    """Bilinear interaction for feature fusion"""
    
    def __init__(self, embedding_dim: int, entropy_dim: int, output_dim: int):
        super().__init__()
        self.bilinear = nn.Bilinear(embedding_dim, entropy_dim, output_dim)
        self.fc_emb = nn.Linear(embedding_dim, output_dim)
        self.fc_ent = nn.Linear(entropy_dim, output_dim)
        
    def forward(self, embedding_features: torch.Tensor, entropy_features: torch.Tensor) -> torch.Tensor:
        bilinear_out = self.bilinear(embedding_features, entropy_features)
        emb_out = self.fc_emb(embedding_features)
        ent_out = self.fc_ent(entropy_features)
        return bilinear_out + emb_out + ent_out


class EnhancedLengthPredictor(nn.Module):
    """
    Enhanced Length Predictor with multi-modal feature fusion.
    
    Supports:
    - Embedding features only (baseline)
    - Entropy features only (ablation)
    - Combined features with various fusion methods
    
    Fusion methods:
    - concat: Simple concatenation
    - attention: Attention-based fusion
    - gating: Gating mechanism
    - bilinear: Bilinear interaction
    """
    
    def __init__(self, config):
        """
        Args:
            config: EnhancedPredictorConfig instance
        """
        super().__init__()
        self.config = config
        
        # Feature encoders
        if config.use_embedding_features:
            self.embedding_encoder = EmbeddingEncoder(config)
            emb_out_dim = config.embedding_hidden_dim
        else:
            self.embedding_encoder = None
            emb_out_dim = 0
            
        if config.use_entropy_features:
            self.entropy_encoder = EntropyEncoder(config)
            ent_out_dim = config.entropy_hidden_dim
        else:
            self.entropy_encoder = None
            ent_out_dim = 0
        
        # Fusion layer
        if config.use_embedding_features and config.use_entropy_features:
            if config.fusion_method == "concat":
                self.fusion = ConcatFusion(emb_out_dim, ent_out_dim, config.fusion_hidden_dim)
            elif config.fusion_method == "attention":
                self.fusion = AttentionFusion(emb_out_dim, ent_out_dim, config.fusion_hidden_dim)
            elif config.fusion_method == "gating":
                self.fusion = GatingFusion(emb_out_dim, ent_out_dim, config.fusion_hidden_dim)
            elif config.fusion_method == "bilinear":
                self.fusion = BilinearFusion(emb_out_dim, ent_out_dim, config.fusion_hidden_dim)
            else:
                self.fusion = ConcatFusion(emb_out_dim, ent_out_dim, config.fusion_hidden_dim)
            fusion_out_dim = config.fusion_hidden_dim
        elif config.use_embedding_features:
            self.fusion = None
            fusion_out_dim = emb_out_dim
        elif config.use_entropy_features:
            self.fusion = None
            fusion_out_dim = ent_out_dim
        else:
            raise ValueError("At least one of embedding or entropy features must be enabled")
        
        # Output layer
        self.dropout = nn.Dropout(config.dropout_rate)
        self.output_fc = nn.Linear(fusion_out_dim, config.num_bins)
        
        # Bin centers for expected length calculation
        self.register_buffer(
            'bin_centers',
            torch.tensor(config.get_bin_centers(), dtype=torch.float32)
        )
        
    def forward(
        self,
        embeddings: Optional[torch.Tensor] = None,
        entropy_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the enhanced predictor.
        
        Args:
            embeddings: LLaMA embeddings [batch_size, embedding_dim]
            entropy_features: Entropy features [batch_size, entropy_feature_dim]
            
        Returns:
            Dict containing logits, probs, and expected_length
        """
        # Encode features
        if self.config.use_embedding_features and embeddings is not None:
            emb_encoded = self.embedding_encoder(embeddings)
        else:
            emb_encoded = None
            
        if self.config.use_entropy_features and entropy_features is not None:
            ent_encoded = self.entropy_encoder(entropy_features)
        else:
            ent_encoded = None
        
        # Fuse features
        if emb_encoded is not None and ent_encoded is not None:
            fused = self.fusion(emb_encoded, ent_encoded)
        elif emb_encoded is not None:
            fused = emb_encoded
        elif ent_encoded is not None:
            fused = ent_encoded
        else:
            raise ValueError("No features provided")
        
        # Output prediction
        fused = self.dropout(fused)
        logits = self.output_fc(fused)
        
        probs = F.softmax(logits, dim=-1)
        expected_length = (probs * self.bin_centers).sum(dim=-1)
        
        return {
            'logits': logits,
            'probs': probs,
            'expected_length': expected_length
        }
    
    def predict_bin(self, embeddings=None, entropy_features=None) -> torch.Tensor:
        """Predict the most likely bin"""
        output = self.forward(embeddings, entropy_features)
        return output['probs'].argmax(dim=-1)
    
    def predict_length(self, embeddings=None, entropy_features=None) -> torch.Tensor:
        """Predict expected length"""
        output = self.forward(embeddings, entropy_features)
        return output['expected_length']


class SequenceEntropyEncoder(nn.Module):
    """
    Encoder for entropy time-series using CNN or RNN.
    Processes raw entropy sequences instead of aggregated features.
    """
    
    def __init__(self, config, encoder_type: str = "cnn"):
        super().__init__()
        self.config = config
        self.encoder_type = encoder_type
        
        if encoder_type == "cnn":
            # 1D CNN for time-series
            self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)  # 2 channels: attn and logits
            self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(64, config.sequence_hidden_dim)
        elif encoder_type in ["lstm", "gru"]:
            rnn_class = nn.LSTM if encoder_type == "lstm" else nn.GRU
            self.rnn = rnn_class(
                input_size=2,  # attn and logits entropy
                hidden_size=config.sequence_hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            self.fc = nn.Linear(config.sequence_hidden_dim * 2, config.sequence_hidden_dim)
        
        self.relu = nn.ReLU()
        
    def forward(self, attn_seq: torch.Tensor, logits_seq: torch.Tensor) -> torch.Tensor:
        """
        Encode entropy sequences.
        
        Args:
            attn_seq: Attention entropy sequence [batch, seq_len]
            logits_seq: Logits entropy sequence [batch, seq_len]
            
        Returns:
            Encoded features [batch, sequence_hidden_dim]
        """
        # Stack into [batch, 2, seq_len] for CNN or [batch, seq_len, 2] for RNN
        if self.encoder_type == "cnn":
            x = torch.stack([attn_seq, logits_seq], dim=1)  # [batch, 2, seq_len]
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)  # [batch, 64]
            x = self.fc(x)
        else:
            x = torch.stack([attn_seq, logits_seq], dim=-1)  # [batch, seq_len, 2]
            x, _ = self.rnn(x)
            x = x[:, -1, :]  # Take last hidden state
            x = self.fc(x)
            
        return self.relu(x)


def compute_enhanced_metrics(
    model: EnhancedLengthPredictor,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    use_entropy: bool = True
) -> Dict[str, float]:
    """
    Compute evaluation metrics for enhanced model.
    """
    model.eval()
    
    total_loss = 0.0
    total_mae = 0.0
    total_correct = 0
    total_top3_correct = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch['embedding'].to(device)
            labels = batch['bin_label'].to(device)
            remaining_lengths = batch['remaining_length'].float().to(device)
            
            entropy_features = None
            if use_entropy and 'entropy_features' in batch:
                entropy_features = batch['entropy_features'].to(device)
            
            # Forward pass
            output = model(embeddings, entropy_features)
            
            # Compute loss
            loss = F.cross_entropy(output['logits'], labels)
            total_loss += loss.item() * len(labels)
            
            # Compute MAE
            mae = torch.abs(output['expected_length'] - remaining_lengths).mean()
            total_mae += mae.item() * len(labels)
            
            # Compute Top-1 accuracy
            predicted_bins = output['logits'].argmax(dim=-1)
            total_correct += (predicted_bins == labels).sum().item()
            
            # Compute Top-3 accuracy
            top3_preds = output['logits'].topk(3, dim=-1).indices
            for i, label in enumerate(labels):
                if label in top3_preds[i]:
                    total_top3_correct += 1
            
            total_samples += len(labels)
            
            all_predictions.extend(output['expected_length'].cpu().tolist())
            all_targets.extend(remaining_lengths.cpu().tolist())
    
    # Compute RMSE
    predictions_arr = torch.tensor(all_predictions)
    targets_arr = torch.tensor(all_targets)
    rmse = torch.sqrt(((predictions_arr - targets_arr) ** 2).mean()).item()
    
    return {
        'loss': total_loss / total_samples,
        'mae': total_mae / total_samples,
        'rmse': rmse,
        'accuracy': total_correct / total_samples,
        'top3_accuracy': total_top3_correct / total_samples
    }


def save_enhanced_model(model: EnhancedLengthPredictor, path: str, config=None):
    """Save enhanced model checkpoint"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_type': 'enhanced'
    }
    torch.save(checkpoint, path)
    print(f"Enhanced model saved to {path}")


def load_enhanced_model(path: str, config=None, device: str = 'cpu') -> EnhancedLengthPredictor:
    """Load enhanced model from checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    
    if config is None:
        config = checkpoint.get('config')
        
    if config is None:
        raise ValueError("Config not found in checkpoint and not provided")
    
    model = EnhancedLengthPredictor(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Enhanced model loaded from {path}")
    return model
