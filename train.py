"""
Training Script for TRAIL Length Prediction
Based on: DON'T STOP ME NOW: EMBEDDING BASED SCHEDULING FOR LLMS (ICLR 2025)

Training Configuration:
- Optimizer: AdamW (with weight decay for regularization)
- Learning Rate Schedule: Cosine Annealing (0.01 → 0)
- Batch Size: 32
- Epochs: 30
- Loss Function: CrossEntropyLoss
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np

from config import ModelConfig, PredictorConfig, DataConfig, get_default_config
from model import LengthPredictor, LengthPredictorEnsemble, compute_loss, compute_metrics, save_model
from model import EnhancedLengthPredictor, compute_enhanced_metrics, save_enhanced_model
from data_utils import (
    AlpacaDataLoader, 
    LLaMAEmbeddingExtractor, 
    EmbeddingDataset,
    create_dataloaders,
    EnhancedLLaMAExtractor,
    create_enhanced_dataloaders
)


class Trainer:
    """Trainer for Length Predictor"""
    
    def __init__(
        self,
        model: LengthPredictor,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: PredictorConfig,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer: AdamW as specified in paper
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler: Cosine Annealing
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=0
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_mae': [],
            'val_loss': [],
            'val_mae': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_mae = 0.0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            embeddings = batch['embedding'].to(self.device)
            labels = batch['bin_label'].to(self.device)
            remaining_lengths = batch['remaining_length'].float().to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(embeddings)
            
            # Compute loss
            loss = self.criterion(output['logits'], labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            with torch.no_grad():
                mae = torch.abs(output['expected_length'] - remaining_lengths).mean()
                
            total_loss += loss.item() * len(labels)
            total_mae += mae.item() * len(labels)
            total_samples += len(labels)
            
            pbar.set_postfix({
                'loss': f"{total_loss / total_samples:.4f}",
                'mae': f"{total_mae / total_samples:.2f}"
            })
        
        avg_loss = total_loss / total_samples
        avg_mae = total_mae / total_samples
        
        return avg_loss, avg_mae
    
    def validate(self) -> Tuple[float, float, float]:
        """Validate on validation set"""
        metrics = compute_metrics(self.model, self.val_loader, self.device)
        return metrics['loss'], metrics['mae'], metrics['accuracy']
    
    def train(self, save_dir: Optional[str] = None, layer_idx: Optional[int] = None):
        """
        Train the model for specified number of epochs.
        
        Args:
            save_dir: Directory to save checkpoints
            layer_idx: Layer index (for naming saved models)
        """
        best_val_mae = float('inf')
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Train
            train_loss, train_mae = self.train_epoch()
            
            # Validate
            val_loss, val_mae, val_accuracy = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f}")
            print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}, Val Acc: {val_accuracy:.4f}")
            
            # Save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                if save_dir:
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    if layer_idx is not None:
                        save_path = os.path.join(save_dir, f"best_model_layer_{layer_idx}.pt")
                    else:
                        save_path = os.path.join(save_dir, "best_model.pt")
                    save_model(self.model, save_path, self.config)
                    
        return self.history


def layer_selection_analysis(
    data: List[Dict],
    data_loader: AlpacaDataLoader,
    model_config: ModelConfig,
    predictor_config: PredictorConfig,
    data_config: DataConfig,
    layers_to_test: Optional[List[int]] = None,
    num_epochs: int = 10  # Fewer epochs for profiling
) -> Dict[int, Dict]:
    """
    Analyze different layers to find the best one for prediction.
    This corresponds to Figure 2 in the paper.
    
    Args:
        data: Dataset samples
        data_loader: AlpacaDataLoader instance
        model_config: Model configuration
        predictor_config: Predictor configuration
        data_config: Data configuration
        layers_to_test: List of layer indices to test (default: all 32 layers)
        num_epochs: Number of epochs for each layer (default: 10 for faster profiling)
        
    Returns:
        Dict mapping layer_idx to results (MAE, accuracy, etc.)
    """
    if layers_to_test is None:
        layers_to_test = list(range(model_config.num_layers))
    
    # Extract embeddings
    extractor = LLaMAEmbeddingExtractor(model_config, predictor_config)
    extractor.load_model()
    
    results = {}
    device = model_config.device
    
    for layer_idx in layers_to_test:
        print(f"\n{'='*50}")
        print(f"Testing Layer {layer_idx}")
        print(f"{'='*50}")
        
        # Extract embeddings for this layer
        embeddings = extractor.extract_all_embeddings(
            data=data,
            layer_indices=[layer_idx],
            data_loader=data_loader
        )
        
        if not embeddings or not embeddings.get(layer_idx):
            print(f"No embeddings extracted for layer {layer_idx}")
            continue
            
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            embeddings[layer_idx],
            predictor_config,
            data_config
        )
        
        # Create and train model
        model = LengthPredictor(predictor_config)
        
        # Use fewer epochs for profiling
        temp_config = PredictorConfig(
            input_dim=predictor_config.input_dim,
            hidden_dim=predictor_config.hidden_dim,
            num_bins=predictor_config.num_bins,
            max_length=predictor_config.max_length,
            batch_size=predictor_config.batch_size,
            num_epochs=num_epochs,
            learning_rate=predictor_config.learning_rate,
            weight_decay=predictor_config.weight_decay
        )
        
        trainer = Trainer(model, train_loader, val_loader, temp_config, device)
        history = trainer.train()
        
        # Evaluate on test set
        test_metrics = compute_metrics(model, test_loader, device)
        
        results[layer_idx] = {
            'test_mae': test_metrics['mae'],
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'history': history
        }
        
        print(f"\nLayer {layer_idx} Results:")
        print(f"  Test MAE: {test_metrics['mae']:.2f}")
        print(f"  Test Loss: {test_metrics['loss']:.4f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Find best layer
    best_layer = min(results.keys(), key=lambda x: results[x]['test_mae'])
    print(f"\n{'='*50}")
    print(f"Best Layer: {best_layer} with MAE: {results[best_layer]['test_mae']:.2f}")
    print(f"{'='*50}")
    
    return results


def train_focused_predictor(
    data: List[Dict],
    data_loader: AlpacaDataLoader,
    model_config: ModelConfig,
    predictor_config: PredictorConfig,
    data_config: DataConfig,
    target_layer: int = 11,
    save_dir: str = "./checkpoints"
) -> Tuple[LengthPredictor, Dict]:
    """
    Train predictor on focused layer (after layer selection).
    
    Args:
        data: Dataset samples
        data_loader: AlpacaDataLoader instance
        model_config: Model configuration
        predictor_config: Predictor configuration
        data_config: Data configuration
        target_layer: Layer to use for prediction (default: 11)
        save_dir: Directory to save model
        
    Returns:
        Trained model and training history
    """
    print(f"\n{'='*50}")
    print(f"Training Focused Predictor on Layer {target_layer}")
    print(f"{'='*50}")
    
    # Extract embeddings
    extractor = LLaMAEmbeddingExtractor(model_config, predictor_config)
    extractor.load_model()
    
    embeddings = extractor.extract_all_embeddings(
        data=data,
        layer_indices=[target_layer],
        data_loader=data_loader,
        save_dir=data_config.embeddings_cache_dir
    )
    
    print(f"Total training pairs: {len(embeddings[target_layer])}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        embeddings[target_layer],
        predictor_config,
        data_config
    )
    
    # Create and train model
    model = LengthPredictor(predictor_config)
    device = model_config.device
    
    trainer = Trainer(model, train_loader, val_loader, predictor_config, device)
    history = trainer.train(save_dir=save_dir, layer_idx=target_layer)
    
    # Final evaluation
    test_metrics = compute_metrics(model, test_loader, device)
    
    print(f"\nFinal Test Results:")
    print(f"  MAE: {test_metrics['mae']:.2f}")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    
    return model, history


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train TRAIL Length Predictor")
    parser.add_argument("--mode", type=str, default="train", 
                       choices=["profile", "train"],
                       help="Mode: 'profile' for layer selection, 'train' for focused training")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="Number of samples to use")
    parser.add_argument("--target_layer", type=int, default=11,
                       help="Target layer for training (default: 11)")
    parser.add_argument("--layers", type=str, default=None,
                       help="Comma-separated layer indices for profiling (e.g., '10,11,12')")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                       help="Directory to save models")
    
    args = parser.parse_args()
    
    # Load configs
    configs = get_default_config()
    model_config = configs["model"]
    predictor_config = configs["predictor"]
    data_config = configs["data"]
    
    # Load data
    data_loader = AlpacaDataLoader(data_config)
    data = data_loader.load_dataset(num_samples=args.num_samples)
    
    if args.mode == "profile":
        # Layer selection analysis
        if args.layers:
            layers_to_test = [int(x) for x in args.layers.split(",")]
        else:
            layers_to_test = list(range(32))  # All layers
            
        results = layer_selection_analysis(
            data=data,
            data_loader=data_loader,
            model_config=model_config,
            predictor_config=predictor_config,
            data_config=data_config,
            layers_to_test=layers_to_test
        )
        
        # Save results
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        results_path = os.path.join(args.save_dir, "layer_analysis_results.json")
        
        # Convert to JSON-serializable format
        json_results = {}
        for layer_idx, layer_results in results.items():
            json_results[str(layer_idx)] = {
                'test_mae': float(layer_results['test_mae']),
                'test_loss': float(layer_results['test_loss']),
                'test_accuracy': float(layer_results['test_accuracy'])
            }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"Results saved to {results_path}")
        
    else:
        # Focused training on target layer
        model, history = train_focused_predictor(
            data=data,
            data_loader=data_loader,
            model_config=model_config,
            predictor_config=predictor_config,
            data_config=data_config,
            target_layer=args.target_layer,
            save_dir=args.save_dir
        )
        
        # Save history
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        history_path = os.path.join(args.save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()


# ============================================================================
# Enhanced Training with Entropy Features
# ============================================================================

class EnhancedTrainer:
    """
    Trainer for Enhanced Length Predictor with entropy features.
    Supports ablation studies and various fusion methods.
    """
    
    def __init__(
        self,
        model: EnhancedLengthPredictor,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config,
        device: str = 'cpu',
        use_entropy: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_entropy = use_entropy
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=0
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_mae': [],
            'train_rmse': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_accuracy': [],
            'val_top3_accuracy': [],
            'learning_rate': []
        }
        
    def train_epoch(self) -> Tuple[float, float, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_mae = 0.0
        total_squared_error = 0.0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            embeddings = batch['embedding'].to(self.device)
            labels = batch['bin_label'].to(self.device)
            remaining_lengths = batch['remaining_length'].float().to(self.device)
            
            entropy_features = None
            if self.use_entropy and 'entropy_features' in batch:
                entropy_features = batch['entropy_features'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(embeddings, entropy_features)
            
            # Compute loss
            loss = self.criterion(output['logits'], labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            with torch.no_grad():
                errors = output['expected_length'] - remaining_lengths
                mae = torch.abs(errors).mean()
                squared_error = (errors ** 2).sum()
                
            total_loss += loss.item() * len(labels)
            total_mae += mae.item() * len(labels)
            total_squared_error += squared_error.item()
            total_samples += len(labels)
            
            pbar.set_postfix({
                'loss': f"{total_loss / total_samples:.4f}",
                'mae': f"{total_mae / total_samples:.2f}"
            })
        
        avg_loss = total_loss / total_samples
        avg_mae = total_mae / total_samples
        rmse = np.sqrt(total_squared_error / total_samples)
        
        return avg_loss, avg_mae, rmse
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        return compute_enhanced_metrics(
            self.model, self.val_loader, self.device, self.use_entropy
        )
    
    def train(self, save_dir: Optional[str] = None, experiment_name: str = "enhanced"):
        """
        Train the enhanced model.
        
        Args:
            save_dir: Directory to save checkpoints
            experiment_name: Name for the experiment (used in saved file names)
        """
        best_val_mae = float('inf')
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Train
            train_loss, train_mae, train_rmse = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_mae'].append(train_mae)
            self.history['train_rmse'].append(train_rmse)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_top3_accuracy'].append(val_metrics['top3_accuracy'])
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
            print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f}, Train RMSE: {train_rmse:.2f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val MAE: {val_metrics['mae']:.2f}, "
                  f"Val RMSE: {val_metrics['rmse']:.2f}")
            print(f"Val Acc: {val_metrics['accuracy']:.4f}, Val Top-3 Acc: {val_metrics['top3_accuracy']:.4f}")
            
            # Save best model
            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                if save_dir:
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    save_path = os.path.join(save_dir, f"best_{experiment_name}_model.pt")
                    save_enhanced_model(self.model, save_path, self.config)
                    
        return self.history


def train_enhanced_predictor(
    data: List[Dict],
    data_loader: AlpacaDataLoader,
    model_config,
    enhanced_config,
    entropy_config,
    data_config,
    target_layer: int = 11,
    save_dir: str = "./checkpoints",
    num_initial_tokens: int = 15
) -> Tuple[EnhancedLengthPredictor, Dict]:
    """
    Train enhanced predictor with entropy features.
    
    Args:
        data: Dataset samples
        data_loader: AlpacaDataLoader instance
        model_config: Model configuration
        enhanced_config: EnhancedPredictorConfig instance
        entropy_config: EntropyConfig instance
        data_config: Data configuration
        target_layer: Layer to extract embeddings from
        save_dir: Directory to save model
        num_initial_tokens: K value for entropy collection
        
    Returns:
        Trained model and training history
    """
    print(f"\n{'='*60}")
    print(f"Training Enhanced Predictor with Entropy Features")
    print(f"Target Layer: {target_layer}, K: {num_initial_tokens}")
    print(f"Fusion Method: {enhanced_config.fusion_method}")
    print(f"{'='*60}")
    
    device = model_config.device
    
    # Create extractor
    from config import PredictorConfig
    predictor_config = PredictorConfig()  # For compatibility
    
    extractor = EnhancedLLaMAExtractor(model_config, predictor_config, entropy_config)
    
    # Check for cached data
    cache_path = os.path.join(data_config.embeddings_cache_dir, f"layer_{target_layer}_enhanced.pt")
    if os.path.exists(cache_path):
        print(f"Loading cached enhanced embeddings from {cache_path}")
        samples = extractor.load_enhanced(data_config.embeddings_cache_dir, [target_layer])
    else:
        # Extract embeddings with entropy features
        extractor.load_model()
        samples = extractor.extract_all_enhanced(
            data=data,
            layer_indices=[target_layer],
            data_loader=data_loader,
            save_dir=data_config.embeddings_cache_dir,
            num_initial_tokens=num_initial_tokens
        )
    
    if not samples or target_layer not in samples or not samples[target_layer]:
        raise ValueError(f"No samples extracted for layer {target_layer}")
    
    print(f"Total enhanced samples: {len(samples[target_layer])}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_enhanced_dataloaders(
        samples[target_layer],
        enhanced_config,
        data_config,
        include_entropy=True
    )
    
    # Create model
    model = EnhancedLengthPredictor(enhanced_config)
    
    # Train
    trainer = EnhancedTrainer(
        model, train_loader, val_loader, enhanced_config, device, use_entropy=True
    )
    history = trainer.train(save_dir=save_dir, experiment_name="enhanced")
    
    # Final evaluation
    test_metrics = compute_enhanced_metrics(model, test_loader, device, use_entropy=True)
    
    print(f"\nFinal Test Results:")
    print(f"  MAE: {test_metrics['mae']:.2f}")
    print(f"  RMSE: {test_metrics['rmse']:.2f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Top-3 Accuracy: {test_metrics['top3_accuracy']:.4f}")
    
    return model, history


def run_ablation_study(
    data: List[Dict],
    data_loader: AlpacaDataLoader,
    model_config,
    enhanced_config,
    entropy_config,
    data_config,
    experiment_config,
    target_layer: int = 11,
    save_dir: str = "./results/ablation"
) -> Dict[str, Dict]:
    """
    Run ablation studies to evaluate contribution of different features.
    
    Studies:
    1. Embedding only (baseline)
    2. Entropy only (attention + logits)
    3. Attention entropy only
    4. Logits entropy only
    5. Full model (embedding + all entropy)
    
    Returns:
        Dict mapping study name to results
    """
    print(f"\n{'='*60}")
    print("Running Ablation Studies")
    print(f"{'='*60}")
    
    device = model_config.device
    results = {}
    
    # First, extract all data
    from config import PredictorConfig
    predictor_config = PredictorConfig()
    
    extractor = EnhancedLLaMAExtractor(model_config, predictor_config, entropy_config)
    extractor.load_model()
    
    samples = extractor.extract_all_enhanced(
        data=data,
        layer_indices=[target_layer],
        data_loader=data_loader,
        save_dir=data_config.embeddings_cache_dir
    )
    
    if not samples[target_layer]:
        raise ValueError("No samples extracted")
    
    train_loader, val_loader, test_loader = create_enhanced_dataloaders(
        samples[target_layer],
        enhanced_config,
        data_config,
        include_entropy=True
    )
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Study 1: Embedding only
    print("\n--- Study 1: Embedding Only (Baseline) ---")
    config_emb = type(enhanced_config)(
        embedding_dim=enhanced_config.embedding_dim,
        embedding_hidden_dim=enhanced_config.embedding_hidden_dim,
        entropy_feature_dim=enhanced_config.entropy_feature_dim,
        entropy_hidden_dim=enhanced_config.entropy_hidden_dim,
        fusion_hidden_dim=enhanced_config.fusion_hidden_dim,
        num_bins=enhanced_config.num_bins,
        max_length=enhanced_config.max_length,
        use_embedding_features=True,
        use_entropy_features=False,
        batch_size=enhanced_config.batch_size,
        num_epochs=enhanced_config.num_epochs,
        learning_rate=enhanced_config.learning_rate,
        weight_decay=enhanced_config.weight_decay,
        dropout_rate=enhanced_config.dropout_rate
    )
    
    model_emb = EnhancedLengthPredictor(config_emb)
    trainer_emb = EnhancedTrainer(model_emb, train_loader, val_loader, config_emb, device, use_entropy=False)
    history_emb = trainer_emb.train(save_dir=save_dir, experiment_name="embedding_only")
    results["embedding_only"] = compute_enhanced_metrics(model_emb, test_loader, device, use_entropy=False)
    results["embedding_only"]["history"] = history_emb
    
    # Study 2: Entropy only
    print("\n--- Study 2: Entropy Features Only ---")
    config_ent = type(enhanced_config)(
        embedding_dim=enhanced_config.embedding_dim,
        embedding_hidden_dim=enhanced_config.embedding_hidden_dim,
        entropy_feature_dim=enhanced_config.entropy_feature_dim,
        entropy_hidden_dim=enhanced_config.entropy_hidden_dim,
        fusion_hidden_dim=enhanced_config.fusion_hidden_dim,
        num_bins=enhanced_config.num_bins,
        max_length=enhanced_config.max_length,
        use_embedding_features=False,
        use_entropy_features=True,
        batch_size=enhanced_config.batch_size,
        num_epochs=enhanced_config.num_epochs,
        learning_rate=enhanced_config.learning_rate,
        weight_decay=enhanced_config.weight_decay,
        dropout_rate=enhanced_config.dropout_rate
    )
    
    model_ent = EnhancedLengthPredictor(config_ent)
    trainer_ent = EnhancedTrainer(model_ent, train_loader, val_loader, config_ent, device, use_entropy=True)
    history_ent = trainer_ent.train(save_dir=save_dir, experiment_name="entropy_only")
    results["entropy_only"] = compute_enhanced_metrics(model_ent, test_loader, device, use_entropy=True)
    results["entropy_only"]["history"] = history_ent
    
    # Study 3: Full model
    print("\n--- Study 3: Full Model (Embedding + Entropy) ---")
    model_full = EnhancedLengthPredictor(enhanced_config)
    trainer_full = EnhancedTrainer(model_full, train_loader, val_loader, enhanced_config, device, use_entropy=True)
    history_full = trainer_full.train(save_dir=save_dir, experiment_name="full_model")
    results["full_model"] = compute_enhanced_metrics(model_full, test_loader, device, use_entropy=True)
    results["full_model"]["history"] = history_full
    
    # Print summary
    print(f"\n{'='*60}")
    print("Ablation Study Results Summary")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'Acc':>10} {'Top-3':>10}")
    print("-" * 60)
    for name, metrics in results.items():
        if name != "history":
            print(f"{name:<20} {metrics['mae']:>10.2f} {metrics['rmse']:>10.2f} "
                  f"{metrics['accuracy']:>10.4f} {metrics['top3_accuracy']:>10.4f}")
    
    # Save results
    results_path = os.path.join(save_dir, "ablation_results.json")
    json_results = {}
    for name, metrics in results.items():
        json_results[name] = {
            'mae': float(metrics['mae']),
            'rmse': float(metrics['rmse']),
            'accuracy': float(metrics['accuracy']),
            'top3_accuracy': float(metrics['top3_accuracy'])
        }
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return results


def analyze_k_values(
    data: List[Dict],
    data_loader: AlpacaDataLoader,
    model_config,
    enhanced_config,
    entropy_config,
    data_config,
    k_values: List[int] = [5, 10, 15, 20, 25],
    target_layer: int = 11,
    save_dir: str = "./results/k_analysis"
) -> Dict[int, Dict]:
    """
    Analyze the effect of different K values (number of initial tokens for entropy).
    
    Args:
        k_values: List of K values to test
        
    Returns:
        Dict mapping K value to results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing K Values: {k_values}")
    print(f"{'='*60}")
    
    device = model_config.device
    results = {}
    
    from config import PredictorConfig
    predictor_config = PredictorConfig()
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    for k in k_values:
        print(f"\n--- Testing K = {k} ---")
        
        # Extract with specific K
        extractor = EnhancedLLaMAExtractor(model_config, predictor_config, entropy_config)
        extractor.load_model()
        
        samples = extractor.extract_all_enhanced(
            data=data,
            layer_indices=[target_layer],
            data_loader=data_loader,
            num_initial_tokens=k
        )
        
        if not samples[target_layer]:
            print(f"No samples for K={k}, skipping")
            continue
        
        train_loader, val_loader, test_loader = create_enhanced_dataloaders(
            samples[target_layer],
            enhanced_config,
            data_config,
            include_entropy=True
        )
        
        model = EnhancedLengthPredictor(enhanced_config)
        trainer = EnhancedTrainer(model, train_loader, val_loader, enhanced_config, device, use_entropy=True)
        history = trainer.train(save_dir=save_dir, experiment_name=f"k_{k}")
        
        results[k] = compute_enhanced_metrics(model, test_loader, device, use_entropy=True)
        results[k]["history"] = history
        
        print(f"K={k}: MAE={results[k]['mae']:.2f}, RMSE={results[k]['rmse']:.2f}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("K-Value Analysis Summary")
    print(f"{'='*60}")
    print(f"{'K':>5} {'MAE':>10} {'RMSE':>10} {'Accuracy':>10}")
    print("-" * 40)
    for k, metrics in sorted(results.items()):
        if k != "history":
            print(f"{k:>5} {metrics['mae']:>10.2f} {metrics['rmse']:>10.2f} {metrics['accuracy']:>10.4f}")
    
    # Save results
    results_path = os.path.join(save_dir, "k_analysis_results.json")
    json_results = {}
    for k, metrics in results.items():
        json_results[str(k)] = {
            'mae': float(metrics['mae']),
            'rmse': float(metrics['rmse']),
            'accuracy': float(metrics['accuracy'])
        }
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    return results
