#!/usr/bin/env python
"""
Quick Start Script for TRAIL Length Prediction
This script demonstrates the complete pipeline from data loading to inference.

Usage:
    python quick_start.py

Note: This script requires a GPU with at least 16GB memory for LLaMA-3-8B-Instruct.
"""

import os
import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_default_config, PredictorConfig, ModelConfig, DataConfig
from data_utils import AlpacaDataLoader, LLaMAEmbeddingExtractor, create_dataloaders
from model import LengthPredictor, compute_metrics, save_model
from train import Trainer
from bayesian_smoothing import BayesianSmoother, OnlineLengthPredictor
from evaluate import plot_layer_mae, plot_prediction_heatmap


def check_environment():
    """Check if environment is properly set up"""
    print("=" * 60)
    print("TRAIL Length Prediction - Quick Start")
    print("Based on: DON'T STOP ME NOW (ICLR 2025)")
    print("=" * 60)
    
    print(f"\nPython version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\nWARNING: CUDA not available. Training will be slow on CPU.")
        print("Recommend using a GPU with at least 16GB memory.\n")
    
    return True


def demo_bayesian_smoothing():
    """Demonstrate Bayesian smoothing mechanism"""
    print("\n" + "=" * 60)
    print("Demo 1: Bayesian Smoothing Mechanism")
    print("=" * 60)
    
    config = PredictorConfig()
    smoother = BayesianSmoother(config)
    
    print("\n--- Transition Matrix Structure ---")
    print("The transition matrix encodes the prior that remaining length")
    print("can only decrease as tokens are generated.")
    print(f"\nBin size: {smoother.bin_size:.1f} tokens")
    print(f"Stay probability: {1 - 1/smoother.bin_size:.4f}")
    print(f"Transition probability: {1/smoother.bin_size:.4f}")
    
    print("\nTransition Matrix (first 5x5):")
    print(smoother.transition_matrix[:5, :5].numpy().round(4))
    
    print("\n--- Simulating Token Generation ---")
    
    # Initial prediction: mostly in bin 4-6 (middle bins)
    initial_pred = torch.tensor([0.02, 0.05, 0.10, 0.15, 0.25, 0.20, 0.12, 0.06, 0.03, 0.02])
    state = smoother.initialize(initial_pred)
    
    print(f"Initial expected length: {smoother.get_expected_length(state):.2f} tokens")
    print(f"Initial probabilities: {state.current_prior.numpy().round(3)}")
    
    # Simulate generating tokens
    for i in [20, 50, 100]:
        # Simulate new prediction after generating i tokens
        # The prediction should shift toward smaller bins
        new_pred = torch.zeros(10)
        for j in range(10):
            # Simple simulation: shift distribution left
            src_idx = min(j + i // 30, 9)
            new_pred[j] = initial_pred[src_idx]
        new_pred = new_pred / new_pred.sum()
        
        state, expected_length = smoother.update(state, new_pred, num_steps=i if i == 20 else 30)
        print(f"\nAfter ~{i} tokens generated:")
        print(f"  Expected remaining length: {expected_length:.2f}")
        print(f"  Most likely bin: b{smoother.get_most_likely_bin(state) + 1}")


def demo_predictor_architecture():
    """Demonstrate the MLP predictor architecture"""
    print("\n" + "=" * 60)
    print("Demo 2: MLP Predictor Architecture")
    print("=" * 60)
    
    config = PredictorConfig()
    model = LengthPredictor(config)
    
    print("\n--- Model Architecture ---")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    # Demo forward pass
    print("\n--- Forward Pass Demo ---")
    batch_size = 4
    dummy_input = torch.randn(batch_size, config.input_dim)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {output['logits'].shape}")
    print(f"Output probs shape: {output['probs'].shape}")
    print(f"Expected lengths: {output['expected_length'].tolist()}")
    
    print("\n--- Bin Configuration ---")
    print(f"Number of bins: {config.num_bins}")
    print(f"Max output length: {config.max_length}")
    print(f"Bin boundaries: {config.get_bin_boundaries()}")
    print(f"Bin centers: {[f'{x:.1f}' for x in config.get_bin_centers()]}")


def demo_mini_training(num_samples: int = 100):
    """
    Demo training with synthetic data (no GPU required).
    For full training, use run.py with real data.
    """
    print("\n" + "=" * 60)
    print("Demo 3: Mini Training with Synthetic Data")
    print("=" * 60)
    
    config = PredictorConfig()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generate synthetic training data
    print(f"\nGenerating {num_samples} synthetic training samples...")
    
    # Create synthetic embeddings and labels
    # Simulate: shorter sequences have embeddings with certain patterns
    train_embeddings = []
    train_labels = []
    
    for _ in range(num_samples):
        # Random remaining length
        remaining = torch.randint(0, config.max_length, (1,)).item()
        bin_idx = config.length_to_bin(remaining)
        
        # Create embedding with signal correlated to remaining length
        embedding = torch.randn(config.input_dim)
        # Add signal: first 512 dims correlate with remaining length
        embedding[:512] += (remaining / config.max_length) * 2
        
        train_embeddings.append(embedding)
        train_labels.append(bin_idx)
    
    # Create dataset
    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, embeddings, labels, remaining_lengths):
            self.embeddings = embeddings
            self.labels = labels
            self.remaining_lengths = remaining_lengths
            
        def __len__(self):
            return len(self.embeddings)
        
        def __getitem__(self, idx):
            return {
                'embedding': self.embeddings[idx],
                'bin_label': torch.tensor(self.labels[idx], dtype=torch.long),
                'remaining_length': self.remaining_lengths[idx],
                'position': 0
            }
    
    remaining_lengths = [config.get_bin_centers()[l] for l in train_labels]
    
    dataset = SyntheticDataset(train_embeddings, train_labels, remaining_lengths)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    # Create model and trainer
    model = LengthPredictor(config)
    
    # Mini training config
    mini_config = PredictorConfig()
    mini_config.num_epochs = 5
    mini_config.learning_rate = 0.001
    
    trainer = Trainer(model, train_loader, val_loader, mini_config, device)
    
    print("\n--- Training for 5 epochs ---")
    history = trainer.train()
    
    print("\n--- Training Complete ---")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Final val MAE: {history['val_mae'][-1]:.2f}")
    print(f"Final val accuracy: {history['val_accuracy'][-1]:.4f}")


def show_usage_instructions():
    """Show instructions for full training with real data"""
    print("\n" + "=" * 60)
    print("Full Training Instructions")
    print("=" * 60)
    
    print("""
To run the full training pipeline with real LLaMA embeddings:

1. Install dependencies:
   pip install -r requirements.txt

2. Set up Hugging Face authentication (for LLaMA access):
   huggingface-cli login

3. Profile layers to find the best one (Figure 2 in paper):
   python run.py profile --num_samples 1000 --layers "10,11,12,13,14,15"

4. Train the predictor on the best layer:
   python run.py train --num_samples 1000 --target_layer 11

5. Evaluate the trained model:
   python run.py evaluate --model_path ./checkpoints/best_model_layer_11.pt

6. Run inference on a prompt:
   python run.py inference --model_path ./checkpoints/best_model_layer_11.pt \\
       --instruction "Explain quantum computing in simple terms"

Hardware Requirements:
- GPU with at least 16GB memory (A100, RTX 3090, etc.)
- 32GB system RAM recommended
- ~20GB disk space for model weights and embeddings

Expected Results (based on paper):
- Layer 11 typically provides the best predictions
- MAE with refinement: ~30-40 (compared to ~80 for BERT)
- Bayesian smoothing reduces MAE by ~10-20%
""")


def main():
    """Run all demos"""
    if not check_environment():
        return
    
    # Run demos
    demo_bayesian_smoothing()
    demo_predictor_architecture()
    demo_mini_training()
    show_usage_instructions()
    
    print("\n" + "=" * 60)
    print("Quick Start Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
