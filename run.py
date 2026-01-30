"""
TRAIL Length Prediction - Main Entry Point
Based on: DON'T STOP ME NOW: EMBEDDING BASED SCHEDULING FOR LLMS (ICLR 2025)

This script provides a unified interface for:
1. Layer profiling (finding the best layer for prediction)
2. Training the length predictor
3. Evaluating predictions with Bayesian smoothing
4. Running inference on new prompts
"""

import os
import argparse
import torch
import json
from pathlib import Path
from typing import List, Optional

from config import get_default_config, ModelConfig, PredictorConfig, DataConfig
from model import LengthPredictor, load_model
from data_utils import AlpacaDataLoader, LLaMAEmbeddingExtractor, create_dataloaders
from bayesian_smoothing import BayesianSmoother, OnlineLengthPredictor
from train import layer_selection_analysis, train_focused_predictor
from evaluate import (
    full_evaluation, 
    plot_layer_mae, 
    plot_training_curves,
    compute_output_length_distribution,
    plot_output_length_cdf
)


def profile_layers(args):
    """
    Profile different layers to find the best one for prediction.
    Corresponds to Figure 2 in the paper.
    """
    configs = get_default_config()
    model_config = configs["model"]
    predictor_config = configs["predictor"]
    data_config = configs["data"]
    
    # Parse layers to test
    if args.layers:
        layers_to_test = [int(x) for x in args.layers.split(",")]
    else:
        layers_to_test = list(range(32))  # All layers
    
    # Load data
    data_loader = AlpacaDataLoader(data_config)
    data = data_loader.load_dataset(num_samples=args.num_samples)
    
    # Run layer selection analysis
    results = layer_selection_analysis(
        data=data,
        data_loader=data_loader,
        model_config=model_config,
        predictor_config=predictor_config,
        data_config=data_config,
        layers_to_test=layers_to_test,
        num_epochs=args.profile_epochs
    )
    
    # Save results
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    results_path = os.path.join(args.save_dir, "layer_analysis_results.json")
    
    json_results = {}
    for layer_idx, layer_results in results.items():
        json_results[str(layer_idx)] = {
            'test_mae': float(layer_results['test_mae']),
            'test_loss': float(layer_results['test_loss']),
            'test_accuracy': float(layer_results['test_accuracy'])
        }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Plot results
    plot_layer_mae(
        results,
        save_path=os.path.join(args.save_dir, "layer_mae_analysis.png")
    )
    
    print(f"\nResults saved to {args.save_dir}")


def train(args):
    """
    Train the length predictor on the selected layer.
    """
    configs = get_default_config()
    model_config = configs["model"]
    predictor_config = configs["predictor"]
    data_config = configs["data"]
    
    # Load data
    data_loader = AlpacaDataLoader(data_config)
    data = data_loader.load_dataset(num_samples=args.num_samples)
    
    # Train model
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
    history_path = os.path.join(args.save_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    plot_training_curves(
        history,
        save_path=os.path.join(args.save_dir, "training_curves.png")
    )
    
    print(f"\nModel and history saved to {args.save_dir}")


def evaluate(args):
    """
    Evaluate trained model with Bayesian smoothing.
    """
    configs = get_default_config()
    model_config = configs["model"]
    predictor_config = configs["predictor"]
    data_config = configs["data"]
    
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
    
    print("\n=== Evaluation Complete ===")
    print(f"Results saved to {args.results_dir}")


def inference(args):
    """
    Run inference on a single prompt to demonstrate the prediction pipeline.
    """
    configs = get_default_config()
    model_config = configs["model"]
    predictor_config = configs["predictor"]
    
    # Load predictor
    predictor = load_model(args.model_path, predictor_config, model_config.device)
    
    # Create smoother
    smoother = BayesianSmoother(predictor_config)
    
    # Create online predictor
    online_predictor = OnlineLengthPredictor(predictor, smoother, model_config.device)
    
    # Load LLaMA for embedding extraction
    extractor = LLaMAEmbeddingExtractor(model_config, predictor_config)
    extractor.load_model()
    
    # Format prompt
    from data_utils import AlpacaDataLoader
    data_config = configs["data"]
    data_loader = AlpacaDataLoader(data_config)
    
    prompt = data_loader.format_prompt(args.instruction, args.input_text or "")
    
    print(f"\n=== Inference Demo ===")
    print(f"Instruction: {args.instruction}")
    if args.input_text:
        print(f"Input: {args.input_text}")
    
    # Extract initial embedding (prefill)
    extractor._register_hooks([args.target_layer])
    
    with torch.no_grad():
        tokens = extractor.tokenizer.encode(prompt, return_tensors="pt").to(extractor.model.device)
        _ = extractor.model(tokens)
        
        # Get average embedding
        layer_output = extractor.layer_outputs[args.target_layer]
        initial_embedding = layer_output.mean(dim=1).squeeze(0).cpu()
    
    extractor._remove_hooks()
    
    # Initialize prediction
    initial_length = online_predictor.initialize(initial_embedding)
    
    print(f"\n--- Initial Prediction ---")
    print(f"Expected output length: {initial_length:.2f} tokens")
    
    state = online_predictor.get_current_prediction()
    print(f"Most likely bin: b{state['most_likely_bin'] + 1}")
    print(f"Probability distribution: {state['probabilities'].round(3)}")
    
    print("\n--- Prediction would be refined as tokens are generated ---")
    print("(In actual inference, each generated token's embedding updates the prediction)")


def analyze_dataset(args):
    """
    Analyze the output length distribution of the dataset.
    """
    configs = get_default_config()
    data_config = configs["data"]
    
    # Load data
    data_loader = AlpacaDataLoader(data_config)
    data = data_loader.load_dataset(num_samples=args.num_samples)
    
    # Compute distribution
    dist = compute_output_length_distribution(data, data_loader)
    
    print("\n=== Output Length Distribution ===")
    print(f"Mean: {dist['mean']:.2f}")
    print(f"Median: {dist['median']:.2f}")
    print(f"Std: {dist['std']:.2f}")
    print(f"Min: {dist['min']}")
    print(f"Max: {dist['max']}")
    
    # Plot CDF
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    plot_output_length_cdf(
        dist['lengths'],
        save_path=os.path.join(args.results_dir, "output_length_cdf.png")
    )


def main():
    parser = argparse.ArgumentParser(
        description="TRAIL Length Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile all layers to find the best one (Figure 2)
  python run.py profile --num_samples 1000 --layers "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
  
  # Profile layers 10-15 specifically
  python run.py profile --num_samples 1000 --layers "10,11,12,13,14,15"
  
  # Train predictor on layer 11 (best layer)
  python run.py train --num_samples 1000 --target_layer 11
  
  # Evaluate trained model
  python run.py evaluate --model_path ./checkpoints/best_model_layer_11.pt
  
  # Run inference on a prompt
  python run.py inference --model_path ./checkpoints/best_model_layer_11.pt \\
      --instruction "Write a short poem about AI"
  
  # Analyze dataset distribution
  python run.py analyze --num_samples 5000
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Profile command
    profile_parser = subparsers.add_parser("profile", help="Profile layers for prediction accuracy")
    profile_parser.add_argument("--num_samples", type=int, default=1000,
                               help="Number of samples for profiling")
    profile_parser.add_argument("--layers", type=str, default=None,
                               help="Comma-separated layer indices (e.g., '10,11,12')")
    profile_parser.add_argument("--profile_epochs", type=int, default=10,
                               help="Number of epochs for profiling")
    profile_parser.add_argument("--save_dir", type=str, default="./checkpoints",
                               help="Directory to save results")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train length predictor")
    train_parser.add_argument("--num_samples", type=int, default=1000,
                             help="Number of training samples")
    train_parser.add_argument("--target_layer", type=int, default=11,
                             help="Target layer for training (default: 11)")
    train_parser.add_argument("--save_dir", type=str, default="./checkpoints",
                             help="Directory to save model")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument("--model_path", type=str, required=True,
                            help="Path to trained model")
    eval_parser.add_argument("--num_samples", type=int, default=1000,
                            help="Number of evaluation samples")
    eval_parser.add_argument("--target_layer", type=int, default=11,
                            help="Target layer for evaluation")
    eval_parser.add_argument("--results_dir", type=str, default="./results",
                            help="Directory to save results")
    
    # Inference command
    infer_parser = subparsers.add_parser("inference", help="Run inference demo")
    infer_parser.add_argument("--model_path", type=str, required=True,
                             help="Path to trained model")
    infer_parser.add_argument("--instruction", type=str, required=True,
                             help="Instruction for the model")
    infer_parser.add_argument("--input_text", type=str, default=None,
                             help="Optional input text")
    infer_parser.add_argument("--target_layer", type=int, default=11,
                             help="Target layer for prediction")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze dataset")
    analyze_parser.add_argument("--num_samples", type=int, default=5000,
                               help="Number of samples to analyze")
    analyze_parser.add_argument("--results_dir", type=str, default="./results",
                               help="Directory to save results")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    if args.command == "profile":
        profile_layers(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "inference":
        inference(args)
    elif args.command == "analyze":
        analyze_dataset(args)


if __name__ == "__main__":
    main()
