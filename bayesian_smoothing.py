"""
Bayesian Smoothing for TRAIL Length Prediction
Based on: DON'T STOP ME NOW: EMBEDDING BASED SCHEDULING FOR LLMS (ICLR 2025)

The smoothing process:
1. Initialize: q^(0) = p^(0)
2. For each iteration t:
   a. Update prior: q^(t)_prior = T × q^(t-1)
   b. Combine evidence: q^(t)(i) = q^(t)_prior(i) × p^(t)(i) / normalization
   c. Calculate expected length: L_t = Σ [q^(t)(i) × m_i]

Transition Matrix T:
- T[i,i] = 1 - 1/bin_size (probability of staying in same bin)
- T[i,i+1] = 1/bin_size (probability of transitioning to next lower bin)
- All other entries are 0 (only adjacent smaller bin transitions allowed)
"""

import torch
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BayesianState:
    """State for Bayesian smoothing process"""
    current_prior: torch.Tensor  # Current probability estimate [num_bins]
    iteration: int               # Current iteration number
    history: List[torch.Tensor]  # History of probability estimates


class BayesianSmoother:
    """
    Bayesian smoother for refining length predictions over time.
    
    Physical meaning:
    - As tokens are generated, the remaining length can only decrease (or stay the same)
    - The transition matrix T encodes this prior knowledge
    - Bayesian update combines prior belief with new observed evidence
    """
    
    def __init__(self, config):
        """
        Args:
            config: BayesianConfig or PredictorConfig instance
        """
        self.num_bins = config.num_bins
        self.max_length = config.max_length
        self.bin_size = self.max_length / self.num_bins
        
        # Build transition matrix
        self.transition_matrix = self._build_transition_matrix()
        
        # Compute bin centers for expected length calculation
        self.bin_centers = self._compute_bin_centers()
        
    def _build_transition_matrix(self) -> torch.Tensor:
        """
        Build the transition matrix T.
        
        Structure:
        - Diagonal entries T[i,i]: probability of staying in same bin = 1 - 1/bin_size
        - Sub-diagonal entries T[i,i+1]: probability of moving to smaller bin = 1/bin_size
        - All other entries: 0
        
        Example for k=10 bins with bin_size=51.2:
        T = [
            [0.9805, 0.0195,   0,     0,   ...],
            [   0,   0.9805, 0.0195,  0,   ...],
            [   0,      0,   0.9805, 0.0195, ...],
            ...
        ]
        
        Returns:
            Transition matrix of shape [num_bins, num_bins]
        """
        T = torch.zeros(self.num_bins, self.num_bins)
        
        # Probability of transitioning to adjacent bin (generating one token)
        transition_prob = 1.0 / self.bin_size
        stay_prob = 1.0 - transition_prob
        
        for i in range(self.num_bins):
            # Diagonal: probability of staying in same bin
            T[i, i] = stay_prob
            
            # Sub-diagonal: probability of transitioning from bin i+1 to bin i
            if i < self.num_bins - 1:
                T[i, i + 1] = transition_prob
                
        return T
    
    def _compute_bin_centers(self) -> torch.Tensor:
        """
        Compute center values for each bin.
        
        For bin i: m_i = (b_i + b_{i+1}) / 2 = 128(2i+1)/5 when max_length=512, k=10
        
        Returns:
            Bin centers of shape [num_bins]
        """
        boundaries = [i * self.bin_size for i in range(self.num_bins + 1)]
        centers = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(self.num_bins)]
        return torch.tensor(centers, dtype=torch.float32)
    
    def initialize(self, initial_prediction: torch.Tensor) -> BayesianState:
        """
        Initialize the Bayesian smoothing state with first prediction.
        
        Args:
            initial_prediction: Initial probability distribution from predictor [num_bins]
            
        Returns:
            Initialized BayesianState
        """
        # Ensure input is properly normalized
        initial_prior = initial_prediction / initial_prediction.sum()
        
        return BayesianState(
            current_prior=initial_prior,
            iteration=0,
            history=[initial_prior.clone()]
        )
    
    def update(
        self,
        state: BayesianState,
        new_prediction: torch.Tensor,
        num_steps: int = 1
    ) -> Tuple[BayesianState, torch.Tensor]:
        """
        Update the probability estimate with new prediction.
        
        Steps:
        1. Apply transition matrix to prior (for each step taken)
        2. Combine with new evidence using Bayes' rule
        3. Normalize
        
        Args:
            state: Current BayesianState
            new_prediction: New probability prediction from model [num_bins]
            num_steps: Number of tokens generated since last update (default: 1)
            
        Returns:
            Tuple of (updated state, expected length)
        """
        # Step 1: Update prior by applying transition matrix
        # Each step (token generation) applies the transition
        prior = state.current_prior
        for _ in range(num_steps):
            # T × prior shifts probability mass toward smaller bins
            prior = torch.matmul(self.transition_matrix, prior)
        
        # Step 2: Combine with new evidence (Bayes' rule)
        # posterior ∝ prior × likelihood
        unnormalized_posterior = prior * new_prediction
        
        # Step 3: Normalize
        posterior = unnormalized_posterior / (unnormalized_posterior.sum() + 1e-10)
        
        # Calculate expected length
        expected_length = (posterior * self.bin_centers).sum()
        
        # Update state
        new_state = BayesianState(
            current_prior=posterior,
            iteration=state.iteration + num_steps,
            history=state.history + [posterior.clone()]
        )
        
        return new_state, expected_length
    
    def get_expected_length(self, state: BayesianState) -> torch.Tensor:
        """
        Calculate expected remaining length from current state.
        
        L_t = Σ [q^(t)(i) × m_i]
        
        Args:
            state: Current BayesianState
            
        Returns:
            Expected remaining length
        """
        return (state.current_prior * self.bin_centers).sum()
    
    def get_most_likely_bin(self, state: BayesianState) -> int:
        """
        Get the most likely bin index from current state.
        
        Args:
            state: Current BayesianState
            
        Returns:
            Most likely bin index
        """
        return state.current_prior.argmax().item()


class OnlineLengthPredictor:
    """
    Online predictor that combines MLP predictions with Bayesian smoothing.
    Used during inference to refine predictions as tokens are generated.
    """
    
    def __init__(self, predictor, smoother: BayesianSmoother, device: str = 'cpu'):
        """
        Args:
            predictor: Trained LengthPredictor model
            smoother: BayesianSmoother instance
            device: Device to run on
        """
        self.predictor = predictor.to(device)
        self.predictor.eval()
        self.smoother = smoother
        self.device = device
        self.state = None
        
    def initialize(self, initial_embedding: torch.Tensor) -> float:
        """
        Initialize with prefill embedding.
        
        Args:
            initial_embedding: Average embedding of input tokens [hidden_dim]
            
        Returns:
            Initial expected length prediction
        """
        with torch.no_grad():
            # Get initial prediction from MLP
            embedding = initial_embedding.unsqueeze(0).to(self.device)
            output = self.predictor(embedding)
            initial_probs = output['probs'].squeeze(0).cpu()
            
            # Initialize Bayesian state
            self.state = self.smoother.initialize(initial_probs)
            
            return self.smoother.get_expected_length(self.state).item()
    
    def update(self, token_embedding: torch.Tensor) -> float:
        """
        Update prediction with new token embedding.
        
        Args:
            token_embedding: Embedding of newly generated token [hidden_dim]
            
        Returns:
            Updated expected length prediction
        """
        if self.state is None:
            raise RuntimeError("Must call initialize() before update()")
            
        with torch.no_grad():
            # Get new prediction from MLP
            embedding = token_embedding.unsqueeze(0).to(self.device)
            output = self.predictor(embedding)
            new_probs = output['probs'].squeeze(0).cpu()
            
            # Update Bayesian state
            self.state, expected_length = self.smoother.update(self.state, new_probs)
            
            return expected_length.item()
    
    def get_current_prediction(self) -> dict:
        """
        Get current prediction state.
        
        Returns:
            Dict containing:
                - expected_length: Current expected remaining length
                - probabilities: Current probability distribution
                - most_likely_bin: Most likely bin index
                - iteration: Current iteration number
        """
        if self.state is None:
            return None
            
        return {
            'expected_length': self.smoother.get_expected_length(self.state).item(),
            'probabilities': self.state.current_prior.numpy(),
            'most_likely_bin': self.smoother.get_most_likely_bin(self.state),
            'iteration': self.state.iteration
        }
    
    def reset(self):
        """Reset the predictor state for new sequence"""
        self.state = None


def compute_smoothed_predictions(
    predictions: List[torch.Tensor],
    smoother: BayesianSmoother
) -> List[float]:
    """
    Apply Bayesian smoothing to a sequence of predictions.
    
    Args:
        predictions: List of probability distributions from MLP [num_bins]
        smoother: BayesianSmoother instance
        
    Returns:
        List of smoothed expected length predictions
    """
    smoothed_lengths = []
    state = smoother.initialize(predictions[0])
    smoothed_lengths.append(smoother.get_expected_length(state).item())
    
    for pred in predictions[1:]:
        state, expected_length = smoother.update(state, pred)
        smoothed_lengths.append(expected_length.item())
        
    return smoothed_lengths


def demo_bayesian_smoothing():
    """Demonstrate Bayesian smoothing with example predictions"""
    from config import PredictorConfig
    
    config = PredictorConfig()
    smoother = BayesianSmoother(config)
    
    print("Transition Matrix (first 5x5):")
    print(smoother.transition_matrix[:5, :5])
    print(f"\nBin Centers: {smoother.bin_centers}")
    
    # Simulate predictions
    print("\n=== Demo: Simulating token generation ===")
    
    # Initial prediction: mostly in middle bins
    initial_pred = torch.tensor([0.05, 0.1, 0.15, 0.2, 0.25, 0.15, 0.05, 0.03, 0.01, 0.01])
    state = smoother.initialize(initial_pred)
    print(f"Initial expected length: {smoother.get_expected_length(state):.2f}")
    
    # Simulate 100 token generations
    for i in range(1, 101, 20):
        # Simulate new prediction (gradually shifting to smaller bins)
        shift = i / 100
        new_pred = torch.zeros(10)
        for j in range(10):
            new_pred[j] = initial_pred[min(j + int(shift * 3), 9)]
        new_pred = new_pred / new_pred.sum()
        
        state, expected_length = smoother.update(state, new_pred)
        print(f"After {i} tokens: expected remaining length = {expected_length:.2f}")


if __name__ == "__main__":
    demo_bayesian_smoothing()
