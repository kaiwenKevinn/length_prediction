"""
Data Utilities for TRAIL Length Prediction
Handles Alpaca dataset loading and LLaMA embedding extraction
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Generator
from dataclasses import dataclass
from tqdm import tqdm
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class EmbeddingSample:
    """Single embedding sample with metadata"""
    embedding: torch.Tensor  # Shape: [hidden_dim]
    remaining_length: int    # Number of tokens remaining
    total_length: int        # Total output length
    current_position: int    # Current token position (0 = prefill)
    layer_idx: int          # Which layer the embedding came from
    sample_id: int          # Original sample ID
    # Entropy features (optional, for enhanced model)
    entropy_features: Optional[torch.Tensor] = None  # Shape: [entropy_feature_dim]
    attention_entropy_seq: Optional[List[float]] = None
    logits_entropy_seq: Optional[List[float]] = None


class AlpacaDataLoader:
    """Load and preprocess Alpaca dataset"""
    
    def __init__(self, config):
        self.config = config
        self.dataset = None
        
    def load_dataset(self, num_samples: Optional[int] = None) -> List[Dict]:
        """Load Alpaca dataset from HuggingFace"""
        print("Loading Alpaca dataset...")
        dataset = load_dataset(self.config.dataset_name, split="train")
        
        # Convert to list of dicts
        data = []
        for item in dataset:
            data.append({
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", "")
            })
        
        if num_samples is not None:
            # Set seed for reproducibility
            np.random.seed(self.config.seed)
            indices = np.random.permutation(len(data))[:num_samples]
            data = [data[i] for i in indices]
            
        self.dataset = data
        print(f"Loaded {len(data)} samples")
        return data
    
    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format instruction and input into prompt"""
        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    def split_dataset(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train/val/test"""
        np.random.seed(self.config.seed)
        indices = np.random.permutation(len(data))
        
        train_size = int(len(data) * self.config.train_ratio)
        val_size = int(len(data) * self.config.val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]
        
        return train_data, val_data, test_data


class LLaMAEmbeddingExtractor:
    """Extract embeddings from LLaMA model layers"""
    
    def __init__(self, model_config, predictor_config):
        self.model_config = model_config
        self.predictor_config = predictor_config
        self.model = None
        self.tokenizer = None
        self.layer_outputs = {}
        self.hooks = []
        
    def load_model(self):
        """Load LLaMA model and tokenizer"""
        print(f"Loading model: {self.model_config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=True
        )
        
        # Set padding token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            torch_dtype=self.model_config.dtype,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print("Model loaded successfully")
        
    def _register_hooks(self, layer_indices: List[int]):
        """Register forward hooks to capture layer outputs"""
        self._remove_hooks()  # Clean up existing hooks
        self.layer_outputs = {}
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # output[0] contains hidden states: [batch, seq_len, hidden_dim]
                if isinstance(output, tuple):
                    self.layer_outputs[layer_idx] = output[0].detach()
                else:
                    self.layer_outputs[layer_idx] = output.detach()
            return hook
        
        for layer_idx in layer_indices:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook)
            
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.layer_outputs = {}
        
    def extract_embeddings_for_sample(
        self,
        prompt: str,
        output_text: str,
        layer_indices: List[int],
        sample_id: int = 0
    ) -> Dict[int, List[EmbeddingSample]]:
        """
        Extract embeddings for a single sample during prefill and decode phases.
        
        Returns:
            Dict mapping layer_idx to list of EmbeddingSample
        """
        if self.model is None:
            self.load_model()
            
        # Register hooks
        self._register_hooks(layer_indices)
        
        # Tokenize
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        output_tokens = self.tokenizer.encode(output_text, add_special_tokens=False, return_tensors="pt")
        output_length = output_tokens.shape[1]
        
        # Clamp output length
        output_length = min(output_length, self.predictor_config.max_length)
        
        result = {layer_idx: [] for layer_idx in layer_indices}
        
        with torch.no_grad():
            # === Phase 1: Prefill ===
            # Process the entire prompt
            _ = self.model(prompt_tokens, use_cache=True)
            
            # Extract prefill embeddings (average over all prompt tokens)
            for layer_idx in layer_indices:
                layer_output = self.layer_outputs[layer_idx]  # [1, seq_len, hidden_dim]
                avg_embedding = layer_output.mean(dim=1).squeeze(0).cpu()  # [hidden_dim]
                
                sample = EmbeddingSample(
                    embedding=avg_embedding,
                    remaining_length=output_length,
                    total_length=output_length,
                    current_position=0,
                    layer_idx=layer_idx,
                    sample_id=sample_id
                )
                result[layer_idx].append(sample)
            
            # === Phase 2: Decode (autoregressive generation) ===
            # Simulate token-by-token generation
            current_tokens = prompt_tokens
            past_key_values = None
            
            for pos in range(min(output_length, self.predictor_config.max_length)):
                if pos >= output_tokens.shape[1]:
                    break
                    
                # Get next token
                next_token = output_tokens[:, pos:pos+1].to(self.model.device)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                
                # Forward pass for single token
                outputs = self.model(
                    current_tokens[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                
                # Extract decode embeddings
                remaining = output_length - (pos + 1)
                for layer_idx in layer_indices:
                    layer_output = self.layer_outputs[layer_idx]  # [1, 1, hidden_dim]
                    token_embedding = layer_output.squeeze(0).squeeze(0).cpu()  # [hidden_dim]
                    
                    sample = EmbeddingSample(
                        embedding=token_embedding,
                        remaining_length=remaining,
                        total_length=output_length,
                        current_position=pos + 1,
                        layer_idx=layer_idx,
                        sample_id=sample_id
                    )
                    result[layer_idx].append(sample)
                    
        self._remove_hooks()
        return result
    
    def extract_all_embeddings(
        self,
        data: List[Dict],
        layer_indices: List[int],
        data_loader: AlpacaDataLoader,
        save_dir: Optional[str] = None
    ) -> Dict[int, List[EmbeddingSample]]:
        """
        Extract embeddings for all samples in dataset.
        
        Args:
            data: List of data samples
            layer_indices: List of layer indices to extract
            data_loader: AlpacaDataLoader instance
            save_dir: Optional directory to save embeddings
            
        Returns:
            Dict mapping layer_idx to list of all EmbeddingSample
        """
        all_embeddings = {layer_idx: [] for layer_idx in layer_indices}
        
        for sample_id, sample in enumerate(tqdm(data, desc="Extracting embeddings")):
            prompt = data_loader.format_prompt(sample["instruction"], sample["input"])
            output_text = sample["output"]
            
            if len(output_text.strip()) == 0:
                continue
                
            try:
                sample_embeddings = self.extract_embeddings_for_sample(
                    prompt=prompt,
                    output_text=output_text,
                    layer_indices=layer_indices,
                    sample_id=sample_id
                )
                
                for layer_idx in layer_indices:
                    all_embeddings[layer_idx].extend(sample_embeddings[layer_idx])
                    
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
                continue
                
        if save_dir:
            self._save_embeddings(all_embeddings, save_dir)
            
        return all_embeddings
    
    def _save_embeddings(self, embeddings: Dict[int, List[EmbeddingSample]], save_dir: str):
        """Save embeddings to disk"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        for layer_idx, samples in embeddings.items():
            layer_data = {
                "embeddings": torch.stack([s.embedding for s in samples]),
                "remaining_lengths": torch.tensor([s.remaining_length for s in samples]),
                "total_lengths": torch.tensor([s.total_length for s in samples]),
                "positions": torch.tensor([s.current_position for s in samples]),
                "sample_ids": torch.tensor([s.sample_id for s in samples])
            }
            
            save_path = os.path.join(save_dir, f"layer_{layer_idx}_embeddings.pt")
            torch.save(layer_data, save_path)
            print(f"Saved layer {layer_idx} embeddings to {save_path}")
            
    def load_embeddings(self, save_dir: str, layer_indices: List[int]) -> Dict[int, List[EmbeddingSample]]:
        """Load embeddings from disk"""
        all_embeddings = {}
        
        for layer_idx in layer_indices:
            load_path = os.path.join(save_dir, f"layer_{layer_idx}_embeddings.pt")
            if not os.path.exists(load_path):
                print(f"Warning: {load_path} not found")
                continue
                
            layer_data = torch.load(load_path)
            samples = []
            
            for i in range(len(layer_data["embeddings"])):
                sample = EmbeddingSample(
                    embedding=layer_data["embeddings"][i],
                    remaining_length=layer_data["remaining_lengths"][i].item(),
                    total_length=layer_data["total_lengths"][i].item(),
                    current_position=layer_data["positions"][i].item(),
                    layer_idx=layer_idx,
                    sample_id=layer_data["sample_ids"][i].item()
                )
                samples.append(sample)
                
            all_embeddings[layer_idx] = samples
            print(f"Loaded {len(samples)} embeddings from layer {layer_idx}")
            
        return all_embeddings


class EmbeddingDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for embedding samples"""
    
    def __init__(self, samples: List[EmbeddingSample], predictor_config):
        self.samples = samples
        self.config = predictor_config
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert remaining length to bin index
        bin_idx = self.config.length_to_bin(sample.remaining_length)
        
        return {
            "embedding": sample.embedding,
            "bin_label": torch.tensor(bin_idx, dtype=torch.long),
            "remaining_length": sample.remaining_length,
            "position": sample.current_position
        }


def create_dataloaders(
    embeddings: List[EmbeddingSample],
    predictor_config,
    data_config,
    batch_size: Optional[int] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train/val/test dataloaders"""
    
    # Split by sample_id to ensure no data leakage
    sample_ids = list(set(s.sample_id for s in embeddings))
    np.random.seed(data_config.seed)
    np.random.shuffle(sample_ids)
    
    train_size = int(len(sample_ids) * data_config.train_ratio)
    val_size = int(len(sample_ids) * data_config.val_ratio)
    
    train_ids = set(sample_ids[:train_size])
    val_ids = set(sample_ids[train_size:train_size + val_size])
    test_ids = set(sample_ids[train_size + val_size:])
    
    train_samples = [s for s in embeddings if s.sample_id in train_ids]
    val_samples = [s for s in embeddings if s.sample_id in val_ids]
    test_samples = [s for s in embeddings if s.sample_id in test_ids]
    
    batch_size = batch_size or predictor_config.batch_size
    
    train_loader = torch.utils.data.DataLoader(
        EmbeddingDataset(train_samples, predictor_config),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        EmbeddingDataset(val_samples, predictor_config),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = torch.utils.data.DataLoader(
        EmbeddingDataset(test_samples, predictor_config),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Enhanced Data Utilities with Entropy Features
# ============================================================================

@dataclass
class EnhancedEmbeddingSample:
    """
    Enhanced embedding sample with entropy features.
    Used for the entropy-augmented length predictor.
    """
    embedding: torch.Tensor           # Shape: [hidden_dim]
    entropy_features: torch.Tensor    # Shape: [entropy_feature_dim]
    remaining_length: int
    total_length: int
    current_position: int
    layer_idx: int
    sample_id: int
    # Raw entropy sequences for analysis
    attention_entropy_seq: List[float] = field(default_factory=list)
    logits_entropy_seq: List[float] = field(default_factory=list)


class EnhancedLLaMAExtractor:
    """
    Enhanced extractor that captures embeddings, attention weights, and logits
    for computing entropy features.
    """
    
    def __init__(self, model_config, predictor_config, entropy_config):
        self.model_config = model_config
        self.predictor_config = predictor_config
        self.entropy_config = entropy_config
        self.model = None
        self.tokenizer = None
        self.layer_outputs = {}
        self.attention_outputs = {}
        self.hooks = []
        
        # Import entropy utilities
        from entropy_utils import EntropyCalculator, EntropyFeatureExtractor, EntropyNormalizer
        self.entropy_calculator = EntropyCalculator(entropy_config)
        self.feature_extractor = EntropyFeatureExtractor(entropy_config)
        self.normalizer = EntropyNormalizer(entropy_config.normalization_momentum)
        
    def load_model(self):
        """Load LLaMA model with attention output enabled"""
        print(f"Loading model: {self.model_config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name,
            torch_dtype=self.model_config.dtype,
            device_map="auto",
            trust_remote_code=True,
            output_attentions=True  # Enable attention output
        )
        self.model.eval()
        print("Model loaded successfully with attention output enabled")
        
    def _register_hooks(self, layer_indices: List[int]):
        """Register hooks for both hidden states and attention weights"""
        self._remove_hooks()
        self.layer_outputs = {}
        self.attention_outputs = {}
        
        def make_hidden_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.layer_outputs[layer_idx] = output[0].detach()
                    # Capture attention weights if available
                    if len(output) > 1 and output[1] is not None:
                        self.attention_outputs[layer_idx] = output[1].detach()
                else:
                    self.layer_outputs[layer_idx] = output.detach()
            return hook
        
        for layer_idx in layer_indices:
            layer = self.model.model.layers[layer_idx]
            hook = layer.register_forward_hook(make_hidden_hook(layer_idx))
            self.hooks.append(hook)
            
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.layer_outputs = {}
        self.attention_outputs = {}
    
    def extract_with_entropy(
        self,
        prompt: str,
        output_text: str,
        layer_indices: List[int],
        sample_id: int = 0,
        num_initial_tokens: Optional[int] = None
    ) -> Dict[int, List[EnhancedEmbeddingSample]]:
        """
        Extract embeddings along with entropy features.
        
        Args:
            prompt: Input prompt
            output_text: Expected output text
            layer_indices: Layers to extract from
            sample_id: Sample identifier
            num_initial_tokens: K value for entropy collection (default from config)
            
        Returns:
            Dict mapping layer_idx to list of EnhancedEmbeddingSample
        """
        if self.model is None:
            self.load_model()
            
        K = num_initial_tokens or self.entropy_config.num_initial_tokens
        
        # Register hooks
        self._register_hooks(layer_indices)
        
        # Tokenize
        prompt_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        output_tokens = self.tokenizer.encode(output_text, add_special_tokens=False, return_tensors="pt")
        output_length = min(output_tokens.shape[1], self.predictor_config.max_length)
        
        result = {layer_idx: [] for layer_idx in layer_indices}
        
        # Collect entropy sequences for first K tokens
        attention_entropy_seq = []  # Per-token aggregated attention entropy
        logits_entropy_seq = []     # Per-token logits entropy
        per_layer_attention = {layer_idx: [] for layer_idx in layer_indices}
        
        with torch.no_grad():
            # === Phase 1: Prefill ===
            outputs = self.model(
                prompt_tokens,
                use_cache=True,
                output_attentions=True
            )
            past_key_values = outputs.past_key_values
            
            # Extract prefill embeddings
            prefill_embeddings = {}
            for layer_idx in layer_indices:
                if layer_idx in self.layer_outputs:
                    layer_output = self.layer_outputs[layer_idx]
                    prefill_embeddings[layer_idx] = layer_output.mean(dim=1).squeeze(0).cpu()
            
            # === Phase 2: Decode with entropy collection ===
            current_tokens = prompt_tokens
            
            for pos in range(min(output_length, self.predictor_config.max_length)):
                if pos >= output_tokens.shape[1]:
                    break
                    
                next_token = output_tokens[:, pos:pos+1].to(self.model.device)
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                
                # Forward pass with attention
                outputs = self.model(
                    current_tokens[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=True
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]  # [1, vocab_size]
                
                # Compute logits entropy
                logits_ent = self.entropy_calculator.compute_logits_entropy(
                    logits, 
                    temperature=self.entropy_config.logits_temperature
                ).item()
                
                # Compute attention entropy from each layer
                if outputs.attentions is not None:
                    layer_attn_entropies = []
                    for layer_idx, attn in enumerate(outputs.attentions):
                        if layer_idx in layer_indices:
                            attn_ent = self.entropy_calculator.compute_attention_entropy(attn)
                            mean_attn_ent = attn_ent.mean().item()
                            per_layer_attention[layer_idx].append(mean_attn_ent)
                            layer_attn_entropies.append(mean_attn_ent)
                    
                    # Aggregate attention entropy across layers
                    if layer_attn_entropies:
                        if self.entropy_config.attention_aggregation == "mean":
                            agg_attn_ent = np.mean(layer_attn_entropies)
                        elif self.entropy_config.attention_aggregation == "max":
                            agg_attn_ent = np.max(layer_attn_entropies)
                        else:
                            agg_attn_ent = np.mean(layer_attn_entropies)
                        attention_entropy_seq.append(agg_attn_ent)
                else:
                    attention_entropy_seq.append(0.0)
                
                logits_entropy_seq.append(logits_ent)
                
                # After collecting K tokens, compute entropy features
                remaining = output_length - (pos + 1)
                
                if pos + 1 >= K or pos + 1 == output_length:
                    # Extract entropy features
                    entropy_features = self.feature_extractor.extract_features(
                        attention_entropy_seq[:pos+1],
                        logits_entropy_seq[:pos+1],
                        per_layer_attention
                    )
                    entropy_tensor = entropy_features.to_tensor()
                    
                    # Normalize if enabled
                    if self.entropy_config.normalize_entropy:
                        attn_arr = np.array(attention_entropy_seq[:pos+1])
                        logits_arr = np.array(logits_entropy_seq[:pos+1])
                        self.normalizer.update_stats("attention", attn_arr)
                        self.normalizer.update_stats("logits", logits_arr)
                    
                    # Create samples for each layer
                    for layer_idx in layer_indices:
                        if layer_idx in self.layer_outputs:
                            token_embedding = self.layer_outputs[layer_idx].squeeze(0).squeeze(0).cpu()
                            
                            sample = EnhancedEmbeddingSample(
                                embedding=token_embedding,
                                entropy_features=entropy_tensor,
                                remaining_length=remaining,
                                total_length=output_length,
                                current_position=pos + 1,
                                layer_idx=layer_idx,
                                sample_id=sample_id,
                                attention_entropy_seq=attention_entropy_seq[:pos+1].copy(),
                                logits_entropy_seq=logits_entropy_seq[:pos+1].copy()
                            )
                            result[layer_idx].append(sample)
        
        self._remove_hooks()
        return result
    
    def extract_all_enhanced(
        self,
        data: List[Dict],
        layer_indices: List[int],
        data_loader: AlpacaDataLoader,
        save_dir: Optional[str] = None,
        num_initial_tokens: Optional[int] = None
    ) -> Dict[int, List[EnhancedEmbeddingSample]]:
        """
        Extract enhanced embeddings with entropy features for all samples.
        """
        all_samples = {layer_idx: [] for layer_idx in layer_indices}
        
        for sample_id, sample in enumerate(tqdm(data, desc="Extracting enhanced embeddings")):
            prompt = data_loader.format_prompt(sample["instruction"], sample["input"])
            output_text = sample["output"]
            
            if len(output_text.strip()) == 0:
                continue
                
            try:
                sample_data = self.extract_with_entropy(
                    prompt=prompt,
                    output_text=output_text,
                    layer_indices=layer_indices,
                    sample_id=sample_id,
                    num_initial_tokens=num_initial_tokens
                )
                
                for layer_idx in layer_indices:
                    all_samples[layer_idx].extend(sample_data[layer_idx])
                    
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
                continue
        
        if save_dir:
            self._save_enhanced(all_samples, save_dir)
            
        return all_samples
    
    def _save_enhanced(self, samples: Dict[int, List[EnhancedEmbeddingSample]], save_dir: str):
        """Save enhanced samples to disk"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        for layer_idx, layer_samples in samples.items():
            if not layer_samples:
                continue
                
            data = {
                "embeddings": torch.stack([s.embedding for s in layer_samples]),
                "entropy_features": torch.stack([s.entropy_features for s in layer_samples]),
                "remaining_lengths": torch.tensor([s.remaining_length for s in layer_samples]),
                "total_lengths": torch.tensor([s.total_length for s in layer_samples]),
                "positions": torch.tensor([s.current_position for s in layer_samples]),
                "sample_ids": torch.tensor([s.sample_id for s in layer_samples]),
            }
            
            save_path = os.path.join(save_dir, f"layer_{layer_idx}_enhanced.pt")
            torch.save(data, save_path)
            print(f"Saved layer {layer_idx} enhanced data to {save_path}")
        
        # Save normalizer stats
        if self.entropy_config.normalize_entropy:
            normalizer_path = os.path.join(save_dir, "normalizer_stats.json")
            self.normalizer.save_stats(normalizer_path)
    
    def load_enhanced(
        self,
        save_dir: str,
        layer_indices: List[int]
    ) -> Dict[int, List[EnhancedEmbeddingSample]]:
        """Load enhanced samples from disk"""
        all_samples = {}
        
        for layer_idx in layer_indices:
            load_path = os.path.join(save_dir, f"layer_{layer_idx}_enhanced.pt")
            if not os.path.exists(load_path):
                print(f"Warning: {load_path} not found")
                continue
                
            data = torch.load(load_path)
            samples = []
            
            for i in range(len(data["embeddings"])):
                sample = EnhancedEmbeddingSample(
                    embedding=data["embeddings"][i],
                    entropy_features=data["entropy_features"][i],
                    remaining_length=data["remaining_lengths"][i].item(),
                    total_length=data["total_lengths"][i].item(),
                    current_position=data["positions"][i].item(),
                    layer_idx=layer_idx,
                    sample_id=data["sample_ids"][i].item()
                )
                samples.append(sample)
            
            all_samples[layer_idx] = samples
            print(f"Loaded {len(samples)} enhanced samples from layer {layer_idx}")
        
        return all_samples


class EnhancedEmbeddingDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for enhanced embedding samples with entropy features.
    """
    
    def __init__(
        self,
        samples: List[EnhancedEmbeddingSample],
        predictor_config,
        include_entropy: bool = True
    ):
        self.samples = samples
        self.config = predictor_config
        self.include_entropy = include_entropy
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        bin_idx = self.config.length_to_bin(sample.remaining_length)
        
        item = {
            "embedding": sample.embedding,
            "bin_label": torch.tensor(bin_idx, dtype=torch.long),
            "remaining_length": sample.remaining_length,
            "position": sample.current_position
        }
        
        if self.include_entropy and sample.entropy_features is not None:
            item["entropy_features"] = sample.entropy_features
        
        return item


def create_enhanced_dataloaders(
    samples: List[EnhancedEmbeddingSample],
    predictor_config,
    data_config,
    include_entropy: bool = True,
    batch_size: Optional[int] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders for enhanced samples.
    """
    # Split by sample_id
    sample_ids = list(set(s.sample_id for s in samples))
    np.random.seed(data_config.seed)
    np.random.shuffle(sample_ids)
    
    train_size = int(len(sample_ids) * data_config.train_ratio)
    val_size = int(len(sample_ids) * data_config.val_ratio)
    
    train_ids = set(sample_ids[:train_size])
    val_ids = set(sample_ids[train_size:train_size + val_size])
    test_ids = set(sample_ids[train_size + val_size:])
    
    train_samples = [s for s in samples if s.sample_id in train_ids]
    val_samples = [s for s in samples if s.sample_id in val_ids]
    test_samples = [s for s in samples if s.sample_id in test_ids]
    
    batch_size = batch_size or predictor_config.batch_size
    
    train_loader = torch.utils.data.DataLoader(
        EnhancedEmbeddingDataset(train_samples, predictor_config, include_entropy),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        EnhancedEmbeddingDataset(val_samples, predictor_config, include_entropy),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = torch.utils.data.DataLoader(
        EnhancedEmbeddingDataset(test_samples, predictor_config, include_entropy),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Enhanced Train samples: {len(train_samples)}")
    print(f"Enhanced Val samples: {len(val_samples)}")
    print(f"Enhanced Test samples: {len(test_samples)}")
    
    return train_loader, val_loader, test_loader
