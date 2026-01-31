"""
BelizeChain AI Models

This module contains the core AI models for BelizeChain's federated learning system.
Includes both full-precision and quantized models optimized for Belize's sovereign
digital infrastructure requirements.

Model Versioning:
- Semantic versioning (MAJOR.MINOR.PATCH)
- Git-style hashing for model checkpoints
- Compatibility tracking across federated rounds
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    BitsAndBytesConfig, AutoModelForCausalLM
)
from typing import Optional, Dict, Any
from datetime import datetime
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

# Model version registry
MODEL_VERSION = "1.0.0"  # Update with each breaking change
MODEL_REGISTRY = {
    "1.0.0": {
        "release_date": "2026-01-26",
        "base_model": "nawal-transformer-small",
        "parameters": 117_000_000,
        "training_rounds": 0,
        "breaking_changes": "Pure Nawal transformer - 100% sovereign architecture"
    }
}

class BelizeChainLLM(nn.Module):
    """
    BelizeChain Large Language Model
    
    Core LLM for Belize's sovereign AI infrastructure with support for:
    - Multilingual processing (English, Spanish, Kriol, Garifuna, Maya languages)
    - Belizean legal and regulatory text understanding
    - Privacy-preserving inference
    - Federated learning compatibility
    """
    
    def __init__(
        self,
        model_name: str = "nawal-transformer-small",
        num_labels: int = 2,
        dropout: float = 0.1,
        belizean_vocab_extension: bool = True,
        version: str = MODEL_VERSION
    ):
        super(BelizeChainLLM, self).__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.version = version
        self.created_at = datetime.utcnow()
        self.training_rounds = 0
        self.privacy_epsilon = 0.0
        self.privacy_delta = 0.0
        self.last_updated = self.created_at
        
        # Load base transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add Belizean-specific tokens if enabled
        if belizean_vocab_extension:
            self._extend_belizean_vocabulary()
        
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Classification head for various tasks
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.transformer.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels)
        )
        
        # Language modeling head
        self.lm_head = nn.Linear(
            self.transformer.config.hidden_size, 
            self.transformer.config.vocab_size
        )
        
        logger.info(f"Initialized BelizeChain LLM with {model_name}")
    
    def _extend_belizean_vocabulary(self):
        """Extend tokenizer with Belizean-specific terms"""
        belizean_tokens = [
            # Currencies and Financial Terms
            "bBZD", "Dalla", "Mahogany", "BZD",
            
            # Government and Legal
            "FSC", "GOB", "BLEAC", "MoF", "CBB",
            "BelizeID", "SSI", "KYC", "AML",
            
            # Geographic and Cultural
            "Belize", "Belizean", "Kriol", "Garifuna", 
            "Maya", "Mestizo", "Mennonite",
            "Cayo", "Toledo", "Orange Walk", "Corozal", "Stann Creek",
            
            # Legal Framework Terms
            "pallet-belize-kyc", "PoUW", "XCM",
            "landledger", "belizex",
            
            # Technical Terms
            "substrate", "polkadot", "WASM", "IPFS", "arweave"
        ]
        
        # Add tokens to tokenizer
        self.tokenizer.add_tokens(belizean_tokens)
        logger.info(f"Added {len(belizean_tokens)} Belizean-specific tokens")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        task: str = "classification"
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass supporting multiple tasks
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Target labels (if supervised)
            task: Task type ("classification", "language_modeling")
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        hidden_states = outputs.last_hidden_state
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else hidden_states[:, 0]
        
        result = {"hidden_states": hidden_states}
        
        if task == "classification":
            logits = self.classifier(pooled_output)
            result["logits"] = logits
            
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
                result["loss"] = loss
                
        elif task == "language_modeling":
            lm_logits = self.lm_head(hidden_states)
            result["logits"] = lm_logits
            
            if labels is not None:
                # Shift labels for language modeling
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                result["loss"] = loss
        
        return result

class QuantizedBelizeModel(nn.Module):
    """
    Quantized version of BelizeChain LLM for efficient federated learning
    
    Uses 8-bit or 4-bit quantization to reduce memory footprint and
    communication costs in federated training scenarios.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",  # Temporary - custom Belizean tokenizer in development
        bits: int = 8,
        num_labels: int = 2
    ):
        super(QuantizedBelizeModel, self).__init__()
        
        self.bits = bits
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Configure quantization
        if bits == 8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            raise ValueError(f"Unsupported quantization bits: {bits}")
        
        # Load quantized model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                num_labels=num_labels
            )
        except Exception as e:
            logger.warning(f"Failed to load as classification model: {e}")
            # Fallback to base model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            
            # Add classification head
            self.classification_head = nn.Linear(
                self.model.config.hidden_size, 
                num_labels
            )
        
        logger.info(f"Initialized {bits}-bit quantized BelizeChain model")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for quantized model"""
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels if hasattr(self.model, 'classifier') else None
        )
        
        # Handle different model architectures
        if hasattr(self.model, 'classifier'):
            # Model has built-in classifier
            return outputs
        else:
            # Add classification head to base model
            hidden_states = outputs.last_hidden_state
            pooled_output = hidden_states[:, 0]  # Use [CLS] token
            
            logits = self.classification_head(pooled_output)
            result = {"logits": logits}
            
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
                result["loss"] = loss
            
            return result
    
    def get_memory_footprint(self) -> int:
        """Get model memory footprint in bytes"""
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

class BelizeanLanguageDetector(nn.Module):
    """
    Language detection model for Belizean multilingual context
    
    Detects and classifies text in:
    - English (official language)
    - Spanish (widely spoken)
    - Kriol (creole language)
    - Garifuna (indigenous language)
    - Maya languages (Q'eqchi', Mopan)
    """
    
    def __init__(self, model_name: str = "gpt2"):  # Temporary - custom Belizean tokenizer
        super(BelizeanLanguageDetector, self).__init__()
        
        self.languages = ["english", "spanish", "kriol", "garifuna", "maya"]
        self.num_languages = len(self.languages)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Language classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.transformer.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_languages)
        )
        
        logger.info("Initialized Belizean language detector")
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Detect language of input text"""
        outputs = self.transformer(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.classifier(pooled_output)
        
        return torch.softmax(logits, dim=-1)
    
    def predict_language(self, text: str) -> str:
        """Predict the most likely language for given text"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        )
        
        with torch.no_grad():
            probabilities = self.forward(
                inputs["input_ids"], 
                inputs["attention_mask"]
            )
        
        predicted_idx = torch.argmax(probabilities, dim=-1).item()
        return self.languages[predicted_idx]


# Model versioning and checkpoint management utilities

def compute_model_hash(model: nn.Module) -> str:
    """
    Compute SHA-256 hash of model weights for integrity verification
    
    Args:
        model: PyTorch model
    
    Returns:
        Hexadecimal hash string
    """
    hasher = hashlib.sha256()
    
    for name, param in model.named_parameters():
        hasher.update(param.data.cpu().numpy().tobytes())
    
    return hasher.hexdigest()[:16]  # First 16 chars


def save_versioned_checkpoint(
    model: nn.Module,
    path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save model checkpoint with version metadata
    
    Args:
        model: Model to save
        path: Checkpoint file path
        metadata: Additional metadata (training rounds, privacy budget, etc.)
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "version": getattr(model, 'version', MODEL_VERSION),
        "model_hash": compute_model_hash(model),
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": metadata or {}
    }
    
    if hasattr(model, 'training_rounds'):
        checkpoint["training_rounds"] = model.training_rounds
    
    if hasattr(model, 'privacy_epsilon'):
        checkpoint["privacy_budget"] = {
            "epsilon": model.privacy_epsilon,
            "delta": model.privacy_delta
        }
    
    torch.save(checkpoint, path)
    logger.info(f"✅ Saved versioned checkpoint: {path} (version {checkpoint['version']})")


def load_versioned_checkpoint(
    model: nn.Module,
    path: str,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint and verify version compatibility
    
    Args:
        model: Model to load weights into
        path: Checkpoint file path
        strict: Strict state_dict loading
    
    Returns:
        Checkpoint metadata dictionary
    """
    checkpoint = torch.load(path, map_location='cpu')
    
    # Verify version compatibility
    checkpoint_version = checkpoint.get("version", "unknown")
    current_version = getattr(model, 'version', MODEL_VERSION)
    
    if checkpoint_version != current_version:
        logger.warning(f"⚠️ Version mismatch: checkpoint={checkpoint_version}, current={current_version}")
        
        # Check if versions are compatible (same MAJOR.MINOR)
        if not versions_compatible(checkpoint_version, current_version):
            raise RuntimeError(
                f"Incompatible versions: {checkpoint_version} -> {current_version}. "
                "Model architecture may have changed."
            )
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    
    # Restore metadata
    if hasattr(model, 'training_rounds'):
        model.training_rounds = checkpoint.get("training_rounds", 0)
    
    if "privacy_budget" in checkpoint:
        model.privacy_epsilon = checkpoint["privacy_budget"]["epsilon"]
        model.privacy_delta = checkpoint["privacy_budget"]["delta"]
    
    if hasattr(model, 'last_updated'):
        model.last_updated = datetime.fromisoformat(checkpoint["timestamp"])
    
    logger.info(f"✅ Loaded checkpoint: {path} (hash: {checkpoint.get('model_hash', 'N/A')})")
    
    return checkpoint.get("metadata", {})


def versions_compatible(version1: str, version2: str) -> bool:
    """
    Check if two semantic versions are compatible
    
    Compatible if MAJOR.MINOR match (PATCH differences allowed)
    """
    try:
        v1_parts = version1.split('.')
        v2_parts = version2.split('.')
        
        # Same MAJOR.MINOR = compatible
        return v1_parts[0] == v2_parts[0] and v1_parts[1] == v2_parts[1]
    
    except (IndexError, ValueError):
        # Unknown version format - assume incompatible
        return False


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive model information for versioning
    
    Returns:
        Dictionary with model metadata
    """
    return {
        "version": getattr(model, 'version', 'unknown'),
        "model_name": getattr(model, 'model_name', 'unknown'),
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_rounds": getattr(model, 'training_rounds', 0),
        "created_at": getattr(model, 'created_at', None),
        "last_updated": getattr(model, 'last_updated', None),
        "model_hash": compute_model_hash(model),
        "privacy_budget": {
            "epsilon": getattr(model, 'privacy_epsilon', 0.0),
            "delta": getattr(model, 'privacy_delta', 0.0)
        }
    }


def create_belizechain_model(
    model_type: str = "standard",
    quantization_bits: int = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create BelizeChain models
    
    Args:
        model_type: Type of model ("standard", "quantized", "language_detector")
        quantization_bits: Bits for quantization (4, 8, or None)
        **kwargs: Additional model parameters
    
    Returns:
        Initialized model instance
    """
    if model_type == "quantized" and quantization_bits:
        return QuantizedBelizeModel(bits=quantization_bits, **kwargs)
    elif model_type == "language_detector":
        return BelizeanLanguageDetector(**kwargs)
    else:
        return BelizeChainLLM(**kwargs)

if __name__ == "__main__":
    # Test model creation
    print("Testing BelizeChain model creation...")
    
    # Standard model
    model = create_belizechain_model()
    print(f"Standard model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Quantized model
    quant_model = create_belizechain_model("quantized", quantization_bits=8)
    print(f"8-bit model memory: {quant_model.get_memory_footprint():,} bytes")
    
    # Language detector
    lang_detector = create_belizechain_model("language_detector")
    print(f"Language detector supports: {lang_detector.languages}")
    
    print("Model creation tests completed!")