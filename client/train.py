"""
BelizeChain Federated AI - Local Training Client

This module implements the local participant side of BelizeChain's federated learning
architecture. It handles local model training while preserving data privacy and 
sovereignty requirements for Belize's national digital infrastructure.

Key Features:
- Privacy-preserving local training
- Quantized model support for efficiency
- Azure ML integration for participants
- Secure aggregation preparation
- Compliance with Belize data sovereignty laws
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import flwr as fl
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import os
from pathlib import Path

from .model import BelizeChainLLM, QuantizedBelizeModel  
from .data_loader import BelizeDataLoader, ComplianceDataFilter

# Configure logging for Belize compliance
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('belizechain_federated_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BelizeTrainingConfig:
    """Configuration for BelizeChain federated training participant"""
    participant_id: str
    model_name: str = "belizechain-llm-base"
    learning_rate: float = 1e-4
    batch_size: int = 32
    local_epochs: int = 3
    quantization_bits: int = 8
    compliance_mode: bool = True
    data_sovereignty_check: bool = True
    azure_endpoint: Optional[str] = None
    encryption_key_path: Optional[str] = None

class BelizeChainFederatedClient(fl.client.NumPyClient):
    """
    BelizeChain Federated Learning Client
    
    Implements privacy-preserving federated learning for Belize's sovereign AI infrastructure.
    Ensures all training complies with Belizean data protection and sovereignty requirements.
    """
    
    def __init__(self, config: BelizeTrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.data_loader = None
        self.compliance_filter = ComplianceDataFilter()
        
        logger.info(f"Initializing BelizeChain FL client: {config.participant_id}")
        self._setup_model()
        self._setup_data()
        
    def _setup_model(self):
        """Initialize the local model with quantization for efficiency"""
        try:
            if self.config.quantization_bits < 16:
                self.model = QuantizedBelizeModel(
                    model_name=self.config.model_name,
                    bits=self.config.quantization_bits
                ).to(self.device)
                logger.info(f"Loaded quantized model ({self.config.quantization_bits}-bit)")
            else:
                self.model = BelizeChainLLM(
                    model_name=self.config.model_name
                ).to(self.device)
                logger.info("Loaded full-precision model")
                
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise
            
    def _setup_data(self):
        """Setup local data loader with compliance filtering"""
        try:
            self.data_loader = BelizeDataLoader(
                participant_id=self.config.participant_id,
                batch_size=self.config.batch_size,
                compliance_filter=self.compliance_filter if self.config.compliance_mode else None,
                data_sovereignty_check=self.config.data_sovereignty_check
            )
            logger.info("Data loader initialized with compliance filtering")
            
        except Exception as e:
            logger.error(f"Data loader setup failed: {e}")
            raise

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Extract model parameters for federated aggregation"""
        logger.info("Extracting model parameters for aggregation")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from federated aggregation"""
        logger.info("Setting model parameters from global aggregation")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Local training round with privacy preservation
        
        Args:
            parameters: Global model parameters from aggregation
            config: Training configuration from server
            
        Returns:
            Updated parameters, number of samples, training metrics
        """
        logger.info(f"Starting local training round for participant {self.config.participant_id}")
        
        # Set global parameters
        self.set_parameters(parameters)
        
        # Prepare model for training
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        
        total_loss = 0.0
        num_samples = 0
        
        # Local training loop
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch in self.data_loader.get_train_loader():
                # Compliance check on batch
                if self.config.compliance_mode:
                    batch = self.compliance_filter.filter_batch(batch)
                    if batch is None:
                        continue
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    labels=batch['labels'].to(self.device)
                )
                
                loss = outputs.loss
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_samples += batch['input_ids'].size(0)
                
            total_loss += epoch_loss
            num_samples += epoch_samples
            
            logger.info(f"Epoch {epoch + 1}/{self.config.local_epochs}: "
                       f"Loss = {epoch_loss / max(epoch_samples, 1):.4f}")
        
        # Privacy-preserving parameter extraction
        updated_parameters = self.get_parameters({})
        
        # Apply differential privacy if configured
        if config.get('differential_privacy', False):
            updated_parameters = self._apply_differential_privacy(
                updated_parameters, 
                config.get('privacy_budget', 1.0)
            )
        
        metrics = {
            'loss': total_loss / max(num_samples, 1),
            'num_samples': num_samples,
            'participant_id': self.config.participant_id,
            'compliance_checks_passed': self.compliance_filter.get_stats()
        }
        
        logger.info(f"Local training completed: {metrics}")
        return updated_parameters, num_samples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Local evaluation with privacy preservation
        
        Args:
            parameters: Global model parameters
            config: Evaluation configuration
            
        Returns:
            Loss, number of samples, evaluation metrics
        """
        logger.info("Starting local evaluation")
        
        self.set_parameters(parameters)
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in self.data_loader.get_eval_loader():
                # Compliance filtering
                if self.config.compliance_mode:
                    batch = self.compliance_filter.filter_batch(batch)
                    if batch is None:
                        continue
                
                outputs = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    labels=batch['labels'].to(self.device)
                )
                
                total_loss += outputs.loss.item()
                
                # Calculate accuracy for classification tasks
                if hasattr(outputs, 'logits'):
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    accuracy = (predictions == batch['labels']).float().mean()
                    total_accuracy += accuracy.item()
                
                num_samples += batch['input_ids'].size(0)
        
        avg_loss = total_loss / max(len(self.data_loader.get_eval_loader()), 1)
        avg_accuracy = total_accuracy / max(len(self.data_loader.get_eval_loader()), 1)
        
        metrics = {
            'accuracy': avg_accuracy,
            'participant_id': self.config.participant_id,
            'evaluation_samples': num_samples
        }
        
        logger.info(f"Evaluation completed: Loss={avg_loss:.4f}, Accuracy={avg_accuracy:.4f}")
        return avg_loss, num_samples, metrics

    def _apply_differential_privacy(self, parameters: List[np.ndarray], epsilon: float) -> List[np.ndarray]:
        """Apply differential privacy to model parameters"""
        logger.info(f"Applying differential privacy with ε={epsilon}")
        
        private_parameters = []
        for param in parameters:
            # Add calibrated noise for differential privacy
            sensitivity = np.max(np.abs(param))  # L-infinity sensitivity
            noise_scale = sensitivity / epsilon
            noise = np.random.laplace(0, noise_scale, param.shape)
            private_parameters.append(param + noise)
        
        return private_parameters

def main():
    """Main entry point for BelizeChain federated training client"""
    config = BelizeTrainingConfig(
        participant_id=os.getenv('BELIZECHAIN_PARTICIPANT_ID', 'default_participant'),
        azure_endpoint=os.getenv('AZURE_ML_ENDPOINT'),
        compliance_mode=os.getenv('BELIZECHAIN_COMPLIANCE', 'true').lower() == 'true'
    )
    
    # Initialize federated learning client
    client = BelizeChainFederatedClient(config)
    
    # Start federated learning
    fl.client.start_numpy_client(
        server_address="localhost:8080",  # Configure for your federated server
        client=client
    )

# Re-export GenomeTrainer for backward compatibility
# New code should import from nawal.client.genome_trainer
from .genome_trainer import GenomeTrainer

__all__ = [
    'BelizeTrainingConfig',
    'BelizeChainFederatedClient',
    'GenomeTrainer',  # Exported for backward compatibility
]

if __name__ == "__main__":
    main()