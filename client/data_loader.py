"""
BelizeChain Data Loader

This module handles data loading and preprocessing for BelizeChain's federated learning
system with built-in compliance filtering and data sovereignty protection.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import os
import json
from pathlib import Path
import hashlib
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DataSovereigntyLevel(Enum):
    """Data sovereignty classification levels"""
    PUBLIC = "public"
    RESTRICTED = "restricted" 
    CONFIDENTIAL = "confidential"
    SECRET = "secret"

@dataclass
class ComplianceMetadata:
    """Metadata for compliance and data sovereignty"""
    data_classification: DataSovereigntyLevel
    contains_pii: bool = False
    requires_kyc: bool = False
    geographic_restriction: str = "BZ"  # Belize by default
    retention_period_days: int = 2555  # 7 years default
    encryption_required: bool = True

class ComplianceDataFilter:
    """
    Compliance and data sovereignty filter for BelizeChain federated learning
    
    Ensures all training data meets Belizean legal requirements and
    data protection standards.
    """
    
    def __init__(self):
        self.filtered_count = 0
        self.total_processed = 0
        self.compliance_violations = []
        
        # Load sensitive terms list for Belize
        self.sensitive_patterns = self._load_sensitive_patterns()
        
        logger.info("Initialized compliance data filter")
    
    def _load_sensitive_patterns(self) -> List[str]:
        """Load patterns that require special handling under Belize law"""
        return [
            # Financial sensitive information
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card numbers
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN-like patterns
            r'\b[A-Z]{2}\d{6}[A-Z]\b',  # Belize ID pattern
            
            # Regulatory terms that require special handling
            r'\bmoney[\s-]?laundering\b',
            r'\bterrorist[\s-]?financing\b',
            r'\btax[\s-]?evasion\b',
            
            # Personal identifiers
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\+?501[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Belize phone numbers
        ]
    
    def filter_batch(self, batch: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        """
        Filter a batch for compliance violations
        
        Args:
            batch: Training batch containing input_ids, attention_mask, labels
            
        Returns:
            Filtered batch or None if batch should be rejected
        """
        self.total_processed += 1
        
        # Convert tokens back to text for analysis
        if 'text' not in batch and hasattr(self, 'tokenizer'):
            # Reconstruct text from token IDs for analysis
            texts = self.tokenizer.batch_decode(
                batch['input_ids'], 
                skip_special_tokens=True
            )
        else:
            texts = batch.get('text', [])
        
        # Check each text in batch
        for i, text in enumerate(texts):
            if not self._is_compliant(text):
                self.filtered_count += 1
                logger.warning(f"Filtered non-compliant text in batch item {i}")
                
                # Remove this item from batch
                for key in batch:
                    if isinstance(batch[key], torch.Tensor) and len(batch[key]) > i:
                        # Remove item at index i
                        indices = list(range(len(batch[key])))
                        indices.pop(i)
                        batch[key] = batch[key][indices]
        
        # Return batch if it has remaining items
        if len(batch.get('input_ids', [])) > 0:
            return batch
        else:
            return None
    
    def _is_compliant(self, text: str) -> bool:
        """Check if text meets compliance requirements"""
        import re
        
        # Check for sensitive patterns
        for pattern in self.sensitive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                self.compliance_violations.append({
                    'pattern': pattern,
                    'text_hash': hashlib.sha256(text.encode()).hexdigest()[:16]
                })
                return False
        
        # Additional Belizean-specific compliance checks
        if self._contains_restricted_content(text):
            return False
            
        return True
    
    def _contains_restricted_content(self, text: str) -> bool:
        """Check for content restricted under Belize law"""
        # Add specific Belizean legal restrictions
        restricted_terms = [
            # Content that might violate Belize regulations
            'illegal gambling', 'unlicensed forex', 'ponzi scheme',
            'tax haven abuse', 'sanctions violation'
        ]
        
        text_lower = text.lower()
        for term in restricted_terms:
            if term in text_lower:
                return True
                
        return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get filtering statistics"""
        return {
            'total_processed': self.total_processed,
            'filtered_count': self.filtered_count,
            'compliance_rate': 1.0 - (self.filtered_count / max(self.total_processed, 1))
        }

class BelizeDataset(Dataset):
    """
    Custom dataset for BelizeChain federated learning
    
    Handles Belizean multilingual data with compliance metadata
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        compliance_metadata: Optional[ComplianceMetadata] = None
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.compliance_metadata = compliance_metadata or ComplianceMetadata(
            data_classification=DataSovereigntyLevel.PUBLIC
        )
        
        # Load data
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from various formats"""
        if self.data_path.suffix == '.json':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path)
            return df.to_dict('records')
        else:
            # Load text files
            with open(self.data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            return [{'text': line.strip(), 'label': 0} for line in lines if line.strip()]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data item"""
        item = self.data[idx]
        
        # Extract text and label
        text = item.get('text', item.get('input', ''))
        label = item.get('label', 0)
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text  # Keep original text for compliance filtering
        }

class BelizeDataLoader:
    """
    Data loader for BelizeChain federated learning participants
    
    Handles data loading with built-in compliance filtering and
    data sovereignty protection for Belizean requirements.
    """
    
    def __init__(
        self,
        participant_id: str,
        batch_size: int = 32,
        max_length: int = 512,
        train_split: float = 0.8,
        compliance_filter: Optional[ComplianceDataFilter] = None,
        data_sovereignty_check: bool = True
    ):
        self.participant_id = participant_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.train_split = train_split
        self.compliance_filter = compliance_filter
        self.data_sovereignty_check = data_sovereignty_check
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            'gpt2',  # Temporary - custom Belizean BPE tokenizer in development
            padding_side='right'
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set reference to tokenizer in compliance filter
        if self.compliance_filter:
            self.compliance_filter.tokenizer = self.tokenizer
        
        # Load participant data
        self._load_participant_data()
        
        logger.info(f"Initialized data loader for participant {participant_id}")
    
    def _load_participant_data(self):
        """Load data specific to this participant"""
        data_dir = Path(f"data/participants/{self.participant_id}")
        
        # Create synthetic data if directory doesn't exist
        if not data_dir.exists():
            logger.info(f"Creating synthetic data for participant {self.participant_id}")
            self._create_synthetic_data(data_dir)
        
        # Load training data
        train_file = data_dir / "train.json"
        if train_file.exists():
            self.dataset = BelizeDataset(
                str(train_file),
                self.tokenizer,
                self.max_length
            )
        else:
            # Create minimal dataset
            self.dataset = self._create_minimal_dataset()
        
        # Split dataset
        train_size = int(self.train_split * len(self.dataset))
        val_size = len(self.dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, 
            [train_size, val_size]
        )
    
    def _create_synthetic_data(self, data_dir: Path):
        """Create synthetic training data for testing"""
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample Belizean-context synthetic data
        synthetic_samples = [
            {"text": "The Financial Services Commission of Belize regulates digital assets.", "label": 0},
            {"text": "BelizeChain provides sovereign blockchain infrastructure for Belize.", "label": 1},
            {"text": "Federated learning preserves data privacy across participants.", "label": 1},
            {"text": "KYC compliance is mandatory for financial services in Belize.", "label": 0},
            {"text": "Quantum computing enhances blockchain security capabilities.", "label": 1},
            {"text": "IPFS provides decentralized storage for blockchain applications.", "label": 1},
            {"text": "The Belize Dollar (BZD) is the official currency of Belize.", "label": 0},
            {"text": "Digital identity systems enhance security and privacy.", "label": 1},
            {"text": "Cross-chain bridges enable blockchain interoperability.", "label": 1},
            {"text": "Compliance frameworks ensure regulatory adherence.", "label": 0}
        ]
        
        # Save synthetic data
        with open(data_dir / "train.json", 'w') as f:
            json.dump(synthetic_samples, f, indent=2)
        
        logger.info(f"Created synthetic data at {data_dir}")
    
    def _create_minimal_dataset(self) -> BelizeDataset:
        """Create minimal dataset for testing"""
        minimal_data = [
            {"text": "BelizeChain federated learning test", "label": 1},
            {"text": "Synthetic training sample", "label": 0}
        ]
        
        # Create temporary file
        temp_file = Path("/tmp/minimal_belizechain_data.json")
        with open(temp_file, 'w') as f:
            json.dump(minimal_data, f)
        
        return BelizeDataset(str(temp_file), self.tokenizer, self.max_length)
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
    
    def get_eval_loader(self) -> DataLoader:
        """Get evaluation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function with compliance filtering"""
        # Standard collation
        collated = {}
        for key in batch[0].keys():
            if key == 'text':
                collated[key] = [item[key] for item in batch]
            else:
                collated[key] = torch.stack([item[key] for item in batch])
        
        # Apply compliance filtering if configured
        if self.compliance_filter:
            collated = self.compliance_filter.filter_batch(collated)
            
        return collated if collated else {}

def create_belizean_data_splits(
    data_path: str,
    num_participants: int = 3,
    output_dir: str = "data/federated"
) -> List[str]:
    """
    Create federated data splits for multiple BelizeChain participants
    
    Args:
        data_path: Path to original dataset
        num_participants: Number of federated participants
        output_dir: Output directory for participant data
    
    Returns:
        List of participant data directories
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load original data
    with open(data_path, 'r') as f:
        if data_path.endswith('.json'):
            data = json.load(f)
        else:
            data = [{'text': line.strip(), 'label': 0} for line in f.readlines()]
    
    # Split data among participants
    participant_dirs = []
    samples_per_participant = len(data) // num_participants
    
    for i in range(num_participants):
        participant_id = f"participant_{i+1}"
        participant_dir = output_path / participant_id
        participant_dir.mkdir(exist_ok=True)
        
        # Get participant's data slice
        start_idx = i * samples_per_participant
        end_idx = start_idx + samples_per_participant if i < num_participants - 1 else len(data)
        participant_data = data[start_idx:end_idx]
        
        # Save participant data
        with open(participant_dir / "train.json", 'w') as f:
            json.dump(participant_data, f, indent=2)
        
        participant_dirs.append(str(participant_dir))
        logger.info(f"Created data for {participant_id}: {len(participant_data)} samples")
    
    return participant_dirs

if __name__ == "__main__":
    # Test data loader
    print("Testing BelizeChain data loader...")
    
    # Create test data loader
    data_loader = BelizeDataLoader(
        participant_id="test_participant",
        batch_size=2,
        compliance_filter=ComplianceDataFilter()
    )
    
    # Test training loader
    train_loader = data_loader.get_train_loader()
    for batch in train_loader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Input shape: {batch['input_ids'].shape}")
        break
    
    print("Data loader test completed!")