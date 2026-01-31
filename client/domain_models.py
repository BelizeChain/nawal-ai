"""
Domain-Specific AI Models for BelizeChain Nawal System

This module implements specialized AI models for different national priority domains:
- AgriTech: Agricultural monitoring and optimization
- Marine: Ocean health and conservation
- Education: Personalized learning systems
- Tech: Software and infrastructure monitoring
- General: Multi-purpose AI for government services

Each domain has specific reward multipliers in the Oracle pallet:
- AgriTech: 1.5x (highest priority for food security)
- Marine: 1.4x (critical for marine ecosystem)
- Education: 1.3x (human capital development)
- Tech: 1.1x (infrastructure optimization)
- General: 1.0x (baseline)

Integrates with:
- Oracle pallet IoT data feeds (drones, sensors, phones)
- Proof of Useful Work (PoUW) reward system
- Federated learning orchestration
- Genome evolution for architecture search

Author: BelizeChain Team
License: MIT
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import logging

from .model import BelizeChainLLM
from ..genome.encoding import Genome, ArchitectureLayer, LayerType

logger = logging.getLogger(__name__)


class ModelDomain(Enum):
    """
    Model domain types matching Oracle pallet enum.
    
    Must stay in sync with Rust enum in pallets/oracle/src/types.rs:
    ```rust
    pub enum ModelDomain {
        General = 0,
        AgriTech = 1,
        Marine = 2,
        Education = 3,
        Tech = 4,
    }
    ```
    """
    GENERAL = 0
    AGRITECH = 1
    MARINE = 2
    EDUCATION = 3
    TECH = 4
    
    def reward_multiplier(self) -> float:
        """Get reward multiplier for this domain (matches Rust implementation)"""
        multipliers = {
            ModelDomain.GENERAL: 1.0,
            ModelDomain.AGRITECH: 1.5,
            ModelDomain.MARINE: 1.4,
            ModelDomain.EDUCATION: 1.3,
            ModelDomain.TECH: 1.1,
        }
        return multipliers[self]
    
    def to_index(self) -> int:
        """Convert to u8 index for Oracle pallet submission"""
        return self.value


@dataclass
class DomainDataConfig:
    """Configuration for domain-specific data preprocessing"""
    
    # Image preprocessing
    image_size: Tuple[int, int] = (224, 224)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    
    # Data augmentation
    augment_rotation: bool = True
    augment_flip: bool = True
    augment_brightness: bool = True
    
    # Sensor data preprocessing
    sensor_window_size: int = 100  # Time series window
    sensor_normalization: str = "minmax"  # "minmax", "zscore", or "none"
    
    # Text preprocessing
    max_sequence_length: int = 512
    tokenizer_name: str = "gpt2"  # Temporary - will be replaced with custom Belizean BPE tokenizer
    
    # Domain-specific thresholds
    quality_threshold: float = 0.7
    confidence_threshold: float = 0.8


@dataclass
class ModelArchitecturePreferences:
    """Genome evolution preferences for each domain"""
    
    # Preferred layer types (for genome evolution)
    preferred_layers: List[LayerType] = field(default_factory=list)
    
    # Architecture constraints
    min_layers: int = 3
    max_layers: int = 20
    min_hidden_size: int = 64
    max_hidden_size: int = 1024
    
    # Mutation rates (for genetic algorithm)
    layer_mutation_rate: float = 0.1
    weight_mutation_rate: float = 0.05
    
    # Training preferences
    preferred_batch_size: int = 32
    preferred_learning_rate: float = 1e-4
    gradient_clip: float = 1.0


class DomainModel(ABC):
    """
    Abstract base class for domain-specific AI models.
    
    All domain models must implement:
    - preprocess_data(): Convert raw IoT data to model input
    - forward(): Run inference
    - calculate_improvement(): Compute PoUW reward metric
    - get_architecture_preferences(): Guide genome evolution
    """
    
    def __init__(
        self,
        domain: ModelDomain,
        genome: Optional[Genome] = None,
        config: Optional[DomainDataConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.domain = domain
        self.genome = genome
        self.config = config or DomainDataConfig()
        self.device = device
        
        # Build model from genome
        self.model = self._build_model_from_genome()
        self.model.to(self.device)
        
        # Training state
        self.training_rounds = 0
        self.last_accuracy = 0.0
        self.best_accuracy = 0.0
        
        logger.info(
            f"Initialized {self.__class__.__name__} "
            f"(domain={domain.name}, device={device})"
        )
    
    @abstractmethod
    def preprocess_data(self, raw_data: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess raw IoT data for model input.
        
        Args:
            raw_data: Dictionary containing:
                - 'data': Raw bytes from IoT device
                - 'feed_type': Type of data feed (imagery, sensor, etc.)
                - 'location': GPS coordinates (optional)
                - 'timestamp': Collection timestamp
                - 'metadata': Device-specific metadata
        
        Returns:
            Preprocessed tensor ready for model input
        """
        pass
    
    @abstractmethod
    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run model inference.
        
        Args:
            input_tensor: Preprocessed model input
        
        Returns:
            Dictionary containing:
                - 'predictions': Model predictions
                - 'confidence': Prediction confidence scores
                - 'features': Extracted features (for validation)
        """
        pass
    
    @abstractmethod
    def calculate_improvement(
        self,
        old_predictions: Dict[str, torch.Tensor],
        new_predictions: Dict[str, torch.Tensor],
        ground_truth: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Calculate model improvement metric for PoUW rewards.
        
        Args:
            old_predictions: Predictions before federated update
            new_predictions: Predictions after federated update
            ground_truth: Optional ground truth labels
        
        Returns:
            Improvement score (0.0-1.0) for PoUW quality metric
        """
        pass
    
    @abstractmethod
    def get_architecture_preferences(self) -> ModelArchitecturePreferences:
        """
        Get domain-specific architecture preferences for genome evolution.
        
        Returns:
            Architecture preferences guiding genetic algorithm
        """
        pass
    
    def _build_model_from_genome(self) -> nn.Module:
        """Build PyTorch model from genome architecture"""
        if self.genome is None:
            # Use default architecture
            return self._get_default_architecture()
        
        # Convert genome layers to PyTorch modules
        layers = []
        for arch_layer in self.genome.architecture:
            pytorch_layer = self._genome_layer_to_pytorch(arch_layer)
            if pytorch_layer is not None:
                layers.append(pytorch_layer)
        
        return nn.Sequential(*layers)
    
    @abstractmethod
    def _get_default_architecture(self) -> nn.Module:
        """Get default model architecture if no genome provided"""
        pass
    
    def _genome_layer_to_pytorch(self, layer: ArchitectureLayer) -> Optional[nn.Module]:
        """Convert genome layer to PyTorch module"""
        layer_type = layer.layer_type
        params = layer.parameters
        
        if layer_type == LayerType.LINEAR:
            return nn.Linear(
                params.get('in_features', 512),
                params.get('out_features', 512)
            )
        elif layer_type == LayerType.CONV2D:
            return nn.Conv2d(
                params.get('in_channels', 3),
                params.get('out_channels', 64),
                params.get('kernel_size', 3),
                padding=params.get('padding', 1)
            )
        elif layer_type == LayerType.RELU:
            return nn.ReLU()
        elif layer_type == LayerType.DROPOUT:
            return nn.Dropout(params.get('p', 0.1))
        elif layer_type == LayerType.BATCH_NORM:
            num_features = params.get('num_features', 512)
            return nn.BatchNorm1d(num_features)
        elif layer_type == LayerType.LAYER_NORM:
            normalized_shape = params.get('normalized_shape', 512)
            return nn.LayerNorm(normalized_shape)
        else:
            logger.warning(f"Unknown layer type: {layer_type}")
            return None
    
    def update_training_stats(self, accuracy: float):
        """Update training statistics for PoUW tracking"""
        self.training_rounds += 1
        self.last_accuracy = accuracy
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            logger.info(f"New best accuracy: {accuracy:.4f}")


# ============================================================================
# AGRITECH DOMAIN MODEL
# ============================================================================

class AgriTechModel(DomainModel):
    """
    Agricultural Technology AI Model
    
    Use Cases:
    - Crop health monitoring from drone imagery
    - NDVI (Normalized Difference Vegetation Index) calculation
    - Pest and disease detection
    - Soil quality analysis from sensor data
    - Yield prediction
    - Irrigation optimization
    
    IoT Data Sources:
    - Drone multispectral cameras (RGB + NIR)
    - Soil moisture sensors
    - pH sensors
    - Temperature sensors
    - Weather station data
    
    Reward Multiplier: 1.5x (highest priority)
    Priority: Critical for Belize's food security and export economy
    """
    
    def __init__(
        self,
        genome: Optional[Genome] = None,
        config: Optional[DomainDataConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(
            domain=ModelDomain.AGRITECH,
            genome=genome,
            config=config,
            device=device,
        )
        
        # AgriTech-specific modules
        self.ndvi_processor = self._build_ndvi_processor()
        self.pest_detector = self._build_pest_detector()
        
    def preprocess_data(self, raw_data: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess agricultural data.
        
        Supports:
        - Multispectral drone imagery (RGB + NIR for NDVI)
        - Sensor readings (soil moisture, pH, temperature)
        - Weather data
        """
        feed_type = raw_data.get('feed_type', 'unknown')
        
        if feed_type == 'drone_imagery':
            return self._preprocess_drone_imagery(raw_data['data'])
        elif feed_type == 'sensor_reading':
            return self._preprocess_sensor_data(raw_data['data'])
        elif feed_type == 'weather_data':
            return self._preprocess_weather_data(raw_data['data'])
        else:
            raise ValueError(f"Unknown feed type for AgriTech: {feed_type}")
    
    def _preprocess_drone_imagery(self, image_data: bytes) -> torch.Tensor:
        """Preprocess drone imagery with NDVI calculation"""
        # Convert bytes to PIL Image
        import io
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Resize to standard size
        image = image.resize(self.config.image_size)
        
        # Convert to tensor and normalize
        tensor = torch.from_numpy(np.array(image)).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # Normalize
        mean = torch.tensor(self.config.normalize_mean).view(-1, 1, 1)
        std = torch.tensor(self.config.normalize_std).view(-1, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def _preprocess_sensor_data(self, sensor_data: bytes) -> torch.Tensor:
        """Preprocess time-series sensor data"""
        # Parse sensor readings (assume CSV format)
        readings = np.frombuffer(sensor_data, dtype=np.float32)
        
        # Apply sliding window
        if len(readings) > self.config.sensor_window_size:
            readings = readings[-self.config.sensor_window_size:]
        else:
            # Pad if too short
            readings = np.pad(
                readings,
                (0, self.config.sensor_window_size - len(readings)),
                mode='constant'
            )
        
        # Normalize
        if self.config.sensor_normalization == 'minmax':
            readings = (readings - readings.min()) / (readings.max() - readings.min() + 1e-8)
        elif self.config.sensor_normalization == 'zscore':
            readings = (readings - readings.mean()) / (readings.std() + 1e-8)
        
        return torch.from_numpy(readings).float().unsqueeze(0)
    
    def _preprocess_weather_data(self, weather_data: bytes) -> torch.Tensor:
        """Preprocess weather station data"""
        # Similar to sensor data but with different features
        return self._preprocess_sensor_data(weather_data)
    
    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run AgriTech inference.
        
        Returns:
            - crop_health: Health score (0-1)
            - ndvi: NDVI values for vegetation analysis
            - pest_probability: Pest detection confidence
            - recommendations: Actionable farming recommendations
        """
        input_tensor = input_tensor.to(self.device)
        
        # Run through base model
        features = self.model(input_tensor)
        
        # Domain-specific outputs
        if features.dim() == 4:  # Image data (B, C, H, W)
            # NDVI calculation from multispectral imagery
            ndvi = self.ndvi_processor(features)
            
            # Pest detection
            pest_prob = self.pest_detector(features)
            
            # Crop health score (derived from NDVI + pest detection)
            crop_health = (ndvi * 0.7 + (1 - pest_prob) * 0.3).mean(dim=[1, 2, 3])
            
            return {
                'predictions': crop_health,
                'confidence': torch.sigmoid(crop_health),
                'ndvi': ndvi,
                'pest_probability': pest_prob,
                'features': features,
            }
        else:  # Sensor data (B, T)
            # Time series prediction
            predictions = torch.sigmoid(features[:, -1])  # Last timestep
            
            return {
                'predictions': predictions,
                'confidence': torch.abs(predictions - 0.5) * 2,  # Distance from uncertain (0.5)
                'features': features,
            }
    
    def calculate_improvement(
        self,
        old_predictions: Dict[str, torch.Tensor],
        new_predictions: Dict[str, torch.Tensor],
        ground_truth: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Calculate improvement in crop health prediction accuracy.
        
        Metrics:
        - NDVI accuracy improvement
        - Pest detection improvement
        - Overall crop health score improvement
        """
        if ground_truth is not None:
            # Calculate accuracy improvements
            old_error = torch.abs(old_predictions['predictions'] - ground_truth).mean()
            new_error = torch.abs(new_predictions['predictions'] - ground_truth).mean()
            
            # Improvement is reduction in error
            improvement = max(0.0, (old_error - new_error).item())
            
            # Normalize to 0-1 range
            improvement = min(1.0, improvement * 10)  # Scale up small improvements
        else:
            # Without ground truth, use confidence improvement
            old_conf = old_predictions['confidence'].mean()
            new_conf = new_predictions['confidence'].mean()
            improvement = max(0.0, (new_conf - old_conf).item())
        
        return improvement
    
    def get_architecture_preferences(self) -> ModelArchitecturePreferences:
        """AgriTech prefers CNN architectures for image processing"""
        return ModelArchitecturePreferences(
            preferred_layers=[
                LayerType.CONV2D,
                LayerType.BATCH_NORM,
                LayerType.RELU,
                LayerType.MAX_POOL,
            ],
            min_layers=5,
            max_layers=15,
            min_hidden_size=64,
            max_hidden_size=512,
            preferred_batch_size=16,  # Larger images need smaller batches
            preferred_learning_rate=1e-4,
        )
    
    def _get_default_architecture(self) -> nn.Module:
        """Default CNN architecture for crop monitoring"""
        return nn.Sequential(
            # Convolutional layers
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            # Classification head
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),  # Crop health score
        )
    
    def _build_ndvi_processor(self) -> nn.Module:
        """NDVI calculation module for vegetation analysis"""
        return nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),  # NDVI normalized to 0-1
        )
    
    def _build_pest_detector(self) -> nn.Module:
        """Pest detection module"""
        return nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),  # Pest probability 0-1
        )


# ============================================================================
# MARINE DOMAIN MODEL
# ============================================================================

class MarineModel(DomainModel):
    """
    Marine Conservation AI Model
    
    Use Cases:
    - Coral reef health monitoring
    - Water quality analysis (salinity, dissolved oxygen, pH)
    - Marine species identification and tracking
    - Mangrove forest monitoring
    - Coastal erosion detection
    - Blue Hole ecosystem monitoring
    
    IoT Data Sources:
    - Underwater drone cameras
    - Marine buoy sensors
    - Water quality sensors
    - Coastal monitoring cameras
    
    Reward Multiplier: 1.4x
    Priority: High - critical for Belize's marine ecosystem and tourism
    """
    
    def __init__(
        self,
        genome: Optional[Genome] = None,
        config: Optional[DomainDataConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(
            domain=ModelDomain.MARINE,
            genome=genome,
            config=config,
            device=device,
        )
        
        # Marine-specific modules
        self.coral_health_detector = self._build_coral_health_detector()
        self.species_classifier = self._build_species_classifier()
    
    def preprocess_data(self, raw_data: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess marine data.
        
        Supports:
        - Underwater imagery (coral reefs, marine life)
        - Water quality sensor data
        - Coastal imagery for erosion monitoring
        """
        feed_type = raw_data.get('feed_type', 'unknown')
        
        if feed_type in ['drone_imagery', 'camera_feed']:
            return self._preprocess_underwater_imagery(raw_data['data'])
        elif feed_type == 'sensor_reading':
            return self._preprocess_water_quality(raw_data['data'])
        else:
            raise ValueError(f"Unknown feed type for Marine: {feed_type}")
    
    def _preprocess_underwater_imagery(self, image_data: bytes) -> torch.Tensor:
        """Preprocess underwater imagery with color correction"""
        import io
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Underwater images need special color correction (blue/green dominance)
        # Apply red channel enhancement
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array[:, :, 0] = np.clip(image_array[:, :, 0] * 1.5, 0, 1)  # Enhance red
        
        image = Image.fromarray((image_array * 255).astype(np.uint8))
        image = image.resize(self.config.image_size)
        
        tensor = torch.from_numpy(np.array(image)).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        
        # Normalize
        mean = torch.tensor(self.config.normalize_mean).view(-1, 1, 1)
        std = torch.tensor(self.config.normalize_std).view(-1, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor.unsqueeze(0)
    
    def _preprocess_water_quality(self, sensor_data: bytes) -> torch.Tensor:
        """Preprocess water quality sensor data"""
        readings = np.frombuffer(sensor_data, dtype=np.float32)
        
        # Water quality features: [salinity, dissolved_oxygen, pH, temperature]
        # Apply domain-specific normalization
        if len(readings) >= 4:
            # Standard ranges for marine waters
            ranges = np.array([
                [30, 40],    # Salinity (PSU)
                [5, 10],     # Dissolved oxygen (mg/L)
                [7.5, 8.5],  # pH
                [24, 30],    # Temperature (°C)
            ])
            
            normalized = []
            for i, reading in enumerate(readings[:4]):
                min_val, max_val = ranges[i % 4]
                norm = (reading - min_val) / (max_val - min_val)
                normalized.append(norm)
            
            readings = np.array(normalized)
        
        return torch.from_numpy(readings).float().unsqueeze(0)
    
    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run Marine inference.
        
        Returns:
            - coral_health: Coral health score (0-1)
            - water_quality: Water quality index
            - species_detected: Marine species classification
            - biodiversity_index: Ecosystem health metric
        """
        input_tensor = input_tensor.to(self.device)
        features = self.model(input_tensor)
        
        if features.dim() == 4:  # Image data
            # Coral health assessment
            coral_health = self.coral_health_detector(features)
            
            # Species classification
            species_logits = self.species_classifier(features)
            
            # Biodiversity index (based on species diversity)
            biodiversity = torch.sigmoid(species_logits.max(dim=1)[0])
            
            return {
                'predictions': coral_health.mean(dim=[1, 2, 3]),
                'confidence': torch.sigmoid(coral_health.mean(dim=[1, 2, 3])),
                'coral_health': coral_health,
                'species_detected': species_logits,
                'biodiversity_index': biodiversity,
                'features': features,
            }
        else:  # Sensor data
            # Water quality index
            water_quality = torch.sigmoid(features.mean(dim=1))
            
            return {
                'predictions': water_quality,
                'confidence': torch.abs(water_quality - 0.5) * 2,
                'water_quality': water_quality,
                'features': features,
            }
    
    def calculate_improvement(
        self,
        old_predictions: Dict[str, torch.Tensor],
        new_predictions: Dict[str, torch.Tensor],
        ground_truth: Optional[torch.Tensor] = None,
    ) -> float:
        """Calculate improvement in marine health prediction"""
        if ground_truth is not None:
            old_error = torch.abs(old_predictions['predictions'] - ground_truth).mean()
            new_error = torch.abs(new_predictions['predictions'] - ground_truth).mean()
            improvement = max(0.0, (old_error - new_error).item())
            improvement = min(1.0, improvement * 10)
        else:
            old_conf = old_predictions['confidence'].mean()
            new_conf = new_predictions['confidence'].mean()
            improvement = max(0.0, (new_conf - old_conf).item())
        
        return improvement
    
    def get_architecture_preferences(self) -> ModelArchitecturePreferences:
        """Marine prefers CNN for underwater image processing"""
        return ModelArchitecturePreferences(
            preferred_layers=[
                LayerType.CONV2D,
                LayerType.BATCH_NORM,
                LayerType.RELU,
                LayerType.MAX_POOL,
            ],
            min_layers=5,
            max_layers=15,
            preferred_batch_size=16,
            preferred_learning_rate=1e-4,
        )
    
    def _get_default_architecture(self) -> nn.Module:
        """Default CNN for marine imagery"""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
    
    def _build_coral_health_detector(self) -> nn.Module:
        """Coral health assessment module"""
        return nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def _build_species_classifier(self) -> nn.Module:
        """Marine species classification module"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 50),  # 50 common Belizean marine species
        )


# ============================================================================
# EDUCATION DOMAIN MODEL
# ============================================================================

class EducationModel(DomainModel):
    """
    Education AI Model
    
    Use Cases:
    - Personalized learning path recommendations
    - Student performance prediction and intervention
    - Content recommendation
    - Learning style adaptation
    - Educational resource optimization
    - Multilingual education support (English, Spanish, Kriol, Garifuna)
    
    IoT Data Sources:
    - Student interaction data (tablet/phone usage)
    - Performance metrics
    - Attendance data
    - Learning material engagement
    
    Reward Multiplier: 1.3x
    Priority: High - human capital development for Belize's future
    """
    
    def __init__(
        self,
        genome: Optional[Genome] = None,
        config: Optional[DomainDataConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(
            domain=ModelDomain.EDUCATION,
            genome=genome,
            config=config,
            device=device,
        )
        
        # Education-specific: Use transformer for text understanding
        self.llm = BelizeChainLLM(
            model_name=config.tokenizer_name if config else "gpt2",
            num_labels=10,  # 10 learning outcomes
            dropout=0.1,
        )
        self.llm.to(self.device)
    
    def preprocess_data(self, raw_data: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess education data.
        
        Supports:
        - Student interaction sequences
        - Performance time series
        - Learning content embeddings
        """
        feed_type = raw_data.get('feed_type', 'unknown')
        
        if feed_type == 'phone_collection':
            return self._preprocess_student_data(raw_data['data'])
        else:
            raise ValueError(f"Unknown feed type for Education: {feed_type}")
    
    def _preprocess_student_data(self, data: bytes) -> torch.Tensor:
        """Preprocess student interaction data"""
        # Parse student data (JSON format expected)
        import json
        try:
            student_data = json.loads(data.decode('utf-8'))
            
            # Convert to feature vector
            # Features: [time_spent, completion_rate, quiz_scores, interaction_count]
            features = [
                student_data.get('time_spent_minutes', 0) / 60.0,  # Normalize to hours
                student_data.get('completion_rate', 0) / 100.0,
                student_data.get('average_quiz_score', 0) / 100.0,
                student_data.get('interaction_count', 0) / 100.0,
            ]
            
            tensor = torch.tensor(features, dtype=torch.float32)
            return tensor.unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Failed to parse student data: {e}")
            # Return zero tensor as fallback
            return torch.zeros(1, 4)
    
    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run Education inference.
        
        Returns:
            - performance_prediction: Expected student performance
            - learning_style: Identified learning style
            - intervention_needed: Whether intervention is recommended
            - recommended_content: Content recommendations
        """
        input_tensor = input_tensor.to(self.device)
        
        # Pass through model
        features = self.model(input_tensor)
        
        # Performance prediction (0-100 score)
        performance = torch.sigmoid(features[:, 0]) * 100
        
        # Intervention threshold
        intervention_needed = performance < 60.0
        
        # Learning style classification (visual, auditory, kinesthetic, etc.)
        learning_style_logits = features[:, 1:5]
        learning_style = torch.softmax(learning_style_logits, dim=1)
        
        return {
            'predictions': performance / 100.0,  # Normalize to 0-1
            'confidence': torch.abs(performance - 50) / 50,
            'performance_prediction': performance,
            'learning_style': learning_style,
            'intervention_needed': intervention_needed,
            'features': features,
        }
    
    def calculate_improvement(
        self,
        old_predictions: Dict[str, torch.Tensor],
        new_predictions: Dict[str, torch.Tensor],
        ground_truth: Optional[torch.Tensor] = None,
    ) -> float:
        """Calculate improvement in student performance prediction"""
        if ground_truth is not None:
            old_error = torch.abs(old_predictions['predictions'] - ground_truth).mean()
            new_error = torch.abs(new_predictions['predictions'] - ground_truth).mean()
            improvement = max(0.0, (old_error - new_error).item())
            improvement = min(1.0, improvement * 5)  # Education needs precise predictions
        else:
            old_conf = old_predictions['confidence'].mean()
            new_conf = new_predictions['confidence'].mean()
            improvement = max(0.0, (new_conf - old_conf).item())
        
        return improvement
    
    def get_architecture_preferences(self) -> ModelArchitecturePreferences:
        """Education prefers transformer architectures for sequential data"""
        return ModelArchitecturePreferences(
            preferred_layers=[
                LayerType.LINEAR,
                LayerType.LAYER_NORM,
                LayerType.GELU,
                LayerType.MULTIHEAD_ATTENTION,
            ],
            min_layers=4,
            max_layers=12,
            preferred_batch_size=32,
            preferred_learning_rate=2e-5,
        )
    
    def _get_default_architecture(self) -> nn.Module:
        """Default architecture for student performance prediction"""
        return nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),  # 10 output features
        )


# ============================================================================
# TECH DOMAIN MODEL
# ============================================================================

class TechModel(DomainModel):
    """
    Technology & Infrastructure AI Model
    
    Use Cases:
    - Software quality metrics and bug prediction
    - Infrastructure performance monitoring
    - Security anomaly detection
    - Resource optimization
    - Load balancing recommendations
    - Code quality analysis
    
    IoT Data Sources:
    - Server metrics (CPU, memory, network)
    - Application logs
    - Performance traces
    - Security event data
    
    Reward Multiplier: 1.1x
    Priority: Medium - infrastructure optimization and reliability
    """
    
    def __init__(
        self,
        genome: Optional[Genome] = None,
        config: Optional[DomainDataConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(
            domain=ModelDomain.TECH,
            genome=genome,
            config=config,
            device=device,
        )
        
        # Tech-specific: Time series analysis for metrics
        self.anomaly_detector = self._build_anomaly_detector()
    
    def preprocess_data(self, raw_data: Dict[str, Any]) -> torch.Tensor:
        """
        Preprocess infrastructure metrics.
        
        Supports:
        - Time series metrics (CPU, memory, network)
        - Event sequences
        - Performance traces
        """
        feed_type = raw_data.get('feed_type', 'unknown')
        
        if feed_type in ['sensor_reading', 'phone_collection']:
            return self._preprocess_metrics(raw_data['data'])
        else:
            raise ValueError(f"Unknown feed type for Tech: {feed_type}")
    
    def _preprocess_metrics(self, data: bytes) -> torch.Tensor:
        """Preprocess infrastructure metrics time series"""
        # Parse metrics (assume time series of [cpu, memory, network, disk])
        metrics = np.frombuffer(data, dtype=np.float32)
        
        # Reshape to (timesteps, features)
        if len(metrics) >= self.config.sensor_window_size * 4:
            metrics = metrics[:self.config.sensor_window_size * 4]
            metrics = metrics.reshape(self.config.sensor_window_size, 4)
        else:
            # Pad if needed
            target_len = self.config.sensor_window_size * 4
            metrics = np.pad(metrics, (0, target_len - len(metrics)), mode='constant')
            metrics = metrics.reshape(self.config.sensor_window_size, 4)
        
        # Normalize to 0-100 range (percentage)
        metrics = np.clip(metrics, 0, 100) / 100.0
        
        return torch.from_numpy(metrics).float().unsqueeze(0)  # (1, T, F)
    
    def forward(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Run Tech inference.
        
        Returns:
            - anomaly_score: Infrastructure anomaly detection
            - performance_prediction: Predicted performance
            - resource_recommendations: Optimization suggestions
            - incident_probability: Likelihood of incident
        """
        input_tensor = input_tensor.to(self.device)
        
        # Flatten time series for processing
        batch_size = input_tensor.shape[0]
        input_flat = input_tensor.view(batch_size, -1)
        
        # Pass through model
        features = self.model(input_flat)
        
        # Anomaly detection
        anomaly_score = self.anomaly_detector(features)
        
        # Performance prediction (0-1, higher is better)
        performance = torch.sigmoid(features[:, 0])
        
        # Incident probability (0-1)
        incident_prob = torch.sigmoid(features[:, 1])
        
        return {
            'predictions': performance,
            'confidence': torch.abs(performance - 0.5) * 2,
            'anomaly_score': anomaly_score,
            'performance_prediction': performance,
            'incident_probability': incident_prob,
            'features': features,
        }
    
    def calculate_improvement(
        self,
        old_predictions: Dict[str, torch.Tensor],
        new_predictions: Dict[str, torch.Tensor],
        ground_truth: Optional[torch.Tensor] = None,
    ) -> float:
        """Calculate improvement in infrastructure monitoring"""
        if ground_truth is not None:
            old_error = torch.abs(old_predictions['predictions'] - ground_truth).mean()
            new_error = torch.abs(new_predictions['predictions'] - ground_truth).mean()
            improvement = max(0.0, (old_error - new_error).item())
            improvement = min(1.0, improvement * 10)
        else:
            old_conf = old_predictions['confidence'].mean()
            new_conf = new_predictions['confidence'].mean()
            improvement = max(0.0, (new_conf - old_conf).item())
        
        return improvement
    
    def get_architecture_preferences(self) -> ModelArchitecturePreferences:
        """Tech prefers LSTM/Transformer for time series"""
        return ModelArchitecturePreferences(
            preferred_layers=[
                LayerType.LINEAR,
                LayerType.GELU,
                LayerType.LAYER_NORM,
                LayerType.DROPOUT,
            ],
            min_layers=3,
            max_layers=10,
            preferred_batch_size=64,
            preferred_learning_rate=1e-3,
        )
    
    def _get_default_architecture(self) -> nn.Module:
        """Default architecture for infrastructure metrics"""
        return nn.Sequential(
            nn.Linear(400, 256),  # 100 timesteps * 4 features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
    
    def _build_anomaly_detector(self) -> nn.Module:
        """Anomaly detection module"""
        return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )


# ============================================================================
# DOMAIN MODEL FACTORY
# ============================================================================

class DomainModelFactory:
    """Factory for creating domain-specific models"""
    
    @staticmethod
    def create_model(
        domain: ModelDomain,
        genome: Optional[Genome] = None,
        config: Optional[DomainDataConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> DomainModel:
        """
        Create a domain-specific model.
        
        Args:
            domain: Target domain (AgriTech, Marine, Education, Tech, General)
            genome: Optional genome for architecture evolution
            config: Optional data configuration
            device: Device for model execution
        
        Returns:
            Initialized domain model
        """
        model_classes = {
            ModelDomain.AGRITECH: AgriTechModel,
            ModelDomain.MARINE: MarineModel,
            ModelDomain.EDUCATION: EducationModel,
            ModelDomain.TECH: TechModel,
            ModelDomain.GENERAL: AgriTechModel,  # Use AgriTech as default
        }
        
        model_class = model_classes.get(domain)
        if model_class is None:
            raise ValueError(f"Unknown domain: {domain}")
        
        return model_class(genome=genome, config=config, device=device)
    
    @staticmethod
    def list_available_domains() -> List[ModelDomain]:
        """List all available domains"""
        return list(ModelDomain)
    
    @staticmethod
    def get_domain_info(domain: ModelDomain) -> Dict[str, Any]:
        """Get information about a domain"""
        info = {
            ModelDomain.AGRITECH: {
                "name": "Agricultural Technology",
                "reward_multiplier": 1.5,
                "priority": "Critical",
                "use_cases": [
                    "Crop health monitoring",
                    "NDVI calculation",
                    "Pest detection",
                    "Soil analysis",
                    "Yield prediction",
                ],
                "data_sources": ["Drones", "Soil sensors", "Weather stations"],
            },
            ModelDomain.MARINE: {
                "name": "Marine Conservation",
                "reward_multiplier": 1.4,
                "priority": "High",
                "use_cases": [
                    "Coral reef monitoring",
                    "Water quality analysis",
                    "Species tracking",
                    "Coastal erosion detection",
                ],
                "data_sources": ["Underwater drones", "Marine buoys", "Cameras"],
            },
            ModelDomain.EDUCATION: {
                "name": "Education",
                "reward_multiplier": 1.3,
                "priority": "High",
                "use_cases": [
                    "Personalized learning",
                    "Performance prediction",
                    "Content recommendation",
                    "Intervention detection",
                ],
                "data_sources": ["Student devices", "Learning platforms"],
            },
            ModelDomain.TECH: {
                "name": "Technology & Infrastructure",
                "reward_multiplier": 1.1,
                "priority": "Medium",
                "use_cases": [
                    "Infrastructure monitoring",
                    "Anomaly detection",
                    "Performance optimization",
                    "Security analysis",
                ],
                "data_sources": ["Server metrics", "Application logs", "Traces"],
            },
            ModelDomain.GENERAL: {
                "name": "General Purpose",
                "reward_multiplier": 1.0,
                "priority": "Baseline",
                "use_cases": [
                    "Government services",
                    "Healthcare data",
                    "General monitoring",
                ],
                "data_sources": ["Various IoT devices"],
            },
        }
        
        return info.get(domain, {})


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_quality_score(
    accuracy: int,
    timeliness: int,
    completeness: int,
    consistency: int,
    provenance: int,
) -> int:
    """
    Calculate quality score matching Oracle pallet algorithm.
    
    Weighted factors (0-100 each → final score 0-1000):
    - Accuracy (30%)
    - Timeliness (20%)
    - Completeness (15%)
    - Consistency (20%)
    - Provenance (15%)
    
    Args:
        accuracy: 0-100
        timeliness: 0-100
        completeness: 0-100
        consistency: 0-100
        provenance: 0-100
    
    Returns:
        Quality score 0-1000
    """
    weighted_sum = (
        accuracy * 30 +
        timeliness * 20 +
        completeness * 15 +
        consistency * 20 +
        provenance * 15
    )
    return weighted_sum // 10  # Scale to 0-1000


def prepare_oracle_submission(
    domain: ModelDomain,
    device_id: bytes,
    data: bytes,
    predictions: Dict[str, torch.Tensor],
    quality_metrics: Dict[str, int],
) -> Dict[str, Any]:
    """
    Prepare data submission for Oracle pallet.
    
    Args:
        domain: Model domain
        device_id: IoT device identifier (32 bytes)
        data: Raw data bytes
        predictions: Model predictions
        quality_metrics: Quality assessment {accuracy, timeliness, etc.}
    
    Returns:
        Dictionary ready for Oracle pallet submission
    """
    import hashlib
    
    # Calculate data hash
    data_hash = hashlib.sha256(data).digest()
    
    # Extract confidence for provenance score
    confidence = predictions.get('confidence', torch.tensor([0.5])).mean().item()
    provenance = int(confidence * 100)
    
    # Calculate overall quality score
    quality_score = calculate_quality_score(
        accuracy=quality_metrics.get('accuracy', 80),
        timeliness=quality_metrics.get('timeliness', 100),
        completeness=quality_metrics.get('completeness', 100),
        consistency=quality_metrics.get('consistency', 85),
        provenance=provenance,
    )
    
    return {
        'device_id': device_id,
        'feed_type_index': 10,  # DroneImagery (example)
        'domain_index': domain.to_index(),
        'data': data,
        'data_hash': data_hash,
        'location': None,  # Optional GPS coordinates
        'accuracy': quality_metrics.get('accuracy', 80),
        'timeliness': quality_metrics.get('timeliness', 100),
        'completeness': quality_metrics.get('completeness', 100),
        'consistency': quality_metrics.get('consistency', 85),
        'provenance': provenance,
        'quality_score': quality_score,
    }
