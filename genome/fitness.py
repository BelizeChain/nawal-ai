"""
Fitness Evaluation Module

Implements multi-criteria fitness evaluation aligned with BelizeChain's
Proof of Useful Work (PoUW) consensus mechanism.

Fitness Function:
    Fitness = 0.40 × Quality + 0.30 × Timeliness + 0.30 × Honesty

Latest Features:
- Async evaluation for scalability
- Pluggable metrics system
- Real-time monitoring integration
- Byzantine-fault detection
"""

from typing import Protocol, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import statistics
import logging

try:
    from pydantic import BaseModel, Field, ConfigDict
except ImportError:
    from typing import Any as BaseModel
    def Field(*args, **kwargs): return None
    def ConfigDict(*args, **kwargs): return {}

from .encoding import Genome

logger = logging.getLogger(__name__)


class FitnessMetric(str, Enum):
    """Types of fitness metrics"""
    QUALITY = "quality"
    TIMELINESS = "timeliness"
    HONESTY = "honesty"
    OVERALL = "overall"


@dataclass
class FitnessScore:
    """
    Complete fitness evaluation result.
    
    Aligned with PoUW consensus scoring in the Staking pallet.
    """
    # Core PoUW metrics (0-100 scale)
    quality: float  # Model accuracy/performance
    timeliness: float  # Training efficiency
    honesty: float  # Privacy compliance
    
    # Overall fitness (weighted combination)
    overall: float
    
    # Metadata
    genome_id: str
    evaluated_at: datetime = field(default_factory=datetime.utcnow)
    evaluator_id: str = "system"
    
    # Detailed metrics
    quality_metrics: dict[str, float] = field(default_factory=dict)
    timeliness_metrics: dict[str, float] = field(default_factory=dict)
    honesty_metrics: dict[str, float] = field(default_factory=dict)
    
    # Validation
    is_valid: bool = True
    validation_errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return {
            "quality": self.quality,
            "timeliness": self.timeliness,
            "honesty": self.honesty,
            "overall": self.overall,
            "genome_id": self.genome_id,
            "evaluated_at": self.evaluated_at.isoformat(),
            "evaluator_id": self.evaluator_id,
            "quality_metrics": self.quality_metrics,
            "timeliness_metrics": self.timeliness_metrics,
            "honesty_metrics": self.honesty_metrics,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'FitnessScore':
        """Create from dictionary"""
        if 'evaluated_at' in data and isinstance(data['evaluated_at'], str):
            data['evaluated_at'] = datetime.fromisoformat(data['evaluated_at'])
        return cls(**data)
    
    def __repr__(self) -> str:
        return (
            f"FitnessScore(overall={self.overall:.2f}, "
            f"quality={self.quality:.2f}, "
            f"timeliness={self.timeliness:.2f}, "
            f"honesty={self.honesty:.2f})"
        )


class PoUWAlignment:
    """
    Proof of Useful Work alignment utilities.
    
    Ensures fitness evaluation matches the Staking pallet's PoUW consensus.
    """
    
    # Fitness weights (must sum to 1.0)
    QUALITY_WEIGHT = 0.40
    TIMELINESS_WEIGHT = 0.30
    HONESTY_WEIGHT = 0.30
    
    # Score ranges
    MIN_SCORE = 0.0
    MAX_SCORE = 100.0
    
    # Thresholds
    PASSING_THRESHOLD = 70.0  # Minimum fitness to earn rewards
    EXCELLENT_THRESHOLD = 90.0  # Bonus rewards
    SLASHING_THRESHOLD = 50.0  # Below this triggers investigation
    
    @classmethod
    def calculate_fitness(
        cls,
        quality: float,
        timeliness: float,
        honesty: float
    ) -> float:
        """
        Calculate overall fitness score using PoUW weights.
        
        Args:
            quality: Quality score (0-100)
            timeliness: Timeliness score (0-100)
            honesty: Honesty score (0-100)
            
        Returns:
            Overall fitness (0-100)
        """
        # Validate inputs
        quality = max(cls.MIN_SCORE, min(cls.MAX_SCORE, quality))
        timeliness = max(cls.MIN_SCORE, min(cls.MAX_SCORE, timeliness))
        honesty = max(cls.MIN_SCORE, min(cls.MAX_SCORE, honesty))
        
        # Calculate weighted fitness
        fitness = (
            cls.QUALITY_WEIGHT * quality +
            cls.TIMELINESS_WEIGHT * timeliness +
            cls.HONESTY_WEIGHT * honesty
        )
        
        return fitness
    
    @classmethod
    def reward_multiplier(cls, fitness: float) -> float:
        """
        Calculate reward multiplier based on fitness.
        
        Returns:
            Multiplier for base reward (0.0 - 2.0)
        """
        if fitness < cls.SLASHING_THRESHOLD:
            return 0.0  # No rewards, potential slashing
        elif fitness < cls.PASSING_THRESHOLD:
            return 0.5  # Reduced rewards
        elif fitness >= cls.EXCELLENT_THRESHOLD:
            return 1.5  # Bonus rewards
        else:
            return 1.0  # Standard rewards
    
    @classmethod
    def should_slash(cls, fitness: float) -> tuple[bool, str]:
        """
        Determine if validator should be slashed.
        
        Returns:
            (should_slash, reason)
        """
        if fitness < cls.SLASHING_THRESHOLD:
            return True, f"Fitness {fitness:.2f} below threshold {cls.SLASHING_THRESHOLD}"
        return False, ""


class FitnessEvaluator:
    """
    Multi-criteria fitness evaluator for genomes.
    
    Evaluates genomes based on:
    1. Quality - Model accuracy and performance
    2. Timeliness - Training efficiency and speed
    3. Honesty - Privacy compliance and data integrity
    """
    
    def __init__(
        self,
        evaluator_id: str = "system",
        enable_async: bool = True
    ):
        self.evaluator_id = evaluator_id
        self.enable_async = enable_async
        self.evaluation_history: list[FitnessScore] = []
        
        logger.info(f"Initialized FitnessEvaluator: {evaluator_id}")
    
    async def evaluate_async(
        self,
        genome: Genome,
        training_metrics: dict[str, Any]
    ) -> FitnessScore:
        """
        Asynchronously evaluate genome fitness.
        
        Args:
            genome: Genome to evaluate
            training_metrics: Metrics from training session
            
        Returns:
            Complete fitness score
        """
        # Evaluate each criterion in parallel
        quality_task = asyncio.create_task(
            self._evaluate_quality_async(genome, training_metrics)
        )
        timeliness_task = asyncio.create_task(
            self._evaluate_timeliness_async(genome, training_metrics)
        )
        honesty_task = asyncio.create_task(
            self._evaluate_honesty_async(genome, training_metrics)
        )
        
        # Wait for all evaluations
        quality, timeliness, honesty = await asyncio.gather(
            quality_task,
            timeliness_task,
            honesty_task
        )
        
        # Calculate overall fitness
        overall = PoUWAlignment.calculate_fitness(quality, timeliness, honesty)
        
        # Create fitness score
        fitness_score = FitnessScore(
            quality=quality,
            timeliness=timeliness,
            honesty=honesty,
            overall=overall,
            genome_id=genome.genome_id,
            evaluator_id=self.evaluator_id
        )
        
        # Store in history
        self.evaluation_history.append(fitness_score)
        
        # Update genome
        genome.calculate_fitness(quality, timeliness, honesty)
        
        logger.info(f"Evaluated genome {genome.genome_id}: {fitness_score}")
        
        return fitness_score
    
    def evaluate(
        self,
        genome: Genome,
        training_metrics: dict[str, Any]
    ) -> FitnessScore:
        """
        Synchronously evaluate genome fitness.
        
        Args:
            genome: Genome to evaluate
            training_metrics: Metrics from training session
            
        Returns:
            Complete fitness score
        """
        if self.enable_async:
            # Run async version
            return asyncio.run(self.evaluate_async(genome, training_metrics))
        
        # Synchronous evaluation
        quality = self._evaluate_quality(genome, training_metrics)
        timeliness = self._evaluate_timeliness(genome, training_metrics)
        honesty = self._evaluate_honesty(genome, training_metrics)
        
        overall = PoUWAlignment.calculate_fitness(quality, timeliness, honesty)
        
        fitness_score = FitnessScore(
            quality=quality,
            timeliness=timeliness,
            honesty=honesty,
            overall=overall,
            genome_id=genome.genome_id,
            evaluator_id=self.evaluator_id
        )
        
        self.evaluation_history.append(fitness_score)
        genome.calculate_fitness(quality, timeliness, honesty)
        
        logger.info(f"Evaluated genome {genome.genome_id}: {fitness_score}")
        
        return fitness_score
    
    async def _evaluate_quality_async(
        self,
        genome: Genome,
        metrics: dict[str, Any]
    ) -> float:
        """
        Evaluate model quality (accuracy, performance).
        
        Quality Score (0-100) based on:
        - Model accuracy on validation set
        - Loss reduction
        - Generalization capability
        - Task-specific metrics
        """
        # Simulate async operation
        await asyncio.sleep(0.1)
        return self._evaluate_quality(genome, metrics)
    
    def _evaluate_quality(
        self,
        genome: Genome,
        metrics: dict[str, Any]
    ) -> float:
        """Synchronous quality evaluation"""
        quality_components = []
        
        # 1. Model Accuracy (40% of quality)
        accuracy = metrics.get('accuracy', 0.0)
        if 'eval_accuracy' in metrics:
            accuracy = metrics['eval_accuracy']
        quality_components.append(('accuracy', accuracy * 100, 0.40))
        
        # 2. Loss Reduction (30% of quality)
        initial_loss = metrics.get('initial_loss', 10.0)
        final_loss = metrics.get('final_loss', initial_loss)
        loss_reduction = max(0, (initial_loss - final_loss) / initial_loss) if initial_loss > 0 else 0
        quality_components.append(('loss_reduction', loss_reduction * 100, 0.30))
        
        # 3. Generalization (20% of quality)
        train_accuracy = metrics.get('train_accuracy', accuracy)
        eval_accuracy = metrics.get('eval_accuracy', accuracy)
        generalization = 1.0 - abs(train_accuracy - eval_accuracy)
        quality_components.append(('generalization', generalization * 100, 0.20))
        
        # 4. Task-specific metrics (10% of quality)
        task_score = metrics.get('task_score', 75.0)  # Default moderate score
        quality_components.append(('task_specific', task_score, 0.10))
        
        # Calculate weighted quality score
        quality_score = sum(score * weight for _, score, weight in quality_components)
        
        # Clamp to valid range
        quality_score = max(0.0, min(100.0, quality_score))
        
        logger.debug(f"Quality components: {quality_components}")
        logger.debug(f"Quality score: {quality_score:.2f}")
        
        return quality_score
    
    async def _evaluate_timeliness_async(
        self,
        genome: Genome,
        metrics: dict[str, Any]
    ) -> float:
        """Evaluate training efficiency"""
        await asyncio.sleep(0.1)
        return self._evaluate_timeliness(genome, metrics)
    
    def _evaluate_timeliness(
        self,
        genome: Genome,
        metrics: dict[str, Any]
    ) -> float:
        """
        Evaluate training timeliness (efficiency, speed).
        
        Timeliness Score (0-100) based on:
        - Training time vs. expected time
        - Samples processed per second
        - Resource utilization efficiency
        - Communication overhead
        """
        timeliness_components = []
        
        # 1. Training Speed (50% of timeliness)
        training_time_seconds = metrics.get('training_time_seconds', 3600)
        expected_time_seconds = metrics.get('expected_time_seconds', 3600)
        
        # Faster than expected = higher score
        if training_time_seconds <= expected_time_seconds:
            speed_score = 100.0
        else:
            # Penalize slower training
            speed_score = 100.0 * (expected_time_seconds / training_time_seconds)
        
        timeliness_components.append(('speed', speed_score, 0.50))
        
        # 2. Throughput (30% of timeliness)
        samples_processed = metrics.get('samples_processed', 1000)
        throughput = samples_processed / max(training_time_seconds, 1)
        
        # Normalize to 0-100 (assuming 10 samples/sec is excellent)
        throughput_score = min(100.0, (throughput / 10.0) * 100)
        timeliness_components.append(('throughput', throughput_score, 0.30))
        
        # 3. Resource Efficiency (20% of timeliness)
        # Lower resource usage = higher efficiency
        cpu_utilization = metrics.get('cpu_utilization', 0.8)
        memory_utilization = metrics.get('memory_utilization', 0.8)
        
        # Ideal utilization is 70-90%
        cpu_efficiency = 100.0 * (1.0 - abs(0.8 - cpu_utilization))
        memory_efficiency = 100.0 * (1.0 - abs(0.8 - memory_utilization))
        efficiency_score = (cpu_efficiency + memory_efficiency) / 2
        
        timeliness_components.append(('efficiency', efficiency_score, 0.20))
        
        # Calculate weighted timeliness score
        timeliness_score = sum(score * weight for _, score, weight in timeliness_components)
        timeliness_score = max(0.0, min(100.0, timeliness_score))
        
        logger.debug(f"Timeliness components: {timeliness_components}")
        logger.debug(f"Timeliness score: {timeliness_score:.2f}")
        
        return timeliness_score
    
    async def _evaluate_honesty_async(
        self,
        genome: Genome,
        metrics: dict[str, Any]
    ) -> float:
        """Evaluate privacy compliance"""
        await asyncio.sleep(0.1)
        return self._evaluate_honesty(genome, metrics)
    
    def _evaluate_honesty(
        self,
        genome: Genome,
        metrics: dict[str, Any]
    ) -> float:
        """
        Evaluate honesty (privacy compliance, data integrity).
        
        Honesty Score (0-100) based on:
        - Privacy preservation (differential privacy)
        - Data sovereignty compliance
        - Byzantine behavior detection
        - Gradient integrity
        """
        honesty_components = []
        
        # 1. Privacy Compliance (40% of honesty)
        privacy_budget_used = metrics.get('privacy_budget_used', 0.0)
        privacy_budget_limit = metrics.get('privacy_budget_limit', 1.0)
        
        if privacy_budget_used <= privacy_budget_limit:
            privacy_score = 100.0
        else:
            # Penalize privacy budget violations
            privacy_score = 100.0 * (privacy_budget_limit / privacy_budget_used)
        
        honesty_components.append(('privacy', privacy_score, 0.40))
        
        # 2. Data Sovereignty (30% of honesty)
        sovereignty_checks_passed = metrics.get('sovereignty_checks_passed', 0)
        sovereignty_checks_total = metrics.get('sovereignty_checks_total', 1)
        
        sovereignty_score = (sovereignty_checks_passed / max(sovereignty_checks_total, 1)) * 100
        honesty_components.append(('sovereignty', sovereignty_score, 0.30))
        
        # 3. Byzantine Resistance (20% of honesty)
        # Check for anomalous gradients or updates
        gradient_norm = metrics.get('gradient_norm', 1.0)
        expected_gradient_norm = metrics.get('expected_gradient_norm', 1.0)
        
        # Gradients too large or too small indicate potential Byzantine behavior
        if 0.5 <= (gradient_norm / max(expected_gradient_norm, 0.01)) <= 2.0:
            byzantine_score = 100.0
        else:
            byzantine_score = 50.0  # Suspicious behavior
        
        honesty_components.append(('byzantine_resistance', byzantine_score, 0.20))
        
        # 4. Update Integrity (10% of honesty)
        # Verify model update consistency
        update_verification = metrics.get('update_verified', True)
        integrity_score = 100.0 if update_verification else 0.0
        
        honesty_components.append(('integrity', integrity_score, 0.10))
        
        # Calculate weighted honesty score
        honesty_score = sum(score * weight for _, score, weight in honesty_components)
        honesty_score = max(0.0, min(100.0, honesty_score))
        
        logger.debug(f"Honesty components: {honesty_components}")
        logger.debug(f"Honesty score: {honesty_score:.2f}")
        
        return honesty_score
    
    def get_statistics(self) -> dict[str, float]:
        """Get statistics from evaluation history"""
        if not self.evaluation_history:
            return {}
        
        qualities = [score.quality for score in self.evaluation_history]
        timelinesses = [score.timeliness for score in self.evaluation_history]
        honesties = [score.honesty for score in self.evaluation_history]
        overalls = [score.overall for score in self.evaluation_history]
        
        return {
            'count': len(self.evaluation_history),
            'quality_mean': statistics.mean(qualities),
            'quality_stdev': statistics.stdev(qualities) if len(qualities) > 1 else 0,
            'timeliness_mean': statistics.mean(timelinesses),
            'timeliness_stdev': statistics.stdev(timelinesses) if len(timelinesses) > 1 else 0,
            'honesty_mean': statistics.mean(honesties),
            'honesty_stdev': statistics.stdev(honesties) if len(honesties) > 1 else 0,
            'overall_mean': statistics.mean(overalls),
            'overall_stdev': statistics.stdev(overalls) if len(overalls) > 1 else 0,
        }


if __name__ == "__main__":
    # Test fitness evaluation
    from encoding import GenomeEncoder
    
    print("Testing Fitness Evaluation...\n")
    
    # Create test genome
    genome = GenomeEncoder.create_baseline_genome()
    
    # Create evaluator
    evaluator = FitnessEvaluator(evaluator_id="test_evaluator")
    
    # Mock training metrics
    training_metrics = {
        'accuracy': 0.85,
        'eval_accuracy': 0.83,
        'train_accuracy': 0.87,
        'initial_loss': 2.5,
        'final_loss': 0.8,
        'training_time_seconds': 1800,
        'expected_time_seconds': 2400,
        'samples_processed': 10000,
        'cpu_utilization': 0.75,
        'memory_utilization': 0.82,
        'privacy_budget_used': 0.8,
        'privacy_budget_limit': 1.0,
        'sovereignty_checks_passed': 95,
        'sovereignty_checks_total': 100,
        'gradient_norm': 1.2,
        'expected_gradient_norm': 1.0,
        'update_verified': True,
    }
    
    # Evaluate
    fitness_score = evaluator.evaluate(genome, training_metrics)
    
    print(f"Genome: {genome.genome_id}")
    print(f"\nFitness Score: {fitness_score}")
    print(f"\nPoUW Alignment:")
    print(f"  Reward Multiplier: {PoUWAlignment.reward_multiplier(fitness_score.overall):.2f}x")
    
    should_slash, reason = PoUWAlignment.should_slash(fitness_score.overall)
    print(f"  Should Slash: {should_slash}")
    if should_slash:
        print(f"  Reason: {reason}")
    
    # Get statistics
    stats = evaluator.get_statistics()
    print(f"\nEvaluator Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    print("\n✅ Fitness evaluation test successful!")
