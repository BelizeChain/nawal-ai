"""
Population Management for Genome Evolution

This module manages the population of genomes across generations:
- Population storage and retrieval
- Selection strategies (tournament, roulette, rank)
- Diversity maintenance
- Elite preservation

Aligned with BelizeChain's distributed validator network.

Author: BelizeChain AI Team
Date: October 2025
Python: 3.13+
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol, Callable
from loguru import logger

from .encoding import Genome
from .fitness import FitnessScore
from .operators import SelectionStrategy


# =============================================================================
# Population Configuration
# =============================================================================


@dataclass
class PopulationConfig:
    """Configuration for population management."""
    
    # Population size
    target_size: int = 50              # Target population size
    min_size: int = 20                 # Minimum before regeneration
    max_size: int = 100                # Maximum allowed size
    
    # Selection
    selection_strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT
    tournament_size: int = 5           # For tournament selection
    
    # Elitism
    elitism_count: int = 2             # Top genomes to preserve
    elitism_threshold: float = 0.8     # Minimum fitness for elite (0-1)
    
    # Diversity
    maintain_diversity: bool = True    # Enforce diversity constraints
    min_diversity_score: float = 0.3   # Minimum architectural diversity
    max_similar_genomes: int = 5       # Max genomes with similar architecture
    
    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration."""
        errors = []
        
        if self.min_size < 1:
            errors.append("min_size must be >= 1")
        
        if self.target_size < self.min_size:
            errors.append("target_size must be >= min_size")
        
        if self.max_size < self.target_size:
            errors.append("max_size must be >= target_size")
        
        if self.tournament_size < 2:
            errors.append("tournament_size must be >= 2")
        
        if not (0.0 <= self.elitism_threshold <= 1.0):
            errors.append("elitism_threshold must be in [0, 1]")
        
        if self.elitism_count > self.target_size:
            errors.append("elitism_count must be <= target_size")
        
        return (len(errors) == 0, errors)


# =============================================================================
# Population Statistics
# =============================================================================


@dataclass
class PopulationStatistics:
    """Statistics about the current population."""
    
    generation: int
    population_size: int
    
    # Fitness statistics
    avg_fitness: float
    max_fitness: float
    min_fitness: float
    std_fitness: float
    
    # Component fitness
    avg_quality: float
    avg_timeliness: float
    avg_honesty: float
    
    # Diversity metrics
    unique_architectures: int
    diversity_score: float
    
    # Elite genomes
    elite_count: int
    elite_avg_fitness: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "generation": self.generation,
            "population_size": self.population_size,
            "avg_fitness": self.avg_fitness,
            "max_fitness": self.max_fitness,
            "min_fitness": self.min_fitness,
            "std_fitness": self.std_fitness,
            "avg_quality": self.avg_quality,
            "avg_timeliness": self.avg_timeliness,
            "avg_honesty": self.avg_honesty,
            "unique_architectures": self.unique_architectures,
            "diversity_score": self.diversity_score,
            "elite_count": self.elite_count,
            "elite_avg_fitness": self.elite_avg_fitness,
        }


# =============================================================================
# Population Manager
# =============================================================================


class PopulationManager:
    """
    Manages population of genomes across generations.
    
    Responsibilities:
    - Maintain population within size constraints
    - Select parents for reproduction
    - Preserve elite genomes
    - Ensure population diversity
    - Collect population statistics
    """
    
    def __init__(self, config: PopulationConfig | None = None):
        """
        Initialize population manager.
        
        Args:
            config: Population configuration (uses defaults if None)
        """
        self.config = config or PopulationConfig()
        
        # Validate configuration
        is_valid, errors = self.config.validate()
        if not is_valid:
            raise ValueError(f"Invalid population config: {', '.join(errors)}")
        
        # Population storage
        self.genomes: dict[str, Genome] = {}
        self.elite_genomes: list[str] = []  # Genome IDs of elite
        
        # Statistics
        self.generation_stats: dict[int, PopulationStatistics] = {}
        
        logger.info(
            "Initialized PopulationManager",
            target_size=self.config.target_size,
            selection=self.config.selection_strategy.name,
            elitism=self.config.elitism_count,
        )
    
    def add_genome(self, genome: Genome) -> None:
        """
        Add genome to population.
        
        Args:
            genome: Genome to add
        """
        if genome.genome_id in self.genomes:
            logger.warning(f"Genome {genome.genome_id} already exists, replacing")
        
        self.genomes[genome.genome_id] = genome
        
        # Check population size
        if len(self.genomes) > self.config.max_size:
            self._cull_population()
    
    def remove_genome(self, genome_id: str) -> None:
        """
        Remove genome from population.
        
        Args:
            genome_id: ID of genome to remove
        """
        if genome_id in self.genomes:
            del self.genomes[genome_id]
            
            # Remove from elite if present
            if genome_id in self.elite_genomes:
                self.elite_genomes.remove(genome_id)
    
    def get_genome(self, genome_id: str) -> Genome | None:
        """
        Retrieve genome by ID.
        
        Args:
            genome_id: Genome identifier
        
        Returns:
            Genome if found, None otherwise
        """
        return self.genomes.get(genome_id)
    
    def get_all_genomes(self) -> list[Genome]:
        """Get all genomes in population."""
        return list(self.genomes.values())
    
    def get_elite_genomes(self) -> list[Genome]:
        """Get elite genomes only."""
        return [
            self.genomes[gid]
            for gid in self.elite_genomes
            if gid in self.genomes
        ]
    
    def select_parent(self) -> Genome:
        """
        Select parent genome for reproduction.
        
        Returns:
            Selected parent genome
        """
        match self.config.selection_strategy:
            case SelectionStrategy.TOURNAMENT:
                return self._tournament_selection()
            
            case SelectionStrategy.ROULETTE:
                return self._roulette_selection()
            
            case SelectionStrategy.RANK:
                return self._rank_selection()
            
            case SelectionStrategy.ELITE:
                return self._elite_selection()
            
            case _:
                # Default to random selection
                return random.choice(list(self.genomes.values()))
    
    def select_parents(self, count: int = 2) -> list[Genome]:
        """
        Select multiple parents for reproduction.
        
        Args:
            count: Number of parents to select
        
        Returns:
            List of selected parents
        """
        return [self.select_parent() for _ in range(count)]
    
    def update_elite(self, generation: int) -> None:
        """
        Update elite genomes based on fitness.
        
        Args:
            generation: Current generation number
        """
        # Get genomes with fitness scores
        scored_genomes = [
            g for g in self.genomes.values()
            if g.fitness_score is not None
        ]
        
        if not scored_genomes:
            logger.warning("No scored genomes available for elite selection")
            return
        
        # Sort by fitness (descending)
        scored_genomes.sort(key=lambda g: g.fitness_score or 0.0, reverse=True)
        
        # Select elite genomes
        self.elite_genomes = []
        for genome in scored_genomes[:self.config.elitism_count]:
            # Check if meets elite threshold
            normalized_fitness = (genome.fitness_score or 0.0) / 100.0
            if normalized_fitness >= self.config.elitism_threshold:
                self.elite_genomes.append(genome.genome_id)
        
        logger.info(
            "Elite genomes updated",
            generation=generation,
            elite_count=len(self.elite_genomes),
            elite_ids=self.elite_genomes,
        )
    
    def compute_statistics(self, generation: int) -> PopulationStatistics:
        """
        Compute population statistics.
        
        Args:
            generation: Current generation number
        
        Returns:
            Population statistics
        """
        genomes = list(self.genomes.values())
        
        if not genomes:
            return PopulationStatistics(
                generation=generation,
                population_size=0,
                avg_fitness=0.0,
                max_fitness=0.0,
                min_fitness=0.0,
                std_fitness=0.0,
                avg_quality=0.0,
                avg_timeliness=0.0,
                avg_honesty=0.0,
                unique_architectures=0,
                diversity_score=0.0,
                elite_count=0,
                elite_avg_fitness=0.0,
            )
        
        # Filter genomes with fitness scores
        scored = [g for g in genomes if g.fitness_score is not None]
        
        if not scored:
            fitness_values = [0.0]
            quality_values = [0.0]
            timeliness_values = [0.0]
            honesty_values = [0.0]
        else:
            fitness_values = [g.fitness_score for g in scored]
            quality_values = [g.quality_score or 0.0 for g in scored]
            timeliness_values = [g.timeliness_score or 0.0 for g in scored]
            honesty_values = [g.honesty_score or 0.0 for g in scored]
        
        # Fitness statistics
        avg_fitness = sum(fitness_values) / len(fitness_values)
        max_fitness = max(fitness_values)
        min_fitness = min(fitness_values)
        
        # Standard deviation
        variance = sum((f - avg_fitness) ** 2 for f in fitness_values) / len(fitness_values)
        std_fitness = variance ** 0.5
        
        # Component averages
        avg_quality = sum(quality_values) / len(quality_values)
        avg_timeliness = sum(timeliness_values) / len(timeliness_values)
        avg_honesty = sum(honesty_values) / len(honesty_values)
        
        # Diversity metrics
        unique_architectures = len(set(g.genome_hash for g in genomes))
        diversity_score = unique_architectures / len(genomes) if genomes else 0.0
        
        # Elite statistics
        elite_genomes = self.get_elite_genomes()
        elite_count = len(elite_genomes)
        elite_avg_fitness = (
            sum(g.fitness_score or 0.0 for g in elite_genomes) / elite_count
            if elite_count > 0 else 0.0
        )
        
        stats = PopulationStatistics(
            generation=generation,
            population_size=len(genomes),
            avg_fitness=avg_fitness,
            max_fitness=max_fitness,
            min_fitness=min_fitness,
            std_fitness=std_fitness,
            avg_quality=avg_quality,
            avg_timeliness=avg_timeliness,
            avg_honesty=avg_honesty,
            unique_architectures=unique_architectures,
            diversity_score=diversity_score,
            elite_count=elite_count,
            elite_avg_fitness=elite_avg_fitness,
        )
        
        # Store statistics
        self.generation_stats[generation] = stats
        
        logger.info(
            "Population statistics computed",
            generation=generation,
            size=stats.population_size,
            avg_fitness=f"{stats.avg_fitness:.2f}",
            max_fitness=f"{stats.max_fitness:.2f}",
            diversity=f"{stats.diversity_score:.2f}",
        )
        
        return stats
    
    def get_statistics(self, generation: int | None = None) -> PopulationStatistics | None:
        """
        Get statistics for a specific generation.
        
        Args:
            generation: Generation number (latest if None)
        
        Returns:
            Statistics if available, None otherwise
        """
        if generation is None:
            if not self.generation_stats:
                return None
            generation = max(self.generation_stats.keys())
        
        return self.generation_stats.get(generation)
    
    def _tournament_selection(self) -> Genome:
        """Tournament selection: choose best from random tournament."""
        if len(self.genomes) < self.config.tournament_size:
            # Not enough genomes, use all
            tournament = list(self.genomes.values())
        else:
            # Random tournament
            tournament = random.sample(
                list(self.genomes.values()),
                self.config.tournament_size,
            )
        
        # Select genome with highest fitness
        tournament_scored = [g for g in tournament if g.fitness_score is not None]
        
        if not tournament_scored:
            # No scored genomes, select randomly
            return random.choice(tournament)
        
        return max(tournament_scored, key=lambda g: g.fitness_score or 0.0)
    
    def _roulette_selection(self) -> Genome:
        """Roulette wheel selection: fitness-proportional selection."""
        scored = [g for g in self.genomes.values() if g.fitness_score is not None]
        
        if not scored:
            return random.choice(list(self.genomes.values()))
        
        # Normalize fitness scores (ensure non-negative)
        min_fitness = min(g.fitness_score for g in scored)
        if min_fitness < 0:
            offset = abs(min_fitness)
            fitness_values = [g.fitness_score + offset for g in scored]
        else:
            fitness_values = [g.fitness_score for g in scored]
        
        total_fitness = sum(fitness_values)
        
        if total_fitness == 0:
            return random.choice(scored)
        
        # Spin the wheel
        spin = random.uniform(0, total_fitness)
        cumulative = 0.0
        
        for genome, fitness in zip(scored, fitness_values):
            cumulative += fitness
            if cumulative >= spin:
                return genome
        
        # Fallback (should not reach here)
        return scored[-1]
    
    def _rank_selection(self) -> Genome:
        """Rank-based selection: selection based on rank, not raw fitness."""
        scored = [g for g in self.genomes.values() if g.fitness_score is not None]
        
        if not scored:
            return random.choice(list(self.genomes.values()))
        
        # Sort by fitness
        scored.sort(key=lambda g: g.fitness_score or 0.0)
        
        # Assign ranks (1 to N)
        ranks = list(range(1, len(scored) + 1))
        total_rank = sum(ranks)
        
        # Select based on rank
        spin = random.uniform(0, total_rank)
        cumulative = 0.0
        
        for genome, rank in zip(scored, ranks):
            cumulative += rank
            if cumulative >= spin:
                return genome
        
        return scored[-1]
    
    def _elite_selection(self) -> Genome:
        """Elite selection: always select from elite genomes."""
        elite = self.get_elite_genomes()
        
        if not elite:
            # No elite genomes, fallback to tournament
            return self._tournament_selection()
        
        return random.choice(elite)
    
    def _cull_population(self) -> None:
        """Remove low-fitness genomes when population exceeds max size."""
        if len(self.genomes) <= self.config.max_size:
            return
        
        # Sort by fitness (ascending)
        sorted_genomes = sorted(
            self.genomes.values(),
            key=lambda g: g.fitness_score if g.fitness_score is not None else -1.0,
        )
        
        # Remove lowest fitness genomes
        num_to_remove = len(self.genomes) - self.config.target_size
        
        for genome in sorted_genomes[:num_to_remove]:
            # Don't remove elite genomes
            if genome.genome_id not in self.elite_genomes:
                self.remove_genome(genome.genome_id)
        
        logger.info(
            "Population culled",
            removed=num_to_remove,
            new_size=len(self.genomes),
        )
    
    def calculate_diversity(self) -> float:
        """
        Calculate population diversity score.
        
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(self.genomes) < 2:
            return 1.0
        
        # Count unique genome hashes
        unique_hashes = len(set(g.genome_hash for g in self.genomes.values()))
        
        # Diversity is ratio of unique to total
        diversity = unique_hashes / len(self.genomes)
        
        return diversity
    
    def enforce_diversity(self) -> None:
        """Remove similar genomes to maintain diversity."""
        if not self.config.maintain_diversity:
            return
        
        # Group genomes by architecture hash
        hash_groups: dict[str, list[Genome]] = defaultdict(list)
        for genome in self.genomes.values():
            hash_groups[genome.genome_hash].append(genome)
        
        # Remove excess similar genomes
        for genome_hash, group in hash_groups.items():
            if len(group) > self.config.max_similar_genomes:
                # Sort by fitness (keep best)
                group.sort(key=lambda g: g.fitness_score or 0.0, reverse=True)
                
                # Remove excess
                for genome in group[self.config.max_similar_genomes:]:
                    if genome.genome_id not in self.elite_genomes:
                        self.remove_genome(genome.genome_id)
        
        logger.info(
            "Diversity enforced",
            population_size=len(self.genomes),
            unique_architectures=len(hash_groups),
        )


# =============================================================================
# Backward Compatibility
# =============================================================================

# Old tests expect "Population" class name
Population = PopulationManager


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PopulationConfig",
    "PopulationStatistics",
    "PopulationManager",
    "Population",  # Alias for backward compatibility
]
