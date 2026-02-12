"""
Evolution History Tracking

This module tracks the evolution of genomes across generations:
- Generation records (population snapshots)
- Lineage tracking (family trees)
- Fitness progression over time
- Export for analysis and visualization

Provides transparency for BelizeChain's decentralized AI evolution.

Author: BelizeChain AI Team
Date: October 2025
Python: 3.13+
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from loguru import logger

from .encoding import Genome
from .population import PopulationStatistics


# =============================================================================
# Generation Record
# =============================================================================


@dataclass
class GenerationRecord:
    """
    Complete record of a single generation.
    
    Captures:
    - Generation metadata
    - Population statistics
    - Best genome of generation
    - Complete genome list (IDs only)
    """
    
    generation: int
    timestamp: str
    
    # Statistics
    statistics: PopulationStatistics
    
    # Best genome
    best_genome_id: str
    best_fitness: float
    
    # Population
    genome_ids: list[str]
    population_size: int
    
    # Evolution metadata
    mutations_applied: int = 0
    crossovers_applied: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generation": self.generation,
            "timestamp": self.timestamp,
            "statistics": self.statistics.to_dict(),
            "best_genome_id": self.best_genome_id,
            "best_fitness": self.best_fitness,
            "genome_ids": self.genome_ids,
            "population_size": self.population_size,
            "mutations_applied": self.mutations_applied,
            "crossovers_applied": self.crossovers_applied,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationRecord:
        """Create from dictionary."""
        # Reconstruct PopulationStatistics
        stats_data = data["statistics"]
        statistics = PopulationStatistics(**stats_data)
        
        return cls(
            generation=data["generation"],
            timestamp=data["timestamp"],
            statistics=statistics,
            best_genome_id=data["best_genome_id"],
            best_fitness=data["best_fitness"],
            genome_ids=data["genome_ids"],
            population_size=data["population_size"],
            mutations_applied=data.get("mutations_applied", 0),
            crossovers_applied=data.get("crossovers_applied", 0),
        )


# =============================================================================
# Genome Lineage
# =============================================================================


@dataclass
class GenomeLineage:
    """
    Tracks lineage (family tree) of a genome.
    
    Records:
    - Direct ancestors (parents, grandparents, etc.)
    - Fitness progression through lineage
    - Evolutionary path
    """
    
    genome_id: str
    generation: int
    fitness: float | None
    
    # Parents
    parent_ids: list[str]
    
    # Ancestors (recursive)
    ancestors: list[str] = field(default_factory=list)
    
    # Descendants (children created from this genome)
    descendant_ids: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "genome_id": self.genome_id,
            "generation": self.generation,
            "fitness": self.fitness,
            "parent_ids": self.parent_ids,
            "ancestors": self.ancestors,
            "descendant_ids": self.descendant_ids,
        }
    
    @classmethod
    def from_genome(cls, genome: Genome) -> GenomeLineage:
        """Create lineage from genome."""
        return cls(
            genome_id=genome.genome_id,
            generation=genome.generation,
            fitness=genome.fitness_score,
            parent_ids=genome.parent_genomes.copy(),
            ancestors=[],
            descendant_ids=[],
        )


# =============================================================================
# Evolution History
# =============================================================================


class EvolutionHistory:
    """
    Tracks complete evolution history across all generations.
    
    Responsibilities:
    - Record each generation
    - Track genome lineages
    - Analyze fitness progression
    - Export history for visualization
    - Persist to storage (Pakit DAG, local)
    """
    
    def __init__(self, experiment_name: str = "nawal-evolution"):
        """
        Initialize evolution history.
        
        Args:
            experiment_name: Name of this evolution experiment
        """
        self.experiment_name = experiment_name
        self.start_time = datetime.now(timezone.utc).isoformat()
        
        # Generation records
        self.generations: dict[int, GenerationRecord] = {}
        
        # Genome lineages
        self.lineages: dict[str, GenomeLineage] = {}
        
        # Best genome per generation
        self.best_genomes: dict[int, str] = {}
        
        # Global best genome
        self.global_best_genome_id: str | None = None
        self.global_best_fitness: float = 0.0
        
        logger.info(
            "Initialized EvolutionHistory",
            experiment=experiment_name,
            start_time=self.start_time,
        )
    
    def record_generation(
        self,
        generation: int,
        statistics: PopulationStatistics,
        genomes: list[Genome],
        mutations_applied: int = 0,
        crossovers_applied: int = 0,
    ) -> None:
        """
        Record a complete generation.
        
        Args:
            generation: Generation number
            statistics: Population statistics
            genomes: All genomes in generation
            mutations_applied: Number of mutations in this generation
            crossovers_applied: Number of crossovers in this generation
        """
        # Find best genome
        scored_genomes = [g for g in genomes if g.fitness_score is not None]
        
        if scored_genomes:
            best_genome = max(scored_genomes, key=lambda g: g.fitness_score or 0.0)
            best_genome_id = best_genome.genome_id
            best_fitness = best_genome.fitness_score or 0.0
        else:
            best_genome_id = genomes[0].genome_id if genomes else "unknown"
            best_fitness = 0.0
        
        # Update global best
        if best_fitness > self.global_best_fitness:
            self.global_best_genome_id = best_genome_id
            self.global_best_fitness = best_fitness
        
        # Create generation record
        record = GenerationRecord(
            generation=generation,
            timestamp=datetime.now(timezone.utc).isoformat(),
            statistics=statistics,
            best_genome_id=best_genome_id,
            best_fitness=best_fitness,
            genome_ids=[g.genome_id for g in genomes],
            population_size=len(genomes),
            mutations_applied=mutations_applied,
            crossovers_applied=crossovers_applied,
        )
        
        self.generations[generation] = record
        self.best_genomes[generation] = best_genome_id
        
        # Update lineages
        for genome in genomes:
            self._update_lineage(genome)
        
        logger.info(
            "Generation recorded",
            generation=generation,
            population_size=len(genomes),
            best_genome=best_genome_id,
            best_fitness=f"{best_fitness:.2f}",
        )
    
    def _update_lineage(self, genome: Genome) -> None:
        """Update lineage information for a genome."""
        # Create or update lineage
        if genome.genome_id not in self.lineages:
            self.lineages[genome.genome_id] = GenomeLineage.from_genome(genome)
        else:
            lineage = self.lineages[genome.genome_id]
            lineage.fitness = genome.fitness_score
        
        # Update parent lineages (mark this as descendant)
        for parent_id in genome.parent_genomes:
            if parent_id in self.lineages:
                parent_lineage = self.lineages[parent_id]
                if genome.genome_id not in parent_lineage.descendant_ids:
                    parent_lineage.descendant_ids.append(genome.genome_id)
    
    def get_lineage(self, genome_id: str) -> GenomeLineage | None:
        """
        Get lineage for a specific genome.
        
        Args:
            genome_id: Genome identifier
        
        Returns:
            Genome lineage if found, None otherwise
        """
        return self.lineages.get(genome_id)
    
    def get_ancestors(self, genome_id: str, max_depth: int = 10) -> list[str]:
        """
        Get all ancestors of a genome.
        
        Args:
            genome_id: Genome identifier
            max_depth: Maximum depth to traverse
        
        Returns:
            List of ancestor genome IDs
        """
        ancestors = []
        current_ids = [genome_id]
        
        for _ in range(max_depth):
            if not current_ids:
                break
            
            next_ids = []
            for current_id in current_ids:
                lineage = self.lineages.get(current_id)
                if lineage:
                    for parent_id in lineage.parent_ids:
                        if parent_id not in ancestors:
                            ancestors.append(parent_id)
                            next_ids.append(parent_id)
            
            current_ids = next_ids
        
        return ancestors
    
    def get_descendants(self, genome_id: str, max_depth: int = 10) -> list[str]:
        """
        Get all descendants of a genome.
        
        Args:
            genome_id: Genome identifier
            max_depth: Maximum depth to traverse
        
        Returns:
            List of descendant genome IDs
        """
        descendants = []
        current_ids = [genome_id]
        
        for _ in range(max_depth):
            if not current_ids:
                break
            
            next_ids = []
            for current_id in current_ids:
                lineage = self.lineages.get(current_id)
                if lineage:
                    for descendant_id in lineage.descendant_ids:
                        if descendant_id not in descendants:
                            descendants.append(descendant_id)
                            next_ids.append(descendant_id)
            
            current_ids = next_ids
        
        return descendants
    
    def get_fitness_progression(self) -> list[tuple[int, float]]:
        """
        Get fitness progression over generations.
        
        Returns:
            List of (generation, best_fitness) tuples
        """
        progression = []
        
        for generation in sorted(self.generations.keys()):
            record = self.generations[generation]
            progression.append((generation, record.best_fitness))
        
        return progression
    
    def get_generation_record(self, generation: int) -> GenerationRecord | None:
        """
        Get record for a specific generation.
        
        Args:
            generation: Generation number
        
        Returns:
            Generation record if found, None otherwise
        """
        return self.generations.get(generation)
    
    def get_best_genome_id(self, generation: int | None = None) -> str | None:
        """
        Get ID of best genome.
        
        Args:
            generation: Specific generation (global best if None)
        
        Returns:
            Genome ID if found, None otherwise
        """
        if generation is None:
            return self.global_best_genome_id
        
        return self.best_genomes.get(generation)
    
    def compute_summary(self) -> dict[str, Any]:
        """
        Compute summary statistics of evolution history.
        
        Returns:
            Summary dictionary
        """
        if not self.generations:
            return {
                "experiment_name": self.experiment_name,
                "total_generations": 0,
                "total_genomes": 0,
                "global_best_fitness": 0.0,
            }
        
        total_genomes = len(self.lineages)
        total_generations = len(self.generations)
        
        # Fitness progression
        progression = self.get_fitness_progression()
        initial_fitness = progression[0][1] if progression else 0.0
        final_fitness = progression[-1][1] if progression else 0.0
        improvement = final_fitness - initial_fitness
        
        # Average population size
        avg_population = sum(
            r.population_size for r in self.generations.values()
        ) / total_generations
        
        # Total mutations and crossovers
        total_mutations = sum(
            r.mutations_applied for r in self.generations.values()
        )
        total_crossovers = sum(
            r.crossovers_applied for r in self.generations.values()
        )
        
        return {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time,
            "total_generations": total_generations,
            "total_genomes": total_genomes,
            "avg_population_size": avg_population,
            "global_best_fitness": self.global_best_fitness,
            "global_best_genome_id": self.global_best_genome_id,
            "initial_fitness": initial_fitness,
            "final_fitness": final_fitness,
            "fitness_improvement": improvement,
            "total_mutations": total_mutations,
            "total_crossovers": total_crossovers,
        }
    
    def export_to_json(self, filepath: Path | str) -> None:
        """
        Export complete history to JSON file.
        
        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time,
            "summary": self.compute_summary(),
            "generations": {
                str(gen): record.to_dict()
                for gen, record in self.generations.items()
            },
            "lineages": {
                gid: lineage.to_dict()
                for gid, lineage in self.lineages.items()
            },
            "best_genomes": {
                str(gen): gid
                for gen, gid in self.best_genomes.items()
            },
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(
            "Evolution history exported",
            filepath=str(filepath),
            generations=len(self.generations),
            genomes=len(self.lineages),
        )
    
    def import_from_json(self, filepath: Path | str) -> None:
        """
        Import history from JSON file.
        
        Args:
            filepath: Input file path
        """
        filepath = Path(filepath)
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        self.experiment_name = data["experiment_name"]
        self.start_time = data["start_time"]
        
        # Import generations
        self.generations = {
            int(gen): GenerationRecord.from_dict(record)
            for gen, record in data["generations"].items()
        }
        
        # Import lineages
        self.lineages = {
            gid: GenomeLineage(**lineage)
            for gid, lineage in data["lineages"].items()
        }
        
        # Import best genomes
        self.best_genomes = {
            int(gen): gid
            for gen, gid in data["best_genomes"].items()
        }
        
        # Recompute global best
        if self.generations:
            latest_gen = max(self.generations.keys())
            latest_record = self.generations[latest_gen]
            self.global_best_genome_id = latest_record.best_genome_id
            self.global_best_fitness = latest_record.best_fitness
        
        logger.info(
            "Evolution history imported",
            filepath=str(filepath),
            generations=len(self.generations),
            genomes=len(self.lineages),
        )
    
    def export_for_visualization(self, filepath: Path | str) -> None:
        """
        Export history in format optimized for visualization.
        
        Creates a simplified JSON suitable for web visualization.
        
        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Fitness over time
        fitness_data = [
            {
                "generation": gen,
                "best_fitness": record.best_fitness,
                "avg_fitness": record.statistics.avg_fitness,
                "max_fitness": record.statistics.max_fitness,
                "min_fitness": record.statistics.min_fitness,
            }
            for gen, record in sorted(self.generations.items())
        ]
        
        # Diversity over time
        diversity_data = [
            {
                "generation": gen,
                "diversity_score": record.statistics.diversity_score,
                "unique_architectures": record.statistics.unique_architectures,
                "population_size": record.statistics.population_size,
            }
            for gen, record in sorted(self.generations.items())
        ]
        
        # Component fitness over time
        component_data = [
            {
                "generation": gen,
                "quality": record.statistics.avg_quality,
                "timeliness": record.statistics.avg_timeliness,
                "honesty": record.statistics.avg_honesty,
            }
            for gen, record in sorted(self.generations.items())
        ]
        
        # Lineage tree (simplified)
        lineage_tree = []
        for genome_id, lineage in self.lineages.items():
            lineage_tree.append({
                "id": genome_id,
                "generation": lineage.generation,
                "fitness": lineage.fitness,
                "parents": lineage.parent_ids,
                "children": lineage.descendant_ids,
            })
        
        data = {
            "experiment": self.experiment_name,
            "summary": self.compute_summary(),
            "fitness_progression": fitness_data,
            "diversity_progression": diversity_data,
            "component_progression": component_data,
            "lineage_tree": lineage_tree,
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(
            "Visualization data exported",
            filepath=str(filepath),
        )


# =============================================================================
# Backward Compatibility - InnovationHistory
# =============================================================================


class InnovationHistory:
    """
    Backward compatibility stub for old test API.
    
    Old API tracked innovation IDs for layers and connections.
    New architecture uses UUID-based layer_ids.
    """
    
    def __init__(self):
        self.next_innovation_id = 1
        self._layer_innovations: dict[tuple, int] = {}
        self._connection_innovations: dict[tuple, int] = {}
    
    def register_layer_innovation(
        self,
        layer_type: str,
        params: dict,
    ) -> int:
        """Register a layer innovation and return its ID."""
        # Create hashable key from layer_type and params
        param_key = tuple(sorted(params.items()))
        key = (layer_type, param_key)
        
        if key not in self._layer_innovations:
            self._layer_innovations[key] = self.next_innovation_id
            self.next_innovation_id += 1
        
        return self._layer_innovations[key]
    
    def register_connection_innovation(
        self,
        source_layer: int,
        target_layer: int,
    ) -> int:
        """Register a connection innovation and return its ID."""
        key = (source_layer, target_layer)
        
        if key not in self._connection_innovations:
            self._connection_innovations[key] = self.next_innovation_id
            self.next_innovation_id += 1
        
        return self._connection_innovations[key]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "GenerationRecord",
    "GenomeLineage",
    "EvolutionHistory",
    "InnovationHistory",  # Added for backward compatibility
]
