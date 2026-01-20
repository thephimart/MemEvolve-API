"""Evolution Manager for runtime memory architecture optimization in API proxy."""

import json
import logging
import threading
import time
import random
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, replace
from pathlib import Path

from ..evolution.genotype import MemoryGenotype, GenotypeFactory
from ..evolution.selection import ParetoSelector, FitnessMetrics
from ..evolution.mutation import MutationEngine, RandomMutationStrategy
from ..evolution.diagnosis import DiagnosisEngine
from ..memory_system import MemorySystem
from ..utils.config import MemEvolveConfig

logger = logging.getLogger(__name__)


@dataclass
class EvolutionMetrics:
    """Runtime metrics for evolution."""
    api_requests_total: int = 0
    api_requests_successful: int = 0
    average_response_time: float = 0.0
    memory_retrievals_total: int = 0
    memory_retrievals_successful: int = 0
    average_retrieval_time: float = 0.0
    current_genotype_id: Optional[str] = None
    evolution_cycles_completed: int = 0
    last_evolution_time: Optional[float] = None


@dataclass
class EvolutionResult:
    """Result of an evolution cycle."""
    generation: int
    best_genotype: MemoryGenotype
    fitness_score: float
    improvement: float
    timestamp: float


class EvolutionManager:
    """Manages runtime evolution of memory architectures for API proxy."""

    def __init__(self, config: MemEvolveConfig, memory_system: MemorySystem):
        self.config = config
        self.memory_system = memory_system
        self.metrics = EvolutionMetrics()

        # Evolution components
        try:
            self.selector = ParetoSelector()
            self.mutation_engine = MutationEngine(RandomMutationStrategy())
            self.diagnosis_engine = DiagnosisEngine()
        except Exception as e:
            logger.warning(f"Failed to initialize evolution components: {e}")
            raise

        # Evolution state
        self.current_genotype: Optional[MemoryGenotype] = None
        self.genotype_factory = GenotypeFactory()
        self.population: List[MemoryGenotype] = []
        self.evolution_history: List[EvolutionResult] = []

        # Control flags
        self.is_running = False
        self.evolution_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Performance tracking
        self.request_times: List[float] = []
        self.retrieval_times: List[float] = []

        # Persistence
        self.persistence_file = Path(
            self.config.cache_dir) / "evolution_state.json"
        self.best_genotype: Optional[MemoryGenotype] = None
        self._load_persistent_state()

    def _load_persistent_state(self):
        """Load previously saved evolution state."""
        if self.persistence_file.exists():
            try:
                with open(self.persistence_file, 'r') as f:
                    data = json.load(f)

                # Load best genotype if available
                if 'best_genotype' in data:
                    genotype_dict = data['best_genotype']
                    self.best_genotype = self._dict_to_genotype(genotype_dict)

                # Load evolution history
                if 'evolution_history' in data:
                    self.evolution_history = []
                    for result_dict in data['evolution_history']:
                        genotype_dict = result_dict.pop('best_genotype')
                        genotype = self._dict_to_genotype(genotype_dict)
                        result = EvolutionResult(
                            best_genotype=genotype, **result_dict)
                        self.evolution_history.append(result)

                # Load metrics
                if 'metrics' in data:
                    for key, value in data['metrics'].items():
                        if hasattr(self.metrics, key):
                            setattr(self.metrics, key, value)

                logger.info(
                    f"Loaded evolution state from {self.persistence_file}"
                )

            except Exception as e:
                logger.warning(f"Failed to load evolution state: {e}")

    def _save_persistent_state(self):
        """Save current evolution state to disk."""
        try:
            # Ensure cache directory exists
            self.persistence_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'best_genotype': (
                    self._genotype_to_dict(self.best_genotype)
                    if self.best_genotype else None
                ),
                'evolution_history': [
                    {
                        'generation': result.generation,
                        'fitness_score': result.fitness_score,
                        'improvement': result.improvement,
                        'timestamp': result.timestamp,
                        'best_genotype': self._genotype_to_dict(result.best_genotype)
                    }
                    for result in self.evolution_history
                ],
                'metrics': {
                    'api_requests_total': self.metrics.api_requests_total,
                    'api_requests_successful': self.metrics.api_requests_successful,
                    'average_response_time': self.metrics.average_response_time,
                    'memory_retrievals_total': self.metrics.memory_retrievals_total,
                    'memory_retrievals_successful': self.metrics.memory_retrievals_successful,
                    'average_retrieval_time': self.metrics.average_retrieval_time,
                    'evolution_cycles_completed': self.metrics.evolution_cycles_completed,
                    'last_evolution_time': self.metrics.last_evolution_time,
                }
            }

            with open(self.persistence_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save evolution state: {e}")

    def _genotype_to_dict(self, genotype: MemoryGenotype) -> Dict[str, Any]:
        """Convert genotype to dictionary for serialization."""
        return genotype.to_dict()

    def _dict_to_genotype(self, data: Dict[str, Any]) -> MemoryGenotype:
        """Convert dictionary back to genotype."""
        # Create a new genotype and populate it
        genotype = MemoryGenotype()

        # Helper function to set nested attributes
        def set_nested_attr(obj, path, value):
            parts = path.split('.')
            for part in parts[:-1]:
                if not hasattr(obj, part):
                    setattr(obj, part, type('', (), {})())
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

        # Populate from the data structure
        if 'encode' in data:
            genotype.encode = self._dict_to_encode_config(data['encode'])
        if 'store' in data:
            genotype.store = self._dict_to_store_config(data['store'])
        if 'retrieve' in data:
            genotype.retrieve = self._dict_to_retrieve_config(data['retrieve'])
        if 'manage' in data:
            genotype.manage = self._dict_to_manage_config(data['manage'])

        return genotype

    def _dict_to_encode_config(self, data: Dict[str, Any]):
        """Convert dict to EncodeConfig."""
        from ..evolution.genotype import EncodeConfig
        return EncodeConfig(**data)

    def _dict_to_store_config(self, data: Dict[str, Any]):
        """Convert dict to StoreConfig."""
        from ..evolution.genotype import StoreConfig
        return StoreConfig(**data)

    def _dict_to_retrieve_config(self, data: Dict[str, Any]):
        """Convert dict to RetrieveConfig."""
        from ..evolution.genotype import RetrieveConfig
        return RetrieveConfig(**data)

    def _dict_to_manage_config(self, data: Dict[str, Any]):
        """Convert dict to ManageConfig."""
        from ..evolution.genotype import ManageConfig
        return ManageConfig(**data)

    def start_evolution(self) -> bool:
        """Start the evolution process in background thread."""
        if self.is_running:
            return False

        try:
            self.is_running = True
            self.stop_event.clear()

            # Initialize population
            self._initialize_population()

            # Start evolution thread
            self.evolution_thread = threading.Thread(
                target=self._evolution_loop, daemon=True)
            self.evolution_thread.start()

            return True
        except Exception as e:
            logger.warning(f"Failed to start evolution: {e}")
            self.is_running = False
            return False

    def stop_evolution(self) -> bool:
        """Stop the evolution process."""
        if not self.is_running:
            return False

        self.is_running = False
        self.stop_event.set()

        if self.evolution_thread:
            self.evolution_thread.join(timeout=5.0)

        # Save final state
        self._save_persistent_state()

        return True

    def get_status(self) -> Dict[str, Any]:
        """Get current evolution status."""
        return {
            "is_running": self.is_running,
            "current_genotype": (
                self.current_genotype.get_genome_id()
                if self.current_genotype else None
            ),
            "population_size": len(self.population),
            "evolution_cycles_completed": self.metrics.evolution_cycles_completed,
            "last_evolution_time": self.metrics.last_evolution_time,
            "api_requests_total": self.metrics.api_requests_total,
            "average_response_time": self.metrics.average_response_time,
            "memory_retrievals_total": self.metrics.memory_retrievals_total,
            "average_retrieval_time": self.metrics.average_retrieval_time,
        }

    def record_api_request(self, response_time: float, success: bool = True):
        """Record an API request for performance tracking."""
        self.metrics.api_requests_total += 1
        if success:
            self.metrics.api_requests_successful += 1

        self.request_times.append(response_time)
        # Keep only last 1000 requests for rolling average
        if len(self.request_times) > 1000:
            self.request_times.pop(0)

        self.metrics.average_response_time = sum(
            self.request_times) / len(self.request_times)

    def record_memory_retrieval(self, retrieval_time: float, success: bool = True):
        """Record a memory retrieval for performance tracking."""
        self.metrics.memory_retrievals_total += 1
        if success:
            self.metrics.memory_retrievals_successful += 1

        self.retrieval_times.append(retrieval_time)
        # Keep only last 1000 retrievals for rolling average
        if len(self.retrieval_times) > 1000:
            self.retrieval_times.pop(0)

        self.metrics.average_retrieval_time = sum(
            self.retrieval_times) / len(self.retrieval_times)

    def _initialize_population(self):
        """Initialize the population with current and variant genotypes."""
        # Start with best saved genotype if available, otherwise baseline
        if self.best_genotype:
            current_genotype = self.best_genotype
            logger.info(
                f"Using previously optimized genotype: {current_genotype.get_genome_id()}")
        else:
            current_genotype = self.genotype_factory.create_baseline_genotype()

        self.population = [current_genotype]

        # Add different architecture variants for initial population
        architectures = [
            self.genotype_factory.create_agentkb_genotype(),
            self.genotype_factory.create_lightweight_genotype(),
            self.genotype_factory.create_riva_genotype(),
            self.genotype_factory.create_cerebra_genotype()
        ]

        # Add available architectures, limiting to population size
        for arch in architectures[:self.config.evolution.population_size - 1]:
            self.population.append(arch)

        # Fill remaining slots with baseline variants if needed
        while len(self.population) < self.config.evolution.population_size:
            variant = self.genotype_factory.create_baseline_genotype()
            self.population.append(variant)

        self.current_genotype = current_genotype
        self.metrics.current_genotype_id = current_genotype.get_genome_id()

    def _evolution_loop(self):
        """Main evolution loop running in background thread."""
        for generation in range(self.config.evolution.generations):
            if self.stop_event.is_set():
                break

            try:
                # Evaluate current population
                fitness_scores = self._evaluate_population()

                # Select best genotypes
                selected = self._select_best_genotypes(fitness_scores)

                # Create next generation
                new_population = self._create_next_generation(selected)

                # Update population
                self.population = new_population

                # Apply best genotype to memory system
                best_genotype = max(
                    new_population,
                    key=lambda g: fitness_scores.get(g.get_genome_id(), 0)
                )
                self._apply_genotype_to_memory_system(best_genotype)

                # Update best genotype for persistence
                if (
                    self.best_genotype is None or
                    fitness_scores[best_genotype.get_genome_id()] >
                    fitness_scores.get(self.best_genotype.get_genome_id(), 0)
                ):
                    self.best_genotype = best_genotype

                # Record evolution result
                result = EvolutionResult(
                    generation=generation,
                    best_genotype=best_genotype,
                    fitness_score=fitness_scores[best_genotype.get_genome_id(
                    )],
                    improvement=0.0,  # Calculate relative to previous best
                    timestamp=time.time()
                )
                self.evolution_history.append(result)

                self.metrics.evolution_cycles_completed = generation + 1
                self.metrics.last_evolution_time = time.time()

                # Save state after each generation
                self._save_persistent_state()

                # Sleep between generations
                self.stop_event.wait(60.0)  # 1 minute between generations

            except Exception as e:
                logger.error(f"Evolution cycle {generation} failed: {e}")
                continue

        self.is_running = False

    def _evaluate_population(self) -> Dict[str, float]:
        """Evaluate fitness of current population."""
        fitness_scores = {}

        for genotype in self.population:
            genome_id = genotype.get_genome_id()

            # Use API performance metrics for fitness
            fitness_metrics = FitnessMetrics(
                # Lower is better (inverted)
                performance=self.metrics.average_response_time,
                cost=(
                    self.metrics.memory_retrievals_total /
                    max(1, self.metrics.api_requests_total)
                ),  # Memory load
                retrieval_accuracy=(
                    self.metrics.memory_retrievals_successful /
                    max(1, self.metrics.memory_retrievals_total)
                ),
                storage_efficiency=1.0,  # Placeholder
                response_time=self.metrics.average_response_time
            )

            fitness_score = fitness_metrics.calculate_fitness_score()
            fitness_scores[genome_id] = fitness_score

        return fitness_scores

    def _select_best_genotypes(self, fitness_scores: Dict[str, float]) -> List[MemoryGenotype]:
        """Select best genotypes using tournament selection."""
        selected = []
        population_with_fitness = [
            (g, fitness_scores[g.get_genome_id()]) for g in self.population
        ]

        for _ in range(len(self.population) // 2):  # Select half the population
            # Tournament selection
            tournament = random.sample(
                population_with_fitness, self.config.evolution.tournament_size
            )
            winner = max(tournament, key=lambda x: x[1])
            selected.append(winner[0])

        return selected

    def _create_next_generation(self, selected: List[MemoryGenotype]) -> List[MemoryGenotype]:
        """Create next generation through crossover and mutation."""
        new_population = list(selected)  # Elitism - keep best

        while len(new_population) < self.config.evolution.population_size:
            # Select parents
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)

            # Crossover
            if random.random() < self.config.evolution.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = replace(parent1)

            # Mutation
            if random.random() < self.config.evolution.mutation_rate:
                mutation_result = self.mutation_engine.mutate(
                    child, self.config.evolution.mutation_rate
                )
                child = mutation_result.mutated_genotype

            new_population.append(child)

        return new_population

    def _crossover(self, parent1: MemoryGenotype, parent2: MemoryGenotype) -> MemoryGenotype:
        """Perform crossover between two genotypes."""
        # Simple single-point crossover
        child = replace(parent1)

        # Randomly swap some components
        if random.random() < 0.5:
            child.encode = parent2.encode
        if random.random() < 0.5:
            child.store = parent2.store
        if random.random() < 0.5:
            child.retrieve = parent2.retrieve
        if random.random() < 0.5:
            child.manage = parent2.manage

        return child

    def _apply_genotype_to_memory_system(self, genotype: MemoryGenotype):
        """Apply genotype configuration to the running memory system."""
        try:
            # Update memory system configuration
            # This is a simplified implementation - in practice would need
            # more sophisticated hot-swapping of components
            self.current_genotype = genotype
            self.metrics.current_genotype_id = genotype.get_genome_id()

            # Note: Full implementation would require memory system to support
            # dynamic reconfiguration of its components

        except Exception as e:
            logger.error(
                f"Failed to apply genotype {genotype.get_genome_id()}: {e}")
