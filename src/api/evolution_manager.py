"""Evolution Manager for runtime memory architecture optimization in API proxy."""

import json
import logging
import threading
import time
import random
from typing import Optional, Dict, Any, List, cast
from dataclasses import field
from dataclasses import dataclass, replace
from pathlib import Path

from ..evolution.genotype import MemoryGenotype, GenotypeFactory
from ..evolution.selection import ParetoSelector
from ..evolution.mutation import MutationEngine, RandomMutationStrategy
from ..evolution.diagnosis import DiagnosisEngine
from ..memory_system import MemorySystem, ComponentType
from ..utils.config import MemEvolveConfig
from ..components.encode import ExperienceEncoder
from ..components.retrieve import (
    KeywordRetrievalStrategy,
    SemanticRetrievalStrategy,
    HybridRetrievalStrategy
)
from ..components.manage import SimpleManagementStrategy

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

    # Enhanced metrics for fitness evaluation
    response_quality_score: float = 0.0  # Semantic coherence/context relevance
    retrieval_precision: float = 0.0     # Precision of memory retrieval
    retrieval_recall: float = 0.0        # Recall of memory retrieval
    memory_utilization: float = 0.0      # Storage efficiency
    user_satisfaction_score: float = 0.0  # Future: explicit feedback

    # Rolling window data
    response_times_window: List[float] = field(default_factory=list)
    retrieval_times_window: List[float] = field(default_factory=list)
    quality_scores_window: List[float] = field(default_factory=list)
    window_size: int = 100  # Rolling window size


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

        # Evolution embedding settings
        # These are optimized values found by evolution
        self.evolution_embedding_max_tokens: Optional[int] = None

        # Base model capabilities (maximum allowable values)
        # Priority: env var > auto-detect > fallback
        self.base_embedding_max_tokens = config.embedding.max_tokens or 512

        # Evolution components
        try:
            self.selector = ParetoSelector()
            self.mutation_engine = MutationEngine(
                RandomMutationStrategy(),
                base_max_tokens=self.base_embedding_max_tokens
            )
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
        # Evolution state is persistent data, not temporary cache
        self.persistence_file = Path(self.config.data_dir) / "evolution_state.json"
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

                # Load evolution embedding settings
                if 'evolution_embedding_max_tokens' in data:
                    self.evolution_embedding_max_tokens = data[
                        'evolution_embedding_max_tokens']
                    logger.info(
                        "Loaded evolution embedding_max_tokens: {}".format(
                            self.evolution_embedding_max_tokens))

                # Load evolution history
                if 'evolution_history' in data:
                    self.evolution_history = []
                    for result_dict in data['evolution_history']:
                        genotype_dict = result_dict.pop('best_genotype')
                        genotype = self._dict_to_genotype(genotype_dict)
                        if genotype is not None:
                            result = EvolutionResult(
                                best_genotype=cast(MemoryGenotype, genotype),
                                **result_dict
                            )
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
                'evolution_embedding_max_tokens': self.evolution_embedding_max_tokens,
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
                    'last_evolution_time': self.metrics.last_evolution_time
                }
            }

            with open(self.persistence_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to save evolution state: {e}")

    def _genotype_to_dict(self, genotype: MemoryGenotype) -> Dict[str, Any]:
        """Convert genotype to dictionary for serialization."""
        return genotype.to_dict()

    def _dict_to_genotype(self, data: Dict[str, Any]) -> Optional[MemoryGenotype]:
        """Convert dictionary back to genotype."""
        if data is None:
            return None

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
                # Wait for minimum evaluation period (10 requests per genotype)
                min_requests = max(10, len(self.population) * 10)
                while self.metrics.api_requests_total < min_requests:
                    if self.stop_event.is_set():
                        return
                    time.sleep(1)  # Wait for more data

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

                    # Update evolution embedding settings from best genotype
                    self.evolution_embedding_max_tokens = best_genotype.encode.max_tokens
                    logger.info(
                        f"Evolution embedding settings updated: "
                        f"max_tokens={self.evolution_embedding_max_tokens}"
                    )

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

            # For now, use current metrics for all genotypes
            # TODO: Implement per-genotype evaluation by testing each one
            fitness_score = self.get_fitness_score()
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

    def record_response_quality(self, quality_score: float):
        """Record response quality score (semantic coherence, context relevance)."""
        self.metrics.quality_scores_window.append(quality_score)
        if len(self.metrics.quality_scores_window) > self.metrics.window_size:
            self.metrics.quality_scores_window.pop(0)

        self.metrics.response_quality_score = sum(
            self.metrics.quality_scores_window) / len(self.metrics.quality_scores_window)

    def get_fitness_score(self) -> float:
        """Calculate current fitness score from metrics."""
        # Weighted combination of metrics
        # Higher is better for all metrics
        weights = {
            'success_rate': 0.2,
            'response_time': 0.2,  # Lower is better, so invert
            'retrieval_success': 0.2,
            'quality': 0.2,
            'utilization': 0.2
        }

        success_rate = (self.metrics.api_requests_successful /
                        max(1, self.metrics.api_requests_total))
        # Lower time = higher score
        response_time_score = 1.0 / (1.0 + self.metrics.average_response_time)
        retrieval_success = (self.metrics.memory_retrievals_successful /
                             max(1, self.metrics.memory_retrievals_total))
        quality = self.metrics.response_quality_score
        utilization = self.metrics.memory_utilization

        fitness = (
            weights['success_rate'] * success_rate +
            weights['response_time'] * response_time_score +
            weights['retrieval_success'] * retrieval_success +
            weights['quality'] * quality +
            weights['utilization'] * utilization
        )

        return fitness

    def _apply_genotype_to_memory_system(self, genotype: MemoryGenotype):
        """Apply genotype configuration to the running memory system."""
        try:
            logger.info(
                f"Applying genotype {genotype.get_genome_id()} to memory system")

            # Apply encoder configuration
            if genotype.encode.llm_model:
                encoder = ExperienceEncoder(
                    base_url=self.config.llm.base_url,
                    api_key=self.config.llm.api_key,
                    model=genotype.encode.llm_model,
                    timeout=self.config.llm.timeout
                )
                encoder.initialize_llm()
                self.memory_system.reconfigure_component(
                    ComponentType.ENCODER, encoder)
                logger.info("Applied encoder configuration")

            # Apply retrieval configuration
            retrieval_strategy = self._create_retrieval_strategy(
                genotype.retrieve)
            if retrieval_strategy:
                self.memory_system.reconfigure_component(
                    ComponentType.RETRIEVAL, retrieval_strategy)
                logger.info("Applied retrieval configuration")

            # Apply management configuration
            management_strategy = self._create_management_strategy(
                genotype.manage)
            if management_strategy:
                # Create new memory manager with the strategy
                from ..components.manage import MemoryManager
                memory_manager = MemoryManager(
                    storage_backend=self.memory_system.storage,
                    management_strategy=management_strategy
                )
                self.memory_system.reconfigure_component(
                    ComponentType.MANAGER, memory_manager)
                logger.info("Applied management configuration")

            # Update tracking variables
            self.current_genotype = genotype
            self.metrics.current_genotype_id = genotype.get_genome_id()

            logger.info(
                f"Successfully applied genotype {genotype.get_genome_id()}")

        except Exception as e:
            logger.error(
                f"Failed to apply genotype {genotype.get_genome_id()}: {e}")
            raise

    def _create_retrieval_strategy(self, config):
        """Create retrieval strategy from genotype config."""
        try:
            if config.strategy_type == "keyword":
                return KeywordRetrievalStrategy()
            elif config.strategy_type == "semantic":
                from ..utils.embeddings import create_embedding_function
                embedding_function = create_embedding_function(
                    provider="openai",
                    base_url=self.config.embedding.base_url or self.config.llm.base_url,
                    api_key=self.config.embedding.api_key or self.config.llm.api_key,
                    evolution_manager=self  # Pass evolution manager for embedding overrides
                )
                return SemanticRetrievalStrategy(embedding_function=embedding_function)
            elif config.strategy_type == "hybrid":
                from ..utils.embeddings import create_embedding_function
                embedding_function = create_embedding_function(
                    provider="openai",
                    base_url=self.config.embedding.base_url or self.config.llm.base_url,
                    api_key=self.config.embedding.api_key or self.config.llm.api_key,
                    evolution_manager=self  # Pass evolution manager for embedding overrides
                )
                return HybridRetrievalStrategy(
                    embedding_function=embedding_function,
                    semantic_weight=config.hybrid_semantic_weight,
                    keyword_weight=config.hybrid_keyword_weight
                )
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to create retrieval strategy: {e}")
            return None

    def _create_management_strategy(self, config):
        """Create management strategy from genotype config."""
        try:
            if config.strategy_type == "simple":
                return SimpleManagementStrategy()
            else:
                logger.warning(
                    f"Unknown management strategy type: {config.strategy_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to create management strategy: {e}")
            return None
