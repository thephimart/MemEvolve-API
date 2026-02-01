from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import random
from .genotype import (
    MemoryGenotype,
    EncodeConfig,
    StoreConfig,
    RetrieveConfig,
    ManageConfig
)


@dataclass
class MutationOperation:
    """Represents a single mutation operation."""

    operation_type: str
    description: str
    target_component: str
    before_value: Any
    after_value: Any
    impact_score: float


@dataclass
class MutationResult:
    """Result of applying mutations to a genotype."""

    original_genotype: MemoryGenotype
    mutated_genotype: MemoryGenotype
    mutations_applied: List[MutationOperation]
    mutation_rate: float
    success: bool = True


class MutationStrategy:
    """Base class for mutation strategies."""

    def __init__(
        self,
        base_max_tokens: int = 512,
        boundary_config=None
    ):
        self.base_max_tokens = base_max_tokens
        self.boundary_config = boundary_config
        # Use boundary config for valid tokens
        self.valid_max_tokens = list(range(
            boundary_config.max_tokens_min,
            boundary_config.max_tokens_max + 1,
            boundary_config.token_step_size
        ))

        # Constrain valid options to base model capabilities
        self.valid_max_tokens = [
            t for t in self.valid_max_tokens
            if t <= self.base_max_tokens
        ]

    def mutate(
        self,
        genotype: MemoryGenotype,
        mutation_rate: float
    ) -> MemoryGenotype:
        """Apply mutations to genotype.

        Args:
            genotype: Original genotype to mutate
            mutation_rate: Probability of mutation per parameter

        Returns:
            Mutated genotype
        """
        raise NotImplementedError


class RandomMutationStrategy(MutationStrategy):
    """Randomly mutate genotype parameters using boundary config."""

    def __init__(
        self,
        base_max_tokens: int = 512,
        boundary_config=None
    ):
        super().__init__(base_max_tokens, boundary_config)

        # Use config values from boundary_config
        self.encoding_strategies_options = boundary_config.encoding_strategies_options
        self.strategy_types = boundary_config.retrieval_strategies
        self.manage_strategies = boundary_config.management_strategies
        self.forgetting_strategies = boundary_config.forgetting_strategies

    def mutate(
        self,
        genotype: MemoryGenotype,
        mutation_rate: float
    ) -> MemoryGenotype:
        """Apply random mutations to genotype.

        Args:
            genotype: Original genotype to mutate
            mutation_rate: Probability of mutation per parameter

        Returns:
            Mutated genotype
        """
        mutated = MemoryGenotype(
            encode=self._mutate_encode(genotype.encode, mutation_rate),
            store=self._mutate_store(genotype.store, mutation_rate),
            retrieve=self._mutate_retrieve(genotype.retrieve, mutation_rate),
            manage=self._mutate_manage(genotype.manage, mutation_rate),
            metadata=genotype.metadata.copy()
        )

        return mutated

    def _mutate_encode(
        self,
        config: EncodeConfig,
        mutation_rate: float
    ) -> EncodeConfig:
        """Mutate encode configuration with model capability constraints."""
        if random.random() < mutation_rate:
            strategies = ["lesson", "skill", "tool", "abstraction"]
            current_strategies = config.encoding_strategies

            if random.random() < self.boundary_config.strategy_addition_probability:
                missing = [
                    s for s in strategies if s not in current_strategies]
                if missing:
                    new_strategy = random.choice(missing)
                    config.encoding_strategies = current_strategies + [
                        new_strategy
                    ]
            else:
                if len(current_strategies) > 1:
                    removed = random.choice(current_strategies)
                    config.encoding_strategies = [
                        s for s in current_strategies if s != removed
                    ]

        if random.random() < mutation_rate and self.valid_max_tokens:
            config.max_tokens = random.choice(self.valid_max_tokens)

        if random.random() < mutation_rate:
            config.batch_size = max(
                self.boundary_config.batch_size_min,
                int(config.batch_size * self.boundary_config.batch_size_multiplier
                    if random.random() < self.boundary_config.strategy_addition_probability
                    else config.batch_size / 2)
            )

        if random.random() < mutation_rate:
            if random.random() < self.boundary_config.strategy_addition_probability:
                config.temperature = max(
                    0.0, min(
                        1.0, config.temperature + self.boundary_config.temperature_change_delta))
            else:
                config.temperature = max(
                    0.0, min(
                        1.0, config.temperature - self.boundary_config.temperature_change_delta))

        if random.random() < mutation_rate:
            config.enable_abstractions = not config.enable_abstractions

        if random.random() < mutation_rate and config.enable_abstractions:
            config.min_abstraction_units = max(
                2,
                config.min_abstraction_units +
                2 if random.random() < self.boundary_config.strategy_addition_probability else config.min_abstraction_units -
                2)

        return EncodeConfig(
            encoding_strategies=config.encoding_strategies,
            llm_model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            batch_size=config.batch_size,
            enable_abstractions=config.enable_abstractions,
            min_abstraction_units=config.min_abstraction_units
        )

    def _mutate_store(
        self,
        config: StoreConfig,
        mutation_rate: float
    ) -> StoreConfig:
        """Mutate store configuration.

        Note: All storage parameters are NOT evolved:
        - backend_type, storage_path, vector_index_file
        - enable_persistence, max_storage_size_mb
        These are user-configurable via .env, not evolved
        """
        # No mutations - storage config is user-controlled
        return StoreConfig(
            backend_type=config.backend_type,
            storage_path=config.storage_path,
            vector_index_file=config.vector_index_file,
            max_storage_size_mb=config.max_storage_size_mb
        )

    def _mutate_retrieve(
        self,
        config: RetrieveConfig,
        mutation_rate: float
    ) -> RetrieveConfig:
        """Mutate retrieve configuration."""
        if random.random() < mutation_rate:
            config.strategy_type = random.choice(self.strategy_types)

        if random.random() < mutation_rate:
            config.default_top_k = random.randint(
                self.boundary_config.top_k_min,
                self.boundary_config.top_k_max
            )

        if random.random() < mutation_rate:
            config.similarity_threshold = round(random.uniform(
                self.boundary_config.similarity_threshold_min,
                self.boundary_config.similarity_threshold_max
            ), 2)

        if random.random() < mutation_rate:
            config.semantic_cache_enabled = not config.semantic_cache_enabled

        if config.strategy_type == "hybrid" and random.random() < mutation_rate:
            min_weight, max_weight = self.boundary_config.hybrid_weight_range
            config.hybrid_semantic_weight = round(random.uniform(min_weight, max_weight), 2)
            config.hybrid_keyword_weight = round(
                1.0 - config.hybrid_semantic_weight, 2
            )

        return RetrieveConfig(
            strategy_type=config.strategy_type,
            default_top_k=config.default_top_k,
            similarity_threshold=config.similarity_threshold,
            enable_filters=config.enable_filters,
            semantic_embedding_model=config.semantic_embedding_model,
            semantic_cache_enabled=config.semantic_cache_enabled,
            hybrid_semantic_weight=config.hybrid_semantic_weight,
            hybrid_keyword_weight=config.hybrid_keyword_weight,
            keyword_case_sensitive=config.keyword_case_sensitive
        )

    def _mutate_manage(
        self,
        config: ManageConfig,
        mutation_rate: float
    ) -> ManageConfig:
        """Mutate manage configuration.

        Note: Data persistence parameters are NOT mutated:
        - prune_max_age_days, prune_max_count, prune_by_type
        - deduplicate_enabled, deduplicate_similarity_threshold
        These are controlled via environment variables and applied
        at server startup and periodically during operation.
        """
        if random.random() < mutation_rate:
            config.enable_auto_management = not config.enable_auto_management

        # Note: auto_prune_threshold is evolved (triggers auto-management)
        # but prune_max_age_days, prune_max_count, prune_by_type are NOT
        # These are user-configurable via MEMEVOLVE_MANAGEMENT_* settings

        if random.random() < mutation_rate:
            config.consolidate_enabled = not config.consolidate_enabled

        # Note: deduplicate_enabled and deduplicate_similarity_threshold
        # are NOT evolved - applied at startup and periodically

        if random.random() < mutation_rate:
            config.forgetting_strategy = random.choice(
                self.forgetting_strategies)

        if random.random() < mutation_rate:
            min_percent, max_percent = self.boundary_config.forgetting_percentage_range
            config.forgetting_percentage = round(random.uniform(min_percent, max_percent), 2)

        return ManageConfig(
            strategy_type=config.strategy_type,
            enable_auto_management=config.enable_auto_management,
            # Note: auto_prune_threshold not evolved - user-configurable
            consolidate_enabled=config.consolidate_enabled,
            consolidate_min_units=config.consolidate_min_units,
            forgetting_strategy=config.forgetting_strategy,
            forgetting_percentage=config.forgetting_percentage
        )


class TargetedMutationStrategy(MutationStrategy):
    """Target mutations based on performance feedback."""

    def __init__(self,
                 feedback_weights: Optional[Dict[str,
                                                 float]] = None,
                 base_max_tokens: int = 512,
                 boundary_config=None):
        super().__init__(base_max_tokens, boundary_config)
        self.feedback_weights = feedback_weights or {
            "performance": 1.0,
            "cost": 0.8,
            "retrieval_accuracy": 0.9,
            "storage_efficiency": 0.7
        }

        # Initialize strategy lists from boundary config
        self.encoding_strategies_options = boundary_config.encoding_strategies_options
        self.strategy_types = boundary_config.retrieval_strategies
        self.manage_strategies = boundary_config.management_strategies
        self.forgetting_strategies = boundary_config.forgetting_strategies

    def mutate(
        self,
        genotype: MemoryGenotype,
        mutation_rate: float,
        feedback: Optional[Dict[str, float]] = None
    ) -> MemoryGenotype:
        """Apply targeted mutations based on feedback.

        Args:
            genotype: Original genotype to mutate
            mutation_rate: Base mutation rate
            feedback: Performance feedback metrics

        Returns:
            Mutated genotype
        """
        if feedback is None:
            feedback = {}

        adjusted_rates = self._adjust_mutation_rates(
            mutation_rate, feedback
        )

        mutated = MemoryGenotype(
            encode=self._mutate_encode(
                genotype.encode,
                adjusted_rates.get("encode", mutation_rate)
            ),
            store=self._mutate_store(
                genotype.store,
                adjusted_rates.get("store", mutation_rate)
            ),
            retrieve=self._mutate_retrieve(
                genotype.retrieve,
                adjusted_rates.get("retrieve", mutation_rate)
            ),
            manage=self._mutate_manage(
                genotype.manage,
                adjusted_rates.get("manage", mutation_rate)
            ),
            metadata=genotype.metadata.copy()
        )

        return mutated

    def _adjust_mutation_rates(
        self,
        base_rate: float,
        feedback: Dict[str, float]
    ) -> Dict[str, float]:
        """Adjust mutation rates based on feedback."""
        adjusted = {}

        if "performance" in feedback and feedback["performance"] < 0.7:
            adjusted["encode"] = min(base_rate * 1.5, 1.0)
            adjusted["retrieve"] = min(base_rate * 1.5, 1.0)
        else:
            adjusted["encode"] = base_rate
            adjusted["retrieve"] = base_rate

        if "retrieval_accuracy" in feedback and feedback["retrieval_accuracy"] < 0.7:
            adjusted["retrieve"] = min(adjusted.get(
                "retrieve", base_rate) * 1.3, 1.0)
        elif "retrieval_accuracy" in feedback and feedback["retrieval_accuracy"] > 0.9:
            adjusted["retrieve"] = base_rate * 0.5

        if "cost" in feedback and feedback["cost"] > 0.7:
            adjusted["store"] = min(base_rate * 1.3, 1.0)
            adjusted["manage"] = min(base_rate * 1.3, 1.0)
        elif "cost" in feedback and feedback["cost"] < 0.3:
            adjusted["store"] = base_rate * 0.7
            adjusted["manage"] = base_rate * 0.7

        if "storage_efficiency" in feedback and feedback["storage_efficiency"] < 0.6:
            adjusted["manage"] = min(adjusted.get(
                "manage", base_rate) * 1.3, 1.0)

        return adjusted

    def _mutate_encode(
        self,
        config: EncodeConfig,
        mutation_rate: float
    ) -> EncodeConfig:
        """Mutate encode configuration."""
        if random.random() < mutation_rate:
            strategies = ["lesson", "skill", "tool", "abstraction"]
            current_strategies = config.encoding_strategies

            if random.random() < self.boundary_config.strategy_addition_probability:
                missing = [
                    s for s in strategies if s not in current_strategies]
                if missing:
                    new_strategy = random.choice(missing)
                    config.encoding_strategies = current_strategies + [
                        new_strategy
                    ]
            else:
                if len(current_strategies) > 1:
                    removed = random.choice(current_strategies)
                    config.encoding_strategies = [
                        s for s in current_strategies if s != removed
                    ]

        if random.random() < mutation_rate:
            config.batch_size = max(
                self.boundary_config.batch_size_min,
                int(config.batch_size * self.boundary_config.batch_size_multiplier
                    if random.random() < self.boundary_config.strategy_addition_probability
                    else config.batch_size / 2)
            )

        if random.random() < mutation_rate:
            if random.random() < self.boundary_config.strategy_addition_probability:
                config.temperature = max(
                    0.0, min(
                        1.0, config.temperature + self.boundary_config.temperature_change_delta))
            else:
                config.temperature = max(
                    0.0, min(
                        1.0, config.temperature - self.boundary_config.temperature_change_delta))

        return EncodeConfig(
            encoding_strategies=config.encoding_strategies,
            llm_model=config.llm_model,
            temperature=round(config.temperature, 2),
            max_tokens=config.max_tokens,
            batch_size=int(config.batch_size),
            enable_abstractions=config.enable_abstractions,
            min_abstraction_units=config.min_abstraction_units
        )

    def _mutate_store(
        self,
        config: StoreConfig,
        mutation_rate: float
    ) -> StoreConfig:
        """Mutate store configuration."""
        # Note: backend_type and max_storage_size_mb are NOT mutated
        # These should be user-configurable via .env, not evolved

        return StoreConfig(
            backend_type=config.backend_type,
            storage_path=config.storage_path,
            vector_index_file=config.vector_index_file,
            enable_persistence=config.enable_persistence,
            max_storage_size_mb=config.max_storage_size_mb
        )

    def _mutate_retrieve(
        self,
        config: RetrieveConfig,
        mutation_rate: float
    ) -> RetrieveConfig:
        """Mutate retrieve configuration."""
        if random.random() < mutation_rate:
            if random.random() < self.boundary_config.strategy_addition_probability:
                new_k = config.default_top_k + 2
            else:
                new_k = config.default_top_k - 2
            config.default_top_k = max(
                self.boundary_config.top_k_min,
                min(self.boundary_config.top_k_max, new_k)
            )

        if random.random() < mutation_rate:
            if random.random() < self.boundary_config.strategy_addition_probability:
                new_thresh = config.similarity_threshold + 0.05
            else:
                new_thresh = config.similarity_threshold - 0.05
            config.similarity_threshold = max(
                self.boundary_config.similarity_threshold_min,
                min(self.boundary_config.similarity_threshold_max, new_thresh)
            )

        return RetrieveConfig(
            strategy_type=config.strategy_type,
            default_top_k=config.default_top_k,
            similarity_threshold=round(config.similarity_threshold, 2),
            enable_filters=config.enable_filters,
            semantic_embedding_model=config.semantic_embedding_model,
            semantic_cache_enabled=config.semantic_cache_enabled,
            hybrid_semantic_weight=config.hybrid_semantic_weight,
            hybrid_keyword_weight=config.hybrid_keyword_weight,
            keyword_case_sensitive=config.keyword_case_sensitive
        )

    def _mutate_manage(
        self,
        config: ManageConfig,
        mutation_rate: float
    ) -> ManageConfig:
        """Mutate manage configuration."""
        # Note: auto_prune_threshold and prune_max_age_days are NOT mutated
        # These should be user-configurable via MEMEVOLVE_MANAGEMENT_* settings

        if random.random() < mutation_rate:
            config.consolidate_enabled = not config.consolidate_enabled

        if random.random() < mutation_rate:
            config.deduplicate_enabled = not config.deduplicate_enabled

        return ManageConfig(
            strategy_type=config.strategy_type,
            enable_auto_management=config.enable_auto_management,
            auto_prune_threshold=config.auto_prune_threshold,
            prune_max_age_days=config.prune_max_age_days,
            prune_max_count=config.prune_max_count,
            prune_by_type=config.prune_by_type,
            consolidate_enabled=config.consolidate_enabled,
            consolidate_min_units=config.consolidate_min_units,
            deduplicate_enabled=config.deduplicate_enabled,
            deduplicate_similarity_threshold=config.deduplicate_similarity_threshold,
            forgetting_strategy=config.forgetting_strategy,
            forgetting_percentage=config.forgetting_percentage
        )


class MutationEngine:
    """Engine for applying mutations to genotypes."""

    def __init__(
        self,
        strategy: Optional[MutationStrategy] = None,
        base_max_tokens: int = 512
    ):
        """Initialize mutation engine with base model capabilities.

        Args:
            strategy: Mutation strategy to use
            base_max_tokens: Maximum context window from model
        """
        self.strategy = strategy or RandomMutationStrategy(
            base_max_tokens=base_max_tokens
        )

    def mutate(
        self,
        genotype: MemoryGenotype,
        mutation_rate: float,
        **kwargs
    ) -> MutationResult:
        """Apply mutations to genotype.

        Args:
            genotype: Original genotype
            mutation_rate: Mutation probability
            **kwargs: Additional arguments for mutation strategy

        Returns:
            MutationResult with details
        """
        original = genotype

        try:
            if isinstance(self.strategy, TargetedMutationStrategy):
                mutated = self.strategy.mutate(
                    genotype, mutation_rate, **kwargs
                )
            else:
                mutated = self.strategy.mutate(genotype, mutation_rate)

            mutations = self._track_mutations(original, mutated)

            return MutationResult(
                original_genotype=original,
                mutated_genotype=mutated,
                mutations_applied=mutations,
                mutation_rate=mutation_rate,
                success=True
            )
        except Exception:
            return MutationResult(
                original_genotype=original,
                mutated_genotype=original,
                mutations_applied=[],
                mutation_rate=mutation_rate,
                success=False
            )

    def _track_mutations(
        self,
        original: MemoryGenotype,
        mutated: MemoryGenotype
    ) -> List[MutationOperation]:
        """Track mutations between original and mutated genotype.

        Args:
            original: Original genotype
            mutated: Mutated genotype

        Returns:
            List of mutation operations
        """
        mutations = []

        if original.encode.encoding_strategies != mutated.encode.encoding_strategies:
            mutations.append(MutationOperation(
                operation_type="modify",
                description="Changed encoding strategies",
                target_component="encode",
                before_value=original.encode.encoding_strategies,
                after_value=mutated.encode.encoding_strategies,
                impact_score=0.5
            ))

        if original.store.backend_type != mutated.store.backend_type:
            mutations.append(MutationOperation(
                operation_type="modify",
                description="Changed storage backend",
                target_component="store",
                before_value=original.store.backend_type,
                after_value=mutated.store.backend_type,
                impact_score=0.8
            ))

        if original.retrieve.strategy_type != mutated.retrieve.strategy_type:
            mutations.append(MutationOperation(
                operation_type="modify",
                description="Changed retrieval strategy",
                target_component="retrieve",
                before_value=original.retrieve.strategy_type,
                after_value=mutated.retrieve.strategy_type,
                impact_score=0.7
            ))

        if original.encode.temperature != mutated.encode.temperature:
            mutations.append(MutationOperation(
                operation_type="tune",
                description="Adjusted temperature",
                target_component="encode",
                before_value=original.encode.temperature,
                after_value=mutated.encode.temperature,
                impact_score=0.2
            ))

        if original.retrieve.default_top_k != mutated.retrieve.default_top_k:
            mutations.append(MutationOperation(
                operation_type="tune",
                description="Adjusted top_k",
                target_component="retrieve",
                before_value=original.retrieve.default_top_k,
                after_value=mutated.retrieve.default_top_k,
                impact_score=0.3
            ))

        return mutations

    def batch_mutate(
        self,
        genotypes: List[MemoryGenotype],
        mutation_rate: float,
        **kwargs
    ) -> List[MutationResult]:
        """Apply mutations to multiple genotypes.

        Args:
            genotypes: List of genotypes to mutate
            mutation_rate: Mutation probability
            **kwargs: Additional arguments

        Returns:
            List of mutation results
        """
        results = []

        for genotype in genotypes:
            result = self.mutate(genotype, mutation_rate, **kwargs)
            results.append(result)

        return results
