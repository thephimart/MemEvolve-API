import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Import configuration for boundary access
from ..utils.config import ConfigManager
from ..utils.logging_manager import LoggingManager

logger = LoggingManager.get_logger(__name__)
logger.debug("Evolution genotype module initialized")


@dataclass
class EncodeConfig:
    """Configuration for Encode component.

    Note: Embedding dimension is determined by embedding model's
    native capability and is not evolved (model constraint).
    """

    encoding_strategies: List[str] = field(
        default_factory=lambda: ["lesson", "skill", "tool", "abstraction"]
    )
    llm_model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    batch_size: int = 10
    enable_abstractions: bool = True
    min_abstraction_units: int = 3


@dataclass
class StoreConfig:
    """Configuration for Store component.

    Note: Embedding dimension is controlled globally via config.embedding.dimension,
    which supports auto-detection from model metadata or manual override via
    environment variables.
    """

    backend_type: str = "json"
    index_type: str = "flat"


@dataclass
class RetrieveConfig:
    """Configuration for Retrieve component."""

    strategy_type: str = "hybrid"
    default_top_k: int = 5
    similarity_threshold: float = 0.7
    enable_filters: bool = True
    semantic_embedding_model: Optional[str] = None
    semantic_cache_enabled: bool = True
    keyword_case_sensitive: bool = False
    hybrid_semantic_weight: float = 0.7
    hybrid_keyword_weight: float = 0.3


@dataclass
class ManageConfig:
    """Configuration for Manage component."""

    strategy_type: str = "simple"
    enable_auto_management: bool = True
    prune_max_age_days: Optional[int] = None
    prune_max_count: Optional[int] = None
    prune_by_type: Optional[str] = None
    consolidate_enabled: bool = True
    consolidate_min_units: int = 2
    deduplicate_enabled: bool = True
    deduplicate_similarity_threshold: float = 0.9
    forgetting_strategy: str = "lru"
    forgetting_percentage: float = 0.1


@dataclass
class MemoryGenotype:
    """Memory system genotype for evolution."""

    encode: EncodeConfig = field(default_factory=EncodeConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)
    manage: ManageConfig = field(default_factory=ManageConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate genotype after initialization."""
        self._validate_genotype()

    def _validate_genotype(self):
        """Validate genotype configuration."""
        if not isinstance(self.encode.encoding_strategies, list):
            raise ValueError("encode.encoding_strategies must be a list")

        if self.encode.temperature < 0.0 or self.encode.temperature > 2.0:
            raise ValueError("encode.temperature must be between 0.0 and 2.0")

        if self.encode.max_tokens < 1 or self.encode.max_tokens > 32768:
            raise ValueError("encode.max_tokens must be between 1 and 32768")

        if self.encode.batch_size < 1 or self.encode.batch_size > 100:
            raise ValueError("encode.batch_size must be between 1 and 100")

        if self.retrieve.default_top_k < 1 or self.retrieve.default_top_k > 100:
            raise ValueError("retrieve.default_top_k must be between 1 and 100")

        if self.retrieve.similarity_threshold < 0.0 or self.retrieve.similarity_threshold > 1.0:
            raise ValueError("retrieve.similarity_threshold must be between 0.0 and 1.0")

    def get_genome_id(self) -> str:
        """Generate unique ID for genotype."""
        genotype_dict = self.to_dict()
        genotype_str = json.dumps(genotype_dict, sort_keys=True)
        return hashlib.sha256(genotype_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert genotype to dictionary."""
        return {
            "encode": {
                "encoding_strategies": self.encode.encoding_strategies,
                "llm_model": self.encode.llm_model,
                "temperature": self.encode.temperature,
                "max_tokens": self.encode.max_tokens,
                "batch_size": self.encode.batch_size,
                "enable_abstractions": self.encode.enable_abstractions,
                "min_abstraction_units": self.encode.min_abstraction_units
            },
            "store": {
                "backend_type": self.store.backend_type,
                "index_type": self.store.index_type
            },
            "retrieve": {
                "strategy_type": self.retrieve.strategy_type,
                "default_top_k": self.retrieve.default_top_k,
                "similarity_threshold": self.retrieve.similarity_threshold,
                "enable_filters": self.retrieve.enable_filters,
                "semantic_embedding_model": self.retrieve.semantic_embedding_model,
                "semantic_cache_enabled": self.retrieve.semantic_cache_enabled,
                "keyword_case_sensitive": self.retrieve.keyword_case_sensitive,
                "hybrid_semantic_weight": self.retrieve.hybrid_semantic_weight,
                "hybrid_keyword_weight": self.retrieve.hybrid_keyword_weight
            },
            "manage": {
                "strategy_type": self.manage.strategy_type,
                "enable_auto_management": self.manage.enable_auto_management,
                "prune_max_age_days": self.manage.prune_max_age_days,
                "prune_max_count": self.manage.prune_max_count,
                "prune_by_type": self.manage.prune_by_type,
                "consolidate_enabled": self.manage.consolidate_enabled,
                "consolidate_min_units": self.manage.consolidate_min_units,
                "deduplicate_enabled": self.manage.deduplicate_enabled,
                "deduplicate_similarity_threshold": self.manage.deduplicate_similarity_threshold,
                "forgetting_strategy": self.manage.forgetting_strategy,
                "forgetting_percentage": self.manage.forgetting_percentage
            },
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(
        cls,
        genotype_dict: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'MemoryGenotype':
        """Create genotype from dictionary."""
        encode_data = genotype_dict.get("encode", {})
        store_data = genotype_dict.get("store", {})
        retrieve_data = genotype_dict.get("retrieve", {})
        manage_data = genotype_dict.get("manage", {})

        encode_config = EncodeConfig(
            encoding_strategies=encode_data.get(
                "encoding_strategies", ["lesson", "skill"]
            ),
            llm_model=encode_data.get("llm_model"),
            temperature=encode_data.get("temperature", 0.7),
            max_tokens=encode_data.get("max_tokens", 512),
            batch_size=encode_data.get("batch_size", 10),
            enable_abstractions=encode_data.get("enable_abstractions", True),
            min_abstraction_units=encode_data.get("min_abstraction_units", 3)
        )

        store_config = StoreConfig(
            backend_type=store_data.get("backend_type", "json"),
            index_type=store_data.get("index_type", "flat")
        )

        retrieve_config = RetrieveConfig(
            strategy_type=retrieve_data.get("strategy_type", "hybrid"),
            default_top_k=retrieve_data.get("default_top_k", 5),
            similarity_threshold=retrieve_data.get("similarity_threshold", 0.7),
            enable_filters=retrieve_data.get("enable_filters", True),
            semantic_embedding_model=retrieve_data.get("semantic_embedding_model"),
            semantic_cache_enabled=retrieve_data.get("semantic_cache_enabled", True),
            keyword_case_sensitive=retrieve_data.get("keyword_case_sensitive", False),
            hybrid_semantic_weight=retrieve_data.get("hybrid_semantic_weight", 0.7),
            hybrid_keyword_weight=retrieve_data.get("hybrid_keyword_weight", 0.3)
        )

        manage_config = ManageConfig(
            strategy_type=manage_data.get("strategy_type", "simple"),
            enable_auto_management=manage_data.get("enable_auto_management", True),
            prune_max_age_days=manage_data.get("prune_max_age_days"),
            prune_max_count=manage_data.get("prune_max_count"),
            prune_by_type=manage_data.get("prune_by_type"),
            consolidate_enabled=manage_data.get("consolidate_enabled", True),
            consolidate_min_units=manage_data.get("consolidate_min_units", 2),
            deduplicate_enabled=manage_data.get("deduplicate_enabled", True),
            deduplicate_similarity_threshold=manage_data.get(
                "deduplicate_similarity_threshold", 0.9
            ),
            forgetting_strategy=manage_data.get("forgetting_strategy", "lru"),
            forgetting_percentage=manage_data.get("forgetting_percentage", 0.1)
        )

        final_metadata = metadata or genotype_dict.get("metadata", {})

        return cls(
            encode=encode_config,
            store=store_config,
            retrieve=retrieve_config,
            manage=manage_config,
            metadata=final_metadata
        )


class GenotypeFactory:
    """Factory for generating diverse memory genotypes."""

    @staticmethod
    def create_baseline_genotype() -> MemoryGenotype:
        """Create baseline genotype with sensible defaults."""
        return MemoryGenotype()

    @staticmethod
    def create_agentkb_genotype() -> MemoryGenotype:
        """Create AgentKB genotype respecting boundary constraints."""
        config_manager = ConfigManager()
        boundaries = config_manager.config.evolution_boundaries

        return MemoryGenotype(
            encode=EncodeConfig(
                encoding_strategies=["lesson"],
                enable_abstractions=False
            ),
            store=StoreConfig(),
            retrieve=RetrieveConfig(
                strategy_type="keyword",
                default_top_k=min(3, boundaries.top_k_max)
            ),
            manage=ManageConfig(
                strategy_type="simple",
                enable_auto_management=True
            ),
            metadata={"architecture": "agentkb"}
        )

    @staticmethod
    def create_lightweight_genotype() -> MemoryGenotype:
        """Create Lightweight genotype respecting boundary constraints."""
        config_manager = ConfigManager()
        boundaries = config_manager.config.evolution_boundaries

        return MemoryGenotype(
            encode=EncodeConfig(
                encoding_strategies=["lesson", "skill"],
                batch_size=max(20, boundaries.batch_size_min),
                enable_abstractions=True
            ),
            store=StoreConfig(),
            retrieve=RetrieveConfig(
                strategy_type="keyword",
                default_top_k=min(5, boundaries.top_k_max)
            ),
            manage=ManageConfig(
                strategy_type="simple",
                enable_auto_management=True,
                consolidate_enabled=True
            ),
            metadata={"architecture": "lightweight"}
        )

    @staticmethod
    def create_riva_genotype() -> MemoryGenotype:
        """Create Riva genotype respecting boundary constraints."""
        config_manager = ConfigManager()
        boundaries = config_manager.config.evolution_boundaries

        return MemoryGenotype(
            encode=EncodeConfig(
                encoding_strategies=["lesson", "skill", "abstraction"],
                min_abstraction_units=5,
                enable_abstractions=True
            ),
            store=StoreConfig(),
            retrieve=RetrieveConfig(
                strategy_type="hybrid",
                default_top_k=min(10, boundaries.top_k_max),
                similarity_threshold=min(0.8, boundaries.relevance_threshold_max),
                hybrid_semantic_weight=0.8,
                hybrid_keyword_weight=0.2
            ),
            manage=ManageConfig(
                strategy_type="simple",
                enable_auto_management=True
            ),
            metadata={"architecture": "riva"}
        )

    @staticmethod
    def create_cerebra_genotype() -> MemoryGenotype:
        """Create Cerebra genotype respecting boundary constraints."""
        config_manager = ConfigManager()
        boundaries = config_manager.config.evolution_boundaries

        return MemoryGenotype(
            encode=EncodeConfig(
                encoding_strategies=["tool", "abstraction"],
                batch_size=max(5, boundaries.batch_size_min),
                enable_abstractions=True
            ),
            store=StoreConfig(),
            retrieve=RetrieveConfig(
                strategy_type="semantic",
                default_top_k=min(15, boundaries.top_k_max),  # BOUNDARY ENFORCED
                similarity_threshold=0.6,
                semantic_cache_enabled=True
            ),
            manage=ManageConfig(
                strategy_type="simple",
                enable_auto_management=True,
                consolidate_enabled=True
            ),
            metadata={"architecture": "cerebra"}
        )

    @staticmethod
    def create_random_genotype(
        mutate_probability: float = 0.3
    ) -> MemoryGenotype:
        """Create a random genotype with mutations."""
        baseline = GenotypeFactory.create_baseline_genotype()

        return GenotypeFactory.mutate_genotype(
            baseline,
            mutate_probability
        )

    @staticmethod
    def mutate_genotype(
        genotype: MemoryGenotype,
        mutate_probability: float = 0.3
    ) -> MemoryGenotype:
        """Apply random mutations to a genotype."""
        import random

        genotype_dict = genotype.to_dict()

        def mutate_dict(data: Dict[str, Any], prob: float) -> Dict[str, Any]:
            """Recursively mutate dictionary values."""
            mutated = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    mutated[key] = mutate_dict(value, prob)
                elif isinstance(value, list):
                    # List mutations not yet implemented
                    mutated[key] = value.copy()
                elif isinstance(value, bool):
                    if random.random() < prob:
                        mutated[key] = not value
                    else:
                        mutated[key] = value
                elif isinstance(value, (int, float)):
                    if random.random() < prob:
                        if isinstance(value, int):
                            # Mutate integers
                            if key.endswith("top_k"):
                                # Mutate within reasonable range
                                mutated[key] = max(
                                    1, min(20, value + random.choice([-1, 1, 2, -2])))
                            elif key.endswith("batch_size"):
                                # Mutate batch size within reasonable range
                                mutated[key] = max(
                                    1, min(50, value + random.choice([-5, -2, 2, 5])))
                            elif key.endswith("max_tokens"):
                                # Mutate max tokens within reasonable range
                                mutated[key] = max(
                                    256, min(4096, value + random.choice([-128, -64, 64, 128])))
                            else:
                                # Generic integer mutation
                                mutated[key] = max(0, value + random.choice([-1, 1]))
                        else:
                            # Float mutation
                            if "threshold" in key or "weight" in key:
                                # Mutate thresholds and weights within valid range
                                mutated[key] = max(
                                    0.0, min(1.0, value + random.choice([-0.1, -0.05, 0.05, 0.1])))
                            elif "temperature" in key:
                                # Mutate temperature within valid range
                                mutated[key] = max(
                                    0.0, min(2.0, value + random.choice([-0.1, -0.05, 0.05, 0.1])))
                            else:
                                # Generic float mutation
                                mutated[key] = value + random.choice([-0.1, -0.05, 0.05, 0.1])
                    else:
                        mutated[key] = value
                elif isinstance(value, str):
                    if random.random() < prob:
                        if "strategy_type" in key:
                            # Mutate strategy types
                            if "retrieval" in key:
                                strategies = ["keyword", "semantic", "hybrid"]
                                current_strategies = [value]
                                available = [s for s in strategies if s not in current_strategies]
                                if available:
                                    mutated[key] = random.choice(available)
                                else:
                                    mutated[key] = value
                            elif "management" in key:
                                strategies = ["simple", "advanced"]
                                current_strategies = [value]
                                available = [s for s in strategies if s not in current_strategies]
                                if available:
                                    mutated[key] = random.choice(available)
                                else:
                                    mutated[key] = value
                        elif "backend_type" in key:
                            # Mutate backend types
                            backends = ["json", "vector", "graph"]
                            current_backends = [value]
                            available = [b for b in backends if b not in current_backends]
                            if available:
                                mutated[key] = random.choice(available)
                            else:
                                mutated[key] = value
                        else:
                            mutated[key] = value
                    else:
                        mutated[key] = value
                else:
                    mutated[key] = value
            return mutated

        mutated_dict = mutate_dict(genotype_dict, mutate_probability)

        return MemoryGenotype.from_dict(
            mutated_dict,
            metadata=genotype.metadata.copy()
        )
