from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import hashlib


@dataclass
class EncodeConfig:
    """Configuration for Encode component.

    Note: Embedding dimension is determined by the embedding model's
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
    MEMEVOLVE_EMBEDDING_DIMENSION environment variable.
    """

    backend_type: str = "json"
    storage_path: str = "data/memory.json"
    vector_index_file: Optional[str] = None
    enable_persistence: bool = True
    max_storage_size_mb: Optional[int] = None


@dataclass
class RetrieveConfig:
    """Configuration for Retrieve component."""

    strategy_type: str = "semantic"
    default_top_k: int = 5
    similarity_threshold: float = 0.7
    enable_filters: bool = True

    semantic_embedding_model: Optional[str] = None
    semantic_cache_enabled: bool = True

    hybrid_semantic_weight: float = 0.7
    hybrid_keyword_weight: float = 0.3

    keyword_case_sensitive: bool = False


@dataclass
class ManageConfig:
    """Configuration for Manage component."""

    strategy_type: str = "simple"
    enable_auto_management: bool = True
    auto_prune_threshold: int = 1000

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
    """Represents a memory architecture genotype.

    A genotype fully specifies how to configure the four
    memory components (Encode, Store, Retrieve, Manage).
    """

    encode: EncodeConfig = field(default_factory=EncodeConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)
    manage: ManageConfig = field(default_factory=ManageConfig)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_genome_id(self) -> str:
        """Generate unique genome ID from configuration."""
        genome_str = self.to_json()
        return hashlib.md5(genome_str.encode()).hexdigest()[:8]

    def to_json(self) -> str:
        """Convert genotype to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

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
                "storage_path": self.store.storage_path,
                "vector_index_file": self.store.vector_index_file,
                "enable_persistence": self.store.enable_persistence,
                "max_storage_size_mb": self.store.max_storage_size_mb
            },
            "retrieve": {
                "strategy_type": self.retrieve.strategy_type,
                "default_top_k": self.retrieve.default_top_k,
                "similarity_threshold": self.retrieve.similarity_threshold,
                "enable_filters": self.retrieve.enable_filters,
                "semantic_embedding_model": (
                    self.retrieve.semantic_embedding_model
                ),
                "semantic_cache_enabled": (
                    self.retrieve.semantic_cache_enabled
                ),
                "hybrid_semantic_weight":
                    self.retrieve.hybrid_semantic_weight,
                "hybrid_keyword_weight":
                    self.retrieve.hybrid_keyword_weight,
                "keyword_case_sensitive":
                    self.retrieve.keyword_case_sensitive
            },
            "manage": {
                "strategy_type": self.manage.strategy_type,
                "enable_auto_management":
                    self.manage.enable_auto_management,
                "auto_prune_threshold":
                    self.manage.auto_prune_threshold,
                "prune_max_age_days": self.manage.prune_max_age_days,
                "prune_max_count": self.manage.prune_max_count,
                "prune_by_type": self.manage.prune_by_type,
                "consolidate_enabled":
                    self.manage.consolidate_enabled,
                "consolidate_min_units":
                    self.manage.consolidate_min_units,
                "deduplicate_enabled":
                    self.manage.deduplicate_enabled,
                "deduplicate_similarity_threshold":
                    self.manage.deduplicate_similarity_threshold,
                "forgetting_strategy":
                    self.manage.forgetting_strategy,
                "forgetting_percentage":
                    self.manage.forgetting_percentage
            },
            "metadata": self.metadata
        }

    @classmethod
    def from_json(cls, json_str: str) -> 'MemoryGenotype':
        """Create genotype from JSON string."""
        data = json.loads(json_str)

        encode_data = data.get("encode", {})
        encode_config = EncodeConfig(**encode_data)

        store_data = data.get("store", {})
        store_config = StoreConfig(**store_data)

        retrieve_data = data.get("retrieve", {})
        retrieve_config = RetrieveConfig(**retrieve_data)

        manage_data = data.get("manage", {})
        manage_config = ManageConfig(**manage_data)

        metadata = data.get("metadata", {})

        return cls(
            encode=encode_config,
            store=store_config,
            retrieve=retrieve_config,
            manage=manage_config,
            metadata=metadata
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryGenotype':
        """Create genotype from dictionary."""
        encode_data = data.get("encode", {})
        encode_config = EncodeConfig(**encode_data)

        store_data = data.get("store", {})
        store_config = StoreConfig(**store_data)

        retrieve_data = data.get("retrieve", {})
        retrieve_config = RetrieveConfig(**retrieve_data)

        manage_data = data.get("manage", {})
        manage_config = ManageConfig(**manage_data)

        metadata = data.get("metadata", {})

        return cls(
            encode=encode_config,
            store=store_config,
            retrieve=retrieve_config,
            manage=manage_config,
            metadata=metadata
        )


class GenotypeFactory:
    """Factory for generating diverse memory genotypes."""

    @staticmethod
    def create_baseline_genotype() -> MemoryGenotype:
        """Create baseline genotype with sensible defaults."""
        return MemoryGenotype()

    @staticmethod
    def create_agentkb_genotype() -> MemoryGenotype:
        """Create AgentKB genotype (static baseline)."""
        return MemoryGenotype(
            encode=EncodeConfig(
                encoding_strategies=["lesson"],
                enable_abstractions=False
            ),
            store=StoreConfig(
                backend_type="json",
                enable_persistence=False
            ),
            retrieve=RetrieveConfig(
                strategy_type="keyword",
                default_top_k=3
            ),
            manage=ManageConfig(
                strategy_type="simple",
                enable_auto_management=False
            ),
            metadata={"architecture": "agentkb"}
        )

    @staticmethod
    def create_lightweight_genotype() -> MemoryGenotype:
        """Create Lightweight genotype (trajectory-based)."""
        return MemoryGenotype(
            encode=EncodeConfig(
                encoding_strategies=["lesson", "skill"],
                batch_size=20,
                enable_abstractions=True
            ),
            store=StoreConfig(
                backend_type="json",
                enable_persistence=True
            ),
            retrieve=RetrieveConfig(
                strategy_type="keyword",
                default_top_k=5
            ),
            manage=ManageConfig(
                strategy_type="simple",
                enable_auto_management=True,
                auto_prune_threshold=500,
                prune_max_age_days=30
            ),
            metadata={"architecture": "lightweight"}
        )

    @staticmethod
    def create_riva_genotype() -> MemoryGenotype:
        """Create Riva genotype (agent-centric, domain-aware)."""
        return MemoryGenotype(
            encode=EncodeConfig(
                encoding_strategies=["lesson", "skill", "abstraction"],
                min_abstraction_units=5,
                enable_abstractions=True
            ),
            store=StoreConfig(
                backend_type="vector",
                enable_persistence=True
            ),
            retrieve=RetrieveConfig(
                strategy_type="hybrid",
                default_top_k=10,
                similarity_threshold=0.8,
                hybrid_semantic_weight=0.8,
                hybrid_keyword_weight=0.2
            ),
            manage=ManageConfig(
                strategy_type="simple",
                enable_auto_management=True,
                auto_prune_threshold=1000,
                deduplicate_enabled=True
            ),
            metadata={"architecture": "riva"}
        )

    @staticmethod
    def create_cerebra_genotype() -> MemoryGenotype:
        """Create Cerebra genotype (tool distillation)."""
        return MemoryGenotype(
            encode=EncodeConfig(
                encoding_strategies=["tool", "abstraction"],
                batch_size=5,
                enable_abstractions=True
            ),
            store=StoreConfig(
                backend_type="vector",
                enable_persistence=True,
                vector_index_file="data/cerebra_vectors"
            ),
            retrieve=RetrieveConfig(
                strategy_type="semantic",
                default_top_k=15,
                similarity_threshold=0.6,
                semantic_cache_enabled=True
            ),
            manage=ManageConfig(
                strategy_type="simple",
                enable_auto_management=True,
                auto_prune_threshold=2000,
                consolidate_enabled=True,
                deduplicate_enabled=True
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
                    if random.random() < prob:
                        if key == "encoding_strategies":
                            all_strategies = [
                                "lesson", "skill", "tool", "abstraction"
                            ]
                            value = random.sample(
                                all_strategies,
                                random.randint(1, 3)
                            )
                        elif random.random() < 0.5:
                            value.reverse()
                    mutated[key] = value
                elif isinstance(value, float):
                    if random.random() < prob:
                        if key in ["temperature", "similarity_threshold"]:
                            value = round(random.uniform(0.0, 1.0), 2)
                        elif key in [
                            "hybrid_semantic_weight",
                            "hybrid_keyword_weight"
                        ]:
                            value = round(random.uniform(0.0, 1.0), 2)
                        elif key == "forgetting_percentage":
                            value = round(random.uniform(0.05, 0.5), 2)
                    mutated[key] = value
                elif isinstance(value, int):
                    if random.random() < prob:
                        if key == "default_top_k":
                            value = random.randint(3, 20)
                        elif key == "batch_size":
                            value = random.choice([5, 10, 20, 50])
                        elif key == "auto_prune_threshold":
                            value = random.choice([100, 500, 1000, 2000])
                        elif key == "max_tokens":
                            value = random.choice([256, 512, 1024, 2048])
                    mutated[key] = value
                elif isinstance(value, bool):
                    if random.random() < 0.2:
                        mutated[key] = not value
                    else:
                        mutated[key] = value
                else:
                    mutated[key] = value
            return mutated

        mutated_data = mutate_dict(genotype_dict, mutate_probability)

        return MemoryGenotype.from_dict(mutated_data)
