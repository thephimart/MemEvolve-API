"""Configuration management system for MemEvolve."""

# Standard library imports
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import requests
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Note: Use standard logging to avoid circular imports with logging_manager

# Initialize logger for this module using standard logging
logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """LLM configuration for memory management operations."""
    base_url: Optional[str] = None
    api_key: str = ""
    model: Optional[str] = None
    auto_resolve_models: bool = True
    timeout: int = 600
    max_retries: int = 3
    max_tokens: Optional[int] = None

    def __post_init__(self, global_config=None):
        """Load from environment variables."""
        self.base_url = os.getenv("MEMEVOLVE_MEMORY_BASE_URL", self.base_url)
        self.api_key = os.getenv("MEMEVOLVE_MEMORY_API_KEY", self.api_key)
        self.model = os.getenv("MEMEVOLVE_MEMORY_MODEL", self.model)

        auto_resolve_env = os.getenv("MEMEVOLVE_MEMORY_AUTO_RESOLVE_MODELS")
        if auto_resolve_env is not None:
            self.auto_resolve_models = auto_resolve_env.lower() in ("true", "1", "yes", "on")

        timeout_env = os.getenv("MEMEVOLVE_MEMORY_TIMEOUT", "600")
        try:
            self.timeout = int(timeout_env)
        except ValueError:
            pass


@dataclass
class UpstreamConfig:
    """Upstream API configuration."""
    base_url: Optional[str] = None
    api_key: str = ""
    model: Optional[str] = None
    auto_resolve_models: bool = True
    timeout: int = 600
    max_retries: int = 3
    max_tokens: Optional[int] = None

    def __post_init__(self):
        """Load from environment variables."""
        self.base_url = os.getenv("MEMEVOLVE_UPSTREAM_BASE_URL", self.base_url)
        self.api_key = os.getenv("MEMEVOLVE_UPSTREAM_API_KEY", self.api_key)
        self.model = os.getenv("MEMEVOLVE_UPSTREAM_MODEL", self.model)
        auto_resolve_env = os.getenv("MEMEVOLVE_UPSTREAM_AUTO_RESOLVE_MODELS")
        if auto_resolve_env is not None:
            self.auto_resolve_models = auto_resolve_env.lower() in ("true", "1", "yes", "on")

        timeout_env = os.getenv("MEMEVOLVE_UPSTREAM_TIMEOUT", "600")
        try:
            self.timeout = int(timeout_env)
        except ValueError:
            pass

        # Load max_tokens from env
        max_tokens_env = os.getenv("MEMEVOLVE_UPSTREAM_MAX_TOKENS")
        if max_tokens_env and max_tokens_env.strip():
            try:
                self.max_tokens = int(max_tokens_env)
            except ValueError:
                pass


@dataclass
class StorageConfig:
    """Storage backend configuration."""
    backend_type: str = "json"
    path: str = "./data/memory"
    index_type: str = "flat"

    def __post_init__(self):
        """Load from environment variables."""
        backend_type_env = os.getenv("MEMEVOLVE_STORAGE_BACKEND_TYPE")
        if backend_type_env is not None:
            self.backend_type = backend_type_env

        path_env = os.getenv("MEMEVOLVE_DATA_DIR")
        if path_env is not None:
            self.path = path_env

        index_type_env = os.getenv("MEMEVOLVE_STORAGE_INDEX_TYPE")
        if index_type_env is not None:
            self.index_type = index_type_env


@dataclass
class RetrievalConfig:
    """Retrieval strategy configuration."""
    # All values loaded from .env in __post_init__
    strategy_type: str = ""
    default_top_k: int = 0
    # semantic_weight: REMOVED - use hybrid_semantic_weight instead
    # keyword_weight: REMOVED - use hybrid_keyword_weight instead
    enable_caching: bool = False
    cache_size: int = 0

    # Evolution-managed parameters (loaded from .env)
    # similarity_threshold: REMOVED - use relevance_threshold instead
    # enable_filters: REMOVED - retrieval filtering eliminated
    enable_filters: bool = False
    semantic_cache_enabled: bool = False
    keyword_case_sensitive: bool = False
    semantic_embedding_model: Optional[str] = ""
    hybrid_semantic_weight: float = 0.0
    hybrid_keyword_weight: float = 0.0

    # Legacy aliases for backward compatibility
    @property
    def semantic_weight(self) -> float:
        """Legacy alias for hybrid_semantic_weight."""
        return self.hybrid_semantic_weight

    @property
    def keyword_weight(self) -> float:
        """Legacy alias for hybrid_keyword_weight."""
        return self.hybrid_keyword_weight

    # Memory filtering parameters (loaded from .env)
    relevance_threshold: float = 0.0

    def __post_init__(self):
        """Load from environment variables."""
        strategy_type_env = os.getenv("MEMEVOLVE_RETRIEVAL_STRATEGY_TYPE")
        if strategy_type_env is not None:
            self.strategy_type = strategy_type_env

        # default_top_k is now handled by env_mappings in ConfigManager

        # Legacy semantic_weight and keyword_weight removed - use hybrid versions instead

        enable_caching_env = os.getenv("MEMEVOLVE_RETRIEVAL_ENABLE_CACHING")
        if enable_caching_env is not None:
            self.enable_caching = enable_caching_env.lower() in ("true", "1", "yes", "on")

        cache_size_env = os.getenv("MEMEVOLVE_RETRIEVAL_CACHE_SIZE")
        if cache_size_env:
            try:
                self.cache_size = int(cache_size_env)
            except ValueError:
                pass

        # similarity_threshold: REMOVED - use relevance_threshold instead

        enable_filters_env = os.getenv("MEMEVOLVE_RETRIEVAL_ENABLE_FILTERS")
        if enable_filters_env is not None:
            self.enable_filters = enable_filters_env.lower() in ("true", "1", "yes", "on")

        semantic_cache_env = os.getenv("MEMEVOLVE_RETRIEVAL_SEMANTIC_CACHE_ENABLED")
        if semantic_cache_env is not None:
            self.semantic_cache_enabled = semantic_cache_env.lower() in ("true", "1", "yes", "on")

        keyword_case_env = os.getenv("MEMEVOLVE_RETRIEVAL_KEYWORD_CASE_SENSITIVE")
        if keyword_case_env is not None:
            self.keyword_case_sensitive = keyword_case_env.lower() in ("true", "1", "yes", "on")

        semantic_embedding_model_env = os.getenv("MEMEVOLVE_RETRIEVAL_SEMANTIC_EMBEDDING_MODEL")
        if semantic_embedding_model_env is not None:
            self.semantic_embedding_model = semantic_embedding_model_env

        # Load hybrid weights from environment (required fields)
        hybrid_semantic_env = os.getenv("MEMEVOLVE_RETRIEVAL_HYBRID_SEMANTIC_WEIGHT")
        if hybrid_semantic_env:
            try:
                self.hybrid_semantic_weight = float(hybrid_semantic_env)
            except ValueError:
                pass

        # Load hybrid keyword weight from environment (required field)
        hybrid_keyword_env = os.getenv("MEMEVOLVE_RETRIEVAL_HYBRID_KEYWORD_WEIGHT")
        if hybrid_keyword_env:
            try:
                self.hybrid_keyword_weight = float(hybrid_keyword_env)
            except ValueError:
                pass

        # Memory relevance threshold filtering
        relevance_threshold_env = os.getenv("MEMEVOLVE_MEMORY_RELEVANCE_THRESHOLD")
        if relevance_threshold_env:
            try:
                self.relevance_threshold = float(relevance_threshold_env)
            except ValueError:
                pass


@dataclass
class ManagementConfig:
    """Memory management configuration - all values loaded from .env."""
    enable_auto_management: bool = False
    auto_prune_threshold: int = 0
    auto_consolidate_interval: int = 0
    deduplicate_threshold: float = 0.0
    forgetting_strategy: str = ""
    max_memory_age_days: int = 0

    # Evolution-managed parameters (loaded from .env)
    strategy_type: str = ""
    # prune_max_age_days: REMOVED - redundant with max_memory_age_days
    prune_max_count: Optional[int] = None
    prune_by_type: Optional[str] = None
    consolidate_enabled: bool = False
    consolidate_min_units: int = 0
    deduplicate_enabled: bool = False
    forgetting_percentage: float = 0.0

    def __post_init__(self):
        """Load from environment variables."""
        enable_auto_env = os.getenv(
            "MEMEVOLVE_MANAGEMENT_ENABLE_AUTO_MANAGEMENT")
        if enable_auto_env is not None:
            self.enable_auto_management = enable_auto_env.lower() in ("true", "1", "yes", "on")

        # Load all values from environment (no hardcoded defaults)
        prune_threshold_env = os.getenv("MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD")
        if prune_threshold_env:
            try:
                self.auto_prune_threshold = int(prune_threshold_env)
            except ValueError:
                pass

        consolidate_interval_env = os.getenv("MEMEVOLVE_MANAGEMENT_AUTO_CONSOLIDATE_INTERVAL")
        if consolidate_interval_env:
            try:
                self.auto_consolidate_interval = int(
                    consolidate_interval_env)
            except ValueError:
                pass

        deduplicate_threshold_env = os.getenv("MEMEVOLVE_MANAGEMENT_DEDUPLICATE_THRESHOLD")
        if deduplicate_threshold_env:
            try:
                self.deduplicate_threshold = float(deduplicate_threshold_env)
            except ValueError:
                pass

        forgetting_strategy_env = os.getenv("MEMEVOLVE_MANAGEMENT_FORGETTING_STRATEGY")
        if forgetting_strategy_env is not None:
            self.forgetting_strategy = forgetting_strategy_env

        max_age_env = os.getenv("MEMEVOLVE_MANAGEMENT_MAX_MEMORY_AGE_DAYS")
        if max_age_env:
            try:
                self.max_memory_age_days = int(max_age_env)
            except ValueError:
                pass

        # Evolution-managed parameters (loaded from environment)
        strategy_type_env = os.getenv("MEMEVOLVE_MANAGEMENT_STRATEGY_TYPE")
        if strategy_type_env is not None:
            self.strategy_type = strategy_type_env

        # prune_max_age_days loading removed - redundant with max_memory_age_days

        if self.prune_max_count is None:
            prune_max_count_env = os.getenv("MEMEVOLVE_MANAGEMENT_PRUNE_MAX_COUNT")
            if prune_max_count_env:
                try:
                    self.prune_max_count = int(prune_max_count_env)
                except ValueError:
                    pass

        if self.prune_by_type is None:
            self.prune_by_type = os.getenv("MEMEVOLVE_MANAGEMENT_PRUNE_BY_TYPE", self.prune_by_type)

        consolidate_enabled_env = os.getenv("MEMEVOLVE_MANAGEMENT_CONSOLIDATE_ENABLED")
        if consolidate_enabled_env is not None:
            self.consolidate_enabled = consolidate_enabled_env.lower() in ("true", "1", "yes", "on")
            consolidate_min_env = os.getenv("MEMEVOLVE_MANAGEMENT_CONSOLIDATE_MIN_UNITS")
            if consolidate_min_env:
                try:
                    self.consolidate_min_units = int(consolidate_min_env)
                except ValueError:
                    pass

        deduplicate_enabled_env = os.getenv("MEMEVOLVE_MANAGEMENT_DEDUPLICATE_ENABLED")
        if deduplicate_enabled_env is not None:
            self.deduplicate_enabled = deduplicate_enabled_env.lower() in ("true", "1", "yes", "on")

        forgetting_pct_env = os.getenv("MEMEVOLVE_MANAGEMENT_FORGETTING_PERCENTAGE")
        if forgetting_pct_env:
            try:
                self.forgetting_percentage = float(forgetting_pct_env)
            except ValueError:
                pass


@dataclass
class EncoderConfig:
    """Experience encoder configuration - all values loaded from .env."""
    encoding_strategies: list = field(default_factory=lambda: [])
    enable_abstraction: bool = False
    abstraction_threshold: int = 0
    enable_tool_extraction: bool = False

    # Evolution-managed parameters (loaded from .env)
    max_tokens: int = 0
    batch_size: int = 0
    temperature: float = 0.0
    llm_model: Optional[str] = ""
    enable_abstractions: bool = False
    min_abstraction_units: int = 0

    def __post_init__(self):
        """Load all values from environment variables."""
        strategies_env = os.getenv("MEMEVOLVE_ENCODER_ENCODING_STRATEGIES")
        if strategies_env:
            self.encoding_strategies = [
                s.strip() for s in strategies_env.split(",") if s.strip()]

        enable_abstraction_env = os.getenv("MEMEVOLVE_ENCODER_ENABLE_ABSTRACTION")
        if enable_abstraction_env is not None:
            self.enable_abstraction = enable_abstraction_env.lower() in ("true", "1", "yes", "on")

        abstraction_threshold_env = os.getenv("MEMEVOLVE_ENCODER_ABSTRACTION_THRESHOLD")
        if abstraction_threshold_env:
            try:
                self.abstraction_threshold = int(abstraction_threshold_env)
            except ValueError:
                pass

        enable_tool_env = os.getenv("MEMEVOLVE_ENCODER_ENABLE_TOOL_EXTRACTION")
        if enable_tool_env is not None:
            self.enable_tool_extraction = enable_tool_env.lower() in ("true", "1", "yes", "on")

        # Evolution-managed parameters (loaded from environment)
        max_tokens_env = os.getenv("MEMEVOLVE_ENCODER_MAX_TOKENS")
        if max_tokens_env:
            try:
                self.max_tokens = int(max_tokens_env)
            except ValueError:
                pass

        batch_size_env = os.getenv("MEMEVOLVE_ENCODER_BATCH_SIZE")
        if batch_size_env:
            try:
                self.batch_size = int(batch_size_env)
            except ValueError:
                pass

        temperature_env = os.getenv("MEMEVOLVE_ENCODER_TEMPERATURE")
        if temperature_env:
            try:
                self.temperature = float(temperature_env)
            except ValueError:
                pass

        llm_model_env = os.getenv("MEMEVOLVE_ENCODER_LLM_MODEL")
        if llm_model_env:
            self.llm_model = llm_model_env

        enable_abstractions_env = os.getenv("MEMEVOLVE_ENCODER_ENABLE_ABSTRACTIONS")
        if enable_abstractions_env is not None:
            self.enable_abstractions = enable_abstractions_env.lower() in ("true", "1", "yes", "on")

        min_abstraction_env = os.getenv("MEMEVOLVE_ENCODER_MIN_ABSTRACTION_UNITS")
        if min_abstraction_env:
            try:
                self.min_abstraction_units = int(min_abstraction_env)
            except ValueError:
                pass


@dataclass
class EmbeddingConfig:
    """Embedding API configuration."""
    base_url: Optional[str] = None
    api_key: str = ""
    model: Optional[str] = None
    auto_resolve_models: bool = True
    timeout: int = 60
    max_retries: int = 3
    max_tokens: Optional[int] = None
    dimension: Optional[int] = None

    def __post_init__(self):
        """Load from environment variables."""
        self.base_url = os.getenv("MEMEVOLVE_EMBEDDING_BASE_URL", self.base_url)
        self.api_key = os.getenv("MEMEVOLVE_EMBEDDING_API_KEY", self.api_key)
        self.model = os.getenv("MEMEVOLVE_EMBEDDING_MODEL", self.model)
        auto_resolve_env = os.getenv("MEMEVOLVE_EMBEDDING_AUTO_RESOLVE_MODELS")
        if auto_resolve_env is not None:
            self.auto_resolve_models = auto_resolve_env.lower() in ("true", "1", "yes", "on")
        timeout_env = os.getenv("MEMEVOLVE_EMBEDDING_TIMEOUT", "60")
        try:
            self.timeout = int(timeout_env)
        except ValueError:
            pass

        # Load max_tokens from env (empty means auto-detect)
        # max_tokens is now handled by env_mappings in ConfigManager

        # Priority 1: Check evolution state for dimension override
        if self.dimension is None:
            evolution_dim = self._get_dimension_from_evolution_state()
            if evolution_dim is not None:
                self.dimension = evolution_dim
                logging.debug(f"Using embedding dimension from evolution state: {evolution_dim}")

        # Priority 2: Load max_tokens from env (empty means auto-detect)
        if self.max_tokens is None:
            max_tokens_env = os.getenv("MEMEVOLVE_EMBEDDING_MAX_TOKENS")
            if max_tokens_env and max_tokens_env.strip():
                try:
                    self.max_tokens = int(max_tokens_env)
                    logging.debug(f"Using embedding max_tokens from environment: {self.max_tokens}")
                except ValueError:
                    logging.warning(
                        f"Invalid MEMEVOLVE_EMBEDDING_MAX_TOKENS: {max_tokens_env}, "
                        "using auto-detection")

        # Priority 3: Load dimension from env (empty means auto-detect)
        if self.dimension is None:
            dimension_env = os.getenv("MEMEVOLVE_EMBEDDING_DIMENSION")
            if dimension_env and dimension_env.strip():
                try:
                    self.dimension = int(dimension_env)
                    logging.debug(f"Using embedding dimension from environment: {self.dimension}")
                except ValueError:
                    logging.warning(
                        f"Invalid MEMEVOLVE_EMBEDDING_DIMENSION: {dimension_env}, "
                        "using auto-detection")

        # Unified auto-resolution handled centrally to prevent redundant API calls
        # Individual auto-detection disabled - handled by ConfigManager._resolve_all_auto_configs()
        # if self.max_tokens is None or self.dimension is None:
        #     self._auto_detect_from_models()

    def _auto_detect_from_models(self):
        """Auto-detect max_tokens and dimension from /models endpoint."""
        if not self.base_url:
            return

        try:
            models_endpoint = f"{self.base_url.rstrip('/')}/models"
            response = requests.get(models_endpoint, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Get first model's metadata
            if "data" in data and len(data["data"]) > 0:
                meta = data["data"][0].get("meta", {})

                # Auto-detect max_tokens (context window)
                if self.max_tokens is None:
                    detected_max_tokens = meta.get("n_ctx_train", 512)
                    self.max_tokens = detected_max_tokens
                    logging.debug(
                        f"Auto-detected embedding max_tokens: {detected_max_tokens} "
                        "(from models endpoint)")

                # Auto-detect dimension
                if self.dimension is None:
                    detected_dim = meta.get("n_embd", 768)
                    self.dimension = detected_dim
                    logging.debug(
                        f"Auto-detected embedding dimension: {detected_dim} "
                        "(from models endpoint)")

        except Exception as e:
            logging.warning(
                f"Failed to auto-detect embedding settings from {self.base_url}: {e}")
            # Set fallbacks
            if self.max_tokens is None:
                self.max_tokens = 512
                logging.warning(
                    "Using fallback embedding max_tokens: 512 (default)")
            if self.dimension is None:
                self.dimension = 768
                logging.warning(
                    "Using fallback embedding dimension: 768 (default)")

    def _get_dimension_from_evolution_state(self) -> Optional[int]:
        """Get embedding dimension from evolution state file."""
        try:
            import json
            import os

            # Use centralized data directory
            data_dir = os.getenv("MEMEVOLVE_DATA_DIR", "./data")
            evolution_dir = os.path.join(data_dir, 'evolution')
            evolution_state_path = os.path.join(evolution_dir, 'evolution_state.json')

            if os.path.exists(evolution_state_path):
                with open(evolution_state_path, 'r') as f:
                    evolution_data = json.load(f)
                    # Look for current genotype embedding dimension
                    current_genotype = evolution_data.get('current_genotype', {})
                    if 'embedding_dim' in current_genotype:
                        return current_genotype['embedding_dim']
        except Exception as e:
            logging.debug(f"Failed to read evolution state for dimension: {e}")

        return None

    def get_models_endpoint(self) -> Optional[str]:
        """Get the models endpoint URL for llama.cpp APIs."""
        if self.auto_resolve_models and self.base_url:
            return f"{self.base_url.rstrip('/')}/models"
        return None

    def resolve_available_models(self) -> List[Dict[str, Any]]:
        """Resolve available models from the llama.cpp API."""
        endpoint = self.get_models_endpoint()
        if not endpoint:
            return []

        try:
            response = requests.get(endpoint, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Handle different response formats from llama.cpp
            if "data" in data:
                return data["data"]
            elif isinstance(data, list):
                return data
            else:
                logging.warning(
                    f"Unexpected response format from {endpoint}: {data}")
                return []
        except Exception as e:
            logging.warning(f"Failed to resolve models from {endpoint}: {e}")
            return []


@dataclass
class EvolutionBoundaryConfig:
    """Evolution parameter boundaries loaded from .env."""
    # Field definitions - values loaded from .env in __post_init__
    max_tokens_min: int = 256  # Default fallback
    max_tokens_max: int = 4096  # Default fallback
    top_k_min: int = 2  # Default fallback
    top_k_max: int = 10  # Default fallback
    relevance_threshold_min: float = 0.3  # Default fallback
    relevance_threshold_max: float = 0.8  # Default fallback
    temperature_min: float = 0.0  # Default fallback
    temperature_max: float = 1.0  # Default fallback
    min_requests_per_cycle: int = 50  # Default fallback
    fitness_history_size: int = 100  # Default fallback

    # Mutation operation parameters
    batch_size_min: int = 5
    batch_size_multiplier: float = 2.0
    token_step_size: int = 256

    # Allowed mutation options (user-configurable)
    retrieval_strategies: List[str] = field(
        default_factory=lambda: [
            "keyword", "semantic", "hybrid", "llm_guided"])
    hybrid_weight_range: Tuple[float, float] = (0.0, 1.0)
    encoding_strategies_options: List[List[str]] = field(default_factory=lambda: [
        ["lesson"], ["lesson", "skill"], ["lesson", "skill", "tool"],
        ["lesson", "skill", "abstraction"], ["skill", "tool"], ["tool", "abstraction"]
    ])
    management_strategies: List[str] = field(default_factory=lambda: ["simple", "advanced"])
    forgetting_strategies: List[str] = field(
        default_factory=lambda: [
            "lru", "lfu", "random", "cost_based"])
    forgetting_percentage_range: Tuple[float, float] = (0.05, 0.3)
    cost_threshold_range: Tuple[float, float] = (0.8, 0.95)

    # Mutation operation probabilities and ranges
    strategy_addition_probability: float = 0.5
    temperature_change_delta: float = 0.1

    # Fallback boundary values (when boundary_config not available)
    fallback_top_k_min: int = 3
    fallback_top_k_max: int = 20
    fallback_similarity_min: float = 0.5
    fallback_similarity_max: float = 0.9

    def __post_init__(self):
        """Load from environment variables with fallbacks in config.py."""
        self.max_tokens_min = int(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_MAX_TOKENS_MIN",
                self.max_tokens_min))
        self.max_tokens_max = int(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_MAX_TOKENS_MAX",
                self.max_tokens_max))
        self.max_tokens_min = int(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_MAX_TOKENS_MIN",
                self.max_tokens_min))
        self.max_tokens_max = int(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_MAX_TOKENS_MAX",
                self.max_tokens_max))
        self.top_k_min = int(os.getenv("MEMEVOLVE_EVOLUTION_TOP_K_MIN", self.top_k_min))
        self.top_k_max = int(os.getenv("MEMEVOLVE_EVOLUTION_TOP_K_MAX", self.top_k_max))
        self.relevance_threshold_min = float(
            os.getenv("MEMEVOLVE_EVOLUTION_RELEVANCE_THRESHOLD_MIN", self.relevance_threshold_min))
        self.relevance_threshold_max = float(
            os.getenv("MEMEVOLVE_EVOLUTION_RELEVANCE_THRESHOLD_MAX", self.relevance_threshold_max))
        # similarity_threshold_min/max: REMOVED - use relevance_threshold_min/max instead
        self.temperature_min = float(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_TEMPERATURE_MIN",
                self.temperature_min))
        self.temperature_max = float(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_TEMPERATURE_MAX",
                self.temperature_max))
        self.min_requests_per_cycle = int(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_MIN_REQUESTS_PER_CYCLE",
                self.min_requests_per_cycle))
        self.fitness_history_size = int(
            os.getenv(
                "MEMEVOLVE_FITNESS_HISTORY_SIZE",
                self.fitness_history_size))

        # Mutation operation parameters
        self.batch_size_min = int(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_BATCH_SIZE_MIN",
                self.batch_size_min))
        self.batch_size_multiplier = float(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_BATCH_SIZE_MULTIPLIER",
                self.batch_size_multiplier))
        self.token_step_size = int(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_TOKEN_STEP_SIZE",
                self.token_step_size))

        # Allowed mutation options from environment
        retrieval_env = os.getenv("MEMEVOLVE_EVOLUTION_RETRIEVAL_STRATEGIES")
        if retrieval_env:
            self.retrieval_strategies = [r.strip() for r in retrieval_env.split(",") if r.strip()]

        hybrid_min = float(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_HYBRID_WEIGHT_MIN",
                self.hybrid_weight_range[0]))
        hybrid_max = float(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_HYBRID_WEIGHT_MAX",
                self.hybrid_weight_range[1]))
        self.hybrid_weight_range = (hybrid_min, hybrid_max)

        strategies_env = os.getenv("MEMEVOLVE_EVOLUTION_ENCODING_STRATEGIES")
        if strategies_env:
            self.encoding_strategies_options = [
                [s.strip() for s in strat.split(",") if s.strip()]
                for strat in strategies_env.split("|") if strat.strip()
            ]

        mgmt_env = os.getenv("MEMEVOLVE_EVOLUTION_MANAGEMENT_STRATEGIES")
        if mgmt_env:
            self.management_strategies = [m.strip() for m in mgmt_env.split(",") if m.strip()]

        forgetting_env = os.getenv("MEMEVOLVE_EVOLUTION_FORGETTING_STRATEGIES")
        if forgetting_env:
            self.forgetting_strategies = [f.strip() for f in forgetting_env.split(",") if f.strip()]

        forget_min = float(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_FORGETTING_PERCENTAGE_MIN",
                self.forgetting_percentage_range[0]))
        forget_max = float(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_FORGETTING_PERCENTAGE_MAX",
                self.forgetting_percentage_range[1]))
        self.forgetting_percentage_range = (forget_min, forget_max)

        cost_min = float(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_COST_THRESHOLD_MIN",
                self.cost_threshold_range[0]))
        cost_max = float(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_COST_THRESHOLD_MAX",
                self.cost_threshold_range[1]))
        self.cost_threshold_range = (cost_min, cost_max)

        # Mutation operation probabilities
        self.strategy_addition_probability = float(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_STRATEGY_ADDITION_PROBABILITY",
                self.strategy_addition_probability))
        self.temperature_change_delta = float(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_TEMPERATURE_DELTA",
                self.temperature_change_delta))

        # Fallback boundary values
        self.fallback_top_k_min = int(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_TOP_K_MIN",
                self.fallback_top_k_min))
        self.fallback_top_k_max = int(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_TOP_K_MAX",
                self.fallback_top_k_max))
        self.fallback_similarity_min = float(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_RELEVANCE_THRESHOLD_MIN",
                self.fallback_similarity_min))
        self.fallback_similarity_max = float(
            os.getenv(
                "MEMEVOLVE_EVOLUTION_RELEVANCE_THRESHOLD_MAX",
                self.fallback_similarity_max))


@dataclass
class EvolutionConfig:
    """Evolution framework configuration - all values loaded from .env."""
    enable: bool = False
    population_size: int = 0
    generations: int = 0
    mutation_rate: float = 0.0
    crossover_rate: float = 0.0
    selection_method: str = ""
    tournament_size: int = 0

    # Fitness evaluation weights (loaded from .env)
    fitness_weight_success: float = 0.0
    fitness_weight_tokens: float = 0.0
    fitness_weight_time: float = 0.0
    fitness_weight_retrieval: float = 0.0

    def __post_init__(self):
        """Load from environment variables."""
        enable_env = os.getenv("MEMEVOLVE_ENABLE_EVOLUTION")
        if enable_env is not None:
            self.enable = enable_env.lower() in ("true", "1", "yes", "on")

        population_size_env = os.getenv("MEMEVOLVE_EVOLUTION_POPULATION_SIZE")
        if population_size_env:
            try:
                self.population_size = int(population_size_env)
            except ValueError:
                pass

        generations_env = os.getenv("MEMEVOLVE_EVOLUTION_GENERATIONS")
        if generations_env:
            try:
                self.generations = int(generations_env)
            except ValueError:
                pass

        mutation_rate_env = os.getenv("MEMEVOLVE_EVOLUTION_MUTATION_RATE")
        if mutation_rate_env:
            try:
                self.mutation_rate = float(mutation_rate_env)
            except ValueError:
                pass

        crossover_rate_env = os.getenv("MEMEVOLVE_EVOLUTION_CROSSOVER_RATE")
        if crossover_rate_env:
            try:
                self.crossover_rate = float(crossover_rate_env)
            except ValueError:
                pass

        selection_method_env = os.getenv("MEMEVOLVE_EVOLUTION_SELECTION_METHOD")
        if selection_method_env is not None:
            self.selection_method = selection_method_env

        tournament_size_env = os.getenv("MEMEVOLVE_EVOLUTION_TOURNAMENT_SIZE")
        if tournament_size_env:
            try:
                self.tournament_size = int(tournament_size_env)
            except ValueError:
                pass

        # Load fitness weights from environment
        fitness_weight_success_env = os.getenv("MEMEVOLVE_FITNESS_WEIGHT_SUCCESS")
        if fitness_weight_success_env:
            try:
                self.fitness_weight_success = float(fitness_weight_success_env)
            except ValueError:
                pass

        fitness_weight_tokens_env = os.getenv("MEMEVOLVE_FITNESS_WEIGHT_TOKENS")
        if fitness_weight_tokens_env:
            try:
                self.fitness_weight_tokens = float(fitness_weight_tokens_env)
            except ValueError:
                pass

        fitness_weight_time_env = os.getenv("MEMEVOLVE_FITNESS_WEIGHT_TIME")
        if fitness_weight_time_env:
            try:
                self.fitness_weight_time = float(fitness_weight_time_env)
            except ValueError:
                pass

        fitness_weight_retrieval_env = os.getenv("MEMEVOLVE_FITNESS_WEIGHT_RETRIEVAL")
        if fitness_weight_retrieval_env:
            try:
                self.fitness_weight_retrieval = float(fitness_weight_retrieval_env)
            except ValueError:
                pass

        # Validate configuration
        self._validate_evolution_config()

    def _validate_evolution_config(self):
        """Validate evolution configuration parameters."""
        errors = []

        # Validate population_size
        if not isinstance(self.population_size, int) or self.population_size < 3:
            errors.append(
                f"population_size must be an integer >= 3, got {self.population_size}"
            )
        elif self.population_size > 1000:
            errors.append(
                f"population_size should not exceed 1000 for performance, got {
                    self.population_size}")

        # Validate generations
        if not isinstance(self.generations, int) or self.generations < 1:
            errors.append(
                f"generations must be an integer >= 1, got {self.generations}"
            )
        elif self.generations > 1000:
            errors.append(
                f"generations should not exceed 1000 to prevent excessive runtime, got {
                    self.generations}")

        # Validate mutation_rate
        if not isinstance(
                self.mutation_rate, (int, float)) or not (
                0.0 <= self.mutation_rate <= 1.0):
            errors.append(
                f"mutation_rate must be a float between 0.0 and 1.0, got {self.mutation_rate}"
            )

        # Validate crossover_rate
        if not isinstance(
                self.crossover_rate, (int, float)) or not (
                0.0 <= self.crossover_rate <= 1.0):
            errors.append(
                f"crossover_rate must be a float between 0.0 and 1.0, got {self.crossover_rate}"
            )

        # Validate selection_method
        valid_selection_methods = ["pareto", "tournament", "roulette", "rank"]
        if self.selection_method not in valid_selection_methods:
            errors.append(
                f"selection_method must be one of {valid_selection_methods}, got '{
                    self.selection_method}'")

        # Validate tournament_size (only relevant for tournament selection)
        if not isinstance(self.tournament_size, int) or self.tournament_size < 2:
            errors.append(
                f"tournament_size must be an integer >= 2, got {self.tournament_size}"
            )
        elif self.tournament_size > self.population_size:
            errors.append(
                f"tournament_size ({
                    self.tournament_size}) cannot exceed population_size ({
                    self.population_size})")

        # Check for reasonable combinations
        if self.population_size < self.tournament_size:
            errors.append(
                f"population_size ({
                    self.population_size}) must be >= tournament_size ({
                    self.tournament_size})")

        # Warn about potentially problematic combinations
        warnings = []
        if self.generations > 100 and self.population_size > 50:
            warnings.append(
                "High generations + large population may cause long evolution times"
            )

        if self.mutation_rate > 0.5:
            warnings.append(
                "Very high mutation_rate (>0.5) may prevent convergence"
            )

        if self.crossover_rate < 0.1:
            warnings.append(
                "Very low crossover_rate (<0.1) may reduce diversity"
            )

        # Raise errors if any validation failed
        if errors:
            error_msg = "Evolution configuration validation failed:\n" + \
                "\n".join(f"  - {err}" for err in errors)
            raise ValueError(error_msg)

        # Log warnings
        if warnings:
            # Use standard logging to avoid circular dependency
            import logging as std_logging
            logger = std_logging.getLogger(__name__)
            for warning in warnings:
                logger.warning(f"Evolution config warning: {warning}")


@dataclass
class APIConfig:
    """API server configuration - all values loaded from .env."""
    enable: bool = False
    host: str = ""
    port: int = 0
    memory_integration: bool = False
    memory_retrieval_limit: int = 0

    def __post_init__(self):
        """Load from environment variables."""
        enable_env = os.getenv("MEMEVOLVE_API_ENABLE")
        if enable_env is not None:
            self.enable = enable_env.lower() in ("true", "1", "yes", "on")

        host_env = os.getenv("MEMEVOLVE_API_HOST")
        if host_env is not None:
            self.host = host_env

        port_env = os.getenv("MEMEVOLVE_API_PORT")
        if port_env:
            try:
                self.port = int(port_env)
            except ValueError:
                pass

        mem_int_env = os.getenv("MEMEVOLVE_API_MEMORY_INTEGRATION")
        if mem_int_env is not None:
            self.memory_integration = mem_int_env.lower() in ("true", "1", "yes", "on")


@dataclass
class Neo4jConfig:
    """Neo4j graph database configuration."""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    timeout: int = 30
    max_retries: int = 3
    disabled: bool = False  # Set via MEMEVOLVE_GRAPH_DISABLE_NEO4J to disable Neo4j

    def __post_init__(self):
        """Load from environment variables."""
        uri_env = os.getenv("MEMEVOLVE_NEO4J_URI")
        if uri_env is not None:
            self.uri = uri_env

        user_env = os.getenv("MEMEVOLVE_NEO4J_USER")
        if user_env is not None:
            self.user = user_env

        password_env = os.getenv("MEMEVOLVE_NEO4J_PASSWORD")
        if password_env is not None:
            self.password = password_env

        timeout_env = os.getenv("MEMEVOLVE_NEO4J_TIMEOUT", "30")
        try:
            self.timeout = int(timeout_env)
        except ValueError:
            pass

        retries_env = os.getenv("MEMEVOLVE_NEO4J_MAX_RETRIES", "3")
        try:
            self.max_retries = int(retries_env)
        except ValueError:
            pass

        # Check if Neo4j should be disabled
        disabled_env = os.getenv("MEMEVOLVE_GRAPH_DISABLE_NEO4J", "").lower()
        if disabled_env in ("true", "1", "yes"):
            self.disabled = True


@dataclass
class CycleEvolutionConfig:
    """Evolution cycle trigger configuration - all values loaded from .env."""
    # Legacy enabled field for backward compatibility
    @property
    def enabled(self) -> bool:
        """Legacy alias - evolution enabled status comes from MEMEVOLVE_ENABLE_EVOLUTION."""
        return os.getenv(
            "MEMEVOLVE_ENABLE_EVOLUTION",
            "false").lower() in (
            "true",
            "1",
            "yes",
            "on")

    requests: int = 0
    degradation: float = 0.0
    plateau: int = 0
    hours: int = 0
    cycle_seconds: int = 0

    def __post_init__(self):
        """Load from environment variables."""
        # enabled field removed - redundant with MEMEVOLVE_ENABLE_EVOLUTION
        requests_env = os.getenv("MEMEVOLVE_EVOLUTION_MIN_REQUESTS_PER_CYCLE")
        if requests_env:
            try:
                self.requests = int(requests_env)
            except ValueError:
                pass

        degradation_env = os.getenv("MEMEVOLVE_EVOLUTION_DEGRADATION")
        if degradation_env:
            try:
                self.degradation = float(degradation_env)
            except ValueError:
                pass

        plateau_env = os.getenv("MEMEVOLVE_EVOLUTION_PLATEAU")
        if plateau_env:
            try:
                self.plateau = int(plateau_env)
            except ValueError:
                pass

        hours_env = os.getenv("MEMEVOLVE_EVOLUTION_HOURS")
        if hours_env:
            try:
                self.hours = int(hours_env)
            except ValueError:
                pass

        cycle_seconds_env = os.getenv("MEMEVOLVE_EVOLUTION_CYCLE_SECONDS")
        if cycle_seconds_env:
            try:
                self.cycle_seconds = int(cycle_seconds_env)
            except ValueError:
                pass


# ComponentLoggingConfig removed - logging simplified to global control


@dataclass
class LoggingConfig:
    """Logging configuration - all values loaded from .env."""
    enable: bool = True
    level: str = "INFO"
    log_dir: str = "./logs"
    max_log_size_mb: int = 10

    def __post_init__(self):
        """Load values from environment variables with graceful defaults."""
        # Global logging enable flag
        enable_env = os.getenv("MEMEVOLVE_LOGGING_ENABLE")
        if enable_env is not None:
            self.enable = enable_env.lower() in ("true", "1", "yes", "on")

        # Log level
        level_env = os.getenv("MEMEVOLVE_LOG_LEVEL")
        if level_env:
            self.level = level_env.upper()

        # Log directory
        log_dir_env = os.getenv("MEMEVOLVE_LOGS_DIR")
        if log_dir_env:
            self.log_dir = log_dir_env

        # Max log file size
        size_env = os.getenv("MEMEVOLVE_LOGGING_MAX_LOG_SIZE_MB")
        if size_env:
            try:
                self.max_log_size_mb = int(size_env)
            except ValueError:
                pass


@dataclass
class EncodingPromptConfig:
    """Centralized encoding prompt configuration for eliminating verbosity."""

    # Encoding type descriptions (configurable with environment fallbacks)
    type_descriptions: Dict[str, str] = field(
        default_factory=lambda: {
            "lesson": "generalizable insight",
            "skill": "actionable technique",
            "tool": "reusable function/algorithm",
            "abstraction": "high-level concept"
        }
    )

    # Encoding strategies fallback (hardcoded in config.py as final fallback)
    encoding_strategies_fallback: List[str] = field(
        default_factory=lambda: ["lesson", "skill", "tool", "abstraction"]
    )

    # Chunk processing prompts (replaces encoder.py lines 269-281)
    chunk_processing_instruction: str = "Extract key insight from this experience chunk as JSON."
    chunk_content_instruction: str = "Focus on the specific action, insight, or learning from this chunk."
    chunk_structure_example: str = '{"type": "lesson|skill|tool|abstraction", "content": "Specific insight", "metadata": {"chunk_index": 0}, "tags": ["relevant"]}'

    # Main encoding prompts (replaces encoder.py lines 515-531)
    encoding_instruction: str = "Extract the most important insight from this experience as JSON."
    content_instruction: str = "Return the core action, decision, or learning in 1-2 sentences."
    structure_example: str = '{"type": "lesson|skill|tool|abstraction", "content": "Specific action learned", "metadata": {}, "tags": ["relevant"]}'

    def __post_init__(self):
        """Load from environment variables with config.py fallbacks."""
        # Type descriptions environment mappings
        type_descriptions_env = os.getenv("MEMEVOLVE_TYPE_DESCRIPTIONS")
        if type_descriptions_env:
            # Parse environment variable as JSON or comma-separated pairs
            try:
                import json as json_module
                self.type_descriptions.update(json_module.loads(type_descriptions_env))
            except (json.JSONDecodeError, ValueError):
                # Fallback to comma-separated format: lesson:description1,skill:description2
                pairs = [pair.strip() for pair in type_descriptions_env.split(',')]
                for pair in pairs:
                    if ':' in pair:
                        type_name, description = pair.split(':', 1)
                        self.type_descriptions[type_name.strip()] = description.strip()

        # Note: encoding_strategies_fallback is hardcoded in config.py as final fallback
        # No environment mapping needed - follows architecture guidelines

        # Chunk processing environment mappings
        self.chunk_processing_instruction = os.getenv(
            "MEMEVOLVE_CHUNK_PROCESSING_INSTRUCTION",
            self.chunk_processing_instruction
        )
        self.chunk_content_instruction = os.getenv(
            "MEMEVOLVE_CHUNK_CONTENT_INSTRUCTION",
            self.chunk_content_instruction
        )
        self.chunk_structure_example = os.getenv(
            "MEMEVOLVE_CHUNK_STRUCTURE_EXAMPLE",
            self.chunk_structure_example
        )

        # Main encoding environment mappings
        self.encoding_instruction = os.getenv(
            "MEMEVOLVE_ENCODING_INSTRUCTION",
            self.encoding_instruction
        )
        self.content_instruction = os.getenv(
            "MEMEVOLVE_CONTENT_INSTRUCTION",
            self.content_instruction
        )
        self.structure_example = os.getenv(
            "MEMEVOLVE_STRUCTURE_EXAMPLE",
            self.structure_example
        )


@dataclass
class MemEvolveConfig:
    """Main MemEvolve configuration."""
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    management: ManagementConfig = field(default_factory=ManagementConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    evolution_boundaries: EvolutionBoundaryConfig = field(default_factory=EvolutionBoundaryConfig)
    cycle_evolution: CycleEvolutionConfig = field(default_factory=CycleEvolutionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    # component_logging removed - logging simplified to global control
    api: APIConfig = field(default_factory=APIConfig)
    upstream: UpstreamConfig = field(default_factory=UpstreamConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    encoding_prompts: EncodingPromptConfig = field(default_factory=EncodingPromptConfig)

    # Business value calculation weights
    business_value_token_efficiency_weight: float = 0.7
    business_value_response_quality_weight: float = 0.3

    # Global settings
    api_max_retries: int = field(
        default=3, metadata={
            "help": "Global max retries for all API calls"})
    default_top_k: int = field(
        default=5, metadata={
            "help": "Global default number of memories to retrieve"})

    project_name: str = "memevolve"
    project_root: str = "."
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    logs_dir: str = "./logs"

    def __post_init__(self):
        """Load from environment variables and apply intelligent defaults."""
        # First, ensure all individual configs load their environment variables
        # This ensures upstream is loaded before we apply fallback logic
        for field_name in [
            'memory', 'storage', 'retrieval', 'management', 'encoder',
            'embedding', 'evolution', 'cycle_evolution', 'logging',
            'api', 'upstream'
        ]:
            config_obj = getattr(self, field_name)
            if hasattr(config_obj, '__post_init__'):
                # Call with the appropriate parameters that individual configs expect
                if field_name == 'memory' and hasattr(config_obj, '__post_init__'):
                    config_obj.__post_init__(self)
                else:
                    config_obj.__post_init__()

        project_name_env = os.getenv("MEMEVOLVE_PROJECT_NAME")
        if project_name_env is not None:
            self.project_name = project_name_env

        project_root_env = os.getenv("MEMEVOLVE_PROJECT_ROOT")
        if project_root_env is not None:
            self.project_root = project_root_env

        data_dir_env = os.getenv("MEMEVOLVE_DATA_DIR")
        if data_dir_env is not None:
            self.data_dir = data_dir_env

        cache_dir_env = os.getenv("MEMEVOLVE_CACHE_DIR")
        if cache_dir_env is not None:
            self.cache_dir = cache_dir_env

        logs_dir_env = os.getenv("MEMEVOLVE_LOGS_DIR")
        if logs_dir_env is not None:
            self.logs_dir = logs_dir_env

        # api_max_retries and default_top_k are now handled by env_mappings in ConfigManager

        # Apply endpoint fallback logic according to specification:
        # Priority order for each endpoint:
        # 1. Explicit environment variable for that endpoint
        # 2. Explicit setting in config file
        # 3. Fallback to upstream (for memory/embedding only)
        # 4. Sensible default

        # Ensure upstream is loaded first (it's the fallback target)
        upstream_base_url = self.upstream.base_url
        upstream_api_key = self.upstream.api_key
        upstream_model = self.upstream.model
        upstream_timeout = self.upstream.timeout
        upstream_auto_resolve = self.upstream.auto_resolve_models

        # Apply embedding fallback hierarchy
        # base_url: env → config → upstream → default
        if not self.embedding.base_url and upstream_base_url:
            self.embedding.base_url = upstream_base_url
        # api_key: env → config → upstream → default
        if not self.embedding.api_key and upstream_api_key:
            self.embedding.api_key = upstream_api_key
        # model: env → config → upstream → default
        if not self.embedding.model and upstream_model:
            self.embedding.model = upstream_model
        # timeout: env → config → upstream → default
        if self.embedding.timeout == 60 and upstream_timeout != 600:  # 60 is embedding default
            self.embedding.timeout = upstream_timeout
        # auto_resolve_models: env → config → upstream → default
        if self.embedding.auto_resolve_models and upstream_auto_resolve:
            self.embedding.auto_resolve_models = upstream_auto_resolve

        # Apply memory fallback hierarchy
        # base_url: env → config → upstream → default
        if not self.memory.base_url and upstream_base_url:
            self.memory.base_url = upstream_base_url
        # api_key: env → config → upstream → default
        if not self.memory.api_key and upstream_api_key:
            self.memory.api_key = upstream_api_key
        # model: env → config → upstream → default
        if not self.memory.model and upstream_model:
            self.memory.model = upstream_model
        # timeout: env → config → upstream → default
        if self.memory.timeout == 600 and upstream_timeout != 600:  # 600 is memory default
            self.memory.timeout = upstream_timeout
        # auto_resolve_models: env → config → upstream → default
        if self.memory.auto_resolve_models and upstream_auto_resolve:
            self.memory.auto_resolve_models = upstream_auto_resolve

        # Apply API fallback hierarchy for server settings
        # For API server, we have different defaults but similar fallback logic
        # Note: API server doesn't typically fallback to upstream for host/port,
        # but we apply other settings like timeout and retries
        if self.api.memory_retrieval_limit == 5:  # Only if not explicitly set
            self.api.memory_retrieval_limit = self.default_top_k

        # Load business value weights from environment
        token_eff_weight_env = os.getenv("MEMEVOLVE_BUSINESS_VALUE_EFFICIENCY_WEIGHT")
        if token_eff_weight_env:
            try:
                self.business_value_token_efficiency_weight = float(token_eff_weight_env)
            except ValueError:
                pass

        response_quality_weight_env = os.getenv("MEMEVOLVE_BUSINESS_VALUE_QUALITY_WEIGHT")
        if response_quality_weight_env:
            try:
                self.business_value_response_quality_weight = float(response_quality_weight_env)
            except ValueError:
                pass

        # Validate business value weights sum to approximately 1.0
        total_weight = self.business_value_token_efficiency_weight + self.business_value_response_quality_weight
        if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
            import logging
            logging.warning(
                f"Business value weights sum to {total_weight:.3f}, not 1.0. "
                "This may affect business value calculation accuracy."
            )

        # Propagate global settings to individual configs
        self._propagate_global_settings()

    def _propagate_global_settings(self):
        """Propagate global settings to individual config objects."""
        # Set max_retries for all API configs
        self.memory.max_retries = self.api_max_retries
        self.upstream.max_retries = self.api_max_retries
        self.embedding.max_retries = self.api_max_retries

        # Set default_top_k for all retrieval configs (evolution can override later)
        if not hasattr(self, '_evolution_values_applied'):
            self.retrieval.default_top_k = self.default_top_k
            self.api.memory_retrieval_limit = self.default_top_k

        # Also set it for the memory system config (will be created later)
        # This ensures consistency when the memory system config is created

        # For API wrapper mode, provide smart defaults
        upstream_url = os.getenv("MEMEVOLVE_UPSTREAM_BASE_URL")
        upstream_key = os.getenv("MEMEVOLVE_UPSTREAM_API_KEY")

        if upstream_url:
            # If upstream URL is set but memory LLM URL is not explicitly set, use upstream
            if not os.getenv("MEMEVOLVE_MEMORY_BASE_URL"):
                self.memory.base_url = upstream_url
            if upstream_key and not os.getenv("MEMEVOLVE_MEMORY_API_KEY"):
                self.memory.api_key = upstream_key

            # For embeddings, default to same as upstream unless explicitly set
            if not os.getenv("MEMEVOLVE_EMBEDDING_BASE_URL"):
                self.embedding.base_url = upstream_url
            if upstream_key and not os.getenv("MEMEVOLVE_EMBEDDING_API_KEY"):
                self.embedding.api_key = upstream_key


class ConfigManager:
    """Configuration manager for MemEvolve."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config manager.

        Args:
            config_path: Path to config file (YAML or JSON)
        """
        self.config_path = config_path
        self.config: MemEvolveConfig = MemEvolveConfig()

        # Load in AGENTS.md priority order: 1. file, 2. evolution, 3. env
        if self.config_path and Path(self.config_path).exists():
            self._load_from_file()

        # Load evolution state BEFORE __post_init__ methods override values
        self._load_evolution_state_priority()

        # Load environment variables LAST (lowest priority)
        self._load_from_env()

        # Unified auto-resolution handled by server.py - disabled individual auto-detection
        # self._resolve_all_auto_configs()

    def get_effective_max_tokens(self, service_type: str) -> Optional[int]:
        """
        Get effective max_tokens with proper priority resolution.
        
        Priority order:
        1. Manual .env value (highest priority)
        2. Auto-resolved value from model capabilities
        3. Unlimited/None (fallback, lowest priority)
        
        If both manual and auto-resolved exist: use LOWER of two values
        This prevents exceeding actual model capabilities.
        
        Args:
            service_type: 'upstream', 'memory', or 'embedding'
            
        Returns:
            Effective max_tokens limit or None for unlimited
        """
        import requests
        
        # Map service type to appropriate config
        config_map = {
            'upstream': self.config.upstream,
            'memory': self.config.memory,
            'embedding': self.config.embedding,
            'encoder': self.config.encoder
        }
        
        config = config_map.get(service_type)
        if not config:
            raise ValueError(f"Unknown service type: {service_type}")
        
        # Get manual override from .env
        manual_limit = getattr(config, 'max_tokens', None)
        
        # Get auto-resolved value if enabled
        auto_limit = None
        if config.auto_resolve_models:
            try:
                auto_limit = self._auto_resolve_max_tokens(service_type)
            except Exception as e:
                logger.debug(f"Failed to auto-resolve max_tokens for {service_type}: {e}")
        
        # Apply lower-value rule if both limits exist
        if manual_limit is not None and auto_limit is not None:
            effective_limit = min(manual_limit, auto_limit)
            logger.info(f"Using lower of manual ({manual_limit}) and auto-resolved ({auto_limit}) for {service_type}: {effective_limit}")
            return effective_limit
        elif manual_limit is not None:
            logger.info(f"Using manual max_tokens ({manual_limit}) for {service_type}")
            return manual_limit
        elif auto_limit is not None:
            logger.info(f"Using auto-resolved max_tokens ({auto_limit}) for {service_type}")
            return auto_limit
        else:
            logger.info(f"No max_tokens limit for {service_type} (unlimited)")
            return None
    
    @staticmethod
    def validate_reasonable_limit(limit: Optional[int], service_type: str) -> tuple[bool, str]:
        """Validate auto-resolved limit is reasonable for the service type."""
        if limit is None:
            return True, "No limit to validate"
            
        if service_type == 'embedding':
            # Embedding models typically have strict limits
            if limit > 8192:
                return False, f"Embedding limit {limit} unusually high (>8192)"
            elif limit < 128:
                return False, f"Embedding limit {limit} unusually low (<128)"
            else:
                return True, f"Embedding limit {limit} reasonable"
                
        elif service_type in ['upstream', 'memory']:
            # LLM models typically have higher limits
            if limit > 131072:  # 128k tokens
                return False, f"LLM limit {limit} unusually high (>131k)"
            elif limit < 1024:  # 1k tokens minimum
                return False, f"LLM limit {limit} unusually low (<1k)"
            else:
                return True, f"LLM limit {limit} reasonable"
        else:
            return True, f"Unknown service {service_type} - assuming reasonable"
    
    def resolve_all_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Centralized endpoint resolution following AGENTS.md configuration policy.

        Flow:
        1. Check if upstream URL is configured in .env, if not fail and shutdown server
        2. Check if memory URL is configured, if false assign upstream URL, if true use memory URL
        3. Check if embedding URL is configured, if false assign upstream URL, if true use embedding URL
        4. Make exactly 1 call to each endpoint /models to get model info
        5. Extract relevant parameters from response JSON for each endpoint type
        """
        print("🔍 Centralized endpoint resolution starting...")

        def _ensure_v1_suffix(url: str) -> str:
            """Ensure URL has /v1 suffix for OpenAI-compatible requests."""
            if not url:
                return url

            # Remove trailing slash if present
            url = url.rstrip('/')

            # Add /v1 if not present
            if not url.endswith('/v1'):
                url = f"{url}/v1"

            return url

        if not self.config.upstream.base_url:
            raise RuntimeError("❌ UPSTREAM_BASE_URL not configured in .env - server cannot start")

        # Step 1: Ensure /v1 suffix for all base URLs
        if self.config.upstream.base_url:
            self.config.upstream.base_url = _ensure_v1_suffix(self.config.upstream.base_url)
        if self.config.memory.base_url:
            self.config.memory.base_url = _ensure_v1_suffix(self.config.memory.base_url)
        if self.config.embedding.base_url:
            self.config.embedding.base_url = _ensure_v1_suffix(self.config.embedding.base_url)

        # Step 2: Determine final URLs with fallback logic
        memory_url = self.config.memory.base_url or self.config.upstream.base_url
        embedding_url = self.config.embedding.base_url or self.config.upstream.base_url

        # Update config with resolved URLs
        if memory_url:
            self.config.memory.base_url = memory_url
        if embedding_url:
            self.config.embedding.base_url = embedding_url

        print("Endpoint URLs resolved:")
        print(f"   Upstream: {self.config.upstream.base_url}")
        print(
            f"   Memory: {memory_url} ({
                'configured' if self.config.memory.base_url else 'fallback to upstream'})")
        print(
            f"   Embedding: {embedding_url} ({
                'configured' if self.config.embedding.base_url else 'fallback to upstream'})")

        results = {}

        # Step 2: Make exactly 1 call per endpoint
        endpoints_to_resolve = [
            ("upstream", self.config.upstream.base_url, self.config.upstream.api_key),
            ("memory", memory_url, self.config.memory.api_key),
            ("embedding", embedding_url, self.config.embedding.api_key)
        ]

        for endpoint_type, base_url, api_key in endpoints_to_resolve:
            try:
                print(f"📡 Resolving {endpoint_type} endpoint: {base_url}")
                model_info = self._call_endpoint_once(base_url, api_key)

                if model_info:
                    results[endpoint_type] = model_info

                    # Extract and set model name
                    model_name = model_info.get("id", "unknown")
                    if endpoint_type == "upstream":
                        self.config.upstream.model = model_name
                    elif endpoint_type == "memory":
                        self.config.memory.model = model_name
                    elif endpoint_type == "embedding":
                        self.config.embedding.model = model_name

                    print(f"   ✅ {endpoint_type.title()} resolved: {model_name}")

                    # Extract endpoint-specific parameters from response JSON
                    meta = model_info.get('meta', {})

                    if endpoint_type == "upstream":
                        # Upstream config doesn't have max_tokens/dimension fields
                        # These are only relevant for embedding config
                        pass

                    elif endpoint_type == "memory":
                        # Memory config doesn't have max_tokens/dimension fields
                        # These are only relevant for embedding config
                        pass

                    elif endpoint_type == "embedding":
                        # Embedding needs dimension info and other embedding-specific params
                        if 'n_embd' in meta:
                            self.config.embedding.dimension = meta['n_embd']
                            print(f"   📊 Embedding dimension: {meta['n_embd']}")
                        # Some embedding models might store dimension in different fields
                        elif 'embedding_size' in meta:
                            self.config.embedding.dimension = meta['embedding_size']
                        elif 'dim' in meta:
                            self.config.embedding.dimension = meta['dim']

                else:
                    print(f"   ⚠️ {endpoint_type.title()} resolution failed")
                    results[endpoint_type] = {}

            except Exception as e:
                print(f"   ❌ {endpoint_type.title()} resolution error: {e}")
                results[endpoint_type] = {}

        print(
            f"🎯 Endpoint resolution complete: {sum(1 for r in results.values() if r)}/{len(endpoints_to_resolve)} endpoints resolved")
        return results

    def _call_endpoint_once(self, base_url: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Make exactly one call to /models endpoint and return model info."""
        try:
            # Strip /v1 suffix for /models endpoint call (models endpoint is at base URL)
            models_url = base_url
            if base_url.endswith('/v1'):
                models_url = base_url[:-3]  # Remove /v1

            headers = {}
            if api_key and api_key != "dummy-key":
                headers["Authorization"] = f"Bearer {api_key}"

            response = requests.get(
                f"{models_url}/models",
                headers=headers,
                timeout=10.0
            )
            response.raise_for_status()

            data = response.json()

            if "data" in data and len(data["data"]) > 0:
                return data["data"][0]
            else:
                logging.warning(f"No model data found in response from {models_url}")
                return {}

        except Exception as e:
            logging.warning(f"Failed to call {base_url}: {e}")
            return {}

    def _auto_resolve_max_tokens(self, service_type: str) -> Optional[int]:
        """Auto-resolve max_tokens by querying the service's /models endpoint."""
        return self._resolve_max_tokens_from_endpoint(service_type)
    
    @classmethod
    def _get_service_base_urls(cls, service_type: str) -> List[str]:
        """Get potential base URLs for a service type."""
        base_urls = []
        
        if service_type == "upstream":
            upstream_url = os.getenv("MEMEVOLVE_UPSTREAM_BASE_URL")
            if upstream_url:
                base_urls.append(upstream_url)
        elif service_type == "memory":
            memory_url = os.getenv("MEMEVOLVE_MEMORY_BASE_URL")
            if memory_url:
                base_urls.append(memory_url)
            # Fallback to upstream
            upstream_url = os.getenv("MEMEVOLVE_UPSTREAM_BASE_URL")
            if upstream_url:
                base_urls.append(upstream_url)
        elif service_type == "embedding":
            embedding_url = os.getenv("MEMEVOLVE_EMBEDDING_BASE_URL")
            if embedding_url:
                base_urls.append(embedding_url)
            # Fallback to upstream
            upstream_url = os.getenv("MEMEVOLVE_UPSTREAM_BASE_URL")
            if upstream_url:
                base_urls.append(upstream_url)
        
        return base_urls
    
    @classmethod
    def _is_embedding_using_upstream_fallback(cls) -> bool:
        """Check if embedding service is configured to use upstream fallback."""
        embedding_url = os.getenv("MEMEVOLVE_EMBEDDING_BASE_URL")
        upstream_url = os.getenv("MEMEVOLVE_UPSTREAM_BASE_URL")
        
        # If no explicit embedding URL, it falls back to upstream
        if not embedding_url and upstream_url:
            return True
        
        # If embedding URL is same as upstream URL, it's effectively using upstream
        if embedding_url and upstream_url and embedding_url.rstrip('/') == upstream_url.rstrip('/'):
            return True
        
        return False
    
    @classmethod
    def _get_model_url(cls, service_type: str, model_id: str) -> Optional[str]:
        """Get models endpoint URL for a service."""
        base_urls = cls._get_service_base_urls(service_type)
        
        for base_url in base_urls:
            if not base_url:
                continue
            # Ensure URL has /v1 suffix for OpenAI compatibility
            if not base_url.endswith('/v1'):
                base_url = base_url.rstrip('/') + '/v1'
            # Return /models endpoint (base_url already has /v1)
            return f"{base_url}/models"
        return None
    
    @classmethod
    def _test_model_endpoint(cls, url: str) -> Optional[Dict[str, Any]]:
        """Test a model endpoint and return model information."""
        try:
            import httpx
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url, headers={"Accept": "application/json"})
                if response.status_code == 200:
                    data = response.json()
                    if "data" in data and isinstance(data["data"], list):
                        logger.info(f"Successfully retrieved models from {url}")
                        return data
        except Exception as e:
            logger.debug(f"Failed to connect to {url}: {e}")
        return None
    
    @classmethod
    def _extract_max_tokens_from_models_data(cls, models_data: Dict[str, Any], service_type: str) -> Optional[int]:
        """Extract max_tokens from models API response based on service type."""
        try:
            if not models_data or "data" not in models_data:
                return None
            
            models = models_data["data"]
            if not isinstance(models, list) or not models:
                return None
            
            # Extract token limits from first available model with proper metadata
            for model in models[:3]:  # Check first 3 models
                if not isinstance(model, dict):
                    continue
                
                model_id = model.get("id", "unknown")
                
                # For upstream and memory services, look for context window in various fields
                if service_type in ["upstream", "memory"]:
                    # First check meta field (llama.cpp format)
                    meta = model.get("meta", {})
                    if "n_ctx_train" in meta and isinstance(meta["n_ctx_train"], (int, float)):
                        max_tokens = int(meta["n_ctx_train"])
                        if max_tokens > 1000:  # Reasonable sanity check
                            logger.info(f"Found max_tokens={max_tokens} for {model_id} from meta.n_ctx_train")
                            return max_tokens
                    
                    # Try different field names for context size
                    for field in ["max_context_tokens", "context_length", "max_tokens"]:
                        if field in model and isinstance(model[field], (int, float)):
                            max_tokens = int(model[field])
                            if max_tokens > 1000:  # Reasonable sanity check
                                logger.info(f"Found max_tokens={max_tokens} for {model_id} from {field}")
                                return max_tokens
                    
                    # Try model name parsing (e.g., "claude-3-5-sonnet-20241022" -> 200k)
                    model_name = str(model.get("id", "")).lower()
                    if "claude-3-5-sonnet" in model_name or "claude-3-5-haiku" in model_name:
                        logger.info(f"Detected Claude 3.5 model {model_id}, using 200k tokens")
                        return 200_000
                    elif "claude-3-opus" in model_name or "claude-3-sonnet" in model_name or "claude-3-haiku" in model_name:
                        logger.info(f"Detected Claude 3 model {model_id}, using 200k tokens")
                        return 200_000
                    elif "gpt-4-turbo" in model_name or "gpt-4o" in model_name:
                        logger.info(f"Detected GPT-4 Turbo/GPT-4o model {model_id}, using 128k tokens")
                        return 128_000
                    elif "gpt-4" in model_name:
                        logger.info(f"Detected GPT-4 model {model_id}, using 8192 tokens")
                        return 8192
                    elif "gpt-3.5" in model_name:
                        logger.info(f"Detected GPT-3.5 model {model_id}, using 4096 tokens")
                        return 4096
                
                # For embedding services, look for embedding-specific metadata
                elif service_type == "embedding":
                    # SPECIAL CASE: Check if this is an LLM model being used for embeddings
                    model_id_lower = str(model.get("id", "")).lower()
                    embedding_models = ["embed", "nomic", "text-embedding", "all-minilm", "e5"]
                    is_llm_model = not any(emb in model_id_lower for emb in embedding_models)
                    
                    # Get meta data for analysis
                    meta = model.get("meta", {})
                    
                    # If this looks like an LLM model, prefer n_embd over n_ctx_train
                    if is_llm_model and "n_embd" in meta:
                        n_embd = int(meta["n_embd"])
                        if n_embd > 0:
                            logger.info(f"Embedding service using LLM model {model_id}, using n_embd={n_embd} as max_tokens")
                            return n_embd
                    
                    # Regular embedding model - check n_ctx_train first
                    if "n_ctx_train" in meta and isinstance(meta["n_ctx_train"], (int, float)):
                        max_tokens = int(meta["n_ctx_train"])
                        if max_tokens > 0:
                            logger.info(f"Found embedding max_tokens={max_tokens} for {model_id} from meta.n_ctx_train")
                            return max_tokens
                    
                    # Try embedding-specific context length fields
                    for field in ["max_context_tokens", "context_length", "max_tokens", "n_embed"]:
                        if field in model and isinstance(model[field], (int, float)):
                            max_tokens = int(model[field])
                            if max_tokens > 0:
                                logger.info(f"Found embedding max_tokens={max_tokens} for {model_id} from {field}")
                                return max_tokens
                    
                    # Check model name for common embedding models
                    model_name = str(model.get("id", "")).lower()
                    if "text-embedding-3" in model_name:
                        if "large" in model_name:
                            logger.info(f"Detected text-embedding-3-large model {model_id}, using 8192 tokens")
                            return 8192
                        else:
                            logger.info(f"Detected text-embedding-3-small model {model_id}, using 8192 tokens")
                            return 8192
                    elif "text-embedding-ada" in model_name:
                        logger.info(f"Detected text-embedding-ada model {model_id}, using 8192 tokens")
                        return 8192
                    elif "all-minilm" in model_name:
                        logger.info(f"Detected MiniLM model {model_id}, using 512 tokens")
                        return 512
            
            logger.warning(f"No suitable token limits found in {len(models)} models from {service_type} service")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract max_tokens from models data: {e}")
            return None
    
    @classmethod
    def _extract_embedding_dimensions(cls, models_data: Dict[str, Any]) -> Optional[int]:
        """Extract embedding dimensions from models API response."""
        try:
            if not models_data or "data" not in models_data:
                return None
            
            models = models_data["data"]
            if not isinstance(models, list) or not models:
                return None
            
            # Extract embedding dimensions from first available model
            for model in models[:3]:  # Check first 3 models
                if not isinstance(model, dict):
                    continue
                
                model_id = model.get("id", "unknown")
                
                # First check meta field (llama.cpp format)
                meta = model.get("meta", {})
                if "n_embd" in meta and isinstance(meta["n_embd"], (int, float)):
                    dim = int(meta["n_embd"])
                    if dim > 0:
                        logger.info(f"Found embedding dimensions={dim} for {model_id} from meta.n_embd")
                        return dim
                
                # Try different field names for embedding dimensions
                for field in ["embedding_dim", "dimensions", "dim", "size"]:
                    if field in model and isinstance(model[field], (int, float)):
                        dim = int(model[field])
                        if dim > 0:
                            logger.info(f"Found embedding dimensions={dim} for {model_id} from {field}")
                            return dim
                
                # Try model name parsing for common embedding models
                model_name = str(model.get("id", "")).lower()
                if "text-embedding-3-large" in model_name:
                    logger.info(f"Detected text-embedding-3-large model {model_id}, using 3072 dimensions")
                    return 3072
                elif "text-embedding-3-small" in model_name:
                    logger.info(f"Detected text-embedding-3-small model {model_id}, using 1536 dimensions")
                    return 1536
                elif "text-embedding-ada" in model_name:
                    logger.info(f"Detected text-embedding-ada model {model_id}, using 1536 dimensions")
                    return 1536
                elif "all-minilm-l6-v2" in model_name:
                    logger.info(f"Detected all-MiniLM-L6-v2 model {model_id}, using 384 dimensions")
                    return 384
                elif "nomic-embed" in model_name and "text" in model_name:
                    logger.info(f"Detected Nomic embedding model {model_id}, using 768 dimensions")
                    return 768
            
            logger.warning(f"No suitable embedding dimensions found in {len(models)} models")
            return None
            
        except Exception as e:
            logger.error(f"Failed to extract embedding dimensions from models data: {e}")
            return None
    
    @classmethod
    def _resolve_max_tokens_from_endpoint(cls, service_type: str) -> Optional[int]:
        """Resolve max_tokens by querying the service's /models endpoint."""
        # Get potential model endpoint URLs
        model_url = cls._get_model_url(service_type, "")
        if not model_url:
            logger.debug(f"No model endpoint URL found for {service_type} service")
            return None
        
        # Test the endpoint
        models_data = cls._test_model_endpoint(model_url)
        if not models_data:
            logger.warning(f"Failed to get models data from {model_url}")
            return None
        
        # Extract max_tokens based on service type
        max_tokens = cls._extract_max_tokens_from_models_data(models_data, service_type)
        if max_tokens:
            return max_tokens
        
        # SPECIAL CASE: If embedding service failed to get limits but uses upstream fallback,
        # try to get n_embd from upstream as fallback
        if service_type == "embedding" and cls._is_embedding_using_upstream_fallback():
            logger.info("Embedding service using upstream fallback, trying to get n_embd from upstream")
            upstream_url = cls._get_model_url("upstream", "")
            if upstream_url:
                upstream_models = cls._test_model_endpoint(upstream_url)
                if upstream_models:
                    # Extract n_embd from upstream model for embedding max_tokens
                    if "data" in upstream_models:
                        for model in upstream_models["data"][:1]:  # Check first model
                            meta = model.get("meta", {})
                            if "n_embd" in meta:
                                n_embd = int(meta["n_embd"])
                                logger.info(f"Using upstream n_embd={n_embd} as embedding max_tokens limit")
                                return n_embd
        
        # Final fallback - safe defaults
        if service_type == "embedding":
            logger.warning("All embedding max_tokens resolution methods failed, using safe default: 512")
            return 512  # Safe default: compatible with most embedding models
        else:
            logger.warning(f"All {service_type} max_tokens resolution methods failed, no safe default available")
            return None
    
    @classmethod
    def resolve_embedding_dimensions(cls) -> Optional[int]:
        """Resolve embedding dimensions by querying the embedding service's /models endpoint."""
        # Get potential model endpoint URL
        model_url = cls._get_model_url("embedding", "")
        if not model_url:
            logger.debug("No model endpoint URL found for embedding service")
            return None
        
        # Test the endpoint
        models_data = cls._test_model_endpoint(model_url)
        if not models_data:
            logger.warning("Failed to get models data from embedding service")
            return None
        
        # Extract embedding dimensions
        embedding_dims = cls._extract_embedding_dimensions(models_data)
        if embedding_dims:
            return embedding_dims
        
        # If embedding resolution failed and service is using upstream fallback,
        # try to get n_embd from upstream as fallback
        if cls._is_embedding_using_upstream_fallback():
            logger.info("Embedding resolution failed, trying upstream n_embd as fallback")
            upstream_url = cls._get_model_url("upstream", "")
            if upstream_url:
                upstream_models = cls._test_model_endpoint(upstream_url)
                if upstream_models and "data" in upstream_models:
                    for model in upstream_models["data"][:1]:  # Check first model
                        meta = model.get("meta", {})
                        if "n_embd" in meta:
                            upstream_n_embd = int(meta["n_embd"])
                            logger.info(f"Using upstream n_embd={upstream_n_embd} as embedding dimensions fallback")
                            return upstream_n_embd
        
        # Final fallback - safe default for common embedding models
        logger.warning("All embedding resolution methods failed, using safe default: 768")
        return 768  # Safe default: works with most embedding models (MiniLM, sentence-transformers)
    
    @classmethod
    def _test_service_connectivity(cls, url: str) -> bool:
        """Test if we can connect to a service URL."""
        try:
            import httpx
            with httpx.Client(timeout=3.0) as client:
                # Ensure URL has /v1 suffix for models endpoint
                if not url.endswith('/v1'):
                    test_url = url.rstrip('/') + '/v1/models'
                else:
                    test_url = url.rstrip('/') + '/models'
                    
                response = client.get(test_url, timeout=2.0)
                if response.status_code == 200:
                    logger.info(f"Service connectivity confirmed: {url}")
                    return True
        except Exception:
            pass
        return False

    def _resolve_all_auto_configs(self):
        """Unified auto-resolution to prevent redundant API calls."""
        # Temporarily disabled for testing - server auto-resolution handles this
        pass

    def _load_config(self):
        """Load configuration following AGENTS.md priority hierarchy."""
        # This method is deprecated - loading moved to __init__ for proper order
        pass

    def _load_from_file(self):
        """Load configuration from file."""
        if self.config_path is None:
            return

        path = Path(self.config_path)

        if not path.exists():
            return

        try:
            if path.suffix in [".yaml", ".yml"]:
                with open(path, 'r') as f:
                    data = yaml.safe_load(f)
            elif path.suffix == ".json":
                with open(path, 'r') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")

            self._apply_config_dict(data)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load config from {self.config_path}: {e}")

    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            # Upstream
            "MEMEVOLVE_UPSTREAM_BASE_URL": (("upstream", "base_url"), None),
            "MEMEVOLVE_UPSTREAM_API_KEY": (("upstream", "api_key"), None),
            "MEMEVOLVE_UPSTREAM_MODEL": (("upstream", "model"), None),
            "MEMEVOLVE_UPSTREAM_AUTO_RESOLVE_MODELS": (
                ("upstream", "auto_resolve_models"),
                lambda x: x.lower() in ("true", "1", "yes", "on")
            ),
            "MEMEVOLVE_UPSTREAM_TIMEOUT": (("upstream", "timeout"), int),
            "MEMEVOLVE_UPSTREAM_MAX_TOKENS": (("upstream", "max_tokens"), int),
            # Storage
            "MEMEVOLVE_STORAGE_BACKEND_TYPE": (("storage", "backend_type"), None),
            "MEMEVOLVE_STORAGE_DATA_DIR": (("storage", "path"), None),
            "MEMEVOLVE_STORAGE_INDEX_TYPE": (("storage", "index_type"), None),
            # Retrieval
            "MEMEVOLVE_RETRIEVAL_STRATEGY_TYPE": (("retrieval", "strategy_type"), None),


            "MEMEVOLVE_RETRIEVAL_ENABLE_CACHING": (("retrieval", "enable_caching"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_RETRIEVAL_CACHE_SIZE": (("retrieval", "cache_size"), int),
            "MEMEVOLVE_MEMORY_RELEVANCE_THRESHOLD": (("retrieval", "relevance_threshold"), float),
            "MEMEVOLVE_RETRIEVAL_DEFAULT_TOP_K": (("retrieval", "default_top_k"), int),
            "MEMEVOLVE_RETRIEVAL_SEMANTIC_CACHE_ENABLED": (("retrieval", "semantic_cache_enabled"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_RETRIEVAL_KEYWORD_CASE_SENSITIVE": (("retrieval", "keyword_case_sensitive"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_RETRIEVAL_SEMANTIC_EMBEDDING_MODEL": (("retrieval", "semantic_embedding_model"), None),
            "MEMEVOLVE_RETRIEVAL_HYBRID_SEMANTIC_WEIGHT": (("retrieval", "hybrid_semantic_weight"), float),
            "MEMEVOLVE_RETRIEVAL_HYBRID_KEYWORD_WEIGHT": (("retrieval", "hybrid_keyword_weight"), float),
            # Management
            "MEMEVOLVE_MANAGEMENT_ENABLE_AUTO_MANAGEMENT": (("management", "enable_auto_management"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD": (("management", "auto_prune_threshold"), int),
            "MEMEVOLVE_MANAGEMENT_AUTO_CONSOLIDATE_INTERVAL": (("management", "auto_consolidate_interval"), int),
            "MEMEVOLVE_MANAGEMENT_DEDUPLICATE_THRESHOLD": (("management", "deduplicate_threshold"), float),
            "MEMEVOLVE_MANAGEMENT_FORGETTING_STRATEGY": (("management", "forgetting_strategy"), None),
            "MEMEVOLVE_MANAGEMENT_MAX_MEMORY_AGE_DAYS": (("management", "max_memory_age_days"), int),
            "MEMEVOLVE_MANAGEMENT_STRATEGY_TYPE": (("management", "strategy_type"), None),
            "MEMEVOLVE_MANAGEMENT_PRUNE_MAX_COUNT": (("management", "prune_max_count"), int),
            "MEMEVOLVE_MANAGEMENT_PRUNE_BY_TYPE": (("management", "prune_by_type"), None),
            "MEMEVOLVE_MANAGEMENT_CONSOLIDATE_ENABLED": (("management", "consolidate_enabled"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_MANAGEMENT_CONSOLIDATE_MIN_UNITS": (("management", "consolidate_min_units"), int),
            "MEMEVOLVE_MANAGEMENT_DEDUPLICATE_ENABLED": (("management", "deduplicate_enabled"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_MANAGEMENT_FORGETTING_PERCENTAGE": (("management", "forgetting_percentage"), float),
            # Encoder
            "MEMEVOLVE_ENCODER_ENCODING_STRATEGIES": (("encoder", "encoding_strategies"), lambda x: [s.strip() for s in x.split(",") if s.strip()]),
            "MEMEVOLVE_ENCODER_ENABLE_ABSTRACTION": (("encoder", "enable_abstraction"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_ENCODER_ABSTRACTION_THRESHOLD": (("encoder", "abstraction_threshold"), int),
            "MEMEVOLVE_ENCODER_ENABLE_TOOL_EXTRACTION": (("encoder", "enable_tool_extraction"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_ENCODER_MAX_TOKENS": (("encoder", "max_tokens"), int),
            "MEMEVOLVE_ENCODER_BATCH_SIZE": (("encoder", "batch_size"), int),
            "MEMEVOLVE_ENCODER_TEMPERATURE": (("encoder", "temperature"), float),
            "MEMEVOLVE_ENCODER_LLM_MODEL": (("encoder", "llm_model"), None),
            "MEMEVOLVE_ENCODER_ENABLE_ABSTRACTIONS": (("encoder", "enable_abstractions"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_ENCODER_MIN_ABSTRACTION_UNITS": (("encoder", "min_abstraction_units"), int),
            # Evolution
            "MEMEVOLVE_EVOLUTION_POPULATION_SIZE": (("evolution", "population_size"), int),
            "MEMEVOLVE_EVOLUTION_GENERATIONS": (("evolution", "generations"), int),
            "MEMEVOLVE_EVOLUTION_MUTATION_RATE": (("evolution", "mutation_rate"), float),
            "MEMEVOLVE_EVOLUTION_CROSSOVER_RATE": (("evolution", "crossover_rate"), float),
            "MEMEVOLVE_EVOLUTION_SELECTION_METHOD": (("evolution", "selection_method"), None),
            "MEMEVOLVE_EVOLUTION_TOURNAMENT_SIZE": (("evolution", "tournament_size"), int),
            "MEMEVOLVE_FITNESS_WEIGHT_SUCCESS": (("evolution", "fitness_weight_success"), float),
            "MEMEVOLVE_FITNESS_WEIGHT_TOKENS": (("evolution", "fitness_weight_tokens"), float),
            "MEMEVOLVE_FITNESS_WEIGHT_TIME": (("evolution", "fitness_weight_time"), float),
            "MEMEVOLVE_FITNESS_WEIGHT_RETRIEVAL": (("evolution", "fitness_weight_retrieval"), float),
            "MEMEVOLVE_ENABLE_EVOLUTION": (("evolution", "enable"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            # Auto-Evolution

            "MEMEVOLVE_CYCLE_EVOLUTION_MIN_REQUESTS_PER_CYCLE": (("cycle_evolution", "requests"), int),
            "MEMEVOLVE_CYCLE_EVOLUTION_DEGRADATION": (("cycle_evolution", "degradation"), float),
            "MEMEVOLVE_CYCLE_EVOLUTION_PLATEAU": (("cycle_evolution", "plateau"), int),
            "MEMEVOLVE_CYCLE_EVOLUTION_HOURS": (("cycle_evolution", "hours"), int),
            "MEMEVOLVE_CYCLE_EVOLUTION_CYCLE_SECONDS": (("cycle_evolution", "cycle_seconds"), int),
            # Logging
            "MEMEVOLVE_LOG_LEVEL": (("logging", "level"), None),
            "MEMEVOLVE_LOGGING_FORMAT": (("logging", "format"), None),
            "MEMEVOLVE_LOGGING_MAX_LOG_SIZE_MB": (("logging", "max_log_size_mb"), int),
            # Component Logging
            "MEMEVOLVE_LOG_API_SERVER_ENABLE": (("component_logging", "api_server_enable"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_LOG_MIDDLEWARE_ENABLE": (("component_logging", "middleware_enable"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_LOG_MEMORY_ENABLE": (("component_logging", "memory_enable"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_LOG_EVOLUTION_ENABLE": (("component_logging", "evolution_enable"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_LOG_MEMEVOLVE_ENABLE": (("component_logging", "memevolve_enable"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_LOG_OPERATION_ENABLE": (("component_logging", "operation_log_enable"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            # API
            "MEMEVOLVE_API_ENABLE": (("api", "enable"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_API_HOST": (("api", "host"), None),
            "MEMEVOLVE_API_PORT": (("api", "port"), int),
            "MEMEVOLVE_API_MEMORY_INTEGRATION": (("api", "memory_integration"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_API_MEMORY_RETRIEVAL_LIMIT": (("api", "memory_retrieval_limit"), int),
            # Neo4j
            "MEMEVOLVE_NEO4J_URI": (("neo4j", "uri"), None),
            "MEMEVOLVE_NEO4J_USER": (("neo4j", "user"), None),
            "MEMEVOLVE_NEO4J_PASSWORD": (("neo4j", "password"), None),
            "MEMEVOLVE_NEO4J_TIMEOUT": (("neo4j", "timeout"), int),
            "MEMEVOLVE_NEO4J_MAX_RETRIES": (("neo4j", "max_retries"), int),
            # Evolution Boundary mappings - all fallbacks in config.py
            "MEMEVOLVE_EVOLUTION_MAX_TOKENS_MIN": (("evolution_boundaries", "max_tokens_min"), int),
            "MEMEVOLVE_EVOLUTION_MAX_TOKENS_MAX": (("evolution_boundaries", "max_tokens_max"), int),
            "MEMEVOLVE_EVOLUTION_TOP_K_MIN": (("evolution_boundaries", "top_k_min"), int),
            "MEMEVOLVE_EVOLUTION_TOP_K_MAX": (("evolution_boundaries", "top_k_max"), int),
            "MEMEVOLVE_EVOLUTION_RELEVANCE_THRESHOLD_MIN": (("evolution_boundaries", "relevance_threshold_min"), float),
            "MEMEVOLVE_EVOLUTION_RELEVANCE_THRESHOLD_MAX": (("evolution_boundaries", "relevance_threshold_max"), float),

            "MEMEVOLVE_EVOLUTION_TEMPERATURE_MIN": (("evolution_boundaries", "temperature_min"), float),
            "MEMEVOLVE_EVOLUTION_TEMPERATURE_MAX": (("evolution_boundaries", "temperature_max"), float),
            "MEMEVOLVE_EVOLUTION_BOUNDARY_MIN_REQUESTS_PER_CYCLE": (("evolution_boundaries", "min_requests_per_cycle"), int),
            "MEMEVOLVE_FITNESS_HISTORY_SIZE": (("evolution_boundaries", "fitness_history_size"), int),

            # Encoding prompt mappings - all fallbacks in config.py
            "MEMEVOLVE_CHUNK_PROCESSING_INSTRUCTION": (("encoding_prompts", "chunk_processing_instruction"), str),
            "MEMEVOLVE_CHUNK_CONTENT_INSTRUCTION": (("encoding_prompts", "chunk_content_instruction"), str),
            "MEMEVOLVE_CHUNK_STRUCTURE_EXAMPLE": (("encoding_prompts", "chunk_structure_example"), str),
            "MEMEVOLVE_ENCODING_INSTRUCTION": (("encoding_prompts", "encoding_instruction"), str),
            "MEMEVOLVE_CONTENT_INSTRUCTION": (("encoding_prompts", "content_instruction"), str),
            "MEMEVOLVE_STRUCTURE_EXAMPLE": (("encoding_prompts", "structure_example"), str),
            # Note: MEMEVOLVE_TYPE_DESCRIPTIONS is handled in EncodingPromptConfig.__post_init__() to parse comma-separated format
            # Do not map here to avoid overwriting the correctly parsed dict with the raw string

            # Global Settings
            "MEMEVOLVE_API_MAX_RETRIES": (("api_max_retries",), int),
            "MEMEVOLVE_DEFAULT_TOP_K": (("default_top_k",), int),
            # Business Value Weights
            "MEMEVOLVE_BUSINESS_VALUE_EFFICIENCY_WEIGHT": (("business_value_token_efficiency_weight",), float),
            "MEMEVOLVE_BUSINESS_VALUE_QUALITY_WEIGHT": (("business_value_response_quality_weight",), float),
            # Project
            "MEMEVOLVE_PROJECT_NAME": (("project_name",), None),
            "MEMEVOLVE_PROJECT_ROOT": (("project_root",), None),
            "MEMEVOLVE_GLOBAL_DATA_DIR": (("data_dir",), None),
            "MEMEVOLVE_CACHE_DIR": (("cache_dir",), None),
            "MEMEVOLVE_LOGS_DIR": (("logs_dir",), None),
        }

        # Skip evolution-overridden environment variables per AGENTS.md policy
        evolution_protected_vars = set()
        if hasattr(self, '_evolution_values_applied'):
            evolution_protected_vars = {
                "MEMEVOLVE_RETRIEVAL_STRATEGY_TYPE",
                "MEMEVOLVE_RETRIEVAL_HYBRID_SEMANTIC_WEIGHT",
                "MEMEVOLVE_RETRIEVAL_HYBRID_KEYWORD_WEIGHT"
            }

        for env_var, (path_parts, converter) in env_mappings.items():
            if env_var in evolution_protected_vars:
                # Skip evolution-overridden values per AGENTS.md policy
                continue
            value = os.getenv(env_var)
            if value is None:
                continue

            # Special handling for embedding auto-detection variables
            # Empty string means "use auto-detection"
            if env_var in ["MEMEVOLVE_EMBEDDING_MAX_TOKENS", "MEMEVOLVE_EMBEDDING_DIMENSION"]:
                if value == "":
                    # Skip setting to allow auto-detection in __post_init__
                    continue
            elif value == "":
                # For other variables, empty means skip
                continue

            obj = self.config
            for part in path_parts[:-1]:
                obj = getattr(obj, part)

            # Apply converter if provided
            if converter is not None:
                value = converter(value)

            setattr(obj, path_parts[-1], value)

    def _apply_config_dict(self, data: Dict[str, Any]):
        """Apply configuration dictionary to config object."""
        for section, section_config in data.items():
            if hasattr(self.config, section):
                section_obj = getattr(self.config, section)
                if isinstance(section_obj,
                              (MemoryConfig, StorageConfig, RetrievalConfig,
                               ManagementConfig, EncoderConfig, EmbeddingConfig,
                               EvolutionConfig, CycleEvolutionConfig, LoggingConfig,
                               APIConfig, UpstreamConfig)):
                    for key, value in section_config.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
                else:
                    setattr(self.config, section, section_config)

    def _load_evolution_state_priority(self):
        """Load evolution state with highest priority per AGENTS.md policy.

        Priority: evolution_state.json > .env > config.py defaults
        Only applies when MEMEVOLVE_ENABLE_EVOLUTION=true and file exists.
        """
        # Check evolution enabled status
        evolution_enabled = os.getenv(
            "MEMEVOLVE_ENABLE_EVOLUTION", "false").lower() in (
            "true", "1", "yes", "on")

        if not evolution_enabled:
            return

        # Check evolution state file exists
        evolution_state_path = os.path.join(
            os.getenv("MEMEVOLVE_DATA_DIR", "./data"),
            "evolution",
            "evolution_state.json"
        )

        if not os.path.exists(evolution_state_path):
            return

        try:
            with open(evolution_state_path, 'r') as f:
                state = json.load(f)

            # Apply genotype overrides using dot notation per AGENTS.md sync rules
            if 'best_genotype' in state:
                self._apply_genotype_overrides(state['best_genotype'])

        except Exception as e:
            # Log error but continue with other config sources
            print(f"Warning: Failed to load evolution state: {e}")

    def _apply_genotype_overrides(self, genotype):
        """Apply genotype configuration overrides following AGENTS.md sync rules.

        Evolution updates ConfigManager using dot notation: config_manager.update(retrieval.default_top_k=7)
        """
        # Apply encode configuration
        if 'encode' in genotype:
            encode = genotype['encode']
            self.config.encoder.batch_size = encode.get(
                'batch_size', self.config.encoder.batch_size)
            self.config.encoder.max_tokens = encode.get(
                'max_tokens', self.config.encoder.max_tokens)
            self.config.encoder.temperature = encode.get(
                'temperature', self.config.encoder.temperature)
            self.config.encoder.enable_abstractions = encode.get(
                'enable_abstractions', self.config.encoder.enable_abstractions)
            self.config.encoder.min_abstraction_units = encode.get(
                'min_abstraction_units', self.config.encoder.min_abstraction_units)
            if encode.get('encoding_strategies'):
                self.config.encoder.encoding_strategies = encode['encoding_strategies']

        # Apply retrieve configuration
        if 'retrieve' in genotype:
            retrieve = genotype['retrieve']
            self.config.retrieval.strategy_type = retrieve.get(
                'strategy_type', self.config.retrieval.strategy_type)
            self.config.retrieval.default_top_k = retrieve.get(
                'default_top_k', self.config.retrieval.default_top_k)
            # similarity_threshold: REMOVED - use relevance_threshold instead
            # enable_filters: REMOVED - retrieval filtering eliminated
            self.config.retrieval.enable_filters = False
            self.config.retrieval.semantic_cache_enabled = retrieve.get(
                'semantic_cache_enabled', self.config.retrieval.semantic_cache_enabled)
            self.config.retrieval.keyword_case_sensitive = retrieve.get(
                'keyword_case_sensitive', self.config.retrieval.keyword_case_sensitive)
            self.config.retrieval.hybrid_semantic_weight = retrieve.get(
                'hybrid_semantic_weight', self.config.retrieval.hybrid_semantic_weight)
            self.config.retrieval.hybrid_keyword_weight = retrieve.get(
                'hybrid_keyword_weight', self.config.retrieval.hybrid_keyword_weight)

            # Mark evolution values as applied
            self._evolution_values_applied = True

        # Apply manage configuration
        if 'manage' in genotype:
            manage = genotype['manage']
            self.config.management.strategy_type = manage.get(
                'strategy_type', self.config.management.strategy_type)
            self.config.management.enable_auto_management = manage.get(
                'enable_auto_management', self.config.management.enable_auto_management)
            # prune_max_age_days removed - redundant with max_memory_age_days
            self.config.management.prune_max_count = manage.get(
                'prune_max_count', self.config.management.prune_max_count)
            self.config.management.prune_by_type = manage.get(
                'prune_by_type', self.config.management.prune_by_type)
            self.config.management.consolidate_enabled = manage.get(
                'consolidate_enabled', self.config.management.consolidate_enabled)
            self.config.management.consolidate_min_units = manage.get(
                'consolidate_min_units', self.config.management.consolidate_min_units)
            self.config.management.deduplicate_enabled = manage.get(
                'deduplicate_enabled', self.config.management.deduplicate_enabled)
            self.config.management.deduplicate_threshold = manage.get(
                'deduplicate_similarity_threshold', self.config.management.deduplicate_threshold)
            self.config.management.forgetting_strategy = manage.get(
                'forgetting_strategy', self.config.management.forgetting_strategy)
            self.config.management.forgetting_percentage = manage.get(
                'forgetting_percentage', self.config.management.forgetting_percentage)

    def save_to_file(self, output_path: str):
        """Save current configuration to file.

        Args:
            output_path: Path to save config file
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        try:
            if output.suffix in [".yaml", ".yml"]:
                with open(output, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            elif output.suffix == ".json":
                with open(output, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {output.suffix}")
        except Exception as e:
            raise RuntimeError(f"Failed to save config to {output_path}: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self.config)

    def update(self, **kwargs):
        """Update configuration values.

        Args:
            **kwargs: Configuration updates in dot notation (e.g., memory.base_url="...")
        """
        for key_path, value in kwargs.items():
            keys = key_path.split('.')
            obj = self.config

            for key in keys[:-1]:
                if hasattr(obj, key):
                    obj = getattr(obj, key)
                else:
                    raise AttributeError(f"Invalid config path: {key_path}")

            if hasattr(obj, keys[-1]):
                setattr(obj, keys[-1], value)
            else:
                raise AttributeError(f"Invalid config path: {key_path}")

    def get(self, key_path: str, default=None):
        """Get configuration value by dot notation.

        Args:
            key_path: Configuration path (e.g., "memory.base_url")
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        obj = self.config

        try:
            for key in keys:
                obj = getattr(obj, key)
            return obj
        except AttributeError:
            return default

    def validate(self) -> bool:
        """Validate configuration.

        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate memory base_url
            assert isinstance(self.config.memory.base_url, str)
            # Allow empty base_url for local LLMs

            # Validate retrieval config
            assert isinstance(self.config.retrieval.strategy_type, str)
            assert self.config.retrieval.strategy_type in [
                "keyword", "semantic", "hybrid", "llm_guided"]
            assert isinstance(self.config.retrieval.default_top_k, int)
            assert self.config.retrieval.default_top_k > 0

            # Validate hybrid weights if using hybrid strategy
            if self.config.retrieval.strategy_type == "hybrid":
                assert isinstance(self.config.retrieval.hybrid_semantic_weight, (int, float))
                assert isinstance(self.config.retrieval.hybrid_keyword_weight, (int, float))
                # Weights should sum to approximately 1.0 (allowing small rounding errors)
                total_weight = self.config.retrieval.hybrid_semantic_weight + \
                    self.config.retrieval.hybrid_keyword_weight
                assert abs(total_weight -
                           1.0) < 0.01, f"Hybrid weights must sum to 1.0, got {total_weight}"

            # Validate storage config
            assert self.config.storage.backend_type in ["json", "vector", "graph"]

            # Validate management config
            assert isinstance(self.config.management.enable_auto_management, bool)
            if self.config.management.enable_auto_management:
                assert isinstance(self.config.management.auto_prune_threshold, int)
                assert self.config.management.auto_prune_threshold > 0

            # Validate encoder config
            assert isinstance(self.config.encoder.encoding_strategies, list)
            assert len(self.config.encoder.encoding_strategies) > 0
            for strategy in self.config.encoder.encoding_strategies:
                assert strategy in ["lesson", "skill", "tool", "abstraction"]

            # Validate evolution config
            if self.config.evolution.enable:
                assert isinstance(self.config.evolution.population_size, int)
                assert self.config.evolution.population_size >= 3
                assert isinstance(self.config.evolution.generations, int)
                assert self.config.evolution.generations >= 1
                assert isinstance(self.config.evolution.mutation_rate, (int, float))
                assert 0.0 <= self.config.evolution.mutation_rate <= 1.0
                assert isinstance(self.config.evolution.crossover_rate, (int, float))
                assert 0.0 <= self.config.evolution.crossover_rate <= 1.0

            return True
        except (AssertionError, TypeError, AttributeError) as e:
            import logging
            logging.error(f"Configuration validation failed: {e}")
            return False

    @staticmethod
    def get_default_config() -> MemEvolveConfig:
        """Get default configuration.

        Returns:
            Default MemEvolveConfig instance
        """
        return MemEvolveConfig()

    @staticmethod
    def get_architecture_config(architecture: str) -> Dict[str, Any]:
        """Get preset configuration for specific architecture.

        Args:
            architecture: Architecture name (agentkb, lightweight, riva, cerebra)

        Returns:
            Configuration dictionary for architecture
        """
        configs = {
            "agentkb": {
                "storage": {"backend_type": "json", "path": "./data/agentkb"},
                "retrieval": {"strategy_type": "keyword", "default_top_k": 3},
                "management": {"enable_auto_management": False},
                "encoder": {"encoding_strategies": ["lesson"]}
            },
            "lightweight": {
                "storage": {"backend_type": "json", "path": "./data/lightweight"},
                "retrieval": {"strategy_type": "hybrid", "default_top_k": 5},
                "management": {
                    "enable_auto_management": True,
                    "auto_prune_threshold": 500
                },
                "encoder": {"encoding_strategies": ["lesson", "skill"]}
            },
            "riva": {
                "storage": {"backend_type": "vector", "path": "./data/riva"},
                "retrieval": {
                    "strategy_type": "hybrid",
                    "default_top_k": 7,
                    "hybrid_semantic_weight": 0.8,
                    "hybrid_keyword_weight": 0.2
                },
                "management": {
                    "enable_auto_management": True,
                    "auto_prune_threshold": 1000
                },
                "encoder": {
                    "encoding_strategies": ["lesson", "skill", "tool"],
                    "enable_tool_extraction": True
                }
            },
            "cerebra": {
                "storage": {"backend_type": "vector", "path": "./data/cerebra"},
                "retrieval": {
                    "strategy_type": "hybrid",
                    "default_top_k": 10,
                    "hybrid_semantic_weight": 0.9,
                    "hybrid_keyword_weight": 0.1,
                    "enable_caching": True
                },
                "management": {
                    "enable_auto_management": True,
                    "auto_prune_threshold": 2000,
                    "auto_consolidate_interval": 50
                },
                "encoder": {
                    "encoding_strategies": ["lesson", "skill", "tool", "abstraction"],
                    "enable_abstraction": True,
                    "abstraction_threshold": 5,
                    "enable_tool_extraction": True
                }
            }
        }

        return configs.get(architecture, {})


def load_config(config_path: Optional[str] = None) -> MemEvolveConfig:
    """Load configuration with proper environment variable override and validation.

    Args:
        config_path: Path to config file (YAML or JSON)

    Returns:
        MemEvolveConfig instance
    """
    manager = ConfigManager(config_path)
    if not manager.validate():
        raise ValueError("Invalid configuration")
    return manager.config


def save_config(config: MemEvolveConfig, output_path: str):
    """Save configuration to file.

    Args:
        config: MemEvolveConfig instance
        output_path: Path to save config file
    """
    manager = ConfigManager()
    manager.config = config
    manager.save_to_file(output_path)
