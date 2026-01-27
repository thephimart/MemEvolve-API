"""Configuration management system for MemEvolve."""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from dotenv import load_dotenv
import requests
import logging

# Load environment variables from .env file
load_dotenv()


@dataclass
class MemoryConfig:
    """LLM configuration for memory management operations."""
    base_url: str = ""
    api_key: str = ""
    model: Optional[str] = None
    auto_resolve_models: bool = True
    timeout: int = 600
    max_retries: int = 3

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
    base_url: str = ""
    api_key: str = ""
    model: Optional[str] = None
    auto_resolve_models: bool = True
    timeout: int = 600
    max_retries: int = 3

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


@dataclass
class StorageConfig:
    """Storage backend configuration."""
    backend_type: str = "json"
    path: str = "./data/memory"
    index_type: str = "flat"

    def __post_init__(self):
        """Load from environment variables."""
        if self.backend_type == "json":
            backend_type_env = os.getenv("MEMEVOLVE_STORAGE_BACKEND_TYPE")
            if backend_type_env is not None:
                self.backend_type = backend_type_env

        if self.path == "./data/memory":
            path_env = os.getenv("MEMEVOLVE_STORAGE_PATH")
            if path_env is not None:
                self.path = path_env

        if self.index_type == "flat":
            index_type_env = os.getenv("MEMEVOLVE_STORAGE_INDEX_TYPE")
            if index_type_env is not None:
                self.index_type = index_type_env


@dataclass
class RetrievalConfig:
    """Retrieval strategy configuration."""
    strategy_type: str = "hybrid"
    default_top_k: int = 3
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    enable_caching: bool = True
    cache_size: int = 1000

    def __post_init__(self):
        """Load from environment variables."""
        if self.strategy_type == "hybrid":
            strategy_type_env = os.getenv("MEMEVOLVE_RETRIEVAL_STRATEGY_TYPE")
            if strategy_type_env is not None:
                self.strategy_type = strategy_type_env

        # default_top_k is now handled by env_mappings in ConfigManager

        if self.semantic_weight == 0.7:
            semantic_weight_env = os.getenv(
                "MEMEVOLVE_RETRIEVAL_SEMANTIC_WEIGHT")
            if semantic_weight_env:
                try:
                    self.semantic_weight = float(semantic_weight_env)
                except ValueError:
                    pass

        if self.keyword_weight == 0.3:
            keyword_weight_env = os.getenv(
                "MEMEVOLVE_RETRIEVAL_KEYWORD_WEIGHT")
            if keyword_weight_env:
                try:
                    self.keyword_weight = float(keyword_weight_env)
                except ValueError:
                    pass

        if self.enable_caching:
            enable_caching_env = os.getenv(
                "MEMEVOLVE_RETRIEVAL_ENABLE_CACHING")
            if enable_caching_env is not None:
                self.enable_caching = enable_caching_env.lower() in ("true", "1", "yes", "on")

        if self.cache_size == 1000:
            cache_size_env = os.getenv("MEMEVOLVE_RETRIEVAL_CACHE_SIZE")
            if cache_size_env:
                try:
                    self.cache_size = int(cache_size_env)
                except ValueError:
                    pass


@dataclass
class ManagementConfig:
    """Memory management configuration."""
    enable_auto_management: bool = True
    auto_prune_threshold: int = 1000
    auto_consolidate_interval: int = 100
    deduplicate_threshold: float = 0.9
    forgetting_strategy: str = "lru"
    max_memory_age_days: int = 365

    def __post_init__(self):
        """Load from environment variables."""
        if self.enable_auto_management:
            enable_auto_env = os.getenv(
                "MEMEVOLVE_MANAGEMENT_ENABLE_AUTO_MANAGEMENT")
            if enable_auto_env is not None:
                self.enable_auto_management = enable_auto_env.lower() in ("true", "1", "yes", "on")

        if self.auto_prune_threshold == 1000:
            prune_threshold_env = os.getenv(
                "MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD")
            if prune_threshold_env:
                try:
                    self.auto_prune_threshold = int(prune_threshold_env)
                except ValueError:
                    pass

        if self.auto_consolidate_interval == 100:
            consolidate_interval_env = os.getenv(
                "MEMEVOLVE_MANAGEMENT_AUTO_CONSOLIDATE_INTERVAL")
            if consolidate_interval_env:
                try:
                    self.auto_consolidate_interval = int(
                        consolidate_interval_env)
                except ValueError:
                    pass

        if self.deduplicate_threshold == 0.9:
            deduplicate_threshold_env = os.getenv(
                "MEMEVOLVE_MANAGEMENT_DEDUPLICATE_THRESHOLD")
            if deduplicate_threshold_env:
                try:
                    self.deduplicate_threshold = float(
                        deduplicate_threshold_env)
                except ValueError:
                    pass

        if self.forgetting_strategy == "lru":
            forgetting_strategy_env = os.getenv(
                "MEMEVOLVE_MANAGEMENT_FORGETTING_STRATEGY")
            if forgetting_strategy_env is not None:
                self.forgetting_strategy = forgetting_strategy_env

        if self.max_memory_age_days == 365:
            max_age_env = os.getenv("MEMEVOLVE_MANAGEMENT_MAX_MEMORY_AGE_DAYS")
            if max_age_env:
                try:
                    self.max_memory_age_days = int(max_age_env)
                except ValueError:
                    pass


@dataclass
class EncoderConfig:
    """Experience encoder configuration."""
    encoding_strategies: list = field(
        default_factory=lambda: ["lesson", "skill"])
    enable_abstraction: bool = True
    abstraction_threshold: int = 10
    enable_tool_extraction: bool = True

    def __post_init__(self):
        """Load from environment variables."""
        if self.encoding_strategies == ["lesson", "skill"]:
            strategies_env = os.getenv("MEMEVOLVE_ENCODER_ENCODING_STRATEGIES")
            if strategies_env:
                self.encoding_strategies = [
                    s.strip() for s in strategies_env.split(",") if s.strip()]

        if self.enable_abstraction:
            enable_abstraction_env = os.getenv(
                "MEMEVOLVE_ENCODER_ENABLE_ABSTRACTION")
            if enable_abstraction_env is not None:
                self.enable_abstraction = enable_abstraction_env.lower() in ("true",
                                                                             "1", "yes", "on")

        if self.abstraction_threshold == 10:
            abstraction_threshold_env = os.getenv(
                "MEMEVOLVE_ENCODER_ABSTRACTION_THRESHOLD")
            if abstraction_threshold_env:
                try:
                    self.abstraction_threshold = int(abstraction_threshold_env)
                except ValueError:
                    pass

        if self.enable_tool_extraction:
            enable_tool_env = os.getenv(
                "MEMEVOLVE_ENCODER_ENABLE_TOOL_EXTRACTION")
            if enable_tool_env is not None:
                self.enable_tool_extraction = enable_tool_env.lower() in ("true", "1", "yes", "on")


@dataclass
class EmbeddingConfig:
    """Embedding API configuration."""
    base_url: str = ""
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

        # Priority 4: Auto-detect from models endpoint if not set
        if self.max_tokens is None or self.dimension is None:
            self._auto_detect_from_models()

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
class EvolutionConfig:
    """Evolution framework configuration."""
    enable: bool = False
    population_size: int = 10
    generations: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.5
    selection_method: str = "pareto"
    tournament_size: int = 3

    # Fitness evaluation weights
    fitness_weight_success: float = 0.4
    fitness_weight_tokens: float = 0.3
    fitness_weight_time: float = 0.2
    fitness_weight_retrieval: float = 0.1

    def __post_init__(self):
        """Load from environment variables."""
        if not self.enable:
            enable_env = os.getenv("MEMEVOLVE_ENABLE_EVOLUTION")
            if enable_env is not None:
                self.enable = enable_env.lower() in ("true", "1", "yes", "on")

        if self.population_size == 10:
            population_size_env = os.getenv(
                "MEMEVOLVE_EVOLUTION_POPULATION_SIZE")
            if population_size_env:
                try:
                    self.population_size = int(population_size_env)
                except ValueError:
                    pass

        if self.generations == 20:
            generations_env = os.getenv("MEMEVOLVE_EVOLUTION_GENERATIONS")
            if generations_env:
                try:
                    self.generations = int(generations_env)
                except ValueError:
                    pass

        if self.mutation_rate == 0.1:
            mutation_rate_env = os.getenv("MEMEVOLVE_EVOLUTION_MUTATION_RATE")
            if mutation_rate_env:
                try:
                    self.mutation_rate = float(mutation_rate_env)
                except ValueError:
                    pass

        if self.crossover_rate == 0.5:
            crossover_rate_env = os.getenv(
                "MEMEVOLVE_EVOLUTION_CROSSOVER_RATE")
            if crossover_rate_env:
                try:
                    self.crossover_rate = float(crossover_rate_env)
                except ValueError:
                    pass

        if self.selection_method == "pareto":
            selection_method_env = os.getenv(
                "MEMEVOLVE_EVOLUTION_SELECTION_METHOD")
            if selection_method_env is not None:
                self.selection_method = selection_method_env

        if self.tournament_size == 3:
            tournament_size_env = os.getenv(
                "MEMEVOLVE_EVOLUTION_TOURNAMENT_SIZE")
            if tournament_size_env:
                try:
                    self.tournament_size = int(tournament_size_env)
                except ValueError:
                    pass

        # Load fitness weights from environment
        if self.fitness_weight_success == 0.4:
            fitness_weight_success_env = os.getenv("MEMEVOLVE_FITNESS_WEIGHT_SUCCESS")
            if fitness_weight_success_env:
                try:
                    self.fitness_weight_success = float(fitness_weight_success_env)
                except ValueError:
                    pass

        if self.fitness_weight_tokens == 0.3:
            fitness_weight_tokens_env = os.getenv("MEMEVOLVE_FITNESS_WEIGHT_TOKENS")
            if fitness_weight_tokens_env:
                try:
                    self.fitness_weight_tokens = float(fitness_weight_tokens_env)
                except ValueError:
                    pass

        if self.fitness_weight_time == 0.2:
            fitness_weight_time_env = os.getenv("MEMEVOLVE_FITNESS_WEIGHT_TIME")
            if fitness_weight_time_env:
                try:
                    self.fitness_weight_time = float(fitness_weight_time_env)
                except ValueError:
                    pass

        if self.fitness_weight_retrieval == 0.1:
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
            import logging
            logger = logging.getLogger(__name__)
            for warning in warnings:
                logger.warning(f"Evolution config warning: {warning}")


@dataclass
class APIConfig:
    """API server configuration."""
    enable: bool = True
    host: str = "127.0.0.1"
    port: int = 11436
    memory_integration: bool = True
    memory_retrieval_limit: int = 5

    def __post_init__(self):
        """Load from environment variables."""
        if self.enable:
            enable_env = os.getenv("MEMEVOLVE_API_ENABLE")
            if enable_env is not None:
                self.enable = enable_env.lower() in ("true", "1", "yes", "on")

        if self.host == "127.0.0.1":
            host_env = os.getenv("MEMEVOLVE_API_HOST")
            if host_env is not None:
                self.host = host_env

        if self.port == 11436:
            port_env = os.getenv("MEMEVOLVE_API_PORT")
            if port_env:
                try:
                    self.port = int(port_env)
                except ValueError:
                    pass

        if self.memory_integration:
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

    def __post_init__(self):
        """Load from environment variables."""
        if self.uri == "bolt://localhost:7687":
            uri_env = os.getenv("MEMEVOLVE_NEO4J_URI")
            if uri_env is not None:
                self.uri = uri_env

        if self.user == "neo4j":
            user_env = os.getenv("MEMEVOLVE_NEO4J_USER")
            if user_env is not None:
                self.user = user_env

        if self.password == "password":
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


@dataclass
class AutoEvolutionConfig:
    """Auto-evolution trigger configuration."""
    enabled: bool = True
    requests: int = 100
    degradation: float = 0.2
    plateau: int = 5
    hours: int = 24
    cycle_seconds: int = 600

    def __post_init__(self):
        """Load from environment variables."""
        if self.enabled:
            enabled_env = os.getenv("MEMEVOLVE_AUTO_EVOLUTION_ENABLED")
            if enabled_env is not None:
                self.enabled = enabled_env.lower() in ("true", "1", "yes", "on")

        if self.requests == 100:
            requests_env = os.getenv("MEMEVOLVE_AUTO_EVOLUTION_REQUESTS")
            if requests_env:
                try:
                    self.requests = int(requests_env)
                except ValueError:
                    pass

        if self.degradation == 0.2:
            degradation_env = os.getenv("MEMEVOLVE_AUTO_EVOLUTION_DEGRADATION")
            if degradation_env:
                try:
                    self.degradation = float(degradation_env)
                except ValueError:
                    pass

        if self.plateau == 5:
            plateau_env = os.getenv("MEMEVOLVE_AUTO_EVOLUTION_PLATEAU")
            if plateau_env:
                try:
                    self.plateau = int(plateau_env)
                except ValueError:
                    pass

        if self.hours == 24:
            hours_env = os.getenv("MEMEVOLVE_AUTO_EVOLUTION_HOURS")
            if hours_env:
                try:
                    self.hours = int(hours_env)
                except ValueError:
                    pass

        if self.cycle_seconds == 600:
            cycle_seconds_env = os.getenv("MEMEVOLVE_AUTO_EVOLUTION_CYCLE_SECONDS")
            if cycle_seconds_env:
                try:
                    self.cycle_seconds = int(cycle_seconds_env)
                except ValueError:
                    pass


@dataclass
class ComponentLoggingConfig:
    """Component-specific logging configuration."""
    api_server_enable: bool = False
    middleware_enable: bool = False
    memory_enable: bool = False
    experiment_enable: bool = False

    def __post_init__(self):
        """Load from environment variables."""
        api_server_env = os.getenv("MEMEVOLVE_LOG_API_SERVER_ENABLE")
        if api_server_env is not None:
            self.api_server_enable = api_server_env.lower() in ("true", "1", "yes", "on")

        middleware_env = os.getenv("MEMEVOLVE_LOG_MIDDLEWARE_ENABLE")
        if middleware_env is not None:
            self.middleware_enable = middleware_env.lower() in ("true", "1", "yes", "on")

        memory_env = os.getenv("MEMEVOLVE_LOG_MEMORY_ENABLE")
        if memory_env is not None:
            self.memory_enable = memory_env.lower() in ("true", "1", "yes", "on")

        experiment_env = os.getenv("MEMEVOLVE_LOG_EXPERIMENT_ENABLE")
        if experiment_env is not None:
            self.experiment_enable = experiment_env.lower() in ("true", "1", "yes", "on")


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = "./logs/memevolve.log"
    enable_operation_log: bool = True
    max_log_size_mb: int = 100

    def __post_init__(self):
        """Load from environment variables."""
        if self.level == "INFO":
            level_env = os.getenv("MEMEVOLVE_LOG_LEVEL")
            if level_env is not None:
                self.level = level_env

        if self.format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s":
            format_env = os.getenv("MEMEVOLVE_LOGGING_FORMAT")
            if format_env is not None:
                self.format = format_env

        if self.log_file is None:
            log_file_env = os.getenv("MEMEVOLVE_LOGGING_LOG_FILE")
            if log_file_env is not None:
                self.log_file = log_file_env

        if self.enable_operation_log:
            enable_op_log_env = os.getenv(
                "MEMEVOLVE_LOGGING_ENABLE_OPERATION_LOG")
            if enable_op_log_env is not None:
                self.enable_operation_log = enable_op_log_env.lower() in ("true", "1", "yes", "on")

        if self.max_log_size_mb == 100:
            max_log_size_env = os.getenv("MEMEVOLVE_LOGGING_MAX_LOG_SIZE_MB")
            if max_log_size_env:
                try:
                    self.max_log_size_mb = int(max_log_size_env)
                except ValueError:
                    pass


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
    auto_evolution: AutoEvolutionConfig = field(default_factory=AutoEvolutionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    component_logging: ComponentLoggingConfig = field(default_factory=ComponentLoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    upstream: UpstreamConfig = field(default_factory=UpstreamConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)

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
            'embedding', 'evolution', 'auto_evolution', 'logging',
            'component_logging', 'api', 'upstream'
        ]:
            config_obj = getattr(self, field_name)
            if hasattr(config_obj, '__post_init__'):
                # Call with the appropriate parameters that individual configs expect
                if field_name == 'memory' and hasattr(config_obj, '__post_init__'):
                    config_obj.__post_init__(self)
                else:
                    config_obj.__post_init__()

        if self.project_name == "memevolve":
            project_name_env = os.getenv("MEMEVOLVE_PROJECT_NAME")
            if project_name_env is not None:
                self.project_name = project_name_env

        if self.project_root == ".":
            project_root_env = os.getenv("MEMEVOLVE_PROJECT_ROOT")
            if project_root_env is not None:
                self.project_root = project_root_env

        if self.data_dir == "./data":
            data_dir_env = os.getenv("MEMEVOLVE_DATA_DIR")
            if data_dir_env is not None:
                self.data_dir = data_dir_env

        if self.cache_dir == "./cache":
            cache_dir_env = os.getenv("MEMEVOLVE_CACHE_DIR")
            if cache_dir_env is not None:
                self.cache_dir = cache_dir_env

        if self.logs_dir == "./logs":
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
        upstream_max_retries = self.upstream.max_retries

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

        # Propagate global settings to individual configs
        self._propagate_global_settings()

    def _propagate_global_settings(self):
        """Propagate global settings to individual config objects."""
        # Set max_retries for all API configs
        self.memory.max_retries = self.api_max_retries
        self.upstream.max_retries = self.api_max_retries
        self.embedding.max_retries = self.api_max_retries

        # Set default_top_k for all retrieval configs
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
        self._load_config()

    def _load_config(self):
        """Load configuration from file or environment."""
        if self.config_path and Path(self.config_path).exists():
            self._load_from_file()
        self._load_from_env()

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
            # Memory LLM
            "MEMEVOLVE_MEMORY_BASE_URL": (("memory", "base_url"), None),
            "MEMEVOLVE_MEMORY_API_KEY": (("memory", "api_key"), None),
            "MEMEVOLVE_MEMORY_MODEL": (("memory", "model"), None),
            "MEMEVOLVE_MEMORY_AUTO_RESOLVE_MODELS": (
                ("memory", "auto_resolve_models"),
                lambda x: x.lower() in ("true", "1", "yes", "on")
            ),
            "MEMEVOLVE_MEMORY_TIMEOUT": (("memory", "timeout"), int),

            # Embedding
            "MEMEVOLVE_EMBEDDING_BASE_URL": (("embedding", "base_url"), None),
            "MEMEVOLVE_EMBEDDING_API_KEY": (("embedding", "api_key"), None),
            "MEMEVOLVE_EMBEDDING_MODEL": (("embedding", "model"), None),
            "MEMEVOLVE_EMBEDDING_AUTO_RESOLVE_MODELS": (
                ("embedding", "auto_resolve_models"),
                lambda x: x.lower() in ("true", "1", "yes", "on")
            ),
            "MEMEVOLVE_EMBEDDING_TIMEOUT": (("embedding", "timeout"), int),
            "MEMEVOLVE_EMBEDDING_MAX_TOKENS": (("embedding", "max_tokens"), int),
            "MEMEVOLVE_EMBEDDING_DIMENSION": (("embedding", "dimension"), int),
            # Upstream
            "MEMEVOLVE_UPSTREAM_BASE_URL": (("upstream", "base_url"), None),
            "MEMEVOLVE_UPSTREAM_API_KEY": (("upstream", "api_key"), None),
            "MEMEVOLVE_UPSTREAM_MODEL": (("upstream", "model"), None),
            "MEMEVOLVE_UPSTREAM_AUTO_RESOLVE_MODELS": (
                ("upstream", "auto_resolve_models"),
                lambda x: x.lower() in ("true", "1", "yes", "on")
            ),
            "MEMEVOLVE_UPSTREAM_TIMEOUT": (("upstream", "timeout"), int),
            # Storage
            "MEMEVOLVE_STORAGE_BACKEND_TYPE": (("storage", "backend_type"), None),
            "MEMEVOLVE_STORAGE_PATH": (("storage", "path"), None),
            "MEMEVOLVE_STORAGE_INDEX_TYPE": (("storage", "index_type"), None),
            # Retrieval
            "MEMEVOLVE_RETRIEVAL_STRATEGY_TYPE": (("retrieval", "strategy_type"), None),
            "MEMEVOLVE_RETRIEVAL_TOP_K": (("retrieval", "default_top_k"), int),
            "MEMEVOLVE_RETRIEVAL_SEMANTIC_WEIGHT": (("retrieval", "semantic_weight"), float),
            "MEMEVOLVE_RETRIEVAL_KEYWORD_WEIGHT": (("retrieval", "keyword_weight"), float),
            "MEMEVOLVE_RETRIEVAL_ENABLE_CACHING": (("retrieval", "enable_caching"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_RETRIEVAL_CACHE_SIZE": (("retrieval", "cache_size"), int),
            # Management
            "MEMEVOLVE_MANAGEMENT_ENABLE_AUTO_MANAGEMENT": (("management", "enable_auto_management"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_MANAGEMENT_AUTO_PRUNE_THRESHOLD": (("management", "auto_prune_threshold"), int),
            "MEMEVOLVE_MANAGEMENT_AUTO_CONSOLIDATE_INTERVAL": (("management", "auto_consolidate_interval"), int),
            "MEMEVOLVE_MANAGEMENT_DEDUPLICATE_THRESHOLD": (("management", "deduplicate_threshold"), float),
            "MEMEVOLVE_MANAGEMENT_FORGETTING_STRATEGY": (("management", "forgetting_strategy"), None),
            "MEMEVOLVE_MANAGEMENT_MAX_MEMORY_AGE_DAYS": (("management", "max_memory_age_days"), int),
            # Encoder
            "MEMEVOLVE_ENCODER_ENCODING_STRATEGIES": (("encoder", "encoding_strategies"), lambda x: [s.strip() for s in x.split(",") if s.strip()]),
            "MEMEVOLVE_ENCODER_ENABLE_ABSTRACTION": (("encoder", "enable_abstraction"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_ENCODER_ABSTRACTION_THRESHOLD": (("encoder", "abstraction_threshold"), int),
            "MEMEVOLVE_ENCODER_ENABLE_TOOL_EXTRACTION": (("encoder", "enable_tool_extraction"), lambda x: x.lower() in ("true", "1", "yes", "on")),
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
            "MEMEVOLVE_AUTO_EVOLUTION_ENABLED": (("auto_evolution", "enabled"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_AUTO_EVOLUTION_REQUESTS": (("auto_evolution", "requests"), int),
            "MEMEVOLVE_AUTO_EVOLUTION_DEGRADATION": (("auto_evolution", "degradation"), float),
            "MEMEVOLVE_AUTO_EVOLUTION_PLATEAU": (("auto_evolution", "plateau"), int),
            "MEMEVOLVE_AUTO_EVOLUTION_HOURS": (("auto_evolution", "hours"), int),
            "MEMEVOLVE_AUTO_EVOLUTION_CYCLE_SECONDS": (("auto_evolution", "cycle_seconds"), int),
            # Logging
            "MEMEVOLVE_LOG_LEVEL": (("logging", "level"), None),
            "MEMEVOLVE_LOGGING_FORMAT": (("logging", "format"), None),
            "MEMEVOLVE_LOGGING_LOG_FILE": (("logging", "log_file"), None),
            "MEMEVOLVE_LOGGING_ENABLE_OPERATION_LOG": (("logging", "enable_operation_log"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_LOGGING_MAX_LOG_SIZE_MB": (("logging", "max_log_size_mb"), int),
            # Component Logging
            "MEMEVOLVE_LOG_API_SERVER_ENABLE": (("component_logging", "api_server_enable"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_LOG_MIDDLEWARE_ENABLE": (("component_logging", "middleware_enable"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_LOG_MEMORY_ENABLE": (("component_logging", "memory_enable"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_LOG_EXPERIMENT_ENABLE": (("component_logging", "experiment_enable"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            # API
            "MEMEVOLVE_API_ENABLE": (("api", "enable"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_API_HOST": (("api", "host"), None),
            "MEMEVOLVE_API_PORT": (("api", "port"), int),
            "MEMEVOLVE_API_MEMORY_INTEGRATION": (("api", "memory_integration"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            # Neo4j
            "MEMEVOLVE_NEO4J_URI": (("neo4j", "uri"), None),
            "MEMEVOLVE_NEO4J_USER": (("neo4j", "user"), None),
            "MEMEVOLVE_NEO4J_PASSWORD": (("neo4j", "password"), None),
            "MEMEVOLVE_NEO4J_TIMEOUT": (("neo4j", "timeout"), int),
            "MEMEVOLVE_NEO4J_MAX_RETRIES": (("neo4j", "max_retries"), int),

            # Global Settings
            "MEMEVOLVE_API_MAX_RETRIES": (("api_max_retries",), int),
            "MEMEVOLVE_DEFAULT_TOP_K": (("default_top_k",), int),
            # Project
            "MEMEVOLVE_PROJECT_NAME": (("project_name",), None),
            "MEMEVOLVE_PROJECT_ROOT": (("project_root",), None),
            "MEMEVOLVE_DATA_DIR": (("data_dir",), None),
            "MEMEVOLVE_CACHE_DIR": (("cache_dir",), None),
            "MEMEVOLVE_LOGS_DIR": (("logs_dir",), None),
        }

        for env_var, (path_parts, converter) in env_mappings.items():
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
                               EvolutionConfig, AutoEvolutionConfig, LoggingConfig,
                               ComponentLoggingConfig, APIConfig, UpstreamConfig)):
                    for key, value in section_config.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
                else:
                    setattr(self.config, section, section_config)

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
            assert isinstance(self.config.memory.base_url, str)
            # Allow empty base_url for local LLMs

            assert isinstance(self.config.retrieval.default_top_k, int)
            assert self.config.retrieval.default_top_k > 0

            assert 0 <= self.config.retrieval.semantic_weight <= 1
            assert 0 <= self.config.retrieval.keyword_weight <= 1

            assert self.config.storage.backend_type in [
                "json", "vector", "graph"]

            return True
        except AssertionError:
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
                    "semantic_weight": 0.8,
                    "keyword_weight": 0.2
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
                    "semantic_weight": 0.9,
                    "keyword_weight": 0.1,
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
    """Load configuration from file or defaults.

    Args:
        config_path: Path to config file

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
