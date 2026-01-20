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
class LLMConfig:
    """LLM configuration."""
    base_url: str = ""
    api_key: str = ""
    model: Optional[str] = None
    auto_resolve_models: bool = True
    timeout: int = 120
    max_retries: int = 3

    def __post_init__(self):
        """Load from environment variables."""
        # Only override with env vars if the value is still the default
        if self.base_url == "":
            base_url_env = os.getenv("MEMEVOLVE_LLM_BASE_URL")
            if base_url_env is not None:
                self.base_url = base_url_env

        if self.api_key == "":
            api_key_env = os.getenv("MEMEVOLVE_LLM_API_KEY")
            if api_key_env is not None:
                self.api_key = api_key_env

        if self.model is None:
            model_env = os.getenv("MEMEVOLVE_LLM_MODEL")
            if model_env is not None:
                self.model = model_env

        if self.auto_resolve_models == True:
            auto_resolve_env = os.getenv("MEMEVOLVE_LLM_AUTO_RESOLVE_MODELS")
            if auto_resolve_env is not None:
                self.auto_resolve_models = auto_resolve_env.lower() in ("true", "1", "yes", "on")

        if self.timeout == 120:
            timeout_env = os.getenv("MEMEVOLVE_LLM_TIMEOUT")
            if timeout_env:
                try:
                    self.timeout = int(timeout_env)
                except ValueError:
                    pass

        if self.max_retries == 3:
            max_retries_env = os.getenv("MEMEVOLVE_LLM_MAX_RETRIES")
            if max_retries_env:
                try:
                    self.max_retries = int(max_retries_env)
                except ValueError:
                    pass


@dataclass
class StorageConfig:
    """Storage backend configuration."""
    backend_type: str = "json"
    path: str = "./data/memory"
    vector_dim: int = 768
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

        if self.vector_dim == 768:
            vector_dim_env = os.getenv("MEMEVOLVE_STORAGE_VECTOR_DIM")
            if vector_dim_env:
                try:
                    self.vector_dim = int(vector_dim_env)
                except ValueError:
                    pass

        if self.index_type == "flat":
            index_type_env = os.getenv("MEMEVOLVE_STORAGE_INDEX_TYPE")
            if index_type_env is not None:
                self.index_type = index_type_env


@dataclass
class RetrievalConfig:
    """Retrieval strategy configuration."""
    strategy_type: str = "hybrid"
    default_top_k: int = 5
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

        if self.default_top_k == 5:
            top_k_env = os.getenv("MEMEVOLVE_RETRIEVAL_TOP_K")
            if top_k_env:
                try:
                    self.default_top_k = int(top_k_env)
                except ValueError:
                    pass

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

        if self.enable_caching == True:
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
        if self.enable_auto_management == True:
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

        if self.enable_abstraction == True:
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

        if self.enable_tool_extraction == True:
            enable_tool_env = os.getenv(
                "MEMEVOLVE_ENCODER_ENABLE_TOOL_EXTRACTION")
            if enable_tool_env is not None:
                self.enable_tool_extraction = enable_tool_env.lower() in ("true", "1", "yes", "on")


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    base_url: str = ""
    api_key: str = ""
    model: Optional[str] = None
    auto_resolve_models: bool = True

    def __post_init__(self):
        """Load from environment variables."""
        if self.base_url == "":
            base_url_env = os.getenv("MEMEVOLVE_EMBEDDING_BASE_URL")
            if base_url_env is not None:
                self.base_url = base_url_env

        if self.api_key == "":
            api_key_env = os.getenv("MEMEVOLVE_EMBEDDING_API_KEY")
            if api_key_env is not None:
                self.api_key = api_key_env

        if self.model is None:
            model_env = os.getenv("MEMEVOLVE_EMBEDDING_MODEL")
            if model_env is not None:
                self.model = model_env

        if self.auto_resolve_models == True:
            auto_resolve_env = os.getenv(
                "MEMEVOLVE_EMBEDDING_AUTO_RESOLVE_MODELS")
            if auto_resolve_env is not None:
                self.auto_resolve_models = auto_resolve_env.lower() in ("true", "1", "yes", "on")

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

    def __post_init__(self):
        """Load from environment variables."""
        if self.enable == False:
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
                f"population_size should not exceed 1000 for performance, got {self.population_size}"
            )

        # Validate generations
        if not isinstance(self.generations, int) or self.generations < 1:
            errors.append(
                f"generations must be an integer >= 1, got {self.generations}"
            )
        elif self.generations > 1000:
            errors.append(
                f"generations should not exceed 1000 to prevent excessive runtime, got {self.generations}"
            )

        # Validate mutation_rate
        if not isinstance(self.mutation_rate, (int, float)) or not (0.0 <= self.mutation_rate <= 1.0):
            errors.append(
                f"mutation_rate must be a float between 0.0 and 1.0, got {self.mutation_rate}"
            )

        # Validate crossover_rate
        if not isinstance(self.crossover_rate, (int, float)) or not (0.0 <= self.crossover_rate <= 1.0):
            errors.append(
                f"crossover_rate must be a float between 0.0 and 1.0, got {self.crossover_rate}"
            )

        # Validate selection_method
        valid_selection_methods = ["pareto", "tournament", "roulette", "rank"]
        if self.selection_method not in valid_selection_methods:
            errors.append(
                f"selection_method must be one of {valid_selection_methods}, got '{self.selection_method}'"
            )

        # Validate tournament_size (only relevant for tournament selection)
        if not isinstance(self.tournament_size, int) or self.tournament_size < 2:
            errors.append(
                f"tournament_size must be an integer >= 2, got {self.tournament_size}"
            )
        elif self.tournament_size > self.population_size:
            errors.append(
                f"tournament_size ({self.tournament_size}) cannot exceed population_size ({self.population_size})"
            )

        # Check for reasonable combinations
        if self.population_size < self.tournament_size:
            errors.append(
                f"population_size ({self.population_size}) must be >= tournament_size ({self.tournament_size})"
            )

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
    port: int = 8001
    upstream_base_url: str = "http://localhost:8000/v1"
    upstream_api_key: Optional[str] = None
    memory_integration: bool = True
    memory_retrieval_limit: int = 5

    def __post_init__(self):
        """Load from environment variables."""
        if self.enable == True:
            enable_env = os.getenv("MEMEVOLVE_API_ENABLE")
            if enable_env is not None:
                self.enable = enable_env.lower() in ("true", "1", "yes", "on")

        if self.host == "127.0.0.1":
            host_env = os.getenv("MEMEVOLVE_API_HOST")
            if host_env is not None:
                self.host = host_env

        if self.port == 8001:
            port_env = os.getenv("MEMEVOLVE_API_PORT")
            if port_env:
                try:
                    self.port = int(port_env)
                except ValueError:
                    pass

        if self.upstream_base_url == "http://localhost:8000/v1":
            upstream_env = os.getenv("MEMEVOLVE_UPSTREAM_BASE_URL")
            if upstream_env is not None:
                self.upstream_base_url = upstream_env

        if self.upstream_api_key is None:
            upstream_key_env = os.getenv("MEMEVOLVE_UPSTREAM_API_KEY")
            if upstream_key_env is not None:
                self.upstream_api_key = upstream_key_env

        if self.memory_integration == True:
            mem_int_env = os.getenv("MEMEVOLVE_API_MEMORY_INTEGRATION")
            if mem_int_env is not None:
                self.memory_integration = mem_int_env.lower() in ("true", "1", "yes", "on")

        if self.memory_retrieval_limit == 5:
            mem_limit_env = os.getenv("MEMEVOLVE_API_MEMORY_RETRIEVAL_LIMIT")
            if mem_limit_env:
                try:
                    self.memory_retrieval_limit = int(mem_limit_env)
                except ValueError:
                    pass


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

        if self.enable_operation_log == True:
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
    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    management: ManagementConfig = field(default_factory=ManagementConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)

    project_name: str = "memevolve"
    project_root: str = "."
    data_dir: str = "./data"
    cache_dir: str = "./cache"

    def __post_init__(self):
        """Load from environment variables and apply intelligent defaults."""
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

        # For API wrapper mode, provide smart defaults
        upstream_url = os.getenv("MEMEVOLVE_UPSTREAM_BASE_URL")
        upstream_key = os.getenv("MEMEVOLVE_UPSTREAM_API_KEY")

        if upstream_url:
            # If upstream URL is set but memory LLM URL is not explicitly set, use upstream
            if not os.getenv("MEMEVOLVE_LLM_BASE_URL"):
                self.llm.base_url = upstream_url
            if upstream_key and not os.getenv("MEMEVOLVE_LLM_API_KEY"):
                self.llm.api_key = upstream_key

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
            # LLM
            "MEMEVOLVE_LLM_BASE_URL": (("llm", "base_url"), None),
            "MEMEVOLVE_LLM_API_KEY": (("llm", "api_key"), None),
            "MEMEVOLVE_LLM_MODEL": (("llm", "model"), None),
            "MEMEVOLVE_LLM_AUTO_RESOLVE_MODELS": (("llm", "auto_resolve_models"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_LLM_TIMEOUT": (("llm", "timeout"), int),
            "MEMEVOLVE_LLM_MAX_RETRIES": (("llm", "max_retries"), int),
            # Embedding
            "MEMEVOLVE_EMBEDDING_BASE_URL": (("embedding", "base_url"), None),
            "MEMEVOLVE_EMBEDDING_API_KEY": (("embedding", "api_key"), None),
            "MEMEVOLVE_EMBEDDING_MODEL": (("embedding", "model"), None),
            "MEMEVOLVE_EMBEDDING_AUTO_RESOLVE_MODELS": (("embedding", "auto_resolve_models"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            # Storage
            "MEMEVOLVE_STORAGE_BACKEND_TYPE": (("storage", "backend_type"), None),
            "MEMEVOLVE_STORAGE_PATH": (("storage", "path"), None),
            "MEMEVOLVE_STORAGE_VECTOR_DIM": (("storage", "vector_dim"), int),
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
            "MEMEVOLVE_ENABLE_EVOLUTION": (("evolution", "enable"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            # Logging
            "MEMEVOLVE_LOG_LEVEL": (("logging", "level"), None),
            "MEMEVOLVE_LOGGING_FORMAT": (("logging", "format"), None),
            "MEMEVOLVE_LOGGING_LOG_FILE": (("logging", "log_file"), None),
            "MEMEVOLVE_LOGGING_ENABLE_OPERATION_LOG": (("logging", "enable_operation_log"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_LOGGING_MAX_LOG_SIZE_MB": (("logging", "max_log_size_mb"), int),
            # API
            "MEMEVOLVE_API_ENABLE": (("api", "enable"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_API_HOST": (("api", "host"), None),
            "MEMEVOLVE_API_PORT": (("api", "port"), int),
            "MEMEVOLVE_UPSTREAM_BASE_URL": (("api", "upstream_base_url"), None),
            "MEMEVOLVE_UPSTREAM_API_KEY": (("api", "upstream_api_key"), None),
            "MEMEVOLVE_API_MEMORY_INTEGRATION": (("api", "memory_integration"), lambda x: x.lower() in ("true", "1", "yes", "on")),
            "MEMEVOLVE_API_MEMORY_RETRIEVAL_LIMIT": (("api", "memory_retrieval_limit"), int),
            # Project
            "MEMEVOLVE_PROJECT_NAME": (("project_name",), None),
            "MEMEVOLVE_PROJECT_ROOT": (("project_root",), None),
            "MEMEVOLVE_DATA_DIR": (("data_dir",), None),
            "MEMEVOLVE_CACHE_DIR": (("cache_dir",), None),
        }

        for env_var, (path_parts, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is None or value == "":
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
                              (LLMConfig, StorageConfig, RetrievalConfig,
                               ManagementConfig, EncoderConfig, EmbeddingConfig,
                               EvolutionConfig, LoggingConfig, APIConfig)):
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
            **kwargs: Configuration updates in dot notation (e.g., llm.base_url="...")
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
            key_path: Configuration path (e.g., "llm.base_url")
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
            assert isinstance(self.config.llm.base_url, str)
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
