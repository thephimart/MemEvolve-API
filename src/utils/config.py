"""Configuration management system for MemEvolve."""

import os
import json
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class LLMConfig:
    """LLM configuration."""
    base_url: str = field(default_factory=lambda: os.getenv("MEMEVOLVE_LLM_BASE_URL", "http://localhost:11434/v1"))
    api_key: str = field(default_factory=lambda: os.getenv("MEMEVOLVE_LLM_API_KEY", ""))
    model: Optional[str] = field(default_factory=lambda: os.getenv("MEMEVOLVE_LLM_MODEL"))
    timeout: int = 120
    max_retries: int = 3


@dataclass
class StorageConfig:
    """Storage backend configuration."""
    backend_type: str = "json"
    path: str = field(default_factory=lambda: os.getenv("MEMEVOLVE_STORAGE_PATH", "./data/memory"))
    vector_dim: int = 768
    index_type: str = "flat"


@dataclass
class RetrievalConfig:
    """Retrieval strategy configuration."""
    strategy_type: str = "hybrid"
    default_top_k: int = 5
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    enable_caching: bool = True
    cache_size: int = 1000


@dataclass
class ManagementConfig:
    """Memory management configuration."""
    enable_auto_management: bool = True
    auto_prune_threshold: int = 1000
    auto_consolidate_interval: int = 100
    deduplicate_threshold: float = 0.9
    forgetting_strategy: str = "lru"
    max_memory_age_days: int = 365


@dataclass
class EncoderConfig:
    """Experience encoder configuration."""
    encoding_strategies: list = field(
        default_factory=lambda: ["lesson", "skill"])
    enable_abstraction: bool = True
    abstraction_threshold: int = 10
    enable_tool_extraction: bool = True


@dataclass
class EvolutionConfig:
    """Evolution framework configuration."""
    population_size: int = 10
    generations: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.5
    selection_method: str = "pareto"
    tournament_size: int = 3


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = field(default_factory=lambda: os.getenv("MEMEVOLVE_LOG_LEVEL", "INFO"))
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    enable_operation_log: bool = True
    max_log_size_mb: int = 100


@dataclass
class MemEvolveConfig:
    """Main MemEvolve configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    management: ManagementConfig = field(default_factory=ManagementConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    project_name: str = "memevolve"
    project_root: str = field(default_factory=lambda: os.getenv("MEMEVOLVE_PROJECT_ROOT", "."))
    data_dir: str = field(default_factory=lambda: os.getenv("MEMEVOLVE_DATA_DIR", "./data"))
    cache_dir: str = field(default_factory=lambda: os.getenv("MEMEVOLVE_CACHE_DIR", "./cache"))


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
            "MEMEVOLVE_LLM_BASE_URL": ("llm", "base_url"),
            "MEMEVOLVE_LLM_API_KEY": ("llm", "api_key"),
            "MEMEVOLVE_LLM_MODEL": ("llm", "model"),
            "MEMEVOLVE_STORAGE_PATH": ("storage", "path"),
            "MEMEVOLVE_RETRIEVAL_TOP_K": ("retrieval", "default_top_k", int),
            "MEMEVOLVE_LOG_LEVEL": ("logging", "level"),
            "MEMEVOLVE_PROJECT_ROOT": ("project_root",),
        }

        for env_var, path_parts in env_mappings.items():
            value = os.getenv(env_var)
            if value is None:
                continue

            obj = self.config
            for part in path_parts[:-1]:
                obj = getattr(obj, part)

            if len(path_parts) == 2 and isinstance(path_parts[1], type):
                setattr(obj, path_parts[0], path_parts[1](value))
            else:
                setattr(obj, path_parts[-1], value)

    def _apply_config_dict(self, data: Dict[str, Any]):
        """Apply configuration dictionary to config object."""
        for section, section_config in data.items():
            if hasattr(self.config, section):
                section_obj = getattr(self.config, section)
                if isinstance(section_obj,
                              (LLMConfig, StorageConfig, RetrievalConfig,
                               ManagementConfig, EncoderConfig,
                               EvolutionConfig, LoggingConfig)):
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
            assert len(self.config.llm.base_url) > 0

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
